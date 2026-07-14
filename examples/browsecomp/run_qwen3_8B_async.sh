#!/bin/bash
# BrowseComp-Plus RL training (GRPO) for Qwen3-8B — fully async, disaggregated.
#
# Topology: 1 training node + N rollout nodes (default 3), i.e. 8 training
# GPUs + 24 inference GPUs on 4x8-GPU nodes. Rollout generation runs in a
# persistent background worker (examples/fully_async/fully_async_rollout.py)
# so long, uneven BrowseComp trajectories never block the training step;
# training drains completed groups. TIS corrects for the off-policy gap.
#
# Multi-node launch: start ray on every node, then run this script on the
# head node only.
#   head node:    MASTER_ADDR=<head-ip> NUM_NODES=4 bash run_qwen3_8B_async.sh
#   worker nodes: ray start --address=<head-ip>:6379 --num-gpus 8
# (On a single 8-GPU node for debugging: NUM_NODES=1 TRAIN_GPUS=4 works too —
# rollout gets the remaining 4 GPUs.)
#
# Same prerequisites as run_qwen3_8B.sh (search server, judge, data, ckpt).
ulimit -n 65536
set -ex

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force || true
pkill -9 ray 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true

# Use a fresh Ray temp dir per run. Reusing a fixed directory can leave a stale
# GCS/session value behind and trigger Ray session mismatch on the next start.
TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_csq_${USER:-user}_$$}"
mkdir -p "${TEMP_DIR}"

cleanup() {
   ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
FULLY_ASYNC_DIR="$(cd -- "${SCRIPT_DIR}/../fully_async" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"
# ---------------------------------------------------------------------------
# Topology — override via environment.
# ---------------------------------------------------------------------------
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
TRAIN_NUM_NODES=${TRAIN_NUM_NODES:-1}
TRAIN_GPUS=${TRAIN_GPUS:-${NUM_GPUS_PER_NODE}}
ROLLOUT_NUM_GPUS=$(( NUM_NODES * NUM_GPUS_PER_NODE - TRAIN_NUM_NODES * TRAIN_GPUS ))

# ---------------------------------------------------------------------------
# Paths — override via environment.
# ---------------------------------------------------------------------------
HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/Qwen3-8B}
REF_LOAD=${REF_LOAD:-/workspace/Qwen3-8B_torch_dist}
DATA_DIR=${DATA_DIR:-/workspace/data/browsecomp}

# ---------------------------------------------------------------------------
# BrowseComp environment — consumed by browsecomp_agent.py / browsecomp_rm.py.
# ---------------------------------------------------------------------------
export LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:?"export LOCAL_SEARCH_URL to the BrowseComp-Plus search server"}
if [ "${BROWSECOMP_EM_ONLY_REWARD:-0}" != "1" ]; then
   : "${GRADER_API_KEY:?export GRADER_API_KEY (or set OPENAI_API_KEY) for the LLM judge}"
fi
export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-100}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-2048}
export BROWSECOMP_MUST_SEARCH=${BROWSECOMP_MUST_SEARCH:-1}
export BROWSECOMP_ENABLE_THINKING=${BROWSECOMP_ENABLE_THINKING:-0}
export BROWSECOMP_SEARCH_MAX_TOPK=${BROWSECOMP_SEARCH_MAX_TOPK:-5}
export BROWSECOMP_SEARCH_SNIPPET_WORDS=${BROWSECOMP_SEARCH_SNIPPET_WORDS:-256}
export BROWSECOMP_OPEN_PAGE_WORDS=${BROWSECOMP_OPEN_PAGE_WORDS:-2048}

# sglang context must not exceed the model's native max (40960 for Qwen3-8B);
# keep the training/agent budget BELOW it so a large final-turn observation
# doesn't overflow mid-rollout (see README "operational lessons").
SGLANG_CTX_LEN=${SGLANG_CTX_LEN:-40960}
MAX_SEQ_LEN=$((SGLANG_CTX_LEN - 4096))

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   # --load /root/Qwen3-8B_miles/
   # --save /root/Qwen3-8B_miles/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   # Fully-async worker has no eval mode; run eval through the standard
   # synchronous path instead.
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout

   --prompt-data ${DATA_DIR}/bc_train.jsonl
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT:-3000}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-32768}
   --rollout-temperature ${ROLLOUT_TEMPERATURE:-1}

   --eval-interval ${EVAL_INTERVAL:-10}
   --eval-prompt-data browsecomp ${DATA_DIR}/bc_test.jsonl
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT:-1}
   --eval-temperature ${EVAL_TEMPERATURE:-0.6}

   --global-batch-size ${GLOBAL_BATCH_SIZE:-256}
   --balance-data

   ${SKIP_EVAL_BEFORE_TRAIN:+--skip-eval-before-train}

   # staleness control for fully async: recycle groups older than 2 versions
   --max-weight-staleness ${MAX_WEIGHT_STALENESS:-2}
)

SESSION_ARGS=(
   # Freeze in-flight requests during weight updates (the default 'retract'
   # + flush never drains under the fully-async continuous request stream).
   --pause-generation-mode in_place

)

PERF_ARGS=(
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-4}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE:-2}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-20480}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # async rollouts are off-policy; TIS uses rollout logprobs to correct the mismatch
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-dev}"
      --wandb-group "${WANDB_GROUP:-browsecomp_qwen3-8B-async}"
   )
   if [ -n "${WANDB_KEY:-}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.8}
   --sglang-context-length ${SGLANG_CTX_LEN}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path browsecomp_agent.generate
   --custom-rm-path browsecomp_rm.reward_func
)

# launch the master node of ray in container (worker nodes must already have
# joined via `ray start --address=${MASTER_ADDR}:6379 --num-gpus 8`)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --temp-dir ${TEMP_DIR} --num-gpus ${NUM_GPUS_PER_NODE} --num-cpus 64 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${FULLY_ASYNC_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"GRADER_API_KEY\": \"${GRADER_API_KEY:-${OPENAI_API_KEY}}\",
    \"GRADER_BASE_URL\": \"${GRADER_BASE_URL:-}\",
    \"GRADER_MODEL\": \"${GRADER_MODEL:-}\",
    \"GRADER_FALLBACK_MODEL\": \"${GRADER_FALLBACK_MODEL:-}\",
    \"GRADER_API_VERSION\": \"${GRADER_API_VERSION:-}\",
    \"BROWSECOMP_MAX_TURNS\": \"${BROWSECOMP_MAX_TURNS}\",
    \"BROWSECOMP_TURN_MAX_NEW_TOKENS\": \"${BROWSECOMP_TURN_MAX_NEW_TOKENS}\",
    \"BROWSECOMP_MUST_SEARCH\": \"${BROWSECOMP_MUST_SEARCH}\",
    \"BROWSECOMP_ENABLE_THINKING\": \"${BROWSECOMP_ENABLE_THINKING}\",
    \"BROWSECOMP_SEARCH_MAX_TOPK\": \"${BROWSECOMP_SEARCH_MAX_TOPK}\",
    \"BROWSECOMP_SEARCH_SNIPPET_WORDS\": \"${BROWSECOMP_SEARCH_SNIPPET_WORDS}\",
    \"BROWSECOMP_OPEN_PAGE_WORDS\": \"${BROWSECOMP_OPEN_PAGE_WORDS}\",
    \"BROWSECOMP_MAX_SEQ_LEN\": \"${MAX_SEQ_LEN}\",
    \"BROWSECOMP_EM_ONLY_REWARD\": \"${BROWSECOMP_EM_ONLY_REWARD:-0}\"
  }
}"

cd "${REPO_DIR}"
TRAIN_ARGS=(
   --actor-num-nodes ${TRAIN_NUM_NODES}
   --actor-num-gpus-per-node ${TRAIN_GPUS}
   --num-gpus-per-node ${NUM_GPUS_PER_NODE}
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${SESSION_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${CUSTOM_ARGS[@]}"
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py "${TRAIN_ARGS[@]}"
