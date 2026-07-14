#!/bin/bash
# BrowseComp-Plus RL training (GRPO) for Qwen3-8B.
#
# Prerequisites (see README.md):
#   1. BrowseComp-Plus search server running; export LOCAL_SEARCH_URL.
#   2. LLM judge endpoint; export GRADER_API_KEY (+ GRADER_BASE_URL / GRADER_MODEL).
#   3. Data prepared with prepare_data.py into ${DATA_DIR}.
#   4. Qwen3-8B HF checkpoint + torch_dist conversion (see README.md).

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"

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
: "${GRADER_API_KEY:?export GRADER_API_KEY (or set OPENAI_API_KEY) for the LLM judge}"
export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-100}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-2048}
export BROWSECOMP_MUST_SEARCH=${BROWSECOMP_MUST_SEARCH:-1}

# prompt budget (system + user question) / total session budget in tokens,
# matching the FoldAgent reproduction (8192 + 32768).
# sglang context must not exceed the model's native max (40960 for Qwen3-8B);
# keep the training/agent budget BELOW it so a large final-turn observation
# doesn't overflow mid-rollout (see README "operational lessons").
SGLANG_CTX_LEN=40960
MAX_SEQ_LEN=$((SGLANG_CTX_LEN - 4096))

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   # --load /root/Qwen3-8B_miles/
   # --save /root/Qwen3-8B_miles/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/bc_train.jsonl
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1

   --eval-interval 10
   --eval-prompt-data browsecomp ${DATA_DIR}/bc_test.jsonl
   --n-samples-per-eval-prompt 1
   --eval-temperature 0.6

   --global-batch-size 256
   --balance-data
)

SESSION_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group browsecomp_qwen3-8B
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
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

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
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
    \"BROWSECOMP_MAX_SEQ_LEN\": \"${MAX_SEQ_LEN}\"
  }
}"

cd "${REPO_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${SESSION_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
