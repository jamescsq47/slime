#!/bin/bash
# BrowseComp-Plus GRPO — fully async, disaggregated, GB300, parameterized.
# Experiment matrix driver for docs/experiments/browsecomp-length-penalty.md.
#
#   MODEL_SIZE = 8B | 32B          (model args file + checkpoints)
#   MODE       = baseline | length_penalty
#
# Topology per cell: 1 train node (4 GPUs, TP=4) + 3 rollout nodes
# (12 GPUs, 3 sglang engines). Runs INSIDE the slime container on the
# TRAIN node after ray workers joined.
# Expects: MASTER_ADDR, LOCAL_SEARCH_URL, GRADER_*, WANDB_* exported.

set -ex

export PYTHONBUFFERED=16

MODEL_SIZE=${MODEL_SIZE:?"set MODEL_SIZE=8B|32B"}
MODE=${MODE:?"set MODE=baseline|length_penalty|length_penalty_global_ref|length_penalty_trunc"}
case "${MODEL_SIZE}" in 8B|32B) ;; *) echo "bad MODEL_SIZE"; exit 1;; esac
case "${MODE}" in baseline|length_penalty|length_penalty_global_ref|length_penalty_trunc) ;; *) echo "bad MODE"; exit 1;; esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
BASE=${BASE:-/data/home/syang}
FULLY_ASYNC_DIR=${REPO_DIR}/examples/fully_async
source "${REPO_DIR}/scripts/models/qwen3-${MODEL_SIZE}.sh"

NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-4}
TRAIN_NUM_NODES=1
TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_NUM_GPUS=$(( NUM_NODES * NUM_GPUS_PER_NODE - TRAIN_NUM_NODES * TRAIN_GPUS ))

HF_CHECKPOINT=${HF_CHECKPOINT:-${BASE}/models/Qwen3-${MODEL_SIZE}}
REF_LOAD=${REF_LOAD:-${BASE}/models/Qwen3-${MODEL_SIZE}_torch_dist}
DATA_DIR=${DATA_DIR:-${SCRIPT_DIR}/data}
SAVE_DIR=${SAVE_DIR:-${BASE}/ckpts/browsecomp_qwen3-${MODEL_SIZE}-${MODE}}
mkdir -p "${SAVE_DIR}"

export LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:?"export LOCAL_SEARCH_URL to the BrowseComp-Plus search server"}
: "${GRADER_API_KEY:?export GRADER_API_KEY (or set OPENAI_API_KEY) for the LLM judge}"
export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-100}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-2048}
export BROWSECOMP_MUST_SEARCH=${BROWSECOMP_MUST_SEARCH:-1}

# Both Qwen3-8B and Qwen3-32B have a 40960-token native context; keep the
# training/agent budget below it (README operational lessons).
SGLANG_CTX_LEN=40960
MAX_SEQ_LEN=$((SGLANG_CTX_LEN - 4096))

# ---------------------------------------------------------------------------
# Length-penalty cell configuration (doc: browsecomp-length-penalty.md)
# ---------------------------------------------------------------------------
MODEL_TAG=$(echo "qwen3-${MODEL_SIZE}" | tr '[:upper:]' '[:lower:]')
if [ "${MODE}" != "baseline" ]; then
   # Common length-penalty base config (identical across all penalty variants
   # so each variant ablates exactly one factor vs plain length_penalty).
   export BROWSECOMP_LENGTH_PENALTY_ENABLED=1
   export BROWSECOMP_LENGTH_PENALTY_SUCCESS_THRESHOLD=1.0
   export BROWSECOMP_LENGTH_PENALTY_BETA=0.10
   export BROWSECOMP_LENGTH_PENALTY_CAP=0.20
   export BROWSECOMP_LENGTH_PENALTY_REF_QUANTILE=0.25
   export BROWSECOMP_LENGTH_PENALTY_REL_SLACK=0.10
   export BROWSECOMP_LENGTH_PENALTY_ABS_SLACK=16.0
   export BROWSECOMP_LENGTH_PENALTY_SUCCESS_FLOOR=0.80
   export BROWSECOMP_LENGTH_PENALTY_REQUIRE_COMPLETED=1
   export BROWSECOMP_LENGTH_PENALTY_LOG_STATS=1
   PENALTY_TAG=true
   case "${MODE}" in
      length_penalty)
         RUN_NAME="browsecomp-b300-${MODEL_TAG}-grpo-length-penalty-beta0.10-cap0.20"
         ;;
      length_penalty_global_ref)
         # Ablation G: within-group reference -> cross-batch EMA reference.
         export BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF=1
         export BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF_ALPHA=0.1
         RUN_NAME="browsecomp-b300-${MODEL_TAG}-grpo-lp-globalref-alpha0.10"
         ;;
      length_penalty_trunc)
         # Ablation T: + flat penalty on budget-wasting failed rollouts.
         export BROWSECOMP_LENGTH_PENALTY_TRUNC_PENALTY=0.05
         RUN_NAME="browsecomp-b300-${MODEL_TAG}-grpo-lp-truncpen0.05"
         ;;
   esac
else
   export BROWSECOMP_LENGTH_PENALTY_ENABLED=0
   RUN_NAME="browsecomp-b300-${MODEL_TAG}-grpo-baseline"
   PENALTY_TAG=false
fi
export WANDB_TAGS="cluster=b300,model=${MODEL_TAG},mode=${MODE},length_penalty_enabled=${PENALTY_TAG}"

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --save ${SAVE_DIR}
   --save-interval 20
   --no-save-optim
   --no-save-rng
)
if [ "${RESUME:-0}" = "1" ]; then
   CKPT_ARGS+=(--load ${SAVE_DIR} --no-load-optim --no-load-rng)
fi

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout

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

   # In-training eval disabled on the async path except the step-0 baseline
   # (README operational lessons); evaluate checkpoints offline.
   --eval-interval ${EVAL_INTERVAL:-100000}
   --eval-prompt-data browsecomp ${DATA_DIR}/bc_test.jsonl
   --n-samples-per-eval-prompt 1
   --eval-temperature 0.6

   --global-batch-size 256
   --balance-data

   --max-weight-staleness 2
)
# The step-0 eval burst (150 concurrent sessions) racing the fully-async
# worker startup killed 686's RolloutManager with SYSTEM_ERROR; the base
# model's step-0 score is already known (~0.23), so skip it by default.
if [ "${SKIP_EVAL_BEFORE_TRAIN:-1}" = "1" ]; then
   ROLLOUT_ARGS+=(--skip-eval-before-train)
fi

SESSION_ARGS=(
   --pause-generation-mode in_place
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 40960
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT:-browsecomp-b300}
   --wandb-group ${RUN_NAME}
)
if [ -n "${WANDB_KEY:-}" ]; then
   WANDB_ARGS+=(--wandb-key ${WANDB_KEY})
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
   --sglang-context-length ${SGLANG_CTX_LEN}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path browsecomp_agent.generate
   --custom-rm-path browsecomp_rm.reward_func
)
if [ "${MODE}" != "baseline" ]; then
   CUSTOM_ARGS+=(--custom-reward-post-process-path length_reward.post_process_rewards)
fi

export MASTER_ADDR=${MASTER_ADDR:?"export MASTER_ADDR (train node routable IP)"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS_PER_NODE} --disable-usage-stats

EXPECTED_GPUS=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
for i in $(seq 1 120); do
   GPUS=$(python3 -c "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
   [ "${GPUS}" -ge "${EXPECTED_GPUS}" ] && break
   echo "waiting for ray workers: ${GPUS}/${EXPECTED_GPUS} GPUs"; sleep 10
done
[ "${GPUS}" -ge "${EXPECTED_GPUS}" ] || { echo "ray workers never joined"; exit 1; }

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
    \"BROWSECOMP_MAX_SEQ_LEN\": \"${MAX_SEQ_LEN}\",
    \"BROWSECOMP_LENGTH_PENALTY_ENABLED\": \"${BROWSECOMP_LENGTH_PENALTY_ENABLED}\",
    \"BROWSECOMP_LENGTH_PENALTY_SUCCESS_THRESHOLD\": \"${BROWSECOMP_LENGTH_PENALTY_SUCCESS_THRESHOLD:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_BETA\": \"${BROWSECOMP_LENGTH_PENALTY_BETA:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_CAP\": \"${BROWSECOMP_LENGTH_PENALTY_CAP:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_REF_QUANTILE\": \"${BROWSECOMP_LENGTH_PENALTY_REF_QUANTILE:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_REL_SLACK\": \"${BROWSECOMP_LENGTH_PENALTY_REL_SLACK:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_ABS_SLACK\": \"${BROWSECOMP_LENGTH_PENALTY_ABS_SLACK:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_SUCCESS_FLOOR\": \"${BROWSECOMP_LENGTH_PENALTY_SUCCESS_FLOOR:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_REQUIRE_COMPLETED\": \"${BROWSECOMP_LENGTH_PENALTY_REQUIRE_COMPLETED:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_LOG_STATS\": \"${BROWSECOMP_LENGTH_PENALTY_LOG_STATS:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF\": \"${BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF_ALPHA\": \"${BROWSECOMP_LENGTH_PENALTY_GLOBAL_REF_ALPHA:-}\",
    \"BROWSECOMP_LENGTH_PENALTY_TRUNC_PENALTY\": \"${BROWSECOMP_LENGTH_PENALTY_TRUNC_PENALTY:-}\",
    \"WANDB_TAGS\": \"${WANDB_TAGS}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"HOME\": \"${BASE}\"
  }
}"

cd "${REPO_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${TRAIN_NUM_NODES} \
   --actor-num-gpus-per-node ${TRAIN_GPUS} \
   --num-gpus-per-node ${NUM_GPUS_PER_NODE} \
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
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
