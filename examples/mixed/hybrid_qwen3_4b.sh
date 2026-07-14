#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

pkill -u $(whoami) -f "ray_csq" 2>/dev/null
sleep 1
rm -rf /tmp/ray_csq
mkdir -p /tmp/ray_csq
TEMP_DIR="/tmp/ray_csq"
set -e
ulimit -n 65536
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
# NVLS requires a compatible NVSwitch fabric; pairwise NVLink alone is not
# sufficient and can leave NCCL collectives spinning indefinitely.
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
echo "NCCL_NVLS_ENABLE: ${NCCL_NVLS_ENABLE} (detected ${NVLINK_COUNT} NVLink references)"
# This host has isolated NVLink pairs. Keep P2P on those links, but make NCCL
# use shared memory instead of direct P2P across PCIe host bridges.
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
echo "NCCL_P2P_LEVEL: ${NCCL_P2P_LEVEL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BROWSECOMP_DIR="$(cd -- "${SCRIPT_DIR}/../browsecomp" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-1.7B.sh"

MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
echo "=== Running mixed BrowseComp + Retool async benchmark: mode=${MODE} ==="

# BrowseComp environment — consumed by browsecomp_agent.py / browsecomp_rm.py.
export LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:?"export LOCAL_SEARCH_URL to the BrowseComp-Plus search server"}
if [ "${BROWSECOMP_EM_ONLY_REWARD:-0}" != "1" ]; then
   if [ -z "${GRADER_API_KEY:-${OPENAI_API_KEY:-}}" ]; then
      echo "export GRADER_API_KEY (or OPENAI_API_KEY) for the BrowseComp LLM judge"
      exit 1
   fi
fi
export GRADER_FALLBACK_MODEL=${GRADER_FALLBACK_MODEL:-${GRADER_MODEL:-}}
export GRADER_API_KEY=${GRADER_API_KEY:-${OPENAI_API_KEY:-}}
export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-100}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-2048}
export BROWSECOMP_MUST_SEARCH=${BROWSECOMP_MUST_SEARCH:-1}
export BROWSECOMP_ENABLE_THINKING=${BROWSECOMP_ENABLE_THINKING:-0}
export BROWSECOMP_SEARCH_MAX_TOPK=${BROWSECOMP_SEARCH_MAX_TOPK:-5}
export BROWSECOMP_SEARCH_SNIPPET_WORDS=${BROWSECOMP_SEARCH_SNIPPET_WORDS:-256}
export BROWSECOMP_OPEN_PAGE_WORDS=${BROWSECOMP_OPEN_PAGE_WORDS:-2048}

SGLANG_CTX_LEN=${SGLANG_CTX_LEN:-40960}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-2}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-20480}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-4096}
MIXED_RETOOL_MAX_RESPONSE_LEN=${MIXED_RETOOL_MAX_RESPONSE_LEN:-8192}
MIXED_BROWSECOMP_MAX_RESPONSE_LEN=${MIXED_BROWSECOMP_MAX_RESPONSE_LEN:-36864}
BROWSECOMP_MAX_SEQ_LEN=${BROWSECOMP_MAX_SEQ_LEN:-${MIXED_BROWSECOMP_MAX_RESPONSE_LEN}}

CKPT_ARGS=(
   --hf-checkpoint /workspace/Qwen3-1.7B
   #--hf-checkpoint /root/Qwen3-1.7B-FP8
   --ref-load /workspace/Qwen3-1.7B_torch_dist
   # --load /root/Qwen3-1.7B_slime/
#    --save /workspace/Qwen3-1.7B_async_math/
#    --save-interval 100
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project mixed-qwen3-1.7b-sync
   --wandb-group qwen3-1.7B-browsecomp-retool-async-0.5-51200-51200
)

# Let wandb read the credential from its standard environment variable. Passing
# --wandb-key would expose it in the Ray entrypoint and argument dump.
export WANDB_API_KEY=${WANDB_KEY:-${WANDB_API_KEY:-}}

PROMPT_SET=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl
BROWSECOMP_DATA_DIR=${BROWSECOMP_DATA_DIR:-/workspace/data/browsecomp}

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout ${NUM_ROLLOUT:-500}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
   --rollout-max-response-len ${MIXED_BROWSECOMP_MAX_RESPONSE_LEN}
   --rollout-temperature 1

   --global-batch-size ${GLOBAL_BATCH_SIZE:-256}
   --num-steps-per-rollout 1
   --balance-data
   --rollout-health-check-interval 10
   --rollout-health-check-timeout 10
   --save-debug-rollout-data /workspace/slime/examples/mixed/debug/test/rollout_{rollout_id}.pt
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-2}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1 # 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
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
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
   --sglang-context-length ${SGLANG_CTX_LEN}
)

CUSTOM_ARGS=(
   --data-source-path custom_data_source.CustomDataSource
   --custom-generate-function-path generate_with_hybrid.generate_unified
   --custom-rm-path generate_with_hybrid.reward_func_unified
   --math-data-path /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
   --qa-data-path ${BROWSECOMP_DATA_DIR}/bc_train.jsonl
   --math-ratio 0.5
   --mask-offpolicy-math 51200
   --mask-offpolicy-qa 51200
   # --batch-alternation \
   # --math-batches-per-cycle 100 \
   # --qa-batches-per-cycle 0 \
#    --dynamic-alternation \
#    --lag-version 3 \
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

# launch the master node of ray in container

# single node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
ray start --head --node-ip-address ${MASTER_ADDR} --temp-dir ${TEMP_DIR} --dashboard-port ${DASHBOARD_PORT} --num-gpus 8 --num-cpus 64 --disable-usage-stats

# Wait for the Ray dashboard and job agent to be fully ready
echo "Waiting for Ray agent to initialize..."
for i in {1..30}; do
  if curl -s http://${MASTER_ADDR}:${DASHBOARD_PORT}/api/jobs/ 1>/dev/null; then
    echo "Ray agent is up!"
    break
  fi
  sleep 1
done

#multi-node
# a10 main
# export MASTER_ADDR="10.0.1.170"
# ray start --head \
#     --node-ip-address=${MASTER_ADDR} \
#     --port=6379 \
#     --num-gpus 8 --num-cpus 64 --disable-usage-stats
# a11 worker 
# export MASTER_ADDR="10.0.1.170"
# export MY_NODE_IP=$(hostname -I | awk '{print $1}')
# ray start --address="${MASTER_ADDR}:6379" \
#           --node-ip-address="${MY_NODE_IP}" \
#           --num-gpus 8 \
#           --num-cpus 64 \
#           --disable-usage-stats


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${BROWSECOMP_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
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
    \"BROWSECOMP_MAX_SEQ_LEN\": \"${BROWSECOMP_MAX_SEQ_LEN}\",
    \"MIXED_RETOOL_MAX_RESPONSE_LEN\": \"${MIXED_RETOOL_MAX_RESPONSE_LEN}\",
    \"MIXED_BROWSECOMP_MAX_RESPONSE_LEN\": \"${MIXED_BROWSECOMP_MAX_RESPONSE_LEN}\",
    \"BROWSECOMP_EM_ONLY_REWARD\": \"${BROWSECOMP_EM_ONLY_REWARD:-0}\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\",
    \"NCCL_P2P_LEVEL\": \"${NCCL_P2P_LEVEL}\"
  }
}"

MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
FULLY_ASYNC_MAX_PARTIAL_SPAN=${FULLY_ASYNC_MAX_PARTIAL_SPAN:-3}
echo "=== Running mixed BrowseComp + Retool async benchmark: mode=${MODE} ==="

# --- Mode-specific flags ---
MODE_ARGS=()
case "${MODE}" in
    one_step_off)
        ;;
    fully_async)
        MODE_ARGS+=(
            --fully-async-debug-version-tracking
        )
        ;;
    window_partial)
        MODE_ARGS+=(
            --fully-async-debug-version-tracking
            --fully-async-buffer-policy window_evict
            --fully-async-version-window "${FULLY_ASYNC_VERSION_WINDOW}"
            --fully-async-max-completed-samples "${FULLY_ASYNC_MAX_COMPLETED_SAMPLES}"
            --fully-async-eviction-policy "${FULLY_ASYNC_EVICTION_POLICY}"
            --partial-rollout
            # --mask-offpolicy-in-partial-rollout
            --fully-async-max-partial-span "${FULLY_ASYNC_MAX_PARTIAL_SPAN}"
        )
        ;;
    staleness_partial)
        MODE_ARGS+=(
            --fully-async-debug-version-tracking
            --fully-async-buffer-policy legacy_backpressure
            --staleness-threshold 0.5
            --partial-rollout
            # --mask-offpolicy-in-partial-rollout
        )
        ;;
    *)
        echo "Unknown MODE: ${MODE}. Use one of: one_step_off, fully_async, window_partial, staleness_partial"
        exit 1
        ;;
esac

ray job submit --address="http://${MASTER_ADDR}:${DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${MODE_ARGS[@]}
