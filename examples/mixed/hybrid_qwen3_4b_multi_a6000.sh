#!/bin/bash
ulimit -n 65536
# for rerun the task - 多节点环境下不要执行这些停止命令！
# 如果需要重启，请在各个节点上手动执行清理
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# sleep 3
# pkill -9 ray

set -euo pipefail

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# A6000 集群网络：l0/l7/l8 的 IB 网卡均为 ibs8，HCA 为 mlx5_0。
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ibs8}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0}
# Scope the metadata-collective workaround to this launcher. Other slime
# launchers keep the historical NCCL path by default.
export SLIME_DYNAMIC_BATCH_SYNC_BACKEND=${SLIME_DYNAMIC_BATCH_SYNC_BACKEND:-gloo_world}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BROWSECOMP_DIR="$(cd -- "${SCRIPT_DIR}/../browsecomp" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
FULLY_ASYNC_MAX_PARTIAL_SPAN=${FULLY_ASYNC_MAX_PARTIAL_SPAN:-3}
echo "=== Running mixed BrowseComp + Retool async benchmark: mode=${MODE}==="

# BrowseComp environment — consumed by browsecomp_agent.py / browsecomp_rm.py.
export LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:?"export LOCAL_SEARCH_URL to the BrowseComp-Plus search server"}
if [ "${BROWSECOMP_EM_ONLY_REWARD:-0}" != "1" ]; then
   if [ -z "${GRADER_API_KEY:-${OPENAI_API_KEY:-}}" ]; then
      echo "export GRADER_API_KEY (or OPENAI_API_KEY) for the BrowseComp LLM judge"
      exit 1
   fi
fi
export GRADER_FALLBACK_MODEL=${GRADER_FALLBACK_MODEL:-${GRADER_MODEL:-}}
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
MIXED_RETOOL_MAX_RESPONSE_LEN=${MIXED_RETOOL_MAX_RESPONSE_LEN:-8192}
MIXED_BROWSECOMP_MAX_RESPONSE_LEN=${MIXED_BROWSECOMP_MAX_RESPONSE_LEN:-36864}
BROWSECOMP_MAX_SEQ_LEN=${BROWSECOMP_MAX_SEQ_LEN:-${MIXED_BROWSECOMP_MAX_RESPONSE_LEN}}
SAVE_PATH=${SAVE_PATH:-/workspace/Qwen3-4B-mixed-browsecomp-retool0.5-a6000/}

# With exactly l0/l7/l8 in Ray, placement bundles are sorted by IP:
#   actor (16 GPUs): l0 10.0.1.150 + l7 10.0.1.157
#   rollout (8 GPUs): l8 10.0.1.158
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}
TRAIN_NODE_IPS=${TRAIN_NODE_IPS:-"10.0.1.150,10.0.1.157"}
ROLLOUT_NODE_IP=${ROLLOUT_NODE_IP:-"10.0.1.158"}
# l0 currently advertises Ray GPUs but its container cannot initialize CUDA.
# Pin the lightweight job driver to l7; the placement group still assigns
# actor GPUs to l0+l7 and rollout GPUs to l8.
TRAIN_DRIVER_NODE_IP=${TRAIN_DRIVER_NODE_IP:-"10.0.1.157"}

CKPT_ARGS=(
   --hf-checkpoint /workspace/Qwen3-4B
   --ref-load /workspace/Qwen3-4B_torch_dist
   # --load /root/Qwen3-4B_slime/
   --save ${SAVE_PATH}
   --save-interval 100
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project mixed-qwen3-4b-a6000
   --wandb-group qwen3-4B-browsecomp-retool-async-a6000
)
PROMPT_SET=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl
BROWSECOMP_DATA_DIR=${BROWSECOMP_DATA_DIR:-/workspace/data/browsecomp}

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout ${NUM_ROLLOUT:-800}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
   --rollout-max-response-len ${MIXED_BROWSECOMP_MAX_RESPONSE_LEN}
   --rollout-temperature 1

   --global-batch-size ${GLOBAL_BATCH_SIZE:-256}
   --num-steps-per-rollout 1
   --balance-data
   --rollout-health-check-interval 30
   --rollout-health-check-timeout 30
   # --use-rollout-logprobs 
   # --save-debug-rollout-data /workspace/slime/examples/hybrid/debug/ratio_0.5_51200_51200/rollout_{rollout_id}.pt
)

PERF_ARGS=(
   # A6000 topology is four NVLink pairs per node. TP=2 stays inside each
   # NVLink pair; CP=2 then communicates between pairs within one NUMA node.
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
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   # --kl-loss-coef 0.00
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
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.5}
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
   --count-aware-alternation
   --math-batches-per-cycle 16
   --qa-batches-per-cycle 16
   # --batch-alternation 
   # --mask-offpolicy-in-partial-rollout 
#    --dynamic-alternation
#    --dynamic-alternation-alpha 1 # lag-based ratio weight; final=(1-alpha)*math-ratio + alpha*lag-ratio
#    --dynamic-alternation-warmup-steps 5  # use fixed math-ratio for first 5 policy versions
#    --dynamic-alternation-min-math-ratio 0.2
#    --dynamic-alternation-max-math-ratio 0.8
#    --enable-tool-delay
#    --tool-delay-mean 25
#    --tool-delay-variance 500
#    --tool-delay-check-interval 0.5
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

# 多节点配置 - 假设Ray已经在head节点启动
# 根据head节点启动脚本的配置
export MASTER_ADDR="${MASTER_ADDR:-10.0.1.150}"
export RAY_PORT="${RAY_PORT:-6379}"

# 检查Ray集群状态 (ray status 需要指定 redis 端口而不是 http 控制台端口)
echo "Checking Ray cluster status at ${MASTER_ADDR}:${RAY_PORT}..."
export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
ray status || echo "Warning: Ray cluster may not be running properly"

# The PACK placement policy is deterministic only when these are the exact
# live GPU nodes. Fail early instead of silently putting inference on a
# training node.
export TRAIN_NODE_IPS ROLLOUT_NODE_IP ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
python - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
train_nodes = set(os.environ["TRAIN_NODE_IPS"].split(","))
rollout_node = os.environ["ROLLOUT_NODE_IP"]
expected = train_nodes | {rollout_node}
live_gpu_nodes = {
    node["NodeManagerAddress"]: int(node["Resources"].get("GPU", 0))
    for node in ray.nodes()
    if node["Alive"] and node["Resources"].get("GPU", 0) > 0
}
ray.shutdown()

if set(live_gpu_nodes) != expected:
    raise SystemExit(
        "Ray topology mismatch: expected GPU nodes "
        f"{sorted(expected)}, got {live_gpu_nodes}. "
        "Use exactly l0, l7, and l8 for deterministic placement."
    )
actor_gpus = int(os.environ["ACTOR_GPUS_PER_NODE"])
rollout_gpus = int(os.environ["ROLLOUT_NUM_GPUS"])
if any(live_gpu_nodes[ip] != actor_gpus for ip in train_nodes):
    raise SystemExit(
        f"Training nodes must expose exactly {actor_gpus} GPUs each, got {live_gpu_nodes}"
    )
if live_gpu_nodes[rollout_node] != rollout_gpus:
    raise SystemExit(
        f"Inference node {rollout_node} must expose exactly {rollout_gpus} Ray GPUs, "
        f"got {live_gpu_nodes}"
    )

print(
    "Validated topology: training on "
    f"{sorted(train_nodes)}; inference on {rollout_node} ({rollout_gpus} GPUs)."
)
PY

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${BROWSECOMP_DIR}\",
    \"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"WANDB_API_KEY\": \"${WANDB_KEY:-${WANDB_API_KEY:-}}\",
    \"LOCAL_SEARCH_URL\": \"${LOCAL_SEARCH_URL}\",
    \"GRADER_API_KEY\": \"${GRADER_API_KEY:-${OPENAI_API_KEY:-}}\",
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
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE}\",
    \"NCCL_IB_HCA\": \"${NCCL_IB_HCA}\",
    \"SLIME_DYNAMIC_BATCH_SYNC_BACKEND\": \"${SLIME_DYNAMIC_BATCH_SYNC_BACKEND}\"
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

# 提交作业到Ray集群
# RAY_DASHBOARD_PORT 必须和 head 节点 ray start --dashboard-port 保持一致。
ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --entrypoint-resources="{\"node:${TRAIN_DRIVER_NODE_IP}\": 0.001}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /workspace/slime/train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} \
   --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
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
