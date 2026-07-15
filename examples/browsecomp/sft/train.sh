#!/bin/bash
# Full-parameter SFT of Qwen3-8B on filtered BrowseComp trajectories.
set -euo pipefail

ulimit -n 65536

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
source "${REPO_DIR}/scripts/models/qwen3-8B.sh"

HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/Qwen3-8B}
REF_LOAD=${REF_LOAD:-/workspace/Qwen3-8B_torch_dist}
SFT_DATA=${SFT_DATA:-/workspace/data/browsecomp/browsecomp_qwen3_8b_sft.jsonl}
SAVE_PATH=${SAVE_PATH:-/workspace/Qwen3-8B-browsecomp-sft}
NUM_GPUS=${NUM_GPUS:-8}

[[ -s "${SFT_DATA}" ]] || { echo "Missing or empty SFT_DATA: ${SFT_DATA}" >&2; exit 2; }
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No usable NVIDIA GPU is visible (nvidia-smi -L failed)." >&2
  exit 2
fi

# A slime/Megatron save directory can be resumed directly. This matters for a
# 100-epoch job: preserve the dataset cursor, optimizer, scheduler and rollout
# id instead of silently restarting from the base checkpoint.
LOAD_ARGS=()
if [[ -s "${SAVE_PATH}/latest_checkpointed_iteration.txt" ]]; then
  LOAD_ARGS=(--load "${SAVE_PATH}")
  echo "Resuming SFT from ${SAVE_PATH} at iteration $(<"${SAVE_PATH}/latest_checkpointed_iteration.txt")"
fi

GPU_MEMORY_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -oE 'NV[0-9]+' | wc -l || true)

# Safe long-context defaults.  On 80GB NVLink/NVSwitch, TP=1 gives more DP
# replicas.  48GB or PCIe-only hosts use TP=2 to reduce model-state pressure.
if [[ -n "${GPU_MEMORY_MIB}" && "${GPU_MEMORY_MIB}" -ge 70000 && "${NVLINK_COUNT}" -gt 0 ]]; then
  DEFAULT_TP=1
else
  DEFAULT_TP=2
fi
TP=${TP:-${DEFAULT_TP}}
CP=${CP:-1}
PP=${PP:-1}
MODEL_PARALLEL=$((TP * CP * PP))
if (( NUM_GPUS % MODEL_PARALLEL != 0 )); then
  echo "NUM_GPUS=${NUM_GPUS} must be divisible by TP*CP*PP=${MODEL_PARALLEL}" >&2
  exit 2
fi
DP=$((NUM_GPUS / MODEL_PARALLEL))

# CP is disabled because FlashAttention 2.8's variable-length CP path hangs on
# this Ampere host.  The longest retained sample is 31,717 tokens, so leave a
# small full-sequence packing margin.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}
SFT_MAX_SEQ_LEN=${SFT_MAX_SEQ_LEN:-12288}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}
NUM_EPOCH=${NUM_EPOCH:-100}

echo "Qwen3-8B SFT topology: GPUs=${NUM_GPUS} memory=${GPU_MEMORY_MIB:-unknown}MiB NVLinkRefs=${NVLINK_COUNT} TP=${TP} CP=${CP} PP=${PP} DP=${DP}"
echo "SFT data=${SFT_DATA} epochs=${NUM_EPOCH} save=${SAVE_PATH}"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --dashboard-port "${DASHBOARD_PORT}" --disable-usage-stats

HAS_NVLINK=0
[[ "${NVLINK_COUNT}" -gt 0 ]] && HAS_NVLINK=1
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"SLIME_DYNAMIC_BATCH_SYNC_BACKEND\":\"gloo_dp\",\"SFT_MAX_SEQ_LEN\":\"${SFT_MAX_SEQ_LEN}\",\"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\":\"3600\"}}"

cd "${REPO_DIR}"
ray job submit --address="http://${MASTER_ADDR}:${DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" -- \
  python3 train_async.py \
    --actor-num-nodes 1 --actor-num-gpus-per-node "${NUM_GPUS}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_LOAD}" \
    "${LOAD_ARGS[@]}" \
    --save "${SAVE_PATH}" --save-interval 100 \
    --rollout-function-path examples.browsecomp.sft.sft_rollout.generate_rollout \
    --prompt-data "${SFT_DATA}" --input-key messages --rollout-shuffle \
    --num-epoch "${NUM_EPOCH}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --loss-type custom_loss \
    --custom-loss-function-path examples.browsecomp.sft.sft_loss.sft_masked_loss \
    --calculate-per-token-loss \
    --log-probs-chunk-size 512 \
    --disable-compute-advantages-and-returns --debug-train-only \
    --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}" \
    --context-parallel-size "${CP}" --sequence-parallel \
    --distributed-timeout-minutes 30 \
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer \
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --optimizer adam --lr 1e-6 --min-lr 1e-7 --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 --weight-decay 0.05 \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash
