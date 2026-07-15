#!/bin/bash
# Generate scored BrowseComp train trajectories only; no Megatron training.
set -euo pipefail

ulimit -n 65536

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
BROWSECOMP_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${REPO_DIR}/scripts/models/qwen3-8B.sh"

HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/Qwen3-8B}
DATA_DIR=${DATA_DIR:-/workspace/data/browsecomp}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_DIR}/qwen3_8b_sft_rollouts}
# The vendored search server lands on physical GPU0 even when launched with a
# remapped CUDA_VISIBLE_DEVICES. Reserve GPU0 and use seven TP=1 rollout
# replicas on physical GPU1-7.
GENERATION_GPUS=${GENERATION_GPUS:-1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GENERATION_GPUS}}
IFS=',' read -ra VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${NUM_GPUS:-${#VISIBLE_GPU_LIST[@]}}
N_SAMPLES=${N_SAMPLES:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-34}
# 680 train questions / 34 prompts per batch = exactly 20 rounds.
NUM_ROLLOUT=${NUM_ROLLOUT:-20}

: "${LOCAL_SEARCH_URL:?export LOCAL_SEARCH_URL to the BrowseComp-Plus search server}"
if [[ "${BROWSECOMP_EM_ONLY_REWARD:-0}" != "1" ]]; then
  if [[ -z "${GRADER_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
    echo "export GRADER_API_KEY or OPENAI_API_KEY" >&2
    exit 2
  fi
fi

# Qwen3-8B BF16 plus a full 40K KV cache is tight on 48GB.  Use TP=2 there;
# 80GB GPUs can run eight independent TP=1 engines for higher throughput.
GPU_MEMORY_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
if (( NUM_GPUS % 2 == 1 )); then
  DEFAULT_INFER_TP=1
elif [[ -n "${GPU_MEMORY_MIB}" && "${GPU_MEMORY_MIB}" -ge 70000 ]]; then
  DEFAULT_INFER_TP=1
else
  DEFAULT_INFER_TP=2
fi
INFER_TP=${INFER_TP:-${DEFAULT_INFER_TP}}
if (( NUM_GPUS % INFER_TP != 0 )); then
  echo "NUM_GPUS=${NUM_GPUS} must be divisible by INFER_TP=${INFER_TP}" >&2
  exit 2
fi

export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-60}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-1536}
export BROWSECOMP_MUST_SEARCH=1
export BROWSECOMP_JUDGE_CONSENSUS=${BROWSECOMP_JUDGE_CONSENSUS:-1}
export BROWSECOMP_ENABLE_THINKING=${BROWSECOMP_ENABLE_THINKING:-0}
export BROWSECOMP_SEARCH_MAX_TOPK=${BROWSECOMP_SEARCH_MAX_TOPK:-5}
export BROWSECOMP_SEARCH_SNIPPET_WORDS=${BROWSECOMP_SEARCH_SNIPPET_WORDS:-256}
export BROWSECOMP_OPEN_PAGE_WORDS=${BROWSECOMP_OPEN_PAGE_WORDS:-2048}
export BROWSECOMP_MAX_SEQ_LEN=${BROWSECOMP_MAX_SEQ_LEN:-36864}

mkdir -p "${OUTPUT_DIR}"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_browsecomp_sft}

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --temp-dir "${RAY_TEMP_DIR}" --dashboard-port "${DASHBOARD_PORT}" --disable-usage-stats

RUNTIME_ENV_JSON=$(RUNTIME_PYTHONPATH="/root/Megatron-LM:${SCRIPT_DIR}:${BROWSECOMP_DIR}" python3 -c 'import json, os; print(json.dumps({"env_vars": {k: v for k, v in os.environ.items() if k.startswith("BROWSECOMP_") or k in {"LOCAL_SEARCH_URL", "GRADER_API_KEY", "OPENAI_API_KEY", "GRADER_BASE_URL", "GRADER_MODEL", "GRADER_FALLBACK_MODEL", "GRADER_API_VERSION"}} | {"PYTHONPATH": os.environ["RUNTIME_PYTHONPATH"], "CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}))')

cd "${REPO_DIR}"
ray job submit --address="http://${MASTER_ADDR}:${DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" -- \
  python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --rollout-num-gpus "${NUM_GPUS}" \
    --debug-rollout-only \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --prompt-data "${DATA_DIR}/bc_train.jsonl" \
    --input-key prompt --label-key label --metadata-key metadata \
    --rollout-shuffle --rollout-seed 47 \
    --num-rollout "${NUM_ROLLOUT}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES}" \
    --rollout-max-response-len 36864 \
    --rollout-temperature 0.8 --rollout-top-p 0.95 \
    --global-batch-size $((ROLLOUT_BATCH_SIZE * N_SAMPLES)) \
    --save-debug-rollout-data "${OUTPUT_DIR}/rollout_{rollout_id}.pt" \
    --rollout-num-gpus-per-engine "${INFER_TP}" \
    --sglang-mem-fraction-static 0.72 \
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-2}" \
    --sglang-context-length 40960 \
    --custom-generate-function-path sft_agent.generate \
    --custom-rm-path sft_rm.reward_func

echo "Rollouts saved under ${OUTPUT_DIR}"
echo "Export with: python examples/browsecomp/sft/export_sft.py --input '${OUTPUT_DIR}/*.pt' --output '${DATA_DIR}/browsecomp_qwen3_8b_sft.jsonl'"
