#!/bin/bash
# Sample correct Qwen3-8B DAPO-math trajectories using mixed-RL generation/RM.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
MIXED_UTILS="${REPO_DIR}/examples/browsecomp/mixed_sft"
source "${REPO_DIR}/scripts/models/qwen3-8B.sh"

HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/Qwen3-8B}
REF_LOAD=${REF_LOAD:-/workspace/Qwen3-8B_torch_dist}
MATH_DATA=${MATH_DATA:-/workspace/data/dapo-math-17k/dapo-math-17k.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/data/dapo-math-17k/qwen3_8b_correct_rollouts}
NUM_ROLLOUT=${NUM_ROLLOUT:-3}

[[ -s "${MATH_DATA}" ]] || { echo "Missing MATH_DATA: ${MATH_DATA}" >&2; exit 2; }
nvidia-smi -L >/dev/null || { echo "No usable GPU visible." >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --dashboard-port 8265 --num-gpus 8 --num-cpus 64 --disable-usage-stats

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM:${REPO_DIR}/examples/mixed\",\"MIXED_RETOOL_MAX_RESPONSE_LEN\":\"8192\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}"
cd "${REPO_DIR}"
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json="${RUNTIME_ENV_JSON}" -- \
  python3 train.py --debug-rollout-only --rollout-num-gpus 8 \
  "${MODEL_ARGS[@]}" --hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_LOAD}" \
  --prompt-data "${MATH_DATA}" --input-key prompt --label-key label --apply-chat-template --rollout-shuffle --rollout-seed 47 \
  --rm-type dapo --reward-key score --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size 32 --n-samples-per-prompt 8 \
  --rollout-max-response-len 8192 --rollout-temperature 1 --global-batch-size 256 --num-steps-per-rollout 1 --balance-data \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 --sequence-parallel \
  --use-dynamic-batch-size --max-tokens-per-gpu 10240 --attention-backend flash \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.70 --sglang-server-concurrency 16 --sglang-context-length 16384 \
  --custom-generate-function-path generate_with_hybrid.generate_unified --custom-rm-path generate_with_hybrid.reward_func_unified \
  --save-debug-rollout-data "${OUTPUT_DIR}/rollout_{rollout_id}.pt"

python3 "${MIXED_UTILS}/export_correct_math_sft.py" --input "${OUTPUT_DIR}/rollout_*.pt" \
  --output "${OUTPUT_DIR}/qwen3_8b_dapo_math_correct_sft.jsonl" --target 558
