#!/bin/bash
# Staged Qwen3-8B BrowseComp SFT -> mixed BrowseComp/Retool GRPO launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

STAGE=${STAGE:-sft}
SFT_DATA=${SFT_DATA:-/workspace/data/browsecomp/browsecomp_qwen3_8b_sft.jsonl}
SFT_SAVE_PATH=${SFT_SAVE_PATH:-/workspace/Qwen3-8B-browsecomp-sft}
RL_SAVE_PATH=${RL_SAVE_PATH:-/workspace/Qwen3-8B-browsecomp-sft-mixed-rl}
HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/Qwen3-8B}
BASE_REF_LOAD=${BASE_REF_LOAD:-/workspace/Qwen3-8B_torch_dist}

require_file() {
  [[ -s "$1" ]] || { echo "Missing or empty file: $1" >&2; exit 2; }
}

require_checkpoint() {
  [[ -d "$1" && -s "$1/latest_checkpointed_iteration.txt" ]] || {
    echo "Not a completed Megatron checkpoint: $1" >&2
    echo "Run STAGE=sft first, or set SFT_SAVE_PATH to the finished SFT checkpoint." >&2
    exit 2
  }
}

case "${STAGE}" in
  sft)
    require_file "${SFT_DATA}"
    echo "=== BrowseComp SFT: 100 epochs ==="
    SFT_DATA="${SFT_DATA}" SAVE_PATH="${SFT_SAVE_PATH}" \
      HF_CHECKPOINT="${HF_CHECKPOINT}" REF_LOAD="${BASE_REF_LOAD}" NUM_EPOCH=100 \
      bash "${REPO_DIR}/examples/browsecomp/sft/train.sh"
    ;;
  rl)
    require_checkpoint "${SFT_SAVE_PATH}"
    : "${LOCAL_SEARCH_URL:?export LOCAL_SEARCH_URL before starting BrowseComp RL}"
    echo "=== Mixed BrowseComp + Retool RL initialized from ${SFT_SAVE_PATH} ==="
    HF_CHECKPOINT="${HF_CHECKPOINT}" REF_LOAD="${SFT_SAVE_PATH}" SAVE_PATH="${RL_SAVE_PATH}" \
      bash "${SCRIPT_DIR}/hybrid_qwen3_4b_multi_sync.sh"
    ;;
  *)
    echo "STAGE must be 'sft' or 'rl' (got: ${STAGE})" >&2
    exit 2
    ;;
esac
