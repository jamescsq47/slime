#!/bin/bash
# Reuse the BrowseComp SFT launcher with 50/50 correct math trajectories.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
MIXED_UTILS="${REPO_DIR}/examples/browsecomp/mixed_sft"
BROWSECOMP_DATA=${BROWSECOMP_DATA:-/workspace/data/browsecomp/browsecomp_qwen3_8b_sft.jsonl}
MATH_DATA=${MATH_DATA:-/workspace/data/dapo-math-17k/qwen3_8b_correct_rollouts/qwen3_8b_dapo_math_correct_sft.jsonl}
MIXED_DATA=${MIXED_DATA:-/workspace/data/mixed_sft/qwen3_8b_browsecomp_math_50_50.jsonl}
MANIFEST=${MANIFEST:-/workspace/data/mixed_sft/qwen3_8b_browsecomp_math_50_50.manifest.json}
SAVE_PATH=${SAVE_PATH:-/workspace/Qwen3-8B-browsecomp-math-sft-50-50}

python3 "${MIXED_UTILS}/build_mixed_sft.py" --browsecomp "${BROWSECOMP_DATA}" --math "${MATH_DATA}" \
  --output "${MIXED_DATA}" --manifest "${MANIFEST}" --per-source 558 --seed 47

# This short run has no 25-epoch milestone before completion. Keep only the
# newest two checkpoint directories while it runs, so epoch-end saves cannot
# accumulate on disk. The cleaner is read-only with respect to mixed configs.
CLEANER_LOG="${SAVE_PATH}/checkpoint-cleaner.log"
mkdir -p "${SAVE_PATH}"
CHECKPOINT_DIR="${SAVE_PATH}" STEPS_PER_EPOCH=4 KEEP_EVERY_EPOCHS=25 TOTAL_EPOCHS=23 \
  POLL_SECONDS=60 LOCK_FILE="/tmp/qwen3_8b_browsecomp_math_sft_cleanup.lock" \
  bash "${REPO_DIR}/examples/mixed/cleanup_browsecomp_sft_checkpoints.sh" >"${CLEANER_LOG}" 2>&1 &
CLEANER_PID=$!
trap 'kill "${CLEANER_PID}" 2>/dev/null || true' EXIT

# 1,116 records * 23 epochs // global batch 256 = 100 optimizer updates.
# These optimizer settings mirror examples/mixed/hybrid_qwen3_4b_multi_sync.sh.
SFT_DATA="${MIXED_DATA}" SAVE_PATH="${SAVE_PATH}" NUM_EPOCH=23 GLOBAL_BATCH_SIZE=256 ROLLOUT_BATCH_SIZE=256 \
  SFT_LR=1e-6 SFT_MIN_LR=1e-6 SFT_LR_DECAY_STYLE=constant SFT_LR_WARMUP_FRACTION=0 \
  SFT_WEIGHT_DECAY=0.1 SFT_ADAM_BETA1=0.9 SFT_ADAM_BETA2=0.98 \
  bash "${SCRIPT_DIR}/train.sh"
