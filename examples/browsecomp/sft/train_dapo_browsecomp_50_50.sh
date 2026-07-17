#!/bin/bash
# Reuse the BrowseComp SFT launcher with 50/50 correct math trajectories.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
MIXED_UTILS="${REPO_DIR}/examples/browsecomp/mixed_sft"
BROWSECOMP_DATA=${BROWSECOMP_DATA:-/workspace/data/browsecomp/browsecomp_qwen3_8b_sft.jsonl}
MATH_DATA=${MATH_DATA:-/workspace/data/dapo-math-17k/qwen3_8b_correct_rollouts/qwen3_8b_dapo_math_correct_sft.jsonl}
MIXED_DATA=${MIXED_DATA:-/workspace/data/mixed_sft/qwen3_8b_browsecomp_math_50_50_x23.jsonl}
MANIFEST=${MANIFEST:-/workspace/data/mixed_sft/qwen3_8b_browsecomp_math_50_50_x23.manifest.json}
SAVE_PATH=${SAVE_PATH:-/workspace/Qwen3-8B-browsecomp-math-sft-50-50-x23}

python3 "${MIXED_UTILS}/build_mixed_sft.py" --browsecomp "${BROWSECOMP_DATA}" --math "${MATH_DATA}" \
  --output "${MIXED_DATA}" --manifest "${MANIFEST}" --per-source 558 --repeat 23 --seed 47

# (1,116 records * 23 shuffled passes) // global batch 256 = 100 optimizer
# updates. One physical epoch suppresses SLIME's implicit epoch-end saves, so
# this run writes only the final checkpoint at step 100.
# These optimizer settings mirror examples/mixed/hybrid_qwen3_4b_multi_sync.sh.
SFT_DATA="${MIXED_DATA}" SAVE_PATH="${SAVE_PATH}" NUM_EPOCH=1 GLOBAL_BATCH_SIZE=256 ROLLOUT_BATCH_SIZE=256 \
  SFT_LR=1e-6 SFT_MIN_LR=1e-6 SFT_LR_DECAY_STYLE=constant SFT_LR_WARMUP_FRACTION=0 \
  SFT_WEIGHT_DECAY=0.1 SFT_ADAM_BETA1=0.9 SFT_ADAM_BETA2=0.98 \
  bash "${SCRIPT_DIR}/train.sh"
