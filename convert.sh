#!/bin/bash
# Convert megatron distributed checkpoints to HuggingFace format.
#
# Usage: bash convert.sh

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
ORIGIN_HF_DIR="/workspace/Qwen3-8B"   # original HF model dir (tokenizer / config source)
PYTHONPATH_ROOT="/root/Megatron-LM"    # Megatron-LM root for PYTHONPATH
SLIME_ROOT="/workspace/slime"           # slime repo root (where tools/ lives)

# ── Model paths to convert (edit this list) ──────────────────────────────────
MODEL_DIRS=(
    "/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-block4-trial2"
    # "/workspace/Qwen3-8B_hybrid_mask256_512_dynamic"
)
# ─────────────────────────────────────────────────────────────────────────────

if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
    echo "No model paths configured. Please edit MODEL_DIRS in this script."
    exit 1
fi

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    if [ ! -d "$MODEL_DIR" ]; then
        echo "[WARN] Model dir not found, skipping: $MODEL_DIR"
        continue
    fi

    echo "========================================"
    echo "Processing model: $MODEL_DIR"
    echo "========================================"

    for ITER_DIR in "${MODEL_DIR}"/iter_*; do
        [ -d "$ITER_DIR" ] || continue

        # Extract the numeric part: iter_0000099 -> 0000099 -> 99 -> iter099
        ITER_NAME=$(basename "$ITER_DIR")               # iter_0000099
        ITER_NUM=${ITER_NAME#iter_}                      # 0000099
        ITER_NUM_NOZERO=$((10#$ITER_NUM))                # 99  (strip leading zeros)
        OUTPUT_NAME="iter$(printf '%03d' $ITER_NUM_NOZERO)"  # iter099

        INPUT_DIR="$ITER_DIR"
        OUTPUT_DIR="${MODEL_DIR}/${OUTPUT_NAME}"

        if [ -d "$OUTPUT_DIR" ]; then
            echo "[SKIP] Output dir already exists: $OUTPUT_DIR"
            continue
        fi

        echo "[CONVERT] $INPUT_DIR -> $OUTPUT_DIR"

        PYTHONPATH="$PYTHONPATH_ROOT" python "$SLIME_ROOT/tools/convert_torch_dist_to_hf.py" \
            --input-dir  "$INPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --origin-hf-dir "$ORIGIN_HF_DIR"

        echo "[DONE]  $OUTPUT_DIR"
    done

    # ── Cleanup: delete megatron checkpoints and rollout after conversion ──
    echo "----------------------------------------"
    echo "Cleaning up intermediate data in: $MODEL_DIR"

    for ITER_DIR in "${MODEL_DIR}"/iter_*; do
        [ -d "$ITER_DIR" ] || continue
        echo "[DELETE] $ITER_DIR"
        rm -rf "$ITER_DIR"
    done

    # if [ -d "${MODEL_DIR}/rollout" ]; then
    #     echo "[DELETE] ${MODEL_DIR}/rollout"
    #     rm -rf "${MODEL_DIR}/rollout"
    # fi

    echo "Cleanup done for: $MODEL_DIR"
    echo "----------------------------------------"
done

echo "All conversions finished."
