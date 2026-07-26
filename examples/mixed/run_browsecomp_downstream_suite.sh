#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/debug/eval}
mkdir -p "${OUTPUT_DIR}"

MODELS=(
  "Qwen3-8B|/workspace/Qwen3-8B_torch_dist"
  "new-iter099|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-new/iter099_torch_dist"
  "new-iter199|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-new/iter199_torch_dist"
  "new-iter299|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-new/iter299_torch_dist"
  "new-iter399|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-new/iter399_torch_dist"
  "sync-iter099|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-sync/iter099_torch_dist"
  "sync-iter199|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-sync/iter199_torch_dist"
  "sync-iter299|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-sync/iter299_torch_dist"
  "block4-iter099|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-block4/iter099_torch_dist"
  "block4-iter199|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-block4/iter199_torch_dist"
  "block4-iter299|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-block4/iter299_torch_dist"
  "block4-iter399|/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-block4/iter399_torch_dist"
  "sft-iter1699|/workspace/Qwen3-8B-browsecomp-sft/iter1699_torch_dist"
  "sft-iter3331|/workspace/Qwen3-8B-browsecomp-sft/iter3331_torch_dist"
)

for entry in "${MODELS[@]}"; do
  name=${entry%%|*}
  ref_load=${entry#*|}
  if [ -n "${ONLY_MODEL:-}" ] && [ "${name}" != "${ONLY_MODEL}" ]; then
    continue
  fi
  output_path="${OUTPUT_DIR}/${name}-browsecomp.pt"
  if [ ! -d "${ref_load}" ]; then
    echo "Missing checkpoint: ${ref_load}" >&2
    exit 1
  fi
  if [ -s "${output_path}" ] && python "${SCRIPT_DIR}/analyze_browsecomp_passk.py" "${output_path}" >/dev/null 2>&1; then
    echo "Skip complete result: ${output_path}"
    continue
  fi
  echo "Evaluating ${name}: ${ref_load}"
  REF_LOAD="${ref_load}" \
  DEBUG_ROLLOUT_PATH="${output_path}" \
  WANDB_GROUP="browsecomp-downstream-${name}" \
    bash "${SCRIPT_DIR}/eval-browsecomp.sh"
  python "${SCRIPT_DIR}/analyze_browsecomp_passk.py" "${output_path}" >/dev/null
done

paths=()
for entry in "${MODELS[@]}"; do
  name=${entry%%|*}
  paths+=("${OUTPUT_DIR}/${name}-browsecomp.pt")
done
python "${SCRIPT_DIR}/analyze_browsecomp_passk.py" "${paths[@]}" \
  --json-output "${OUTPUT_DIR}/browsecomp-passk.json" \
  --csv-output "${OUTPUT_DIR}/browsecomp-passk.csv" \
  | tee "${OUTPUT_DIR}/browsecomp-passk.md"
