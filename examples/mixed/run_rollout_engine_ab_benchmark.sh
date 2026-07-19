#!/usr/bin/env bash
# Run the three conditions against Slime-launched SGLang routers.
#
# The routers must each serve the same immutable HF_CHECKPOINT, with one GPU
# per engine.  No RL checkpoint or training actor is involved.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/debug/rollout_engine_ab}
ROUTER_16=${ROUTER_16:?URL of the actual 16-engine Slime router}
ROUTER_8=${ROUTER_8:?URL of the actual 8-engine Slime router}
HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/Qwen3-8B}
LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:?export LOCAL_SEARCH_URL for BrowseComp}

# Match examples/mixed/hybrid_qwen3_4b_multi{,_sync}.sh.
export MIXED_RETOOL_MAX_RESPONSE_LEN=${MIXED_RETOOL_MAX_RESPONSE_LEN:-8192}
export MIXED_BROWSECOMP_MAX_RESPONSE_LEN=${MIXED_BROWSECOMP_MAX_RESPONSE_LEN:-36864}
export BROWSECOMP_MAX_SEQ_LEN=${BROWSECOMP_MAX_SEQ_LEN:-36864}
export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-100}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-2048}
export BROWSECOMP_MUST_SEARCH=${BROWSECOMP_MUST_SEARCH:-1}
export BROWSECOMP_ENABLE_THINKING=${BROWSECOMP_ENABLE_THINKING:-0}

mkdir -p "${OUTPUT_DIR}"

run_condition() {
  local name=$1 router=$2 engines=$3
  shift 3
  python "${SCRIPT_DIR}/rollout_engine_ab_benchmark.py" run \
    --name "${name}" --router "${router}" --engines "${engines}" \
    --model "${HF_CHECKPOINT}" --weight-label "base-qwen3-8b" \
    --groups 32 --math-ratio 0.5 --samples-per-group 8 \
    --seed 47 --temperature 1 --top-p 1 --top-k -1 \
    --max-response-len 36864 --context-length 40960 \
    --output-dir "${OUTPUT_DIR}" "$@"
}

run_condition e16_no_partial "${ROUTER_16}" 16
run_condition e8_no_partial "${ROUTER_8}" 8
run_condition e8_partial "${ROUTER_8}" 8 --partial \
  --abort-after "${ABORT_AFTER:-60}" --max-aborts "${MAX_ABORTS:-2}"
python "${SCRIPT_DIR}/rollout_engine_ab_benchmark.py" summarize --output-dir "${OUTPUT_DIR}"
echo "Results: ${OUTPUT_DIR}/summary.json"
