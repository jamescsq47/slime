#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/bfcl_web_search.yaml}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd):${PYTHONPATH:-}"
export START_SEARCH_SERVER=false
export MODEL_PATH="${MODEL_PATH:-/dataset/model/qwen3/Qwen3-8B}"
# Begin with one worker: public web search is not an unlimited load-test service.
export MODEL_GPUS="${MODEL_GPUS:-0}"
export MODEL_PORTS="${MODEL_PORTS:-27600}"
export ROUTER_PORT="${ROUTER_PORT:-27610}"
export MEM_FRACTION_STATIC=0.80
export MAX_INFLIGHT="${MAX_INFLIGHT:-8}"
export REQUESTS="${REQUESTS:-100}"
export TEMPERATURE=0
export TOP_P=1 TOP_K=-1
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export DISPATCH_POLICY=random PRESERVE_SOURCE_ORDER=true POST_ANALYZER=none
export MODEL_MAX_RESPONSE_LENGTH=32768 MODEL_CONTEXT_LENGTH=40960
export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline/bfcl-web-qwen3-8b-c${MAX_INFLIGHT}}"
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_bfcl_web.py" --config "${WORKLOAD_CONFIG}"
exec bash "${SCRIPT_DIR}/run_colocated_case.sh"
