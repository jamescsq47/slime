#!/usr/bin/env bash
set -euo pipefail

# Four collocated TP=2 workers. GPU7 also hosts BrowseComp retrieval, so the
# whole [3,7] TP group uses GPU7's safe, uniform KV-pool fraction.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3-32B}"
export MODEL_GPU_GROUPS="${MODEL_GPU_GROUPS:-0,4;1,5;2,6;3,7}"
export MODEL_TP_SIZE=2
export MODEL_PORTS="${MODEL_PORTS:-33400 33401 33402 33403}"
export MODEL_MEM_FRACTION_STATICS="${MODEL_MEM_FRACTION_STATICS:-0.80 0.80 0.80 0.60}"
export ROUTER_PORT="${ROUTER_PORT:-33410}"
export SEARCH_GPU="${SEARCH_GPU:-7}"
export SEARCH_PORT="${SEARCH_PORT:-9340}"
export SEARCH_START_AFTER_MODELS=true
export MAX_INFLIGHT="${MAX_INFLIGHT:-192}"
export REQUESTS="${REQUESTS:-680}"
export MATH_RATIO=0
export PRESERVE_SOURCE_ORDER=true
export SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_browsecomp_source_order_n680.json}"
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline/formal-qwen3-32b-tp2-browsecomp-colocated-c192-w300-m1200-20260817-r2}"
export PD_INFERENCE_RETURN_LOGPROB=false

exec bash "${SCRIPT_DIR}/run_colocated_case.sh"
