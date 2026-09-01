#!/usr/bin/env bash
set -euo pipefail

# Formal counterpart of the collocated Qwen3-32B BrowseComp c192 baseline:
# one logical TP=2 Prefill worker and three logical TP=2 Decode workers.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3-32B}"
export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-1,5;2,6;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-30300}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-28300}"
export DECODE_PORTS="${DECODE_PORTS:-30301 30302 30303}"
export ROUTER_PORT="${ROUTER_PORT:-30310}"
export ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-30320}"
export SEARCH_GPU="${SEARCH_GPU:-7}"
export SEARCH_PORT="${SEARCH_PORT:-9350}"
export SEARCH_START_AFTER_MODELS=true
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.80 0.80 0.60}"

export MATH_RATIO=0
export PRESERVE_SOURCE_ORDER=true
export SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_browsecomp_source_order_n680.json}"
export REQUESTS="${REQUESTS:-680}"
export MAX_INFLIGHT="${MAX_INFLIGHT:-192}"
# This launcher profiles steady-state serving.  Keep the validated closed-loop
# workload explicit here instead of inheriting run_pd_servers.sh's low-rate
# open-loop smoke defaults.
export CLOSED_LOOP="${CLOSED_LOOP:-1}"
export ARRIVAL_RATE="${ARRIVAL_RATE:-100}"
export ARRIVAL_DISTRIBUTION="${ARRIVAL_DISTRIBUTION:-fixed}"
export WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
export TEMPERATURE=0
export TOP_P=1
export TOP_K=-1
export PD_PAGE_SIZE=64
export MAX_CONTEXT_LENGTH=40960
export MAX_RESPONSE_LENGTH=36864
export PREFILL_CHUNKED_PREFILL_SIZE=8192
export PREFILL_MAX_PREFILL_TOKENS=16384
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"

export P2D_HOST_STAGING=true
export P2D_HOST_ARENA_GIB_PER_P=32
export SGLANG_AGENTIC_KV_P2D_HOST_STAGING=true
export SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB=32
export FAST_TOOL_THRESHOLD_SECONDS="${FAST_TOOL_THRESHOLD_SECONDS:-2}"
export DIRECT_WAIT_SECONDS="${DIRECT_WAIT_SECONDS:-2}"
export PD_INFERENCE_RETURN_LOGPROB=false
# A valid request may remain in P's tokenizer/scheduler pipeline while older
# work drains.  Do not turn that normal backpressure into an HTTP 500.
export P_ACCEPT_TIMEOUT_SECONDS="${P_ACCEPT_TIMEOUT_SECONDS:-600}"
export PD_LATE_BIND_NUMA_DOMAINS=0
export SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=0
export SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1

# Keep the validated listener range for reproducible launches.
export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/new-method/formal-qwen3-32b-tp2-browsecomp-2p6d-bidir-c192-w300-m1200-20260817-r4}"

exec bash "${SCRIPT_DIR}/run_2p6d_numa_case.sh"
