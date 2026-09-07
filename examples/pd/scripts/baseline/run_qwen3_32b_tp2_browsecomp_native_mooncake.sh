#!/usr/bin/env bash
set -euo pipefail

# Stock SGLang 2P:6D baseline: one logical TP=2 Prefill worker and three
# logical TP=2 Decode workers with native HiCache + Mooncake reverse KV.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export CASE_MODE=native_mooncake
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3-32B}"
export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-1,5;2,6;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-30400}"
export PREFILL_BOOTSTRAP_PORTS="${PREFILL_BOOTSTRAP_PORTS:-28400}"
export DECODE_PORTS="${DECODE_PORTS:-30401 30402 30403}"
export PREFILL_MEM_FRACTION_STATICS="${PREFILL_MEM_FRACTION_STATICS:-0.80}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.80 0.80 0.60}"
export ROUTER_PORT="${ROUTER_PORT:-30410}"
export ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-30420}"
export SEARCH_GPU="${SEARCH_GPU:-7}"
export SEARCH_PORT="${SEARCH_PORT:-9360}"
export SEARCH_START_AFTER_MODELS=true

# HiCache size is per TP rank.  This gives 128 GiB for the P replica and
# 112 GiB for each D replica, matching the current TP=2 comparison setup.
export P_HICACHE_SIZE="${P_HICACHE_SIZE:-64}"
export D_HICACHE_SIZE="${D_HICACHE_SIZE:-56}"
export MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-29451}"
export MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-29452}"
export MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-29480}"
export MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-29403}"
export MOONCAKE_CLIENT_HTTP_PORT="${MOONCAKE_CLIENT_HTTP_PORT:-29490}"

export MATH_RATIO=0
export PRESERVE_SOURCE_ORDER=true
export SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_browsecomp_source_order_n680.json}"
export REQUESTS="${REQUESTS:-680}"
export MAX_INFLIGHT="${MAX_INFLIGHT:-256}"
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export PD_INFERENCE_RETURN_LOGPROB=false
export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline/formal-qwen3-32b-tp2-browsecomp-native-mooncake-2p6d-c256-w300-m1200-20260824-r1}"

exec bash "${SCRIPT_DIR}/run_pd_case.sh"
