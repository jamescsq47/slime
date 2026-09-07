#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-27B BrowseComp validation for the agentic request-generation KV
# pipeline.  There are two logical P and two logical D engines, each TP=2 and
# spread across both NUMA domains: four physical P GPUs plus four physical D
# GPUs.  Unlike the colocated baseline, every model turn crosses P -> D and an
# unfinished turn restores the complete attention + Mamba snapshot D -> P.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_SOURCE="${SGLANG_OVERLAY_ROOT:-/homes/siqic/sglang-agentic-mamba/python}"

export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_mamba/bin}"
export PATH="${PD_ENV_BIN}:${PATH}"
export SGLANG_OVERLAY_ROOT="${SGLANG_SOURCE}"
export PYTHONPATH="${SGLANG_SOURCE}:${PD_DIR}:${PYTHONPATH:-}"
export PD_DATA_ROOT="${PD_DATA_ROOT:-/homes/siqic/data}"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.5-27B}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/browsecomp_qwen35_source_order.yaml}"

export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4;1,5}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-2,6;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-37700 37701}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-38700 38701}"
export DECODE_PORTS="${DECODE_PORTS:-37702 37703}"
export ROUTER_PORT="${ROUTER_PORT:-37710}"
export ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-37720}"
export AGENTIC_DIRECT_BASE_PORT="${AGENTIC_DIRECT_BASE_PORT:-61700}"

export SEARCH_GPU="${SEARCH_GPU:-7}"
export SEARCH_PORT="${SEARCH_PORT:-8750}"
export SEARCH_START_AFTER_MODELS=true
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.80 0.60}"

export PD_PAGE_SIZE=64
export MAMBA_TRACK_INTERVAL=64
export MAX_CONTEXT_LENGTH=40960
export MAX_RESPONSE_LENGTH=36864
export PREFILL_CHUNKED_PREFILL_SIZE=8192
export PREFILL_MAX_PREFILL_TOKENS=16384

export MATH_RATIO=0
export PRESERVE_SOURCE_ORDER=true
export SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_browsecomp_source_order_n680.json}"
export REQUESTS="${REQUESTS:-680}"
export WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
export MAX_INFLIGHT="${MAX_INFLIGHT:-192}"
export CLOSED_LOOP=1
export ARRIVAL_RATE=100
export ARRIVAL_DISTRIBUTION=fixed
export DISPATCH_POLICY=fixed
export SEED=2026
export TEMPERATURE=0
export TOP_P=1
export TOP_K=-1
export PD_INFERENCE_RETURN_LOGPROB=false
# Qwen3.5 in pd_mamba is validated with Triton attention.  This also keeps
# source/destination output comparisons reproducible and avoids selecting the
# environment's older FlashInfer wheel during server startup.
export PD_DETERMINISTIC_INFERENCE=1
export PD_SERVER_RANDOM_SEED=2026
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export POST_ANALYZER=none

export PD_LATE_BIND_NUMA_DOMAINS=1
export SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=1
export SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1
export MAX_PREFILL_INFLIGHT="${MAX_PREFILL_INFLIGHT:-48}"
export D_TARGET_KV_FRACTION=0.90
export P_ACCEPT_TIMEOUT_SECONDS=600
export P_READY_TIMEOUT_SECONDS=600

export D2P_HOST_ARENA_GIB_PER_P=32
export P2D_HOST_STAGING=true
export P2D_HOST_ARENA_GIB_PER_P=8
export FAST_TOOL_THRESHOLD_SECONDS="${FAST_TOOL_THRESHOLD_SECONDS:-2}"
export DIRECT_WAIT_SECONDS="${DIRECT_WAIT_SECONDS:-2}"
export PD_MAX_TRANSFER_INFLIGHT=8
export P_TO_D_CONSUMERS=24
export SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY=true

export P_READY_BACKPRESSURE_MODE=continuous
export P_READY_REQUEST_CAP=8
export P_READY_TOKEN_CAP_FRACTION=0.25
export P_READY_HBM_HIGH_WATERMARK=0.85

export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/qwen35-27b-tp2-browsecomp-agentic-kv-4p4d-c192-w300-m1200}"
exec bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh"
