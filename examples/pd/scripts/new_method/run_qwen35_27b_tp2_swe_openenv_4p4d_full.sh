#!/usr/bin/env bash
set -euo pipefail

# Full SWE-bench Verified comparison run: exactly one pass over the same 500
# source-ordered tasks as the colocated Qwen3.5-27B baseline.  Physical
# topology: two P and two D replicas, each spanning both NUMA domains at TP=2.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_SOURCE="${SGLANG_OVERLAY_ROOT:-/homes/siqic/sglang-agentic-mamba/python}"

export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_mamba/bin}"
export PATH="${PD_ENV_BIN}:${PATH}"
export SGLANG_OVERLAY_ROOT="${SGLANG_SOURCE}"
export PYTHONPATH="${SGLANG_SOURCE}:${PD_DIR}:${PYTHONPATH:-}"
export PD_DATA_ROOT="${PD_DATA_ROOT:-/tmp/pd-data}"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.5-27B}"
# The recorded 44.4%/46.2% 8K/64-turn colocated repeats used this isolated
# parser combination.
export MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-glm45}"
export MODEL_TOOL_CALL_PARSER="${MODEL_TOOL_CALL_PARSER:-qwen3_coder}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml}"

export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4;1,5}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-2,6;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-37600 37601}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-38600 38601}"
export DECODE_PORTS="${DECODE_PORTS:-37602 37603}"
export ROUTER_PORT="${ROUTER_PORT:-37610}"
export ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-37620}"
export AGENTIC_DIRECT_BASE_PORT="${AGENTIC_DIRECT_BASE_PORT:-61600}"
export PD_SKIP_SEARCH=1

export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.80 0.80}"
export PD_PAGE_SIZE=64
export MAMBA_TRACK_INTERVAL=64
export MAX_CONTEXT_LENGTH=131072
export MAX_RESPONSE_LENGTH=81920
export PREFILL_CHUNKED_PREFILL_SIZE=8192
export PREFILL_MAX_PREFILL_TOKENS=8192

# Baseline-aligned evaluation semantics: no warmup pool and no recycling.
# The run ends only after all 500 source-order tasks have durable verifier rows.
export REQUESTS="${REQUESTS:-500}"
export WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
export MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
export CLOSED_LOOP=false
export ARRIVAL_RATE=100
export ARRIVAL_RATES=100
export ARRIVAL_DISTRIBUTION=fixed
export DISPATCH_POLICY=random
export PRESERVE_SOURCE_ORDER=true
export SEED=2026
export TEMPERATURE=0.6
export TOP_P=0.95
export TOP_K=20
export MIN_P=0
export PD_DETERMINISTIC_INFERENCE=1
export PD_SERVER_RANDOM_SEED=2026
export PD_INFERENCE_RETURN_LOGPROB=false
export SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
export POST_ANALYZER=swe_bench

export PD_LATE_BIND_NUMA_DOMAINS=1
export SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=1
export SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1
# The closed-loop throughput experiments deliberately used a narrow 12-request
# tokenizer gate.  A baseline-aligned SWE-bench pass injects up to c128 tasks
# at once, so split that admission capacity across the two P replicas instead
# of serializing the fixed 500-task evaluation behind the closed-loop gate.
export MAX_PREFILL_INFLIGHT="${MAX_PREFILL_INFLIGHT:-64}"
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

# Keep a drainable P workspace under sustained multi-turn pressure.  Each P
# may compute ahead four ordinary generations per physical D replica, while
# Direct/Host parent recoveries remain exempt in SGLang so D->P traffic can
# always release Decode-side KV.  Without this bound, a full SWE-bench pass can
# fill P with completed P->D generations at the same time that D is filled with
# completed D->P generations, producing a circular capacity wait.
export P_READY_BACKPRESSURE_MODE=continuous
export P_READY_REQUEST_CAP=8
export P_READY_TOKEN_CAP_FRACTION=0.25
export P_READY_HBM_HIGH_WATERMARK=0.85

export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/qwen35-27b-tp2-swe-openenv-agentic-kv-4p4d-c128-full}"
exec bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh"
