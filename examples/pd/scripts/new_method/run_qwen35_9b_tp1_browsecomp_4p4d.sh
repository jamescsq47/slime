#!/usr/bin/env bash
set -euo pipefail

# Isolated pd_mamba compatibility experiment. Do not source the TP2 wrapper:
# it unconditionally resets TP. BrowseComp harness/profile are unchanged.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_mamba/bin}"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-/homes/siqic/sglang-agentic-mamba/python}"
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_DIR}:${PYTHONPATH:-}"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.5-9B}"
export PD_DATA_ROOT="${PD_DATA_ROOT:-/homes/siqic/data}"
export WORKLOAD_CONFIG="${PD_DIR}/configs/experiments/browsecomp_qwen35_source_order.yaml"
export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0;2;4;6}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-1;3;5;7}"
export PREFILL_TP_SIZE=1 DECODE_TP_SIZE=1
export PREFILL_PORTS="${PREFILL_PORTS:-18000 18100 18200 18300}"
export DECODE_PORTS="${DECODE_PORTS:-18400 18500 18600 18700}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-19000 19100 19200 19300}"
export ROUTER_PORT="${ROUTER_PORT:-18800}" ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-18820}"
export AGENTIC_DIRECT_BASE_PORT="${AGENTIC_DIRECT_BASE_PORT:-26000}"
export PD_LATE_BIND_ROUTER_ENTRY="${SCRIPT_DIR}/internal/launch_mamba_late_binding_router.py"
export SEARCH_GPU=7 SEARCH_PORT=8750 SEARCH_START_AFTER_MODELS=true PD_SKIP_SEARCH=0
export SEARCH_SERVER_EMBEDDING_CACHE="${PD_DIR}/data/browsecomp/artifacts/search/corpus_embeddings.pkl"
export MEM_FRACTION_STATIC=0.80 DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.60"
export PD_PAGE_SIZE=64 MAMBA_TRACK_INTERVAL=64
export MAX_CONTEXT_LENGTH=40960 MAX_RESPONSE_LENGTH=36864
export PREFILL_CHUNKED_PREFILL_SIZE=8192 PREFILL_MAX_PREFILL_TOKENS=16384
export MATH_RATIO=0 PRESERVE_SOURCE_ORDER=true
export SCHEDULE_FILE="${PD_DIR}/configs/workloads/fixed_browsecomp_source_order_n680.json"
export REQUESTS=680 WARMUP_REQUESTS=0 MAX_INFLIGHT="${MAX_INFLIGHT:-512}"
export CLOSED_LOOP=1 ARRIVAL_RATE=100 ARRIVAL_DISTRIBUTION=fixed DISPATCH_POLICY=fixed
export SEED=2026 TEMPERATURE=0 TOP_P=1 TOP_K=-1 MIN_P=0
export PD_INFERENCE_RETURN_LOGPROB=false PD_DETERMINISTIC_INFERENCE=1 PD_SERVER_RANDOM_SEED=2026
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}" MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export POST_ANALYZER=none METRICS_INTERVAL=2
# Native BrowseComp tool-role history is prefix-stable. Do not enable the
# shorter prompt checkpoint introduced specifically for SWE user-role results.
export SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT=false
export PD_LATE_BIND_NUMA_DOMAINS=1 SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=1 SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1
export MAX_PREFILL_INFLIGHT="${MAX_PREFILL_INFLIGHT:-48}"
export D_TARGET_KV_FRACTION=0.90 P_ACCEPT_TIMEOUT_SECONDS=600 P_READY_TIMEOUT_SECONDS=600
export D2P_HOST_ARENA_GIB_PER_P=32 P2D_HOST_ARENA_GIB_PER_P=8 P2D_HOST_STAGING=true
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND=file SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND=file
export FAST_TOOL_THRESHOLD_SECONDS=2 DIRECT_WAIT_SECONDS=2 SGLANG_AGENTIC_KV_EARLY_CLAIM_POST_TIMEOUT=2
export PD_MAX_TRANSFER_INFLIGHT=8 P_TO_D_CONSUMERS=24 SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY=true
export P_READY_BACKPRESSURE_MODE=continuous P_READY_REQUEST_CAP=8 P_READY_TOKEN_CAP_FRACTION=0.25 P_READY_HBM_HIGH_WATERMARK=0.85

export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/qwen35-9b-tp1-browsecomp-agentic-kv-4p4d-c512-w300-m1200-$(date -u +%Y%m%dT%H%M%S)}"
mkdir -p "${RUN_DIR}"
[[ ! -e "${RUN_DIR}/config.json" ]] || { echo 'Refusing to reuse an existing experiment' >&2; exit 2; }
# Unique retained control ledger names; the nested supervisor owns arenas and
# dependency-ordered shutdown. Never clean another run's paths or processes.
CONTROL_DIR="$(mktemp -d /dev/shm/pd-q35-9b-browse-control.XXXXXX)"
export SGLANG_AGENTIC_KV_LEDGER_PATH="${CONTROL_DIR}/ledger.json"
export SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH="${CONTROL_DIR}/staging.json"
export SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH="${CONTROL_DIR}/p2d.json"
case_pid=""
cleanup_case() {
  local status=$?
  trap - INT TERM
  if [[ -n "${case_pid}" ]]; then
    kill -TERM -- "-${case_pid}" 2>/dev/null || true
    wait "${case_pid}" 2>/dev/null || true
  fi
  return "${status}"
}
trap cleanup_case EXIT
trap 'exit 130' INT TERM
setsid bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh" &
case_pid=$!
wait "${case_pid}"
case_pid=""
