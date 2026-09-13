#!/usr/bin/env bash
set -euo pipefail

# 100 source-order Verified tasks x 5 independent episodes; not Verified500.
# The isolated Mamba overlay is intentionally not the concurrently edited pd env.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_mamba/bin}"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-/homes/siqic/sglang-agentic-mamba/python}"
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_DIR}:${PYTHONPATH:-}"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.5-9B}"
export PD_DATA_ROOT="${PD_DATA_ROOT:-/tmp/pd-data-first100}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml}"
export MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-qwen3}" MODEL_TOOL_CALL_PARSER=qwen3_coder
export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0;2;4;6}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-1;3;5;7}"
export PREFILL_TP_SIZE=1 DECODE_TP_SIZE=1
export PREFILL_PORTS="${PREFILL_PORTS:-18000 18100 18200 18300}"
export DECODE_PORTS="${DECODE_PORTS:-18400 18500 18600 18700}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-19000 19100 19200 19300}"
export ROUTER_PORT="${ROUTER_PORT:-18800}" ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-18820}"
export AGENTIC_DIRECT_BASE_PORT="${AGENTIC_DIRECT_BASE_PORT:-26000}"
export PD_SKIP_SEARCH=1
export MEM_FRACTION_STATIC=0.80 DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.80"
export PD_PAGE_SIZE=64 MAMBA_TRACK_INTERVAL=64
export MAX_CONTEXT_LENGTH=131072 MAX_RESPONSE_LENGTH=81920
export PREFILL_CHUNKED_PREFILL_SIZE=8192 PREFILL_MAX_PREFILL_TOKENS=8192
export REQUESTS="${REQUESTS:-500}" WARMUP_REQUESTS=0 MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
export CLOSED_LOOP=false ARRIVAL_RATES=100 ARRIVAL_DISTRIBUTION=fixed
export DISPATCH_POLICY=random PRESERVE_SOURCE_ORDER=true SEED=2026
export TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0
export PD_DETERMINISTIC_INFERENCE=1 PD_SERVER_RANDOM_SEED=2026
export PD_INFERENCE_RETURN_LOGPROB=false SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
export PD_LATE_BIND_NUMA_DOMAINS=1 SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=1 SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1
export MAX_PREFILL_INFLIGHT="${MAX_PREFILL_INFLIGHT:-32}"
export D_TARGET_KV_FRACTION=0.90 P_ACCEPT_TIMEOUT_SECONDS=600 P_READY_TIMEOUT_SECONDS=600
export D2P_HOST_ARENA_GIB_PER_P=32 P2D_HOST_ARENA_GIB_PER_P=8 P2D_HOST_STAGING=true
# This overlay uses tmpfs files, NOT the other agent's new memfd transport.
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND=file SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND=file
export FAST_TOOL_THRESHOLD_SECONDS=2 DIRECT_WAIT_SECONDS=2
export SGLANG_AGENTIC_KV_EARLY_CLAIM_POST_TIMEOUT=2
export PD_MAX_TRANSFER_INFLIGHT=8 P_TO_D_CONSUMERS=24 SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY=true
export P_READY_BACKPRESSURE_MODE=continuous P_READY_REQUEST_CAP=8
export P_READY_TOKEN_CAP_FRACTION=0.25 P_READY_HBM_HIGH_WATERMARK=0.85
export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/qwen35-9b-tp1-swe-openenv-4p4d-c128-first100x5-$(date -u +%Y%m%dT%H%M%S)}"
mkdir -p "${RUN_DIR}"
export PD_SWE_RUN_ID="${PD_SWE_RUN_ID:-$(basename -- "${RUN_DIR}")}"
export PD_SWE_PROGRESS_FILE="${RUN_DIR}/episode_progress.jsonl"
[[ ! -e "${RUN_DIR}/requests.completed.jsonl" ]] || { echo 'Run already contains episodes' >&2; exit 2; }
existing_containers="$(docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")"
[[ -z "${existing_containers}" ]] || { echo 'Run label already owns containers' >&2; exit 2; }
case_pid=""
cleanup_swe_pd() {
  local status=$? owned_containers=()
  trap - INT TERM
  if [[ -n "${case_pid}" ]]; then
    # Let the nested supervisor finish its dependency-ordered GPU shutdown.
    kill -TERM -- "-${case_pid}" 2>/dev/null || true
    wait "${case_pid}" 2>/dev/null || true
  fi
  mapfile -t owned_containers < <(timeout 15 docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")
  if (( ${#owned_containers[@]} )); then
    timeout 30 docker rm -f "${owned_containers[@]}" || true
  fi
  return "${status}"
}
trap cleanup_swe_pd EXIT
trap 'exit 130' INT TERM
export INFERENCE_ENTRY="${SCRIPT_DIR}/internal/inference_checkpointed.py"
export PD_LATE_BIND_ROUTER_ENTRY="${SCRIPT_DIR}/internal/launch_mamba_late_binding_router.py"
# The shared wrapper overrides BOOTSTRAP_PORT. Isolate ledger names explicitly.
CONTROL_DIR="$(mktemp -d /dev/shm/pd-q35-9b-control.XXXXXX)"
export SGLANG_AGENTIC_KV_LEDGER_PATH="${CONTROL_DIR}/ledger.json"
export SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH="${CONTROL_DIR}/staging.json"
export SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH="${CONTROL_DIR}/p2d.json"
# Retain these small control ledgers for fault diagnosis; pipeline owns and
# cleans its independently allocated large arenas after its GPU children exit.
export PD_Q35_CONTROL_DIR="${CONTROL_DIR}"
setsid bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh" &
case_pid=$!
wait "${case_pid}"
case_pid=""
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/internal/analyze_mamba_swe_run.py" "${RUN_DIR}"
