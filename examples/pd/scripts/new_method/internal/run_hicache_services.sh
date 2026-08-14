#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PD_DIR}"

# Isolated 1P:nD PD case runner used by the scheduling matrix. It never edits
# the shared conda environment; an optional private SGLang copy is selected by
# SGLANG_OVERLAY_ROOT/PYTHONPATH.
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/new-method/manual-agentic-pd}"
PD_RUN_QWEN_SCRIPT="${PD_RUN_QWEN_SCRIPT:-${SCRIPT_DIR}/run_pd_servers.sh}"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd/bin}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-8080}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-9003}"
MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-50052}"
MOONCAKE_CLIENT_HTTP_PORT="${MOONCAKE_CLIENT_HTTP_PORT:-9300}"
MOONCAKE_STORE_SIZE="${MOONCAKE_STORE_SIZE:-256 GB}"
MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO="${MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO:-0.85}"
MOONCAKE_EVICTION_RATIO="${MOONCAKE_EVICTION_RATIO:-0.10}"
MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-$(hostname -I | awk '{print $1}')}"
PD_HICACHE_PREFETCH_THRESHOLD="${PD_HICACHE_PREFETCH_THRESHOLD:-64}"
PD_HICACHE_PREFETCH_TIMEOUT_BASE="${PD_HICACHE_PREFETCH_TIMEOUT_BASE:-5}"
PD_HICACHE_PREFETCH_TIMEOUT_PER_KI_TOKEN="${PD_HICACHE_PREFETCH_TIMEOUT_PER_KI_TOKEN:-0.5}"
# Resolve the imported package instead of assuming a site-packages layout.
# The pd environment may intentionally use an editable sglang-agentic checkout.
SGLANG_PACKAGE_ROOT="$("${PD_ENV_BIN}/python" -c \
  'from pathlib import Path; import sglang; print(Path(sglang.__file__).resolve().parent)')"
SGLANG_SCHEDULER_OUTPUT="${SGLANG_PACKAGE_ROOT}/srt/managers/scheduler_output_processor_mixin.py"
SGLANG_DECODE_OFFLOAD="${SGLANG_PACKAGE_ROOT}/srt/disaggregation/decode_kvcache_offload_manager.py"
SGLANG_DECODE_MIXIN="${SGLANG_PACKAGE_ROOT}/srt/disaggregation/decode.py"
if ! grep -q "Keep the complete KV history resident" "${SGLANG_SCHEDULER_OUTPUT}" \
  || ! grep -q "pending_responses" "${SGLANG_DECODE_OFFLOAD}" \
  || ! grep -q "pop_ready_responses" "${SGLANG_DECODE_MIXIN}"; then
  echo "Refusing unsafe Decode offload: apply ${PD_DIR}/patches/sglang_finish_only_decode_offload.patch to the pd environment first." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/logs"
service_pids=()
experiment_pid=""
cleanup_started=0

process_group_has_live_members() {
  local pgid="$1"
  # A native wrapper may have a zombie leader while its worker threads still
  # own sockets.  Check all tasks in the setsid group before declaring it dead.
  ps -eLo pgid=,stat= | awk -v target="${pgid}" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

cleanup() {
  local pid alive index task_id
  local deadline
  # This function runs from an EXIT trap under `set -e`.  A false arithmetic
  # command has status 1 and used to abort the trap before any child process
  # was terminated, leaving Mooncake process groups behind on startup errors.
  if (( cleanup_started == 1 )); then
    return
  fi
  cleanup_started=1
  trap - INT TERM
  if [[ -n "${experiment_pid}" ]] && process_group_has_live_members "${experiment_pid}"; then
    kill -TERM -- "-${experiment_pid}" 2>/dev/null || true
    wait "${experiment_pid}" 2>/dev/null || true
  fi
  # Stop dependants before their control plane: Mooncake client, then master.
  for ((index=${#service_pids[@]} - 1; index >= 0; index--)); do
    pid="${service_pids[index]}"
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  deadline=$((SECONDS + ${PD_AUX_SERVICE_GRACEFUL_SHUTDOWN_SECONDS:-30}))
  while (( SECONDS < deadline )); do
    alive=0
    for ((index=${#service_pids[@]} - 1; index >= 0; index--)); do
      pid="${service_pids[index]}"
      process_group_has_live_members "${pid}" && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for ((index=${#service_pids[@]} - 1; index >= 0; index--)); do
    pid="${service_pids[index]}"
    if process_group_has_live_members "${pid}"; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
      while read -r task_id; do
        [[ -n "${task_id}" ]] && kill -KILL "${task_id}" 2>/dev/null || true
      done < <(ps -eLo lwp=,pgid= | awk -v target="${pid}" '$2 == target { print $1 }')
    fi
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_http() {
  local name="$1"
  local url="$2"
  local pid="$3"
  # Faulting a 128 GiB Host HiCache can take well over one minute even on an
  # otherwise idle machine.  Keep the timeout configurable and long enough
  # that initialization is not mistaken for a crashed service.
  local deadline=$((SECONDS + ${SERVICE_STARTUP_TIMEOUT_SECONDS:-300}))
  while (( SECONDS < deadline )); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before becoming healthy" >&2
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for ${name}: ${url}" >&2
  return 1
}

setsid "${PD_ENV_BIN}/mooncake_master" \
  --rpc_port="${MOONCAKE_MASTER_PORT}" \
  --enable_http_metadata_server=true \
  --http_metadata_server_port="${MOONCAKE_METADATA_PORT}" \
  --eviction_high_watermark_ratio="${MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO}" \
  --eviction_ratio="${MOONCAKE_EVICTION_RATIO}" \
  --metrics_port="${MOONCAKE_METRICS_PORT}" \
  >"${RUN_DIR}/logs/mooncake-master.log" 2>&1 &
service_pids+=("$!")
wait_http mooncake-master "http://127.0.0.1:${MOONCAKE_METRICS_PORT}/health" "${service_pids[-1]}"

setsid "${PD_ENV_BIN}/mooncake_client" \
  --host=127.0.0.1 \
  --port="${MOONCAKE_CLIENT_PORT}" \
  --global_segment_size="${MOONCAKE_STORE_SIZE}" \
  --master_server_address="127.0.0.1:${MOONCAKE_MASTER_PORT}" \
  --metadata_server="http://127.0.0.1:${MOONCAKE_METADATA_PORT}/metadata" \
  --protocol=tcp \
  --threads=8 \
  --enable_http_server=true \
  --http_port="${MOONCAKE_CLIENT_HTTP_PORT}" \
  >"${RUN_DIR}/logs/mooncake-client.log" 2>&1 &
service_pids+=("$!")
wait_http mooncake-client "http://127.0.0.1:${MOONCAKE_CLIENT_HTTP_PORT}/health" "${service_pids[-1]}"

mooncake_config="{\"master_server_address\":\"127.0.0.1:${MOONCAKE_MASTER_PORT}\",\"local_hostname\":\"${MOONCAKE_LOCAL_HOSTNAME}\",\"metadata_server\":\"http://127.0.0.1:${MOONCAKE_METADATA_PORT}/metadata\",\"global_segment_size\":\"0\",\"protocol\":\"tcp\",\"device_name\":\"\",\"prefetch_threshold\":${PD_HICACHE_PREFETCH_THRESHOLD},\"prefetch_timeout_base\":${PD_HICACHE_PREFETCH_TIMEOUT_BASE},\"prefetch_timeout_per_ki_token\":${PD_HICACHE_PREFETCH_TIMEOUT_PER_KI_TOKEN}}"

setsid env \
  PATH="${PD_ENV_BIN}:${PATH}" \
  SLIME_HTTP_READ_TIMEOUT_SECONDS="${SLIME_HTTP_READ_TIMEOUT_SECONDS:-3600}" \
  RUN_DIR="${RUN_DIR}" \
  PREFILL_GPU="${PREFILL_GPU:-0}" \
  DECODE_GPUS="${DECODE_GPUS:-1}" \
  DECODE_PORTS="${DECODE_PORTS:-35401}" \
  SEARCH_GPU="${SEARCH_GPU:-2}" \
  PREFILL_PORT="${PREFILL_PORT:-35400}" \
  ROUTER_PORT="${ROUTER_PORT:-35402}" \
  ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-29400}" \
  BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-9540}" \
  SEARCH_PORT="${SEARCH_PORT:-8540}" \
  MATH_RATIO="${MATH_RATIO:-0.5}" \
  DISPATCH_POLICY="${DISPATCH_POLICY:-fixed}" \
  SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n4096.json}" \
  SEED="${SEED:-2026}" \
  REQUESTS="${REQUESTS:-4096}" \
  WARMUP_REQUESTS=0 \
  MAX_INFLIGHT="${MAX_INFLIGHT:-128}" \
  CLOSED_LOOP=1 \
  CLOSED_LOOP_WARMUP_MIN_SECONDS="${CLOSED_LOOP_WARMUP_MIN_SECONDS:-600}" \
  CLOSED_LOOP_WARMUP_COMPLETIONS="${CLOSED_LOOP_WARMUP_COMPLETIONS:-128}" \
  CLOSED_LOOP_RECENT_SECONDS="${CLOSED_LOOP_RECENT_SECONDS:-180}" \
  CLOSED_LOOP_MAX_WARMUP_SECONDS="${CLOSED_LOOP_MAX_WARMUP_SECONDS:-2400}" \
  CLOSED_LOOP_MEASUREMENT_SECONDS="${CLOSED_LOOP_MEASUREMENT_SECONDS:-1200}" \
  ARRIVAL_RATE=100 \
  ARRIVAL_DISTRIBUTION=fixed \
  TEMPERATURE=0 \
  TOP_P=1 \
  TOP_K=-1 \
  METRICS_INTERVAL=2 \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}" \
  PD_HICACHE_STORAGE_BACKEND=mooncake \
  PD_HICACHE_STORAGE_DIR="${PD_HICACHE_STORAGE_DIR:-${RUN_DIR}/hicache}" \
  PD_HICACHE_STORAGE_PREFETCH_POLICY="${PD_HICACHE_STORAGE_PREFETCH_POLICY:-timeout}" \
  PD_HICACHE_PREFETCH_THRESHOLD="${PD_HICACHE_PREFETCH_THRESHOLD}" \
  PD_HICACHE_STORAGE_EXTRA_CONFIG="${mooncake_config}" \
  PD_PREFILL_HICACHE_SIZE_GB="${PD_PREFILL_HICACHE_SIZE_GB:-128}" \
  PD_DECODE_HICACHE_SIZE_GB="${PD_DECODE_HICACHE_SIZE_GB:-56}" \
  PD_HICACHE_MEM_LAYOUT="${PD_HICACHE_MEM_LAYOUT:-page_first}" \
  PD_PAGE_SIZE="${PD_PAGE_SIZE:-64}" \
  PD_ENABLE_DECODE_OFFLOAD_KVCACHE="${PD_ENABLE_DECODE_OFFLOAD_KVCACHE:-1}" \
  PD_MAX_TRANSFER_INFLIGHT="${PD_MAX_TRANSFER_INFLIGHT:-0}" \
  PD_P_READY_DIR="${PD_P_READY_DIR:-}" \
  SGLANG_PD_P_READY_PHASE="${SGLANG_PD_P_READY_PHASE:-finished}" \
  PYTHONPATH="${SGLANG_OVERLAY_ROOT:+${SGLANG_OVERLAY_ROOT}:}${PYTHONPATH:-}" \
  bash "${PD_RUN_QWEN_SCRIPT}" &
experiment_pid=$!
wait "${experiment_pid}"
experiment_pid=""
