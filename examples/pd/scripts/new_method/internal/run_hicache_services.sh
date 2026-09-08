#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PD_DIR}"

# Compatibility entry point retained for existing experiment scripts.  The
# new method has no Mooncake service: lifecycle metadata lives in /dev/shm and
# KV payloads use only Direct and the two Shared Host Arenas.
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/new-method/manual-agentic-pd}"
PD_RUN_QWEN_SCRIPT="${PD_RUN_QWEN_SCRIPT:-${SCRIPT_DIR}/run_pd_servers.sh}"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd/bin}"
mkdir -p "${RUN_DIR}/logs"

experiment_pid=""
cleanup() {
  trap - INT TERM
  if [[ -n "${experiment_pid}" ]]; then
    kill -TERM -- "-${experiment_pid}" 2>/dev/null || true
    wait "${experiment_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

setsid env \
  PATH="${PD_ENV_BIN}:${PATH}" \
  RUN_DIR="${RUN_DIR}" \
  PD_HICACHE_STORAGE_BACKEND="" \
  PD_HICACHE_STORAGE_EXTRA_CONFIG="" \
  PD_HICACHE_SIZE_GB=0 \
  PD_PREFILL_HICACHE_SIZE_GB=0 \
  PD_DECODE_HICACHE_SIZE_GB=0 \
  PD_ENABLE_DECODE_OFFLOAD_KVCACHE=0 \
  SGLANG_AGENTIC_KV_METADATA_DIR="${SGLANG_AGENTIC_KV_METADATA_DIR:-${PD_P_READY_DIR}/snapshot-metadata}" \
  PYTHONPATH="${SGLANG_OVERLAY_ROOT:+${SGLANG_OVERLAY_ROOT}:}${PYTHONPATH:-}" \
  bash "${PD_RUN_QWEN_SCRIPT}" &
experiment_pid=$!
wait "${experiment_pid}"
experiment_pid=""
