#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-9540}"
DECODE_GPUS="${DECODE_GPUS:-1}"

created_ready_dir=0
created_ledger=0
created_staging_ledger=0
created_host_arena=0
service_runner_pid=""
if [[ -z "${PD_P_READY_DIR:-}" ]]; then
  PD_P_READY_DIR="$(mktemp -d "/dev/shm/sglang-agentic-p-ready-${BOOTSTRAP_PORT}.XXXXXX")"
  created_ready_dir=1
fi
if [[ -z "${SGLANG_AGENTIC_KV_LEDGER_PATH:-}" ]]; then
  SGLANG_AGENTIC_KV_LEDGER_PATH="/dev/shm/sglang-agentic-kv-ledger-${BOOTSTRAP_PORT}.json"
  created_ledger=1
  rm -f -- "${SGLANG_AGENTIC_KV_LEDGER_PATH}"
fi
if [[ -z "${SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH:-}" ]]; then
  SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH="${SGLANG_AGENTIC_KV_LEDGER_PATH}.staging"
  created_staging_ledger=1
  rm -f -- "${SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH}"
fi
if [[ -z "${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR:-}" ]]; then
  SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR="$(mktemp -d "/dev/shm/sglang-agentic-host-arena-${BOOTSTRAP_PORT}.XXXXXX")"
  created_host_arena=1
fi

cleanup_ready_dir() {
  if (( created_ready_dir == 1 )) && [[ -d "${PD_P_READY_DIR}" ]]; then
    # Early-claim markers live in nested arrivals/routes/finals directories.
    # Remove the complete per-run mktemp tree after every owned service group
    # has exited; a maxdepth=1 cleanup leaves thousands of stale markers.
    find "${PD_P_READY_DIR}" -mindepth 1 -depth -delete
    rmdir "${PD_P_READY_DIR}" 2>/dev/null || true
  fi
  if (( created_ledger == 1 )); then
    rm -f -- "${SGLANG_AGENTIC_KV_LEDGER_PATH}"
  fi
  if (( created_staging_ledger == 1 )); then
    rm -f -- "${SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH}"
  fi
  if (( created_host_arena == 1 )) && [[ -d "${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR}" ]]; then
    find "${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR}" -mindepth 1 -type f -delete
    find "${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR}" -depth -type d -empty -delete
  fi
}

cleanup_pipeline() {
  local deadline
  if [[ -n "${service_runner_pid}" ]] && kill -0 "${service_runner_pid}" 2>/dev/null; then
    kill -TERM -- "-${service_runner_pid}" 2>/dev/null || true
    deadline=$((SECONDS + 45))
    while kill -0 "${service_runner_pid}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
    kill -KILL -- "-${service_runner_pid}" 2>/dev/null || true
    wait "${service_runner_pid}" 2>/dev/null || true
  fi
  cleanup_ready_dir
}
trap cleanup_pipeline EXIT
trap 'exit 130' INT TERM

read -r -a decode_gpu_array <<<"${DECODE_GPUS}"
decode_writers="${#decode_gpu_array[@]}"

export SGLANG_AGENTIC_KV_LIFECYCLE=true
export SGLANG_AGENTIC_KV_CAPACITY_GIB="${SGLANG_AGENTIC_KV_CAPACITY_GIB:-256}"
export SGLANG_AGENTIC_KV_D_WRITERS="${SGLANG_AGENTIC_KV_D_WRITERS:-${decode_writers}}"
export SGLANG_AGENTIC_KV_LEDGER_PATH
export SGLANG_AGENTIC_KV_HOST_STAGING="${SGLANG_AGENTIC_KV_HOST_STAGING:-true}"
export SGLANG_AGENTIC_KV_D_HOSTLESS="${SGLANG_AGENTIC_KV_D_HOSTLESS:-true}"
export SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB="${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB:-128}"
export SGLANG_AGENTIC_KV_RELAY_ENABLED="${SGLANG_AGENTIC_KV_RELAY_ENABLED:-true}"
export SGLANG_AGENTIC_KV_RELAY_D2H_GIBPS="${SGLANG_AGENTIC_KV_RELAY_D2H_GIBPS:-21.0}"
export SGLANG_AGENTIC_KV_DIRECT_CROSS_NUMA_GIBPS="${SGLANG_AGENTIC_KV_DIRECT_CROSS_NUMA_GIBPS:-7.45}"
export SGLANG_AGENTIC_KV_RELAY_NVLINK_GIBPS="${SGLANG_AGENTIC_KV_RELAY_NVLINK_GIBPS:-220.0}"
export SGLANG_AGENTIC_KV_RELAY_STALE_SECONDS="${SGLANG_AGENTIC_KV_RELAY_STALE_SECONDS:-5.0}"
# A 64 MiB relay slot turns a typical 0.6--1.5 GiB agent snapshot into
# 10--24 cross-process state handshakes.  Two 256 MiB slots cost only 512 MiB
# of D HBM while reducing that control-plane amplification by roughly 4x.
export SGLANG_AGENTIC_KV_STAGING_SLOT_MIB="${SGLANG_AGENTIC_KV_STAGING_SLOT_MIB:-256}"
export SGLANG_AGENTIC_KV_STAGING_SLOTS="${SGLANG_AGENTIC_KV_STAGING_SLOTS:-2}"
export SGLANG_AGENTIC_KV_P_HOST_HIGH_WATERMARK="${SGLANG_AGENTIC_KV_P_HOST_HIGH_WATERMARK:-0.80}"
export SGLANG_AGENTIC_KV_P_HOST_LOW_WATERMARK="${SGLANG_AGENTIC_KV_P_HOST_LOW_WATERMARK:-0.70}"
export SGLANG_AGENTIC_KV_P_HOST_HARD_WATERMARK="${SGLANG_AGENTIC_KV_P_HOST_HARD_WATERMARK:-0.90}"
export SGLANG_AGENTIC_KV_HIGH_WATERMARK="${SGLANG_AGENTIC_KV_HIGH_WATERMARK:-0.90}"
if [[ -z "${SGLANG_AGENTIC_KV_TOOL_MEAN_SECONDS+x}" ]]; then
  SGLANG_AGENTIC_KV_TOOL_MEAN_SECONDS='{}'
fi
export SGLANG_AGENTIC_KV_TOOL_MEAN_SECONDS
export SGLANG_AGENTIC_KV_READY_TIMEOUT="${SGLANG_AGENTIC_KV_READY_TIMEOUT:-120}"
export SGLANG_AGENTIC_KV_STALE_SECONDS="${SGLANG_AGENTIC_KV_STALE_SECONDS:-300}"
export SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD="${SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD:-2.0}"
export SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT="${SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT:-2.0}"
# Split the fast-path deadline into the two phases required by the agentic
# pipeline.  The first deadline classifies the tool as fast; only after the
# parent turn actually returns does D start the P-admission deadline.  Once P
# immediately allocates exact-size P pages from the arrival marker;
# DIRECT_HANDSHAKE_TIMEOUT bounds receiver setup/DMA, not request scheduling.
export SGLANG_AGENTIC_KV_EARLY_CLAIM="${SGLANG_AGENTIC_KV_EARLY_CLAIM:-true}"
export SGLANG_AGENTIC_KV_EARLY_CLAIM_POST_TIMEOUT="${SGLANG_AGENTIC_KV_EARLY_CLAIM_POST_TIMEOUT:-${SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT}}"
export SGLANG_AGENTIC_KV_DIRECT_D_HBM_HIGH_WATERMARK="${SGLANG_AGENTIC_KV_DIRECT_D_HBM_HIGH_WATERMARK:-0.85}"
export SGLANG_AGENTIC_KV_DIRECT_MANIFEST_POLL_INTERVAL="${SGLANG_AGENTIC_KV_DIRECT_MANIFEST_POLL_INTERVAL:-0.10}"
export SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT="${SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT:-8}"
# Do not impose an artificial request/token credit on Prefill.  Native SGLang
# admission remains the final HBM-safety boundary, while the waiting queue
# still preserves Direct > slow recovery > new priority.  Experiments can
# explicitly restore continuous/hysteresis backpressure through the env var.
export SGLANG_PD_P_READY_BACKPRESSURE_MODE="${SGLANG_PD_P_READY_BACKPRESSURE_MODE:-disabled}"
export SGLANG_PD_P_READY_REQUEST_CAP="${SGLANG_PD_P_READY_REQUEST_CAP:-0}"
# Direct receives use independent, exact-size page allocations.  They outrank
# slow recovery and new Prefill; eight concurrent rooms absorb short bursts
# without reserving a permanent HBM buffer.
export SGLANG_AGENTIC_KV_DIRECT_IO_CAP="${SGLANG_AGENTIC_KV_DIRECT_IO_CAP:-8}"
export SGLANG_AGENTIC_KV_P_DIRECT_RESERVE_TOKENS="${SGLANG_AGENTIC_KV_P_DIRECT_RESERVE_TOKENS:-40000}"
export SGLANG_PD_DECODE_ENABLE_RADIX_CACHE="${SGLANG_PD_DECODE_ENABLE_RADIX_CACHE:-true}"
# Bound slow-path pressure on Decode.  Only one gather+D2H chunk is submitted
# at a time on each D; the background worker returns to Decode before issuing
# the next chunk.  Completed KV releases are likewise committed one request at
# a time on the allocator-owning scheduler thread.
export SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS="${SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS:-512}"
export SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS="${SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS:-512}"
export SGLANG_DECODE_IO_MAX_COMMITS_PER_STEP="${SGLANG_DECODE_IO_MAX_COMMITS_PER_STEP:-1}"
# Once the scheduler has allocated exact P pages and launched reverse NIXL,
# keep receiver progress independent of long Prefill forwards.  The worker
# only polls transport; allocator/Radix ownership remains scheduler-only.
export SGLANG_AGENTIC_KV_P_DIRECT_PROGRESS_INTERVAL_SECONDS="${SGLANG_AGENTIC_KV_P_DIRECT_PROGRESS_INTERVAL_SECONDS:-0.005}"
# The Decode transport/control worker must not rescan the shared agentic
# ledger on every scheduler tick.  A 20 ms control-plane cadence prevents the
# background Python thread from contending with Decode for the GIL, while the
# independent NIXL completion worker keeps its 2 ms data-plane cadence.
export SGLANG_AGENTIC_KV_D_CONTROL_POLL_SECONDS="${SGLANG_AGENTIC_KV_D_CONTROL_POLL_SECONDS:-0.02}"
# P->D transport is the Decode supply path, so it has a dedicated high-rate
# worker. Preallocation and agentic lifecycle control run on separate workers
# and cannot block it. Releases are still coalesced briefly so the paged
# allocator can free a burst with one torch.unique/torch.cat.
export SGLANG_DECODE_TRANSFER_PROGRESS_INTERVAL_SECONDS="${SGLANG_DECODE_TRANSFER_PROGRESS_INTERVAL_SECONDS:-0.005}"
export SGLANG_DECODE_PREALLOC_PROGRESS_INTERVAL_SECONDS="${SGLANG_DECODE_PREALLOC_PROGRESS_INTERVAL_SECONDS:-0.005}"
export SGLANG_DECODE_IO_COMMIT_INTERVAL_SECONDS="${SGLANG_DECODE_IO_COMMIT_INTERVAL_SECONDS:-0.02}"
# Sender completion polling is also transport work. Keep it off the P
# scheduler so a slow NIXL status query cannot freeze Prefill admission.
export SGLANG_PREFILL_TRANSFER_ASYNC_PROGRESS="${SGLANG_PREFILL_TRANSFER_ASYNC_PROGRESS:-1}"
export SGLANG_PREFILL_TRANSFER_PROGRESS_INTERVAL_SECONDS="${SGLANG_PREFILL_TRANSFER_PROGRESS_INTERVAL_SECONDS:-0.005}"
# Bound ordinary work in P's HTTP/tokenizer pipeline.  Direct receive no
# longer waits for this pipeline: the router marker independently triggers
# P-page allocation and reverse NIXL, then the tokenized request binds the KV.
export SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT="${SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT:-4}"
# P-ready backpressure can intentionally leave a fresh request below parent
# work for minutes.  Waiting for the scheduler's `.accepted` marker must use
# the same lifecycle horizon as P-ready, otherwise the router returns a 500
# every 30 seconds while the original request is still valid in P's queue.
export SGLANG_PD_LATE_BIND_ACCEPT_TIMEOUT_S="${SGLANG_PD_LATE_BIND_ACCEPT_TIMEOUT_S:-${PD_LATE_BIND_READY_TIMEOUT_S:-600}}"
# Mooncake's TCP transport connection pool exists in the installed 0.3.12
# build but is opt-in.  Agentic request-generation manifests are updated
# frequently; without pooling every tiny update creates a short-lived TCP
# connection and a sustained run can exhaust the kernel ephemeral-port range.
export MC_TCP_ENABLE_CONNECTION_POOL="${MC_TCP_ENABLE_CONNECTION_POOL:-1}"

setsid env \
  BOOTSTRAP_PORT="${BOOTSTRAP_PORT}" \
  DECODE_GPUS="${DECODE_GPUS}" \
  PD_PAGE_SIZE="${PD_PAGE_SIZE:-64}" \
  PD_PREFILL_HICACHE_WRITE_POLICY="${PD_PREFILL_HICACHE_WRITE_POLICY:-write_back}" \
  PD_ENABLE_DECODE_OFFLOAD_KVCACHE=1 \
  PD_MAX_TRANSFER_INFLIGHT="${PD_MAX_TRANSFER_INFLIGHT:-8}" \
  PD_P_READY_DIR="${PD_P_READY_DIR}" \
  MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO="${MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO:-0.98}" \
  MOONCAKE_EVICTION_RATIO="${MOONCAKE_EVICTION_RATIO:-0.02}" \
  bash "${SCRIPT_DIR}/run_hicache_services.sh" &
service_runner_pid=$!
wait "${service_runner_pid}"
service_runner_pid=""
