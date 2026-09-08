#!/usr/bin/env bash

# Shared process-lifecycle helpers for reproducible PD experiments.  Every
# long-lived service is placed in its own setsid process group and registered
# here so normal completion, Ctrl-C, and launcher failure all use the same
# bounded TERM -> KILL cleanup path.

PD_SERVICE_PGIDS=()

pd_group_has_live_members() {
  local pgid="$1"
  ps -eLo pgid=,stat= | awk -v target="${pgid}" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

pd_track_group() {
  PD_SERVICE_PGIDS+=("$1")
}

pd_stop_group() {
  local pgid="$1" deadline alive
  pd_group_has_live_members "${pgid}" || { wait "${pgid}" 2>/dev/null || true; return; }
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  deadline=$((SECONDS + ${PD_CLEANUP_GRACE_SECONDS:-30}))
  while (( SECONDS < deadline )); do
    pd_group_has_live_members "${pgid}" || break
    sleep 1
  done
  if pd_group_has_live_members "${pgid}"; then
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
  wait "${pgid}" 2>/dev/null || true
}

pd_cleanup_all() {
  local index pgid deadline any_live
  trap - EXIT INT TERM
  # Stop every service first, then share one grace period.  Waiting up to the
  # full grace period for each group serially can outlive the parent launcher;
  # later groups would then be adopted by pid 1 and survive as GPU orphans.
  for ((index=${#PD_SERVICE_PGIDS[@]} - 1; index >= 0; index--)); do
    pgid="${PD_SERVICE_PGIDS[index]}"
    if pd_group_has_live_members "${pgid}"; then
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + ${PD_CLEANUP_GRACE_SECONDS:-30}))
  while (( SECONDS < deadline )); do
    any_live=0
    for pgid in "${PD_SERVICE_PGIDS[@]}"; do
      if pd_group_has_live_members "${pgid}"; then
        any_live=1
        break
      fi
    done
    (( any_live == 0 )) && break
    sleep 1
  done
  for ((index=${#PD_SERVICE_PGIDS[@]} - 1; index >= 0; index--)); do
    pgid="${PD_SERVICE_PGIDS[index]}"
    if pd_group_has_live_members "${pgid}"; then
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
    wait "${pgid}" 2>/dev/null || true
  done
  PD_SERVICE_PGIDS=()
}

pd_signal_exit() {
  exit 130
}

pd_install_cleanup_traps() {
  trap pd_cleanup_all EXIT
  trap pd_signal_exit INT TERM
}

pd_wait_http() {
  local name="$1" url="$2" pgid="$3"
  local deadline=$((SECONDS + ${4:-900}))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if ! pd_group_has_live_members "${pgid}"; then
      echo "${name} exited before becoming healthy" >&2
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for ${name}: ${url}" >&2
  return 1
}

pd_check_gpu_idle() {
  local gpu="$1" used
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( used > ${PD_IDLE_GPU_MAX_MIB:-1024} )); then
    echo "GPU ${gpu} is occupied: ${used} MiB" >&2
    return 1
  fi
}

pd_check_port_free() {
  local port="$1"
  if ss -H -ltn "sport = :${port}" | grep -q .; then
    echo "TCP port ${port} is already in use" >&2
    return 1
  fi
}
