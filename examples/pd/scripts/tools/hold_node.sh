#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
pd_install_cleanup_traps

HOLDER_ROOT="${HOLDER_ROOT:-${PD_DIR}/runs/node-holder-1p7d-retool}"
mkdir -p "${HOLDER_ROOT}"
iteration=0
while true; do
  iteration=$((iteration + 1))
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  run_dir="${HOLDER_ROOT}/${stamp}-iteration-${iteration}"
  echo "[$(date -u --iso-8601=seconds)] holder starting ${run_dir}"
  setsid env \
    RUN_DIR="${run_dir}" PD_ENV_BIN=/homes/siqic/anaconda3/envs/pd/bin \
    PREFILL_GPU=0 DECODE_GPUS='1 2 3 4 5 6 7' \
    DECODE_PORTS='27701 27702 27703 27704 27705 27706 27707' \
    PREFILL_PORT=27700 ROUTER_PORT=27710 ROUTER_PROMETHEUS_PORT=27720 \
    BOOTSTRAP_PORT=28700 SEARCH_GPU=7 SEARCH_PORT=8770 PD_SKIP_SEARCH=1 \
    MOONCAKE_MASTER_PORT=57751 MOONCAKE_CLIENT_PORT=57752 \
    MOONCAKE_METADATA_PORT=57780 MOONCAKE_METRICS_PORT=57703 \
    MOONCAKE_CLIENT_HTTP_PORT=57790 MATH_RATIO=1.0 DISPATCH_POLICY=random \
    REQUESTS=16384 MAX_INFLIGHT=512 WARMUP_SECONDS=300 \
    MAX_WARMUP_SECONDS=420 MEASURE_SECONDS=43200 DIRECT_WAIT_SECONDS=10 \
    bash "${PD_DIR}/scripts/new_method/run_1p5d_case.sh" &
  child=$!; pd_track_group "${child}"
  set +e
  wait "${child}"
  status=$?
  set -e
  echo "[$(date -u --iso-8601=seconds)] holder segment exited ${status}; restart in 10s"
  sleep 10
done
