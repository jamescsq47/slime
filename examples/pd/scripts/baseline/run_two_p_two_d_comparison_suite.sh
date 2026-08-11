#!/usr/bin/env bash
set -euo pipefail

# Reproducible 2P:2D comparison using the same fixed Mixed 1:1 workload as
# run_four_gpu_comparison_suite.sh.  Cases run sequentially on model GPUs
# 0-3; GPU 7 is reserved for the BrowseComp embedding/search server.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/baseline/four-gpu-2p2d-c256-s2026-w300-m1200}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
MAX_INFLIGHT="${MAX_INFLIGHT:-256}"
REQUESTS="${REQUESTS:-4096}"
SEED="${SEED:-2026}"
SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n4096.json}"
SEARCH_GPU="${SEARCH_GPU:-7}"

mkdir -p "${RUN_ROOT}"

complete_run() {
  local run_dir="$1"
  [[ -f "${run_dir}/offload_analysis_summary.json" ]] || return 1
  "${PD_ENV_BIN}/python" - "${run_dir}/offload_analysis_summary.json" "${MEASURE_SECONDS}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
complete = (
    summary.get("status") == "steady_churn"
    and summary.get("measurement_seconds", 0) >= float(sys.argv[2]) - 2
)
raise SystemExit(0 if complete else 1)
PY
}

run_case() {
  local name="$1"
  shift
  local run_dir="${RUN_ROOT}/${name}"
  if complete_run "${run_dir}"; then
    echo "skip completed ${name}: ${run_dir}"
    return
  fi
  echo "start ${name} at $(date -u +%FT%TZ)"
  env RUN_DIR="${run_dir}" PD_ENV_BIN="${PD_ENV_BIN}" \
    WARMUP_SECONDS="${WARMUP_SECONDS}" MEASURE_SECONDS="${MEASURE_SECONDS}" \
    MAX_INFLIGHT="${MAX_INFLIGHT}" REQUESTS="${REQUESTS}" SEED="${SEED}" \
    SCHEDULE_FILE="${SCHEDULE_FILE}" SEARCH_GPU="${SEARCH_GPU}" \
    SLIME_HTTP_READ_TIMEOUT_SECONDS=3600 "$@"
  echo "finish ${name} at $(date -u +%FT%TZ)"
}

run_case pd-no-reverse-2p2d \
  CASE_MODE=no_reverse PREFILL_GPUS='0 1' PREFILL_PORTS='28500 28501' \
  PREFILL_BOOTSTRAP_PORTS='29500 29501' DECODE_GPUS='2 3' \
  DECODE_PORTS='28502 28503' ROUTER_PORT=28510 \
  ROUTER_PROMETHEUS_PORT=28520 SEARCH_PORT=8850 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case pd-native-mooncake-2p2d \
  CASE_MODE=native_mooncake PREFILL_GPUS='0 1' PREFILL_PORTS='28600 28601' \
  PREFILL_BOOTSTRAP_PORTS='29600 29601' DECODE_GPUS='2 3' \
  DECODE_PORTS='28602 28603' ROUTER_PORT=28610 \
  ROUTER_PROMETHEUS_PORT=28620 SEARCH_PORT=8860 \
  MOONCAKE_MASTER_PORT=58651 MOONCAKE_CLIENT_PORT=58652 \
  MOONCAKE_METADATA_PORT=58680 MOONCAKE_METRICS_PORT=58603 \
  MOONCAKE_CLIENT_HTTP_PORT=58690 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/compare_two_p_two_d_baselines.py" \
  --run-root "${RUN_ROOT}"
echo "2P:2D baseline comparison complete: ${RUN_ROOT}"
