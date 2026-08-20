#!/usr/bin/env bash
set -euo pipefail

# Eight sequential stock-SGLang PD baselines on four model GPUs:
#   * BrowseComp-only: 3P:1D and 2P:2D, each with and without native Mooncake.
#   * Retool-only:     1P:3D and 2P:2D, each with and without native Mooncake.
# GPU 7 hosts the BrowseComp search model.  The fixed schedules make task type
# and row order deterministic across the two KV policies for each workload.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/baseline/pure-pd-c256-s2026-w300-m1200}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
MAX_INFLIGHT="${MAX_INFLIGHT:-256}"
REQUESTS="${REQUESTS:-4096}"
SEED="${SEED:-2026}"
SEARCH_GPU="${SEARCH_GPU:-7}"
BROWSECOMP_DATA="${BROWSECOMP_DATA:-${PD_DIR}/runs-host/baseline/refresh-pure-colocated-4gpu-c256-s2026-w300-m1200-20260812/input/bc_train_repeated_n4096.jsonl}"
RETOOL_DATA="${RETOOL_DATA:-/homes/siqic/data/dapo-math-17k/dapo-math-17k.jsonl}"
BROWSECOMP_SCHEDULE="${BROWSECOMP_SCHEDULE:-${PD_DIR}/configs/workloads/fixed_browsecomp_s2026_n4096.json}"
RETOOL_SCHEDULE="${RETOOL_SCHEDULE:-${PD_DIR}/configs/workloads/fixed_retool_s2026_n4096.json}"

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
    SEARCH_GPU="${SEARCH_GPU}" SLIME_HTTP_READ_TIMEOUT_SECONDS=3600 "$@"
  echo "finish ${name} at $(date -u +%FT%TZ)"
}

# BrowseComp-only, 3P:1D.
run_case browsecomp-no-reverse-3p1d \
  CASE_MODE=no_reverse MATH_RATIO=0 QA_DATA="${BROWSECOMP_DATA}" \
  SCHEDULE_FILE="${BROWSECOMP_SCHEDULE}" \
  PREFILL_GPUS='0 1 2' PREFILL_PORTS='30100 30101 30102' \
  PREFILL_BOOTSTRAP_PORTS='31100 31101 31102' DECODE_GPUS='3' \
  DECODE_PORTS='30103' ROUTER_PORT=30110 ROUTER_PROMETHEUS_PORT=30120 \
  SEARCH_PORT=9010 bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case browsecomp-native-mooncake-3p1d \
  CASE_MODE=native_mooncake MATH_RATIO=0 QA_DATA="${BROWSECOMP_DATA}" \
  SCHEDULE_FILE="${BROWSECOMP_SCHEDULE}" \
  PREFILL_GPUS='0 1 2' PREFILL_PORTS='30200 30201 30202' \
  PREFILL_BOOTSTRAP_PORTS='31200 31201 31202' DECODE_GPUS='3' \
  DECODE_PORTS='30203' ROUTER_PORT=30210 ROUTER_PROMETHEUS_PORT=30220 \
  SEARCH_PORT=9020 MOONCAKE_MASTER_PORT=60251 MOONCAKE_CLIENT_PORT=60252 \
  MOONCAKE_METADATA_PORT=60280 MOONCAKE_METRICS_PORT=60203 \
  MOONCAKE_CLIENT_HTTP_PORT=60290 bash "${SCRIPT_DIR}/run_pd_case.sh"

# Retool-only, 2P:2D.
run_case retool-no-reverse-2p2d \
  CASE_MODE=no_reverse MATH_RATIO=1 MATH_DATA="${RETOOL_DATA}" \
  SCHEDULE_FILE="${RETOOL_SCHEDULE}" \
  PREFILL_GPUS='0 1' PREFILL_PORTS='30300 30301' \
  PREFILL_BOOTSTRAP_PORTS='31300 31301' DECODE_GPUS='2 3' \
  DECODE_PORTS='30302 30303' ROUTER_PORT=30310 ROUTER_PROMETHEUS_PORT=30320 \
  SEARCH_PORT=9030 bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case retool-native-mooncake-2p2d \
  CASE_MODE=native_mooncake MATH_RATIO=1 MATH_DATA="${RETOOL_DATA}" \
  SCHEDULE_FILE="${RETOOL_SCHEDULE}" \
  PREFILL_GPUS='0 1' PREFILL_PORTS='30400 30401' \
  PREFILL_BOOTSTRAP_PORTS='31400 31401' DECODE_GPUS='2 3' \
  DECODE_PORTS='30402 30403' ROUTER_PORT=30410 ROUTER_PROMETHEUS_PORT=30420 \
  SEARCH_PORT=9040 MOONCAKE_MASTER_PORT=60451 MOONCAKE_CLIENT_PORT=60452 \
  MOONCAKE_METADATA_PORT=60480 MOONCAKE_METRICS_PORT=60403 \
  MOONCAKE_CLIENT_HTTP_PORT=60490 bash "${SCRIPT_DIR}/run_pd_case.sh"

# Retool-only, 1P:3D.
run_case retool-no-reverse-1p3d \
  CASE_MODE=no_reverse MATH_RATIO=1 MATH_DATA="${RETOOL_DATA}" \
  SCHEDULE_FILE="${RETOOL_SCHEDULE}" \
  PREFILL_GPUS='0' PREFILL_PORTS='30500' PREFILL_BOOTSTRAP_PORTS='31500' \
  DECODE_GPUS='1 2 3' DECODE_PORTS='30501 30502 30503' \
  ROUTER_PORT=30510 ROUTER_PROMETHEUS_PORT=30520 SEARCH_PORT=9050 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case retool-native-mooncake-1p3d \
  CASE_MODE=native_mooncake MATH_RATIO=1 MATH_DATA="${RETOOL_DATA}" \
  SCHEDULE_FILE="${RETOOL_SCHEDULE}" \
  PREFILL_GPUS='0' PREFILL_PORTS='30600' PREFILL_BOOTSTRAP_PORTS='31600' \
  DECODE_GPUS='1 2 3' DECODE_PORTS='30601 30602 30603' \
  ROUTER_PORT=30610 ROUTER_PROMETHEUS_PORT=30620 SEARCH_PORT=9060 \
  MOONCAKE_MASTER_PORT=60651 MOONCAKE_CLIENT_PORT=60652 \
  MOONCAKE_METADATA_PORT=60680 MOONCAKE_METRICS_PORT=60603 \
  MOONCAKE_CLIENT_HTTP_PORT=6090 bash "${SCRIPT_DIR}/run_pd_case.sh"

# BrowseComp-only, 2P:2D.
run_case browsecomp-no-reverse-2p2d \
  CASE_MODE=no_reverse MATH_RATIO=0 QA_DATA="${BROWSECOMP_DATA}" \
  SCHEDULE_FILE="${BROWSECOMP_SCHEDULE}" \
  PREFILL_GPUS='0 1' PREFILL_PORTS='30700 30701' \
  PREFILL_BOOTSTRAP_PORTS='31700 31701' DECODE_GPUS='2 3' \
  DECODE_PORTS='30702 30703' ROUTER_PORT=30710 ROUTER_PROMETHEUS_PORT=30720 \
  SEARCH_PORT=9070 bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case browsecomp-native-mooncake-2p2d \
  CASE_MODE=native_mooncake MATH_RATIO=0 QA_DATA="${BROWSECOMP_DATA}" \
  SCHEDULE_FILE="${BROWSECOMP_SCHEDULE}" \
  PREFILL_GPUS='0 1' PREFILL_PORTS='30800 30801' \
  PREFILL_BOOTSTRAP_PORTS='31800 31801' DECODE_GPUS='2 3' \
  DECODE_PORTS='30802 30803' ROUTER_PORT=30810 ROUTER_PROMETHEUS_PORT=30820 \
  SEARCH_PORT=9080 MOONCAKE_MASTER_PORT=60851 MOONCAKE_CLIENT_PORT=60852 \
  MOONCAKE_METADATA_PORT=60880 MOONCAKE_METRICS_PORT=60803 \
  MOONCAKE_CLIENT_HTTP_PORT=6091 bash "${SCRIPT_DIR}/run_pd_case.sh"

"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/compare_pure_workload_pd_baselines.py" \
  --run-root "${RUN_ROOT}"
echo "Pure-workload PD baseline suite complete: ${RUN_ROOT}"
