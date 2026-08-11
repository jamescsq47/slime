#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/baseline/six-gpu-architecture-comparison-c384-s2026}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
MAX_INFLIGHT="${MAX_INFLIGHT:-384}"
mkdir -p "${RUN_ROOT}"

complete_run() {
  local run_dir="$1"
  [[ -f "${run_dir}/offload_analysis_summary.json" ]] || return 1
  python3 - "${run_dir}/offload_analysis_summary.json" "${MEASURE_SECONDS}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("status") == "steady_churn" and x.get("measurement_seconds",0) >= float(sys.argv[2])-2 else 1)
PY
}

run_case() {
  local name="$1"; shift
  local run_dir="${RUN_ROOT}/${name}"
  if complete_run "${run_dir}"; then
    echo "skip completed ${name}"
    return
  fi
  env RUN_DIR="${run_dir}" WARMUP_SECONDS="${WARMUP_SECONDS}" \
    MEASURE_SECONDS="${MEASURE_SECONDS}" MAX_INFLIGHT="${MAX_INFLIGHT}" "$@"
}

run_case colocated-6gpu \
  bash "${SCRIPT_DIR}/run_colocated_case.sh"

run_case pd-no-reverse-1p5d \
  CASE_MODE=no_reverse PREFILL_GPUS='0' PREFILL_PORTS='27400' \
  PREFILL_BOOTSTRAP_PORTS='28400' DECODE_GPUS='1 2 3 4 5' \
  DECODE_PORTS='27401 27402 27403 27404 27405' ROUTER_PORT=27410 \
  ROUTER_PROMETHEUS_PORT=27420 SEARCH_PORT=8740 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case pd-no-reverse-2p4d \
  CASE_MODE=no_reverse PREFILL_GPUS='0 1' PREFILL_PORTS='27500 27501' \
  PREFILL_BOOTSTRAP_PORTS='28500 28501' DECODE_GPUS='2 3 4 5' \
  DECODE_PORTS='27502 27503 27504 27505' ROUTER_PORT=27510 \
  ROUTER_PROMETHEUS_PORT=27520 SEARCH_PORT=8750 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case pd-native-mooncake-1p5d \
  CASE_MODE=native_mooncake PREFILL_GPUS='0' PREFILL_PORTS='27600' \
  PREFILL_BOOTSTRAP_PORTS='28600' DECODE_GPUS='1 2 3 4 5' \
  DECODE_PORTS='27601 27602 27603 27604 27605' ROUTER_PORT=27610 \
  ROUTER_PROMETHEUS_PORT=27620 SEARCH_PORT=8760 \
  MOONCAKE_MASTER_PORT=57651 MOONCAKE_CLIENT_PORT=57652 \
  MOONCAKE_METADATA_PORT=57680 MOONCAKE_METRICS_PORT=57603 \
  MOONCAKE_CLIENT_HTTP_PORT=57690 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

/homes/siqic/anaconda3/envs/pd/bin/python \
  "${SCRIPT_DIR}/../tools/compare_architectures.py" --run-root "${RUN_ROOT}"
echo "baseline comparison suite complete: ${RUN_ROOT}"
