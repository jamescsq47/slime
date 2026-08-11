#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/baseline/pd-native-offload-correctness-c32-s2026-v2}"
mkdir -p "${RUN_ROOT}"

run_case() {
  local mode="$1" port_base="$2" run_dir="${RUN_ROOT}/$1"
  if [[ -f "${run_dir}/capture.json" ]]; then
    echo "skip completed ${mode}"
    return
  fi
  env CASE_MODE="${mode}" RUN_DIR="${run_dir}" PD_ENV_BIN="${PD_ENV_BIN}" \
    P_PORT="$((port_base))" D_PORT="$((port_base + 1))" \
    ROUTER_PORT="$((port_base + 10))" ROUTER_PROMETHEUS_PORT="$((port_base + 20))" \
    BOOTSTRAP_PORT="$((port_base + 1000))" \
    MOONCAKE_MASTER_PORT="$((port_base + 30000 + 51))" \
    MOONCAKE_CLIENT_PORT="$((port_base + 30000 + 52))" \
    MOONCAKE_METADATA_PORT="$((port_base + 30000 + 80))" \
    MOONCAKE_METRICS_PORT="$((port_base + 30000 + 3))" \
    MOONCAKE_CLIENT_HTTP_PORT="$((port_base + 30000 + 90))" \
    bash "${SCRIPT_DIR}/run_pd_correctness_case.sh"
}

run_case no_reverse 28100
run_case hicache_no_decode_offload 28300
run_case native_mooncake 28200

set +e
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/compare_pd_correctness.py" \
  --left "${RUN_ROOT}/no_reverse/capture.json" \
  --right "${RUN_ROOT}/hicache_no_decode_offload/capture.json" \
  --output "${RUN_ROOT}/comparison_hicache_control.json" \
  >"${RUN_ROOT}/comparison_hicache_control.log" 2>&1
control_status=$?
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/compare_pd_correctness.py" \
  --left "${RUN_ROOT}/no_reverse/capture.json" \
  --right "${RUN_ROOT}/native_mooncake/capture.json" \
  --output "${RUN_ROOT}/comparison_native_offload.json" \
  >"${RUN_ROOT}/comparison_native_offload.log" 2>&1
native_status=$?
set -e
cat "${RUN_ROOT}/comparison_hicache_control.log"
cat "${RUN_ROOT}/comparison_native_offload.log"
echo "HiCache-only correctness status: ${control_status}"
echo "native Decode-offload correctness status: ${native_status}"
(( control_status == 0 && native_status == 0 ))
