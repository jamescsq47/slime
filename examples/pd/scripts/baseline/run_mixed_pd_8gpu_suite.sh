#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU stock-SGLang PD comparison on the fixed Mixed 1:1 workload.
# GPU7 also hosts BrowseComp retrieval, so its Decode worker uses a smaller
# static KV pool. Cases are intentionally sequential to avoid interference.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/baseline/mixed-pd-8gpu-c512-s2026-w300-m1200}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
MAX_INFLIGHT="${MAX_INFLIGHT:-512}"
REQUESTS="${REQUESTS:-8192}"
SEED="${SEED:-2026}"
SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n8192.json}"

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
    SCHEDULE_FILE="${SCHEDULE_FILE}" MATH_RATIO=0.5 \
    SEARCH_GPU=7 PD_INFERENCE_RETURN_LOGPROB=false \
    SLIME_HTTP_READ_TIMEOUT_SECONDS=3600 "$@"
  echo "finish ${name} at $(date -u +%FT%TZ)"
}

run_case pd-no-reverse-2p6d \
  CASE_MODE=no_reverse PREFILL_GPUS='0 1' PREFILL_PORTS='33100 33101' \
  PREFILL_BOOTSTRAP_PORTS='34100 34101' \
  PREFILL_MEM_FRACTION_STATICS='0.85 0.85' \
  DECODE_GPUS='2 3 4 5 6 7' \
  DECODE_PORTS='33102 33103 33104 33105 33106 33107' \
  DECODE_MEM_FRACTION_STATICS='0.85 0.85 0.85 0.85 0.85 0.60' \
  ROUTER_PORT=33110 ROUTER_PROMETHEUS_PORT=33120 SEARCH_PORT=9310 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case pd-native-mooncake-2p6d \
  CASE_MODE=native_mooncake PREFILL_GPUS='0 1' PREFILL_PORTS='33200 33201' \
  PREFILL_BOOTSTRAP_PORTS='34200 34201' \
  PREFILL_MEM_FRACTION_STATICS='0.85 0.85' \
  DECODE_GPUS='2 3 4 5 6 7' \
  DECODE_PORTS='33202 33203 33204 33205 33206 33207' \
  DECODE_MEM_FRACTION_STATICS='0.85 0.85 0.85 0.85 0.85 0.60' \
  ROUTER_PORT=33210 ROUTER_PROMETHEUS_PORT=33220 SEARCH_PORT=9320 \
  MOONCAKE_MASTER_PORT=63251 MOONCAKE_CLIENT_PORT=63252 \
  MOONCAKE_METADATA_PORT=63280 MOONCAKE_METRICS_PORT=63203 \
  MOONCAKE_CLIENT_HTTP_PORT=63290 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case pd-no-reverse-4p4d \
  CASE_MODE=no_reverse PREFILL_GPUS='0 1 2 3' \
  PREFILL_PORTS='33300 33301 33302 33303' \
  PREFILL_BOOTSTRAP_PORTS='34300 34301 34302 34303' \
  PREFILL_MEM_FRACTION_STATICS='0.85 0.85 0.85 0.85' \
  DECODE_GPUS='4 5 6 7' DECODE_PORTS='33304 33305 33306 33307' \
  DECODE_MEM_FRACTION_STATICS='0.85 0.85 0.85 0.60' \
  ROUTER_PORT=33310 ROUTER_PROMETHEUS_PORT=33320 SEARCH_PORT=9330 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case pd-native-mooncake-4p4d \
  CASE_MODE=native_mooncake PREFILL_GPUS='0 1 2 3' \
  PREFILL_PORTS='33400 33401 33402 33403' \
  PREFILL_BOOTSTRAP_PORTS='34400 34401 34402 34403' \
  PREFILL_MEM_FRACTION_STATICS='0.85 0.85 0.85 0.85' \
  DECODE_GPUS='4 5 6 7' DECODE_PORTS='33404 33405 33406 33407' \
  DECODE_MEM_FRACTION_STATICS='0.85 0.85 0.85 0.60' \
  ROUTER_PORT=33410 ROUTER_PROMETHEUS_PORT=33420 SEARCH_PORT=9340 \
  MOONCAKE_MASTER_PORT=63451 MOONCAKE_CLIENT_PORT=63452 \
  MOONCAKE_METADATA_PORT=63480 MOONCAKE_METRICS_PORT=63403 \
  MOONCAKE_CLIENT_HTTP_PORT=63490 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

echo "mixed PD 8-GPU suite complete: ${RUN_ROOT}"
