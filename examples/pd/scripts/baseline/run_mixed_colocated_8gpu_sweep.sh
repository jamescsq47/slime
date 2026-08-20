#!/usr/bin/env bash
set -euo pipefail

# Stock-SGLang, 8-worker colocated mixed-workload saturation sweep.
# GPU7 also hosts BrowseComp retrieval, so only that worker uses a smaller KV pool.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/baseline/mixed-colocated-8gpu-saturation-s2026-w300-m1200}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
REQUESTS="${REQUESTS:-8192}"
SEED="${SEED:-2026}"
CONCURRENCIES="${CONCURRENCIES:-512}"
SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n8192.json}"

mkdir -p "${RUN_ROOT}"

complete_run() {
  local run_dir="$1"
  [[ -f "${run_dir}/offload_analysis_summary.json" ]] || return 1
  "${PD_ENV_BIN}/python" - "${run_dir}/offload_analysis_summary.json" "${MEASURE_SECONDS}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
ok = (
    summary.get("status") == "steady_churn"
    and summary.get("measurement_seconds", 0) >= float(sys.argv[2]) - 2
)
raise SystemExit(0 if ok else 1)
PY
}

for concurrency in ${CONCURRENCIES}; do
  run_dir="${RUN_ROOT}/c${concurrency}"
  if complete_run "${run_dir}"; then
    echo "skip completed c${concurrency}: ${run_dir}"
    continue
  fi
  echo "start mixed colocated 8-GPU c${concurrency} at $(date -u +%FT%TZ)"
  env PD_ENV_BIN="${PD_ENV_BIN}" RUN_DIR="${run_dir}" \
    MODEL_GPUS='0 1 2 3 4 5 6 7' \
    MODEL_PORTS='32100 32101 32102 32103 32104 32105 32106 32107' \
    MODEL_MEM_FRACTION_STATICS='0.80 0.80 0.80 0.80 0.80 0.80 0.80 0.60' \
    PD_INFERENCE_RETURN_LOGPROB=false \
    ROUTER_PORT=32110 SEARCH_GPU=7 SEARCH_PORT=9210 \
    MAX_INFLIGHT="${concurrency}" REQUESTS="${REQUESTS}" \
    WARMUP_SECONDS="${WARMUP_SECONDS}" MEASURE_SECONDS="${MEASURE_SECONDS}" \
    SEED="${SEED}" MATH_RATIO=0.5 SCHEDULE_FILE="${SCHEDULE_FILE}" \
    SLIME_HTTP_READ_TIMEOUT_SECONDS=3600 \
    bash "${SCRIPT_DIR}/run_colocated_case.sh"
  echo "finish mixed colocated 8-GPU c${concurrency} at $(date -u +%FT%TZ)"
done

echo "mixed colocated 8-GPU sweep complete: ${RUN_ROOT}"
