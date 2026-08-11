#!/usr/bin/env bash
set -euo pipefail

# Reproducible four-model-GPU comparison of colocated serving, stock PD without
# reverse KV reuse, and stock PD with SGLang's native Mooncake/HiCache path.
# Cases deliberately run sequentially so they see the same GPUs and do not
# contend with one another. GPU 7 is used only by the BrowseComp search model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/baseline/four-gpu-1p3d-c256-s2026-w300-m1200}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
MAX_INFLIGHT="${MAX_INFLIGHT:-256}"
REQUESTS="${REQUESTS:-4096}"
SEED="${SEED:-2026}"
SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n4096.json}"
SEARCH_GPU="${SEARCH_GPU:-7}"

mkdir -p "${RUN_ROOT}"

cat >"${RUN_ROOT}/experiment_config.json" <<EOF
{
  "environment": "${PD_ENV_BIN}",
  "model_gpus": [0, 1, 2, 3],
  "search_gpu": ${SEARCH_GPU},
  "workload": "Mixed Retool/BrowseComp 1:1",
  "schedule_file": "${SCHEDULE_FILE}",
  "requests": ${REQUESTS},
  "max_inflight": ${MAX_INFLIGHT},
  "seed": ${SEED},
  "warmup_seconds": ${WARMUP_SECONDS},
  "measurement_seconds": ${MEASURE_SECONDS},
  "context_length": 40960,
  "page_size": 64,
  "mem_fraction_static": 0.85,
  "temperature": 0,
  "top_p": 1,
  "top_k": -1
}
EOF

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

run_case colocated-4gpu \
  MODEL_GPUS='0 1 2 3' MODEL_PORTS='27700 27701 27702 27703' \
  ROUTER_PORT=27710 SEARCH_PORT=8770 \
  bash "${SCRIPT_DIR}/run_colocated_case.sh"

run_case pd-no-reverse-1p3d \
  CASE_MODE=no_reverse PREFILL_GPUS='0' PREFILL_PORTS='27800' \
  PREFILL_BOOTSTRAP_PORTS='28800' DECODE_GPUS='1 2 3' \
  DECODE_PORTS='27801 27802 27803' ROUTER_PORT=27810 \
  ROUTER_PROMETHEUS_PORT=27820 SEARCH_PORT=8780 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

run_case pd-native-mooncake-1p3d \
  CASE_MODE=native_mooncake PREFILL_GPUS='0' PREFILL_PORTS='27900' \
  PREFILL_BOOTSTRAP_PORTS='28900' DECODE_GPUS='1 2 3' \
  DECODE_PORTS='27901 27902 27903' ROUTER_PORT=27910 \
  ROUTER_PROMETHEUS_PORT=27920 SEARCH_PORT=8790 \
  MOONCAKE_MASTER_PORT=57951 MOONCAKE_CLIENT_PORT=57952 \
  MOONCAKE_METADATA_PORT=57980 MOONCAKE_METRICS_PORT=57903 \
  MOONCAKE_CLIENT_HTTP_PORT=57990 \
  bash "${SCRIPT_DIR}/run_pd_case.sh"

"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/compare_four_gpu_baselines.py" \
  --run-root "${RUN_ROOT}"
echo "four-GPU baseline comparison complete: ${RUN_ROOT}"
