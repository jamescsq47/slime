#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
pd_install_cleanup_traps
cd "${PD_DIR}"

PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
MODEL_PATH="${MODEL_PATH:-/dataset/model/qwen3/Qwen3-8B}"
WORKSPACE_ROOT="$(dirname -- "$(cd -- "${PD_DIR}/../.." && pwd)")"
MATH_DATA="${MATH_DATA:-${WORKSPACE_ROOT}/data/dapo-math-17k/dapo-math-17k.jsonl}"
QA_DATA="${QA_DATA:-${WORKSPACE_ROOT}/data/browsecomp/bc_train.jsonl}"
MODEL_GPUS="${MODEL_GPUS:-0 1 2 3 4 5}"
MODEL_PORTS="${MODEL_PORTS:-27200 27201 27202 27203 27204 27205}"
ROUTER_PORT="${ROUTER_PORT:-27210}"
SEARCH_GPU="${SEARCH_GPU:-6}"
SEARCH_PORT="${SEARCH_PORT:-8720}"
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline-colocated-6gpu-c384}"
MAX_INFLIGHT="${MAX_INFLIGHT:-384}"
REQUESTS="${REQUESTS:-4096}"
SEED="${SEED:-2026}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n4096.json}"
MATH_RATIO="${MATH_RATIO:-0.5}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MODEL_MEM_FRACTION_STATICS="${MODEL_MEM_FRACTION_STATICS:-}"
PRESERVE_SOURCE_ORDER="${PRESERVE_SOURCE_ORDER:-false}"

read -r -a model_gpus <<<"${MODEL_GPUS}"
read -r -a model_ports <<<"${MODEL_PORTS}"
if [[ -n "${MODEL_MEM_FRACTION_STATICS}" ]]; then
  read -r -a model_mem_fraction_statics <<<"${MODEL_MEM_FRACTION_STATICS}"
else
  model_mem_fraction_statics=()
  for _ in "${model_gpus[@]}"; do
    model_mem_fraction_statics+=("${MEM_FRACTION_STATIC}")
  done
fi
(( ${#model_gpus[@]} == ${#model_ports[@]} )) || {
  echo "MODEL_GPUS/MODEL_PORTS length mismatch" >&2; exit 2;
}
(( ${#model_gpus[@]} == ${#model_mem_fraction_statics[@]} )) || {
  echo "MODEL_MEM_FRACTION_STATICS must be empty or match MODEL_GPUS" >&2; exit 2;
}
mkdir -p "${RUN_DIR}/logs"
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd):${PYTHONPATH:-}"
export LOCAL_SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}"
unset SGLANG_AGENTIC_KV_LIFECYCLE SGLANG_AGENTIC_KV_HOST_STAGING \
  SGLANG_AGENTIC_KV_D_HOSTLESS SGLANG_AGENTIC_KV_LEDGER_PATH \
  SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT || true
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_environments.py" \
  --expect baseline --output "${RUN_DIR}/environment.json"

for gpu in "${model_gpus[@]}" "${SEARCH_GPU}"; do pd_check_gpu_idle "${gpu}"; done
for port in "${model_ports[@]}" "${ROUTER_PORT}" "${SEARCH_PORT}"; do pd_check_port_free "${port}"; done

setsid env CUDA_VISIBLE_DEVICES="${SEARCH_GPU}" SEARCH_SERVER_GPU_IDS=0 \
  "${PD_ENV_BIN}/python" search_server.py --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 --port "${SEARCH_PORT}" >"${RUN_DIR}/logs/search.log" 2>&1 &
search_pid=$!; pd_track_group "${search_pid}"
pd_wait_http search "http://127.0.0.1:${SEARCH_PORT}/health" "${search_pid}" 1200

worker_args=()
for index in "${!model_gpus[@]}"; do
  setsid env CUDA_VISIBLE_DEVICES="${model_gpus[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${model_ports[index]}" --context-length 40960 \
    --page-size 64 --mem-fraction-static "${model_mem_fraction_statics[index]}" --enable-metrics \
    --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
    >"${RUN_DIR}/logs/model-${index}.log" 2>&1 &
  model_pid=$!; pd_track_group "${model_pid}"
  pd_wait_http "model-${index}" "http://127.0.0.1:${model_ports[index]}/health" "${model_pid}" 900
  worker_args+=("http://127.0.0.1:${model_ports[index]}")
done

setsid "${PD_ENV_BIN}/python" -m sglang_router.launch_router \
  --worker-urls "${worker_args[@]}" --policy cache_aware \
  --cache-threshold 0.3 --balance-abs-threshold 8 --balance-rel-threshold 1.2 \
  --host 0.0.0.0 --port "${ROUTER_PORT}" >"${RUN_DIR}/logs/router.log" 2>&1 &
router_pid=$!; pd_track_group "${router_pid}"
pd_wait_http router "http://127.0.0.1:${ROUTER_PORT}/health" "${router_pid}" 300

ports_csv="$(IFS=,; echo "${model_ports[*]}")"
inference_order_args=()
if [[ "${PRESERVE_SOURCE_ORDER}" == "true" ]]; then
  inference_order_args+=(--preserve-source-order)
fi
SLIME_HTTP_READ_TIMEOUT_SECONDS="${SLIME_HTTP_READ_TIMEOUT_SECONDS:-3600}" \
"${PD_ENV_BIN}/python" inference.py --model "${MODEL_PATH}" \
  --math-data "${MATH_DATA}" --qa-data "${QA_DATA}" --router-port "${ROUTER_PORT}" \
  --prefill-port "${model_ports[0]}" --prefill-ports "${ports_csv}" \
  --decode-port "${model_ports[0]}" --decode-ports "${ports_csv}" \
  --math-ratio "${MATH_RATIO}" --requests "${REQUESTS}" --warmup-requests 0 \
  --dispatch-policy fixed --schedule-file "${SCHEDULE_FILE}" \
  --request-rate 100 --arrival-distribution fixed --max-inflight "${MAX_INFLIGHT}" \
  --metrics-interval 2 --seed "${SEED}" --temperature 0 --top-p 1 --top-k -1 \
  --closed-loop --closed-loop-warmup-min-seconds "${WARMUP_SECONDS}" \
  --closed-loop-warmup-completions 0 --closed-loop-recent-seconds 120 \
  --closed-loop-max-warmup-seconds "$((WARMUP_SECONDS + 120))" \
  --closed-loop-measurement-seconds "${MEASURE_SECONDS}" --output-dir "${RUN_DIR}" \
  "${inference_order_args[@]}" \
  >"${RUN_DIR}/inference.log" 2>&1
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/analyze_pd_offload.py" --run-dir "${RUN_DIR}"
echo "baseline colocated case complete: ${RUN_DIR}"
