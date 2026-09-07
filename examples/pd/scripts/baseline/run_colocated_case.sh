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
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-}"
MODEL_GPUS="${MODEL_GPUS:-0 1 2 3 4 5}"
MODEL_GPU_GROUPS="${MODEL_GPU_GROUPS:-}"
MODEL_TP_SIZE="${MODEL_TP_SIZE:-1}"
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-40960}"
MODEL_MAX_RESPONSE_LENGTH="${MODEL_MAX_RESPONSE_LENGTH:-36864}"
MODEL_PAGE_SIZE="${MODEL_PAGE_SIZE:-64}"
MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-}"
MODEL_TOOL_CALL_PARSER="${MODEL_TOOL_CALL_PARSER:-}"
MODEL_ATTENTION_BACKEND="${MODEL_ATTENTION_BACKEND:-}"
MODEL_SAMPLING_BACKEND="${MODEL_SAMPLING_BACKEND:-}"
MODEL_PORTS="${MODEL_PORTS:-27200 27201 27202 27203 27204 27205}"
ROUTER_PORT="${ROUTER_PORT:-27210}"
SEARCH_GPU="${SEARCH_GPU:-6}"
SEARCH_PORT="${SEARCH_PORT:-8720}"
START_SEARCH_SERVER="${START_SEARCH_SERVER:-true}"
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline-colocated-6gpu-c384}"
MAX_INFLIGHT="${MAX_INFLIGHT:-384}"
REQUESTS="${REQUESTS:-4096}"
SEED="${SEED:-2026}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
CLOSED_LOOP="${CLOSED_LOOP:-true}"
SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n4096.json}"
DISPATCH_POLICY="${DISPATCH_POLICY:-fixed}"
MATH_RATIO="${MATH_RATIO:-0.5}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MODEL_MEM_FRACTION_STATICS="${MODEL_MEM_FRACTION_STATICS:-}"
PRESERVE_SOURCE_ORDER="${PRESERVE_SOURCE_ORDER:-false}"
SEARCH_START_AFTER_MODELS="${SEARCH_START_AFTER_MODELS:-false}"
POST_ANALYZER="${POST_ANALYZER:-pd_offload}"

if [[ -n "${MODEL_GPU_GROUPS}" ]]; then
  IFS=';' read -r -a model_gpu_groups <<<"${MODEL_GPU_GROUPS}"
else
  read -r -a model_gpu_groups <<<"${MODEL_GPUS}"
fi
read -r -a model_ports <<<"${MODEL_PORTS}"
[[ "${MODEL_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MODEL_TP_SIZE must be a positive integer" >&2; exit 2;
}
model_gpus=()
for group in "${model_gpu_groups[@]}"; do
  IFS=',' read -r -a group_gpus <<<"${group}"
  (( ${#group_gpus[@]} == MODEL_TP_SIZE )) || {
    echo "model GPU group '${group}' has ${#group_gpus[@]} ranks; expected ${MODEL_TP_SIZE}" >&2
    exit 2
  }
  model_gpus+=("${group_gpus[@]}")
done
if [[ -n "${MODEL_MEM_FRACTION_STATICS}" ]]; then
  read -r -a model_mem_fraction_statics <<<"${MODEL_MEM_FRACTION_STATICS}"
else
  model_mem_fraction_statics=()
  for _ in "${model_gpu_groups[@]}"; do
    model_mem_fraction_statics+=("${MEM_FRACTION_STATIC}")
  done
fi
(( ${#model_gpu_groups[@]} == ${#model_ports[@]} )) || {
  echo "model GPU groups/MODEL_PORTS length mismatch" >&2; exit 2;
}
(( ${#model_gpu_groups[@]} == ${#model_mem_fraction_statics[@]} )) || {
  echo "MODEL_MEM_FRACTION_STATICS must be empty or match logical model GPU groups" >&2; exit 2;
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

declare -A checked_gpus=()
gpus_to_check=("${model_gpus[@]}")
if [[ "${START_SEARCH_SERVER}" == "true" ]]; then
  gpus_to_check+=("${SEARCH_GPU}")
fi
for gpu in "${gpus_to_check[@]}"; do
  [[ -n "${checked_gpus[${gpu}]:-}" ]] && continue
  pd_check_gpu_idle "${gpu}"
  checked_gpus[${gpu}]=1
done
ports_to_check=("${model_ports[@]}" "${ROUTER_PORT}")
if [[ "${START_SEARCH_SERVER}" == "true" ]]; then
  ports_to_check+=("${SEARCH_PORT}")
fi
for port in "${ports_to_check[@]}"; do pd_check_port_free "${port}"; done

start_search_server() {
  setsid env CUDA_VISIBLE_DEVICES="${SEARCH_GPU}" SEARCH_SERVER_GPU_IDS=0 \
    "${PD_ENV_BIN}/python" search_server.py --model Qwen/Qwen3-Embedding-8B \
    --corpus Tevatron/browsecomp-plus-corpus \
    --corpus-embedding-dataset miaolu3/browsecomp-plus \
    --host 0.0.0.0 --port "${SEARCH_PORT}" >"${RUN_DIR}/logs/search.log" 2>&1 &
  search_pid=$!; pd_track_group "${search_pid}"
  pd_wait_http search "http://127.0.0.1:${SEARCH_PORT}/health" "${search_pid}" 1200
}

if [[ "${START_SEARCH_SERVER}" == "true" && "${SEARCH_START_AFTER_MODELS}" != "true" ]]; then
  start_search_server
fi

worker_args=()
gpu_numa_node() {
  local node
  node="$(nvidia-smi topo -m | awk -v row="GPU$1" '$1 == row {print $(NF-1); exit}')"
  [[ "${node}" =~ ^[0-9]+$ ]] && printf '%s\n' "${node}" || printf '0\n'
}

for index in "${!model_gpu_groups[@]}"; do
  IFS=',' read -r -a group_gpus <<<"${model_gpu_groups[index]}"
  model_tp_args=()
  model_parser_args=()
  if (( MODEL_TP_SIZE > 1 )); then
    group_numas=()
    for gpu in "${group_gpus[@]}"; do group_numas+=("$(gpu_numa_node "${gpu}")"); done
    model_tp_args+=(--tp-size "${MODEL_TP_SIZE}" --numa-node "${group_numas[@]}")
  fi
  if [[ -n "${MODEL_REASONING_PARSER}" ]]; then
    model_parser_args+=(--reasoning-parser "${MODEL_REASONING_PARSER}")
  fi
  if [[ -n "${MODEL_TOOL_CALL_PARSER}" ]]; then
    model_parser_args+=(--tool-call-parser "${MODEL_TOOL_CALL_PARSER}")
  fi
  if [[ -n "${MODEL_ATTENTION_BACKEND}" ]]; then
    model_parser_args+=(--attention-backend "${MODEL_ATTENTION_BACKEND}")
  fi
  if [[ -n "${MODEL_SAMPLING_BACKEND}" ]]; then
    model_parser_args+=(--sampling-backend "${MODEL_SAMPLING_BACKEND}")
  fi
  setsid env CUDA_VISIBLE_DEVICES="${model_gpu_groups[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${model_ports[index]}" --context-length "${MODEL_CONTEXT_LENGTH}" \
    "${model_tp_args[@]}" \
    "${model_parser_args[@]}" \
    --page-size "${MODEL_PAGE_SIZE}" --mem-fraction-static "${model_mem_fraction_statics[index]}" --enable-metrics \
    --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
    >"${RUN_DIR}/logs/model-${index}.log" 2>&1 &
  model_pid=$!; pd_track_group "${model_pid}"
  pd_wait_http "model-${index}" "http://127.0.0.1:${model_ports[index]}/health" "${model_pid}" 900
  worker_args+=("http://127.0.0.1:${model_ports[index]}")
done

if [[ "${START_SEARCH_SERVER}" == "true" && "${SEARCH_START_AFTER_MODELS}" == "true" ]]; then
  start_search_server
fi

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
workload_args=()
if [[ -n "${WORKLOAD_CONFIG}" ]]; then
  workload_args+=(--workload-config "${WORKLOAD_CONFIG}")
else
  workload_args+=(--math-data "${MATH_DATA}" --qa-data "${QA_DATA}" --math-ratio "${MATH_RATIO}")
fi
dispatch_args=(--dispatch-policy "${DISPATCH_POLICY}")
if [[ "${DISPATCH_POLICY}" == "fixed" || "${DISPATCH_POLICY}" == "profile_balanced" ]]; then
  dispatch_args+=(--schedule-file "${SCHEDULE_FILE}")
fi
loop_args=()
if [[ "${CLOSED_LOOP}" == "true" ]]; then
  loop_args+=(
    --closed-loop
    --closed-loop-warmup-min-seconds "${WARMUP_SECONDS}"
    --closed-loop-warmup-completions 0
    --closed-loop-recent-seconds 120
    --closed-loop-max-warmup-seconds "$((WARMUP_SECONDS + 120))"
    --closed-loop-measurement-seconds "${MEASURE_SECONDS}"
  )
elif [[ "${CLOSED_LOOP}" != "false" ]]; then
  echo "CLOSED_LOOP must be true or false" >&2
  exit 2
fi
SLIME_HTTP_READ_TIMEOUT_SECONDS="${SLIME_HTTP_READ_TIMEOUT_SECONDS:-3600}" \
"${PD_ENV_BIN}/python" inference.py --model "${MODEL_PATH}" \
  "${workload_args[@]}" --router-port "${ROUTER_PORT}" \
  --prefill-port "${model_ports[0]}" --prefill-ports "${ports_csv}" \
  --decode-port "${model_ports[0]}" --decode-ports "${ports_csv}" \
  --requests "${REQUESTS}" --warmup-requests 0 \
  "${dispatch_args[@]}" \
  --request-rate 100 --arrival-distribution fixed --max-inflight "${MAX_INFLIGHT}" \
  --metrics-interval 2 --seed "${SEED}" \
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}" \
  --max-context-length "${MODEL_CONTEXT_LENGTH}" \
  --max-response-length "${MODEL_MAX_RESPONSE_LENGTH}" \
  "${loop_args[@]}" --output-dir "${RUN_DIR}" \
  "${inference_order_args[@]}" \
  >"${RUN_DIR}/inference.log" 2>&1
case "${POST_ANALYZER}" in
  pd_offload)
    "${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/analyze_pd_offload.py" --run-dir "${RUN_DIR}"
    ;;
  swe_bench)
    "${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/analyze_swe_bench_run.py" "${RUN_DIR}"
    ;;
  none)
    ;;
  *)
    echo "unsupported POST_ANALYZER=${POST_ANALYZER}" >&2
    exit 2
    ;;
esac
echo "baseline colocated case complete: ${RUN_DIR}"
