#!/usr/bin/env bash
set -euo pipefail

# Baseline cases intentionally retain stock SGLang HiCache/Mooncake behavior.
unset SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
pd_install_cleanup_traps
cd "${PD_DIR}"

PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
MODEL_PATH="${MODEL_PATH:-/dataset/model/qwen3/Qwen3-8B}"
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-40960}"
MODEL_MAX_RESPONSE_LENGTH="${MODEL_MAX_RESPONSE_LENGTH:-36864}"
MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-}"
MODEL_TOOL_CALL_PARSER="${MODEL_TOOL_CALL_PARSER:-}"
WORKSPACE_ROOT="$(dirname -- "$(cd -- "${PD_DIR}/../.." && pwd)")"
MATH_DATA="${MATH_DATA:-${WORKSPACE_ROOT}/data/dapo-math-17k/dapo-math-17k.jsonl}"
QA_DATA="${QA_DATA:-${WORKSPACE_ROOT}/data/browsecomp/bc_train.jsonl}"
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-}"
CASE_MODE="${CASE_MODE:-no_reverse}"
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline-${CASE_MODE}}"
PREFILL_GPUS="${PREFILL_GPUS:-0}"
DECODE_GPUS="${DECODE_GPUS:-1 2 3 4 5}"
PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-}"
DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-1}"
PREFILL_PORTS="${PREFILL_PORTS:-27100}"
PREFILL_BOOTSTRAP_PORTS="${PREFILL_BOOTSTRAP_PORTS:-28100}"
DECODE_PORTS="${DECODE_PORTS:-27101 27102 27103 27104 27105}"
PREFILL_MEM_FRACTION_STATICS="${PREFILL_MEM_FRACTION_STATICS:-}"
DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-}"
ROUTER_PORT="${ROUTER_PORT:-27110}"
ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-27120}"
SEARCH_GPU="${SEARCH_GPU:-6}"
SEARCH_PORT="${SEARCH_PORT:-8710}"
START_SEARCH_SERVER="${START_SEARCH_SERVER:-true}"
SEARCH_START_AFTER_MODELS="${SEARCH_START_AFTER_MODELS:-false}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-57151}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-57180}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-57103}"
MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-57152}"
MOONCAKE_CLIENT_HTTP_PORT="${MOONCAKE_CLIENT_HTTP_PORT:-57190}"
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
PAGE_SIZE="${PAGE_SIZE:-64}"
MATH_RATIO="${MATH_RATIO:-0.5}"
P_HICACHE_SIZE="${P_HICACHE_SIZE:-128}"
D_HICACHE_SIZE="${D_HICACHE_SIZE:-56}"
HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
P_HICACHE_WRITE_POLICY="${P_HICACHE_WRITE_POLICY:-write_through}"
PRESERVE_SOURCE_ORDER="${PRESERVE_SOURCE_ORDER:-false}"
POST_ANALYZER="${POST_ANALYZER:-pd_offload}"

case "${CASE_MODE}" in
  no_reverse|hicache_no_decode_offload|native_mooncake) ;;
  *) echo "CASE_MODE must be no_reverse, hicache_no_decode_offload, or native_mooncake" >&2; exit 2 ;;
esac

if [[ -n "${PREFILL_GPU_GROUPS}" ]]; then
  IFS=';' read -r -a p_gpu_groups <<<"${PREFILL_GPU_GROUPS}"
else
  read -r -a p_gpu_groups <<<"${PREFILL_GPUS}"
fi
read -r -a p_ports <<<"${PREFILL_PORTS}"
read -r -a p_bootstrap_ports <<<"${PREFILL_BOOTSTRAP_PORTS}"
if [[ -n "${DECODE_GPU_GROUPS}" ]]; then
  IFS=';' read -r -a d_gpu_groups <<<"${DECODE_GPU_GROUPS}"
else
  read -r -a d_gpu_groups <<<"${DECODE_GPUS}"
fi
read -r -a d_ports <<<"${DECODE_PORTS}"
[[ "${PREFILL_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PREFILL_TP_SIZE must be a positive integer" >&2; exit 2;
}
[[ "${DECODE_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "DECODE_TP_SIZE must be a positive integer" >&2; exit 2;
}
p_physical_gpus=()
for group in "${p_gpu_groups[@]}"; do
  IFS=',' read -r -a group_gpus <<<"${group}"
  (( ${#group_gpus[@]} == PREFILL_TP_SIZE )) || {
    echo "Prefill GPU group '${group}' has ${#group_gpus[@]} ranks; expected ${PREFILL_TP_SIZE}" >&2
    exit 2
  }
  p_physical_gpus+=("${group_gpus[@]}")
done
d_physical_gpus=()
for group in "${d_gpu_groups[@]}"; do
  IFS=',' read -r -a group_gpus <<<"${group}"
  (( ${#group_gpus[@]} == DECODE_TP_SIZE )) || {
    echo "Decode GPU group '${group}' has ${#group_gpus[@]} ranks; expected ${DECODE_TP_SIZE}" >&2
    exit 2
  }
  d_physical_gpus+=("${group_gpus[@]}")
done
if [[ -n "${PREFILL_MEM_FRACTION_STATICS}" ]]; then
  read -r -a p_mem_fraction_statics <<<"${PREFILL_MEM_FRACTION_STATICS}"
else
  p_mem_fraction_statics=()
  for _ in "${p_gpu_groups[@]}"; do p_mem_fraction_statics+=(0.85); done
fi
if [[ -n "${DECODE_MEM_FRACTION_STATICS}" ]]; then
  read -r -a d_mem_fraction_statics <<<"${DECODE_MEM_FRACTION_STATICS}"
else
  d_mem_fraction_statics=()
  for _ in "${d_gpu_groups[@]}"; do d_mem_fraction_statics+=(0.85); done
fi
(( ${#p_gpu_groups[@]} == ${#p_ports[@]} && ${#p_gpu_groups[@]} == ${#p_bootstrap_ports[@]} )) || {
  echo "Prefill GPU groups/PREFILL_PORTS/PREFILL_BOOTSTRAP_PORTS length mismatch" >&2; exit 2;
}
(( ${#d_gpu_groups[@]} == ${#d_ports[@]} )) || {
  echo "Decode GPU groups/DECODE_PORTS length mismatch" >&2; exit 2;
}
(( ${#p_gpu_groups[@]} == ${#p_mem_fraction_statics[@]} )) || {
  echo "Prefill GPU groups/PREFILL_MEM_FRACTION_STATICS length mismatch" >&2; exit 2;
}
(( ${#d_gpu_groups[@]} == ${#d_mem_fraction_statics[@]} )) || {
  echo "Decode GPU groups/DECODE_MEM_FRACTION_STATICS length mismatch" >&2; exit 2;
}
if (( ${#p_gpu_groups[@]} < 1 || ${#d_gpu_groups[@]} < 1 )); then
  echo "At least one Prefill GPU and one Decode GPU are required" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/hicache"
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd):${PYTHONPATH:-}"
export LOCAL_SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}"
export MC_TCP_ENABLE_CONNECTION_POOL="${MC_TCP_ENABLE_CONNECTION_POOL:-1}"
# Prove that an inherited shell cannot accidentally enable the custom pd path.
unset SGLANG_AGENTIC_KV_LIFECYCLE SGLANG_AGENTIC_KV_HOST_STAGING \
  SGLANG_AGENTIC_KV_D_HOSTLESS SGLANG_AGENTIC_KV_LEDGER_PATH \
  SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT || true
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_environments.py" \
  --expect baseline --output "${RUN_DIR}/environment.json"

declare -A checked_gpus=()
gpus_to_check=("${p_physical_gpus[@]}" "${d_physical_gpus[@]}")
if [[ "${START_SEARCH_SERVER}" == "true" ]]; then
  gpus_to_check+=("${SEARCH_GPU}")
fi
for gpu in "${gpus_to_check[@]}"; do
  [[ -n "${checked_gpus[${gpu}]:-}" ]] && continue
  pd_check_gpu_idle "${gpu}"
  checked_gpus[${gpu}]=1
done
ports_to_check=("${p_ports[@]}" "${p_bootstrap_ports[@]}" "${d_ports[@]}" \
  "${ROUTER_PORT}" "${ROUTER_PROMETHEUS_PORT}")
if [[ "${START_SEARCH_SERVER}" == "true" ]]; then
  ports_to_check+=("${SEARCH_PORT}")
fi
for port in "${ports_to_check[@]}"; do pd_check_port_free "${port}"; done

mooncake_config=""
if [[ "${CASE_MODE}" != no_reverse ]]; then
  for port in "${MOONCAKE_MASTER_PORT}" "${MOONCAKE_METADATA_PORT}" \
    "${MOONCAKE_METRICS_PORT}" "${MOONCAKE_CLIENT_PORT}" "${MOONCAKE_CLIENT_HTTP_PORT}"; do
    pd_check_port_free "${port}"
  done
  setsid "${PD_ENV_BIN}/mooncake_master" \
    --rpc_port="${MOONCAKE_MASTER_PORT}" --enable_http_metadata_server=true \
    --http_metadata_server_port="${MOONCAKE_METADATA_PORT}" \
    --eviction_high_watermark_ratio=0.85 --eviction_ratio=0.10 \
    --metrics_port="${MOONCAKE_METRICS_PORT}" \
    >"${RUN_DIR}/logs/mooncake-master.log" 2>&1 &
  mooncake_master_pid=$!; pd_track_group "${mooncake_master_pid}"
  pd_wait_http mooncake-master "http://127.0.0.1:${MOONCAKE_METRICS_PORT}/health" "${mooncake_master_pid}" 300
  setsid "${PD_ENV_BIN}/mooncake_client" --host=127.0.0.1 \
    --port="${MOONCAKE_CLIENT_PORT}" --global_segment_size="256 GB" \
    --master_server_address="127.0.0.1:${MOONCAKE_MASTER_PORT}" \
    --metadata_server="http://127.0.0.1:${MOONCAKE_METADATA_PORT}/metadata" \
    --protocol=tcp --threads=8 --enable_http_server=true \
    --http_port="${MOONCAKE_CLIENT_HTTP_PORT}" \
    >"${RUN_DIR}/logs/mooncake-client.log" 2>&1 &
  mooncake_client_pid=$!; pd_track_group "${mooncake_client_pid}"
  pd_wait_http mooncake-client "http://127.0.0.1:${MOONCAKE_CLIENT_HTTP_PORT}/health" "${mooncake_client_pid}" 300
  local_hostname="$(hostname -I | awk '{print $1}')"
  mooncake_config="{\"master_server_address\":\"127.0.0.1:${MOONCAKE_MASTER_PORT}\",\"local_hostname\":\"${local_hostname}\",\"metadata_server\":\"http://127.0.0.1:${MOONCAKE_METADATA_PORT}/metadata\",\"global_segment_size\":\"0\",\"protocol\":\"tcp\",\"device_name\":\"\",\"prefetch_threshold\":64,\"prefetch_timeout_base\":5,\"prefetch_timeout_per_ki_token\":0.5}"
fi

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

gpu_numa_node() {
  local node
  node="$(nvidia-smi topo -m | awk -v row="GPU$1" '$1 == row {print $(NF-1); exit}')"
  [[ "${node}" =~ ^[0-9]+$ ]] && printf '%s\n' "${node}" || printf '0\n'
}

for index in "${!p_gpu_groups[@]}"; do
  IFS=',' read -r -a group_gpus <<<"${p_gpu_groups[index]}"
  p_tp_args=()
  p_parser_args=()
  if (( PREFILL_TP_SIZE > 1 )); then
    group_numas=()
    for gpu in "${group_gpus[@]}"; do group_numas+=("$(gpu_numa_node "${gpu}")"); done
    p_tp_args+=(--tp-size "${PREFILL_TP_SIZE}" --numa-node "${group_numas[@]}")
  fi
  if [[ -n "${MODEL_REASONING_PARSER}" ]]; then
    p_parser_args+=(--reasoning-parser "${MODEL_REASONING_PARSER}")
  fi
  if [[ -n "${MODEL_TOOL_CALL_PARSER}" ]]; then
    p_parser_args+=(--tool-call-parser "${MODEL_TOOL_CALL_PARSER}")
  fi
  p_args=(
    --model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${p_ports[index]}"
    --context-length "${MODEL_CONTEXT_LENGTH}" --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${p_mem_fraction_statics[index]}"
    --enable-metrics --uvicorn-access-log-exclude-prefixes /get_load /metrics /health
    --disaggregation-mode prefill --disaggregation-transfer-backend nixl
    --disaggregation-bootstrap-port "${p_bootstrap_ports[index]}"
    "${p_tp_args[@]}"
    "${p_parser_args[@]}"
  )
  if [[ "${CASE_MODE}" == no_reverse ]]; then
    p_args+=(--disable-radix-cache)
  else
    p_args+=(--enable-hierarchical-cache --hicache-size "${P_HICACHE_SIZE}" \
      --hicache-mem-layout "${HICACHE_MEM_LAYOUT}" \
      --hicache-io-backend "${HICACHE_IO_BACKEND}" \
      --hicache-write-policy "${P_HICACHE_WRITE_POLICY}" \
      --hicache-storage-backend mooncake --hicache-storage-prefetch-policy timeout \
      --hicache-storage-backend-extra-config "${mooncake_config}")
  fi
  setsid env CUDA_VISIBLE_DEVICES="${p_gpu_groups[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server "${p_args[@]}" \
    >"${RUN_DIR}/logs/prefill-${index}.log" 2>&1 &
  prefill_pid=$!; pd_track_group "${prefill_pid}"
  pd_wait_http "prefill-${index}" "http://127.0.0.1:${p_ports[index]}/health" "${prefill_pid}" 900
done

for index in "${!d_gpu_groups[@]}"; do
  IFS=',' read -r -a group_gpus <<<"${d_gpu_groups[index]}"
  d_tp_args=()
  d_parser_args=()
  if (( DECODE_TP_SIZE > 1 )); then
    group_numas=()
    for gpu in "${group_gpus[@]}"; do group_numas+=("$(gpu_numa_node "${gpu}")"); done
    d_tp_args+=(--tp-size "${DECODE_TP_SIZE}" --numa-node "${group_numas[@]}")
  fi
  if [[ -n "${MODEL_REASONING_PARSER}" ]]; then
    d_parser_args+=(--reasoning-parser "${MODEL_REASONING_PARSER}")
  fi
  if [[ -n "${MODEL_TOOL_CALL_PARSER}" ]]; then
    d_parser_args+=(--tool-call-parser "${MODEL_TOOL_CALL_PARSER}")
  fi
  d_args=(
    --model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${d_ports[index]}"
    --context-length "${MODEL_CONTEXT_LENGTH}" --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${d_mem_fraction_statics[index]}"
    --enable-metrics --uvicorn-access-log-exclude-prefixes /get_load /metrics /health
    --disaggregation-mode decode --disaggregation-transfer-backend nixl
    "${d_tp_args[@]}"
    "${d_parser_args[@]}"
  )
  if [[ "${CASE_MODE}" == native_mooncake ]]; then
    d_args+=(--disaggregation-decode-enable-offload-kvcache --hicache-size "${D_HICACHE_SIZE}" \
      --hicache-mem-layout "${HICACHE_MEM_LAYOUT}" \
      --hicache-io-backend "${HICACHE_IO_BACKEND}" \
      --hicache-storage-backend mooncake \
      --hicache-storage-backend-extra-config "${mooncake_config}")
  fi
  setsid env CUDA_VISIBLE_DEVICES="${d_gpu_groups[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server "${d_args[@]}" \
    >"${RUN_DIR}/logs/decode-${index}.log" 2>&1 &
  decode_pid=$!; pd_track_group "${decode_pid}"
  pd_wait_http "decode-${index}" "http://127.0.0.1:${d_ports[index]}/health" "${decode_pid}" 900
done

if [[ "${START_SEARCH_SERVER}" == "true" && "${SEARCH_START_AFTER_MODELS}" == "true" ]]; then
  start_search_server
fi

router_args=(--pd-disaggregation --host 0.0.0.0 --port "${ROUTER_PORT}"
  --prometheus-port "${ROUTER_PROMETHEUS_PORT}" --policy power_of_two)
for index in "${!p_ports[@]}"; do
  router_args+=(--prefill "http://127.0.0.1:${p_ports[index]}" "${p_bootstrap_ports[index]}")
done
for port in "${d_ports[@]}"; do router_args+=(--decode "http://127.0.0.1:${port}"); done
setsid "${PD_ENV_BIN}/python" -m sglang_router.launch_router "${router_args[@]}" \
  >"${RUN_DIR}/logs/router.log" 2>&1 &
router_pid=$!; pd_track_group "${router_pid}"
pd_wait_http router "http://127.0.0.1:${ROUTER_PORT}/health" "${router_pid}" 300

p_ports_csv="$(IFS=,; echo "${p_ports[*]}")"
d_ports_csv="$(IFS=,; echo "${d_ports[*]}")"
infer_args=(
  --model "${MODEL_PATH}"
  --router-port "${ROUTER_PORT}"
  --prefill-port "${p_ports[0]}" --prefill-ports "${p_ports_csv}"
  --decode-port "${d_ports[0]}" --decode-ports "${d_ports_csv}"
  --requests "${REQUESTS}" --warmup-requests 0
  --dispatch-policy "${DISPATCH_POLICY}"
  --request-rate 100 --arrival-distribution fixed --max-inflight "${MAX_INFLIGHT}"
  --metrics-interval 2 --seed "${SEED}"
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}"
  --max-context-length "${MODEL_CONTEXT_LENGTH}"
  --max-response-length "${MODEL_MAX_RESPONSE_LENGTH}"
  --output-dir "${RUN_DIR}"
)
if [[ "${DISPATCH_POLICY}" == "fixed" || "${DISPATCH_POLICY}" == "profile_balanced" ]]; then
  infer_args+=(--schedule-file "${SCHEDULE_FILE}")
fi
if [[ "${CLOSED_LOOP}" == "true" ]]; then
  infer_args+=(
    --closed-loop --closed-loop-warmup-min-seconds "${WARMUP_SECONDS}"
    --closed-loop-warmup-completions 0 --closed-loop-recent-seconds 120
    --closed-loop-max-warmup-seconds "$((WARMUP_SECONDS + 120))"
    --closed-loop-measurement-seconds "${MEASURE_SECONDS}"
  )
elif [[ "${CLOSED_LOOP}" != "false" ]]; then
  echo "CLOSED_LOOP must be true or false" >&2
  exit 2
fi
if [[ -n "${WORKLOAD_CONFIG}" ]]; then
  infer_args+=(--workload-config "${WORKLOAD_CONFIG}")
else
  infer_args+=(--math-data "${MATH_DATA}" --qa-data "${QA_DATA}" --math-ratio "${MATH_RATIO}")
fi
if [[ "${CASE_MODE}" == native_mooncake ]]; then
  infer_args+=(--pd-enable-decode-offload-kvcache --pd-hicache-storage-backend mooncake \
    --pd-hicache-storage-dir "${RUN_DIR}/hicache" \
    --pd-hicache-storage-prefetch-policy timeout --pd-hicache-prefetch-threshold 64)
fi
if [[ "${PRESERVE_SOURCE_ORDER}" == "true" ]]; then
  infer_args+=(--preserve-source-order)
fi
SLIME_HTTP_READ_TIMEOUT_SECONDS="${SLIME_HTTP_READ_TIMEOUT_SECONDS:-3600}" \
"${PD_ENV_BIN}/python" inference.py "${infer_args[@]}" >"${RUN_DIR}/inference.log" 2>&1
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
echo "baseline PD case complete: ${RUN_DIR}"
