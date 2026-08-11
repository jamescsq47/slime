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
CASE_MODE="${CASE_MODE:-no_reverse}"
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline-${CASE_MODE}}"
PREFILL_GPUS="${PREFILL_GPUS:-0}"
DECODE_GPUS="${DECODE_GPUS:-1 2 3 4 5}"
PREFILL_PORTS="${PREFILL_PORTS:-27100}"
PREFILL_BOOTSTRAP_PORTS="${PREFILL_BOOTSTRAP_PORTS:-28100}"
DECODE_PORTS="${DECODE_PORTS:-27101 27102 27103 27104 27105}"
ROUTER_PORT="${ROUTER_PORT:-27110}"
ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-27120}"
SEARCH_GPU="${SEARCH_GPU:-6}"
SEARCH_PORT="${SEARCH_PORT:-8710}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-57151}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-57180}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-57103}"
MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-57152}"
MOONCAKE_CLIENT_HTTP_PORT="${MOONCAKE_CLIENT_HTTP_PORT:-57190}"
MAX_INFLIGHT="${MAX_INFLIGHT:-384}"
REQUESTS="${REQUESTS:-4096}"
SEED="${SEED:-2026}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_random_s2026_n4096.json}"
PAGE_SIZE="${PAGE_SIZE:-64}"

case "${CASE_MODE}" in
  no_reverse|native_mooncake) ;;
  *) echo "CASE_MODE must be no_reverse or native_mooncake" >&2; exit 2 ;;
esac

read -r -a p_gpus <<<"${PREFILL_GPUS}"
read -r -a p_ports <<<"${PREFILL_PORTS}"
read -r -a p_bootstrap_ports <<<"${PREFILL_BOOTSTRAP_PORTS}"
read -r -a d_gpus <<<"${DECODE_GPUS}"
read -r -a d_ports <<<"${DECODE_PORTS}"
(( ${#p_gpus[@]} == ${#p_ports[@]} && ${#p_gpus[@]} == ${#p_bootstrap_ports[@]} )) || {
  echo "PREFILL_GPUS/PREFILL_PORTS/PREFILL_BOOTSTRAP_PORTS length mismatch" >&2; exit 2;
}
(( ${#d_gpus[@]} == ${#d_ports[@]} )) || {
  echo "DECODE_GPUS/DECODE_PORTS length mismatch" >&2; exit 2;
}
if (( ${#p_gpus[@]} < 1 || ${#d_gpus[@]} < 1 )); then
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

for gpu in "${p_gpus[@]}" "${d_gpus[@]}" "${SEARCH_GPU}"; do pd_check_gpu_idle "${gpu}"; done
for port in "${p_ports[@]}" "${p_bootstrap_ports[@]}" "${d_ports[@]}" \
  "${ROUTER_PORT}" "${ROUTER_PROMETHEUS_PORT}" "${SEARCH_PORT}"; do pd_check_port_free "${port}"; done

mooncake_config=""
if [[ "${CASE_MODE}" == native_mooncake ]]; then
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

setsid env CUDA_VISIBLE_DEVICES="${SEARCH_GPU}" SEARCH_SERVER_GPU_IDS=0 \
  "${PD_ENV_BIN}/python" search_server.py --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 --port "${SEARCH_PORT}" >"${RUN_DIR}/logs/search.log" 2>&1 &
search_pid=$!; pd_track_group "${search_pid}"
pd_wait_http search "http://127.0.0.1:${SEARCH_PORT}/health" "${search_pid}" 1200

for index in "${!p_gpus[@]}"; do
  p_args=(
    --model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${p_ports[index]}"
    --context-length 40960 --page-size "${PAGE_SIZE}" --mem-fraction-static 0.85
    --enable-metrics --uvicorn-access-log-exclude-prefixes /get_load /metrics /health
    --disaggregation-mode prefill --disaggregation-transfer-backend nixl
    --disaggregation-bootstrap-port "${p_bootstrap_ports[index]}"
  )
  if [[ "${CASE_MODE}" == no_reverse ]]; then
    p_args+=(--disable-radix-cache)
  else
    p_args+=(--enable-hierarchical-cache --hicache-size 128 \
      --hicache-mem-layout page_first --hicache-write-policy write_through \
      --hicache-storage-backend mooncake --hicache-storage-prefetch-policy timeout \
      --hicache-storage-backend-extra-config "${mooncake_config}")
  fi
  setsid env CUDA_VISIBLE_DEVICES="${p_gpus[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server "${p_args[@]}" \
    >"${RUN_DIR}/logs/prefill-${index}.log" 2>&1 &
  prefill_pid=$!; pd_track_group "${prefill_pid}"
  pd_wait_http "prefill-${index}" "http://127.0.0.1:${p_ports[index]}/health" "${prefill_pid}" 900
done

for index in "${!d_gpus[@]}"; do
  d_args=(
    --model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${d_ports[index]}"
    --context-length 40960 --page-size "${PAGE_SIZE}" --mem-fraction-static 0.85
    --enable-metrics --uvicorn-access-log-exclude-prefixes /get_load /metrics /health
    --disaggregation-mode decode --disaggregation-transfer-backend nixl
  )
  if [[ "${CASE_MODE}" == native_mooncake ]]; then
    d_args+=(--disaggregation-decode-enable-offload-kvcache --hicache-size 56 \
      --hicache-mem-layout page_first --hicache-storage-backend mooncake \
      --hicache-storage-backend-extra-config "${mooncake_config}")
  fi
  setsid env CUDA_VISIBLE_DEVICES="${d_gpus[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server "${d_args[@]}" \
    >"${RUN_DIR}/logs/decode-${index}.log" 2>&1 &
  decode_pid=$!; pd_track_group "${decode_pid}"
  pd_wait_http "decode-${index}" "http://127.0.0.1:${d_ports[index]}/health" "${decode_pid}" 900
done

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
  --model "${MODEL_PATH}" --math-data "${MATH_DATA}" --qa-data "${QA_DATA}"
  --router-port "${ROUTER_PORT}"
  --prefill-port "${p_ports[0]}" --prefill-ports "${p_ports_csv}"
  --decode-port "${d_ports[0]}" --decode-ports "${d_ports_csv}"
  --math-ratio 0.5 --requests "${REQUESTS}" --warmup-requests 0
  --dispatch-policy fixed --schedule-file "${SCHEDULE_FILE}"
  --request-rate 100 --arrival-distribution fixed --max-inflight "${MAX_INFLIGHT}"
  --metrics-interval 2 --seed "${SEED}" --temperature 0 --top-p 1 --top-k -1
  --closed-loop --closed-loop-warmup-min-seconds "${WARMUP_SECONDS}"
  --closed-loop-warmup-completions 0 --closed-loop-recent-seconds 120
  --closed-loop-max-warmup-seconds "$((WARMUP_SECONDS + 120))"
  --closed-loop-measurement-seconds "${MEASURE_SECONDS}" --output-dir "${RUN_DIR}"
)
if [[ "${CASE_MODE}" == native_mooncake ]]; then
  infer_args+=(--pd-enable-decode-offload-kvcache --pd-hicache-storage-backend mooncake \
    --pd-hicache-storage-dir "${RUN_DIR}/hicache" \
    --pd-hicache-storage-prefetch-policy timeout --pd-hicache-prefetch-threshold 64)
fi
SLIME_HTTP_READ_TIMEOUT_SECONDS="${SLIME_HTTP_READ_TIMEOUT_SECONDS:-3600}" \
"${PD_ENV_BIN}/python" inference.py "${infer_args[@]}" >"${RUN_DIR}/inference.log" 2>&1
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/analyze_pd_offload.py" --run-dir "${RUN_DIR}"
echo "baseline PD case complete: ${RUN_DIR}"
