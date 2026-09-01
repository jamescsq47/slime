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
CASE_MODE="${CASE_MODE:-no_reverse}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
P_GPU="${P_GPU:-0}"
D_GPU="${D_GPU:-1}"
P_PORT="${P_PORT:-28000}"
D_PORT="${D_PORT:-28001}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-29000}"
ROUTER_PORT="${ROUTER_PORT:-28010}"
ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-28020}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-58051}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-58080}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-58003}"
MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-58052}"
MOONCAKE_CLIENT_HTTP_PORT="${MOONCAKE_CLIENT_HTTP_PORT:-58090}"
CONCURRENCY="${CONCURRENCY:-32}"
FIRST_TURN_TOKENS="${FIRST_TURN_TOKENS:-256}"
INTER_TURN_DELAY="${INTER_TURN_DELAY:-0}"
WORKLOAD_MODE="${WORKLOAD_MODE:-multiturn}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
P_HICACHE_SIZE="${P_HICACHE_SIZE:-128}"
D_HICACHE_SIZE="${D_HICACHE_SIZE:-56}"
MOONCAKE_SIZE="${MOONCAKE_SIZE:-256 GB}"

case "${CASE_MODE}" in
  no_reverse|hicache_no_decode_offload|native_mooncake) ;;
  *) exit 2 ;;
esac
mkdir -p "${RUN_DIR}/logs"
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd):${PYTHONPATH:-}"
unset SGLANG_AGENTIC_KV_LIFECYCLE SGLANG_AGENTIC_KV_HOST_STAGING \
  SGLANG_AGENTIC_KV_D_HOSTLESS SGLANG_AGENTIC_KV_LEDGER_PATH \
  SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT || true
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_environments.py" \
  --expect baseline --output "${RUN_DIR}/environment.json"

for gpu in "${P_GPU}" "${D_GPU}"; do pd_check_gpu_idle "${gpu}"; done
for port in "${P_PORT}" "${D_PORT}" "${BOOTSTRAP_PORT}" "${ROUTER_PORT}" \
  "${ROUTER_PROMETHEUS_PORT}"; do pd_check_port_free "${port}"; done

mooncake_config=""
if [[ "${CASE_MODE}" != no_reverse ]]; then
  for port in "${MOONCAKE_MASTER_PORT}" "${MOONCAKE_METADATA_PORT}" \
    "${MOONCAKE_METRICS_PORT}" "${MOONCAKE_CLIENT_PORT}" "${MOONCAKE_CLIENT_HTTP_PORT}"; do
    pd_check_port_free "${port}"
  done
  setsid "${PD_ENV_BIN}/mooncake_master" --rpc_port="${MOONCAKE_MASTER_PORT}" \
    --enable_http_metadata_server=true --http_metadata_server_port="${MOONCAKE_METADATA_PORT}" \
    --eviction_high_watermark_ratio=0.85 --eviction_ratio=0.10 \
    --metrics_port="${MOONCAKE_METRICS_PORT}" >"${RUN_DIR}/logs/mooncake-master.log" 2>&1 &
  master_pid=$!; pd_track_group "${master_pid}"
  pd_wait_http mooncake-master "http://127.0.0.1:${MOONCAKE_METRICS_PORT}/health" "${master_pid}" 300
  setsid "${PD_ENV_BIN}/mooncake_client" --host=127.0.0.1 --port="${MOONCAKE_CLIENT_PORT}" \
    --global_segment_size="${MOONCAKE_SIZE}" --master_server_address="127.0.0.1:${MOONCAKE_MASTER_PORT}" \
    --metadata_server="http://127.0.0.1:${MOONCAKE_METADATA_PORT}/metadata" \
    --protocol=tcp --threads=8 --enable_http_server=true --http_port="${MOONCAKE_CLIENT_HTTP_PORT}" \
    >"${RUN_DIR}/logs/mooncake-client.log" 2>&1 &
  client_pid=$!; pd_track_group "${client_pid}"
  pd_wait_http mooncake-client "http://127.0.0.1:${MOONCAKE_CLIENT_HTTP_PORT}/health" "${client_pid}" 300
  hostname_ip="$(hostname -I | awk '{print $1}')"
  mooncake_config="{\"master_server_address\":\"127.0.0.1:${MOONCAKE_MASTER_PORT}\",\"local_hostname\":\"${hostname_ip}\",\"metadata_server\":\"http://127.0.0.1:${MOONCAKE_METADATA_PORT}/metadata\",\"global_segment_size\":\"0\",\"protocol\":\"tcp\",\"device_name\":\"\",\"prefetch_threshold\":64,\"prefetch_timeout_base\":5,\"prefetch_timeout_per_ki_token\":0.5}"
fi

p_args=(--model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${P_PORT}" \
  --context-length 40960 --page-size 64 --mem-fraction-static 0.85 --enable-metrics \
  --enable-deterministic-inference --attention-backend triton --random-seed 2026 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}")
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then p_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}"); fi
if [[ "${CASE_MODE}" == no_reverse ]]; then
  p_args+=(--disable-radix-cache)
else
  p_args+=(--enable-hierarchical-cache --hicache-size "${P_HICACHE_SIZE}" --hicache-mem-layout page_first \
    --hicache-write-policy write_through --hicache-storage-backend mooncake \
    --hicache-storage-prefetch-policy timeout --hicache-storage-backend-extra-config "${mooncake_config}")
fi
setsid env CUDA_VISIBLE_DEVICES="${P_GPU}" "${PD_ENV_BIN}/python" -m sglang.launch_server \
  "${p_args[@]}" >"${RUN_DIR}/logs/prefill.log" 2>&1 &
p_pid=$!; pd_track_group "${p_pid}"
pd_wait_http prefill "http://127.0.0.1:${P_PORT}/health" "${p_pid}" 900

d_args=(--model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${D_PORT}" \
  --context-length 40960 --page-size 64 --mem-fraction-static 0.85 --enable-metrics \
  --enable-deterministic-inference --attention-backend triton --random-seed 2026 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl)
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then d_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}"); fi
if [[ "${CASE_MODE}" == native_mooncake ]]; then
  d_args+=(--disaggregation-decode-enable-offload-kvcache --hicache-size "${D_HICACHE_SIZE}" \
    --hicache-mem-layout page_first --hicache-storage-backend mooncake \
    --hicache-storage-backend-extra-config "${mooncake_config}")
fi
setsid env CUDA_VISIBLE_DEVICES="${D_GPU}" "${PD_ENV_BIN}/python" -m sglang.launch_server \
  "${d_args[@]}" >"${RUN_DIR}/logs/decode.log" 2>&1 &
d_pid=$!; pd_track_group "${d_pid}"
pd_wait_http decode "http://127.0.0.1:${D_PORT}/health" "${d_pid}" 900

setsid "${PD_ENV_BIN}/python" -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://127.0.0.1:${P_PORT}" "${BOOTSTRAP_PORT}" \
  --decode "http://127.0.0.1:${D_PORT}" --policy power_of_two \
  --prometheus-port "${ROUTER_PROMETHEUS_PORT}" --host 0.0.0.0 --port "${ROUTER_PORT}" \
  >"${RUN_DIR}/logs/router.log" 2>&1 &
router_pid=$!; pd_track_group "${router_pid}"
pd_wait_http router "http://127.0.0.1:${ROUTER_PORT}/health" "${router_pid}" 300

if [[ "${WORKLOAD_MODE}" == churn ]]; then
  "${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/capture_decode_offload_churn.py" \
    --url "http://127.0.0.1:${ROUTER_PORT}" --decode-url "http://127.0.0.1:${D_PORT}" \
    --model "${MODEL_PATH}" --label "${CASE_MODE}" --output "${RUN_DIR}/capture.json" \
    >"${RUN_DIR}/capture.log" 2>&1
else
  "${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/capture_pd_correctness.py" \
    --url "http://127.0.0.1:${ROUTER_PORT}" --model "${MODEL_PATH}" \
    --label "${CASE_MODE}" --concurrency "${CONCURRENCY}" \
    --first-turn-tokens "${FIRST_TURN_TOKENS}" --inter-turn-delay "${INTER_TURN_DELAY}" \
    --output "${RUN_DIR}/capture.json" \
    >"${RUN_DIR}/capture.log" 2>&1
fi
echo "correctness case complete: ${CASE_MODE} ${RUN_DIR}"
