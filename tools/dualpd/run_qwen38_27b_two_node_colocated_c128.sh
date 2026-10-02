#!/usr/bin/env bash
set -euo pipefail

# Two-node native colocated baseline: four TP=2 replicas per node, one global
# router, SWE-bench Verified 500 tasks, global concurrency 128.

SLIME_ROOT="/homes/siqic/dualpd/pd_multi_node_v3/slime"
ENV_BIN="/homes/siqic/anaconda3/envs/pd_mamba_baseline/bin"
MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.8-27B}"
REMOTE_NODE="${REMOTE_NODE:-a11}"
REMOTE_IP="${REMOTE_IP:-10.0.1.171}"
RUN_ID="${RUN_ID:-qwen38-27b-tp2-colocated-2node-c128-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_ROOT="/tmp/dualpd-baseline/${RUN_ID}"
REMOTE_ROOT="/tmp/dualpd-baseline/${RUN_ID}"
RESULT_ROOT="${SLIME_ROOT}/runs/dualpd/${RUN_ID}"
WORKLOAD_CONFIG="${SLIME_ROOT}/examples/pd/configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml"
DATA_ROOT="/homes/siqic/dualpd/slime/downloads/pd-data"
ROUTER_PORT=34710
LOCAL_PORTS=(34700 34701 34702 34703)
REMOTE_PORTS=(34800 34801 34802 34803)
GPU_GROUPS=(0,4 1,5 2,6 3,7)
LOCAL_PIDS=()

mkdir -p "${LOCAL_ROOT}/logs" "${RESULT_ROOT}"
ssh "${REMOTE_NODE}" "mkdir -p '${REMOTE_ROOT}/logs' '${REMOTE_ROOT}/pids'"

export PATH="${ENV_BIN}:${PATH}"
export PYTHONPATH="${SLIME_ROOT}:$(dirname "${SLIME_ROOT}"):${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PD_DATA_ROOT="${DATA_ROOT}"
export PD_SWE_RUN_ID="${RUN_ID}"
export PD_SWE_PROGRESS_FILE="${RESULT_ROOT}/episode_progress.jsonl"
export PD_INFERENCE_RETURN_LOGPROB=false
export PD_MODEL_HTTP_TRANSPORT=aiohttp
export SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
unset SGLANG_OVERLAY_ROOT SGLANG_ENABLE_DETERMINISTIC_INFERENCE NCCL_ALGO || true
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_AGENTIC_)
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_PD_)

wait_http() {
  local name="$1" url="$2" timeout_seconds="$3" start now
  start="$(date +%s)"
  while ! curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; do
    now="$(date +%s)"
    if (( now - start >= timeout_seconds )); then
      echo "Timed out waiting for ${name}: ${url}" >&2
      return 1
    fi
    sleep 2
  done
  echo "Ready: ${name}"
}

stop_local_group() {
  local pid="$1"
  kill -TERM -- "-${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$? pid deadline owned=()
  trap - EXIT INT TERM
  for pid in "${LOCAL_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && stop_local_group "${pid}"
  done
  ssh "${REMOTE_NODE}" "
    if [[ -d '${REMOTE_ROOT}/pids' ]]; then
      for file in '${REMOTE_ROOT}'/pids/*; do
        [[ -f \"\$file\" ]] || continue
        pid=\$(cat \"\$file\")
        kill -TERM -- -\"\$pid\" 2>/dev/null || true
      done
    fi
  " >/dev/null 2>&1 || true
  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    local alive=0
    for pid in "${LOCAL_PIDS[@]:-}"; do
      [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "${LOCAL_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill -KILL -- "-${pid}" 2>/dev/null || true
  done
  ssh "${REMOTE_NODE}" "
    if [[ -d '${REMOTE_ROOT}/pids' ]]; then
      for file in '${REMOTE_ROOT}'/pids/*; do
        [[ -f \"\$file\" ]] || continue
        pid=\$(cat \"\$file\")
        kill -KILL -- -\"\$pid\" 2>/dev/null || true
      done
    fi
  " >/dev/null 2>&1 || true
  mapfile -t owned < <(timeout 15 docker ps -aq --filter "label=pd.swe.run_id=${RUN_ID}" 2>/dev/null || true)
  if (( ${#owned[@]} )); then timeout 60 docker rm -f "${owned[@]}" >/dev/null 2>&1 || true; fi
  rsync -a "${LOCAL_ROOT}/" "${RESULT_ROOT}/local-services/" >/dev/null 2>&1 || true
  rsync -a "${REMOTE_NODE}:${REMOTE_ROOT}/" "${RESULT_ROOT}/remote-services/" >/dev/null 2>&1 || true
  printf '%s\n' "${status}" >"${RESULT_ROOT}/launcher_exit_code"
  exit "${status}"
}
trap cleanup EXIT INT TERM

for port in "${LOCAL_PORTS[@]}" "${REMOTE_PORTS[@]}" "${ROUTER_PORT}"; do
  if ss -ltnH "sport = :${port}" | grep -q .; then
    echo "Local port ${port} is already in use" >&2
    exit 2
  fi
done
ssh "${REMOTE_NODE}" "
  for port in ${REMOTE_PORTS[*]}; do
    if ss -ltnH \"sport = :\$port\" | grep -q .; then
      echo \"Remote port \$port is already in use\" >&2
      exit 2
    fi
  done
"

for gpu in {0..7}; do
  used="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | wc -l)"
  (( used == 0 )) || { echo "Local GPUs are not idle" >&2; exit 2; }
  break
done
ssh "${REMOTE_NODE}" 'test "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | wc -l)" -eq 0' || {
  echo "Remote GPUs are not idle" >&2
  exit 2
}

model_args=(
  --model-path "${MODEL_PATH}" --host 127.0.0.1
  --tp-size 2 --context-length 131072 --page-size 64
  --mamba-track-interval 64 --mamba-full-memory-ratio 0.9
  --mem-fraction-static 0.80 --chunked-prefill-size 8192
  --max-prefill-tokens 8192 --attention-backend triton
  --sampling-backend flashinfer --random-seed 2026 --numa-node 0 1
  --mamba-radix-cache-strategy extra_buffer --reasoning-parser glm45
  --tool-call-parser qwen3_coder --enable-metrics --skip-server-warmup
  --uvicorn-access-log-exclude-prefixes /get_load /metrics /health
)

for index in "${!LOCAL_PORTS[@]}"; do
  setsid env CUDA_VISIBLE_DEVICES="${GPU_GROUPS[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${ENV_BIN}/python" -m sglang.launch_server "${model_args[@]}" \
    --port "${LOCAL_PORTS[index]}" \
    >"${LOCAL_ROOT}/logs/model-${index}.log" 2>&1 < /dev/null &
  LOCAL_PIDS+=("$!")
  # Serialize first-time kernel compilation/model initialization. Starting all
  # four replicas together can contend on the shared compiler cache and ports.
  wait_http "a10-model-${index}" "http://127.0.0.1:${LOCAL_PORTS[index]}/model_info" 1200
done

for index in "${!REMOTE_PORTS[@]}"; do
  remote_port="${REMOTE_PORTS[index]}"
  remote_group="${GPU_GROUPS[index]}"
  ssh "${REMOTE_NODE}" bash -s -- \
    "${remote_port}" "${remote_group}" "${REMOTE_ROOT}" "${ENV_BIN}" "${MODEL_PATH}" <<'REMOTE'
set -euo pipefail
port="$1"; group="$2"; root="$3"; env_bin="$4"; model="$5"
index=$((port - 34800))
export PATH="${env_bin}:${PATH}"
setsid env CUDA_VISIBLE_DEVICES="${group}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
  "${env_bin}/python" -m sglang.launch_server \
  --model-path "${model}" --host "${REMOTE_IP:-10.0.1.171}" --port "${port}" \
  --tp-size 2 --context-length 131072 --page-size 64 \
  --mamba-track-interval 64 --mamba-full-memory-ratio 0.9 \
  --mem-fraction-static 0.80 --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 --attention-backend triton \
  --sampling-backend flashinfer --random-seed 2026 --numa-node 0 1 \
  --mamba-radix-cache-strategy extra_buffer --reasoning-parser glm45 \
  --tool-call-parser qwen3_coder --enable-metrics --skip-server-warmup \
  --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
  >"${root}/logs/model-${index}.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"${root}/pids/model-${index}"
REMOTE
  wait_http "a11-model-${index}" "http://${REMOTE_IP}:${remote_port}/model_info" 1200
done

worker_urls=()
for port in "${LOCAL_PORTS[@]}"; do worker_urls+=("http://127.0.0.1:${port}"); done
for port in "${REMOTE_PORTS[@]}"; do worker_urls+=("http://${REMOTE_IP}:${port}"); done
setsid "${ENV_BIN}/python" -m sglang_router.launch_router \
  --worker-urls "${worker_urls[@]}" --policy cache_aware \
  --cache-threshold 0.3 --balance-abs-threshold 8 --balance-rel-threshold 1.2 \
  --host 0.0.0.0 --port "${ROUTER_PORT}" \
  >"${LOCAL_ROOT}/logs/router.log" 2>&1 < /dev/null &
LOCAL_PIDS+=("$!")
wait_http router "http://127.0.0.1:${ROUTER_PORT}/health" 300

# Forward remote metrics ports to localhost so inference.py can sample all
# eight collocated engines through its existing single-host metrics interface.
tunnel_args=()
for port in "${REMOTE_PORTS[@]}"; do tunnel_args+=(-L "${port}:${REMOTE_IP}:${port}"); done
setsid ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 \
  -o ServerAliveCountMax=3 "${tunnel_args[@]}" "${REMOTE_NODE}" \
  >"${LOCAL_ROOT}/logs/metrics-tunnel.log" 2>&1 < /dev/null &
LOCAL_PIDS+=("$!")
sleep 2

all_ports="$(IFS=,; printf '%s' "${LOCAL_PORTS[*]},${REMOTE_PORTS[*]}")"
cd "${SLIME_ROOT}/examples/pd"
"${ENV_BIN}/python" "${SLIME_ROOT}/examples/pd/scripts/new_method/internal/inference_checkpointed.py" \
  --model "${MODEL_PATH}" --workload-config "${WORKLOAD_CONFIG}" \
  --router-host 127.0.0.1 --router-port "${ROUTER_PORT}" \
  --prefill-host 127.0.0.1 --prefill-port "${LOCAL_PORTS[0]}" \
  --prefill-ports "${all_ports}" --decode-host 127.0.0.1 \
  --decode-port "${LOCAL_PORTS[0]}" --decode-ports "${all_ports}" \
  --requests 500 --warmup-requests 0 --dispatch-policy random \
  --preserve-source-order --request-rate 100 --arrival-distribution fixed \
  --max-inflight 128 --metrics-interval 2 --seed 2026 \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --max-context-length 131072 --max-response-length 81920 \
  --output-dir "${RESULT_ROOT}/workload" \
  >"${LOCAL_ROOT}/logs/inference.log" 2>&1

"${ENV_BIN}/python" "${SLIME_ROOT}/examples/pd/scripts/tools/analyze_swe_bench_run.py" \
  "${RESULT_ROOT}/workload" >"${LOCAL_ROOT}/logs/analyzer.log" 2>&1
