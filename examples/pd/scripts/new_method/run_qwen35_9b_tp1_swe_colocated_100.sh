#!/usr/bin/env bash
set -euo pipefail

# Matched-environment diagnostic: same pd_mamba source, but native colocated
# serving. No lifecycle, reverse transfer, native HiCache or Mooncake enabled.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
pd_install_cleanup_traps
cd "${PD_DIR}"
export PD_ENV_BIN=/homes/siqic/anaconda3/envs/pd_mamba/bin
export SGLANG_OVERLAY_ROOT=/homes/siqic/sglang-agentic-mamba/python
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd):${PYTHONPATH:-}"
export PD_DATA_ROOT=/tmp/pd-data-first100
export PD_INFERENCE_RETURN_LOGPROB=false SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
export MODEL_PATH=/homes/siqic/Qwen3.5-9B
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml}"
export MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-qwen3}"
export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/qwen35-9b-tp1-swe-colocated-first100-$(date -u +%Y%m%dT%H%M%S)}"
export PD_SWE_RUN_ID="${PD_SWE_RUN_ID:-$(basename -- "${RUN_DIR}")}"
export PD_SWE_PROGRESS_FILE="${RUN_DIR}/episode_progress.jsonl"
export MIN_P=0
# Do not inherit any custom PD activation from an interactive shell.
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_AGENTIC_)
mkdir -p "${RUN_DIR}/logs"
[[ ! -e "${RUN_DIR}/requests.completed.jsonl" ]] || { echo 'Run already contains episodes' >&2; exit 2; }
# Docker containers outlive their host process group if Python receives TERM.
# Refuse a reused label before installing cleanup, so pre-existing containers
# can never be mistaken for this launch's resources.
existing_containers="$(docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")"
[[ -z "${existing_containers}" ]] || { echo 'Run label already owns containers' >&2; exit 2; }
pd_colocated_cleanup() {
  local status=$? owned_containers=()
  pd_cleanup_all
  mapfile -t owned_containers < <(timeout 15 docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")
  if (( ${#owned_containers[@]} )); then
    timeout 30 docker rm -f "${owned_containers[@]}" || true
  fi
  return "${status}"
}
trap pd_colocated_cleanup EXIT
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_environments.py" \
  --expect modified --output "${RUN_DIR}/environment.json"
"${PD_ENV_BIN}/python" -c 'import inference; import data.dispatch; import data.swe_bench_openenv.harness'

ports=(33600 33601 33602 33603 33604 33605 33606 33607)
for index in "${!ports[@]}"; do
  pd_check_gpu_idle "${index}"
  pd_check_port_free "${ports[index]}"
done
pd_check_port_free 33610

worker_pids=()
worker_urls=()
for index in "${!ports[@]}"; do
  mkdir -p "${RUN_DIR}/raw-${index}"
  # Level3 JSON logging retains the raw pre-parser output without truncation.
  # Explicit input_ids remain authoritative; logging does not re-tokenize them.
  setsid env CUDA_VISIBLE_DEVICES="${index}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${ports[index]}" \
    --tp-size 1 --context-length 131072 --page-size 64 --mamba-track-interval 64 \
    --mem-fraction-static 0.80 --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --enable-deterministic-inference --attention-backend triton --random-seed 2026 \
    --reasoning-parser "${MODEL_REASONING_PARSER}" --tool-call-parser qwen3_coder \
    --enable-metrics --skip-server-warmup \
    --log-requests --log-requests-level 3 --log-requests-format json \
    --log-requests-target "${RUN_DIR}/raw-${index}" \
    --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
    >"${RUN_DIR}/logs/model-${index}.log" 2>&1 &
  worker_pids+=("$!"); pd_track_group "$!"
  worker_urls+=("http://127.0.0.1:${ports[index]}")
done
for index in "${!ports[@]}"; do
  pd_wait_http "model-${index}" "${worker_urls[index]}/model_info" "${worker_pids[index]}" 1200
  echo "model-${index} ready"
done
setsid "${PD_ENV_BIN}/python" -m sglang_router.launch_router \
  --worker-urls "${worker_urls[@]}" --policy cache_aware \
  --cache-threshold 0.3 --balance-abs-threshold 8 --balance-rel-threshold 1.2 \
  --host 127.0.0.1 --port 33610 >"${RUN_DIR}/logs/router.log" 2>&1 &
router_pid=$!; pd_track_group "${router_pid}"
pd_wait_http router http://127.0.0.1:33610/health "${router_pid}" 300
ports_csv="$(IFS=,; echo "${ports[*]}")"
setsid "${PD_ENV_BIN}/python" "${SCRIPT_DIR}/internal/inference_checkpointed.py" \
  --model "${MODEL_PATH}" \
  --workload-config "${WORKLOAD_CONFIG}" \
  --router-port 33610 --prefill-port 33600 --prefill-ports "${ports_csv}" \
  --decode-port 33600 --decode-ports "${ports_csv}" \
  --requests 100 --warmup-requests 0 --dispatch-policy random --preserve-source-order \
  --request-rate 100 --arrival-distribution fixed --max-inflight 128 \
  --metrics-interval 2 --seed 2026 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --max-context-length 131072 --max-response-length 81920 --output-dir "${RUN_DIR}" \
  >"${RUN_DIR}/inference.log" 2>&1 &
inference_pid=$!; pd_track_group "${inference_pid}"
wait "${inference_pid}"
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/internal/analyze_mamba_swe_run.py" "${RUN_DIR}"
