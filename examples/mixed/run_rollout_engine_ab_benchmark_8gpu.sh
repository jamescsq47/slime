#!/usr/bin/env bash
# Eight-GPU simulation: 16 engines = 2 workers/GPU; 8 engines = 1 worker/GPU.
# All conditions use the same immutable base Qwen3-8B and SGLang settings.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-/workspace/Qwen3-8B-AWQ}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/debug/rollout_engine_ab_8gpu}
WORK_DIR=${WORK_DIR:-/tmp/slime_rollout_engine_ab_8gpu}
BASE_PORT=${BASE_PORT:-31000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.34}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-40960}
SEARCH_URL=${SEARCH_URL:-http://10.0.1.171:8000}
PIDS=()

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}"

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  PIDS=()
}
trap cleanup EXIT INT TERM

wait_healthy() {
  local url=$1 deadline=$((SECONDS + 900))
  until curl -sf "${url}/health" >/dev/null; do
    (( SECONDS < deadline )) || { echo "Timeout: ${url}" >&2; return 1; }
    sleep 2
  done
}

launch_workers() {
  local engines=$1 batch_start batch_end engine gpu port
  WORKER_URLS=()
  cleanup
  if ss -ltn | grep -Eq ':310(0[0-9]|1[0-5])[[:space:]]'; then
    echo "Benchmark worker ports are already occupied" >&2
    return 1
  fi
  # Never initialize two models on one GPU concurrently: SGLang sizes its
  # memory pool from instantaneous free memory during startup.
  for batch_start in 0 8; do
    (( batch_start < engines )) || break
    batch_end=$((batch_start + 8))
    (( batch_end > engines )) && batch_end=${engines}
    for ((engine=batch_start; engine<batch_end; engine++)); do
      gpu=$((engine % 8))
      port=$((BASE_PORT + engine))
      WORKER_URLS+=("http://127.0.0.1:${port}")
      setsid env CUDA_VISIBLE_DEVICES=${gpu} SLIME_ENABLE_PROFILING=true \
        python -m sglang.launch_server \
          --model-path "${MODEL}" --host 127.0.0.1 --port "${port}" \
          --context-length "${CONTEXT_LENGTH}" \
          --mem-fraction-static "${MEM_FRACTION_STATIC}" \
          --max-running-requests 32 --disable-cuda-graph --enable-metrics \
          >"${WORK_DIR}/worker_${engines}_${engine}.log" 2>&1 &
      PIDS+=("$!")
    done
    for ((engine=batch_start; engine<batch_end; engine++)); do
      wait_healthy "http://127.0.0.1:$((BASE_PORT + engine))"
    done
  done
}

run_condition() {
  local name=$1 engines=$2
  shift 2
  python "${SCRIPT_DIR}/rollout_engine_ab_benchmark.py" run \
    --name "${name}" --worker-urls "${WORKER_URLS[@]}" --engines "${engines}" \
    --model "${MODEL}" --weight-label base-qwen3-8b-awq-8gpu-simulation \
    --topology-note "8-GPU simulation: ${engines} engines; $((engines / 8)) worker(s) per GPU; uniform official Qwen3-8B-AWQ checkpoint" \
    --groups 32 --math-ratio 0.5 --samples-per-group 8 \
    --seed 47 --temperature 1 --top-p 1 --top-k -1 \
    --max-response-len 36864 --context-length "${CONTEXT_LENGTH}" \
    --output-dir "${OUTPUT_DIR}" "$@"
}

export MIXED_RETOOL_MAX_RESPONSE_LEN=${MIXED_RETOOL_MAX_RESPONSE_LEN:-8192}
export MIXED_BROWSECOMP_MAX_RESPONSE_LEN=${MIXED_BROWSECOMP_MAX_RESPONSE_LEN:-36864}
export BROWSECOMP_MAX_SEQ_LEN=${BROWSECOMP_MAX_SEQ_LEN:-36864}
export BROWSECOMP_MAX_TURNS=${BROWSECOMP_MAX_TURNS:-100}
export BROWSECOMP_TURN_MAX_NEW_TOKENS=${BROWSECOMP_TURN_MAX_NEW_TOKENS:-2048}
export BROWSECOMP_MUST_SEARCH=${BROWSECOMP_MUST_SEARCH:-1}
export BROWSECOMP_ENABLE_THINKING=${BROWSECOMP_ENABLE_THINKING:-0}
export LOCAL_SEARCH_URL=${SEARCH_URL}
wait_healthy "${SEARCH_URL}"

if [[ ${ONLY_PARTIAL:-0} == 1 ]]; then
  launch_workers 8
  run_condition e8_partial 8 --partial --abort-after "${ABORT_AFTER:-60}" --max-aborts "${MAX_ABORTS:-2}"
  python "${SCRIPT_DIR}/rollout_engine_ab_benchmark.py" summarize --output-dir "${OUTPUT_DIR}"
  exit 0
fi

launch_workers 16
run_condition e16_no_partial 16
launch_workers 8
run_condition e8_no_partial 8
run_condition e8_partial 8 --partial --abort-after "${ABORT_AFTER:-60}" --max-aborts "${MAX_ABORTS:-2}"
python "${SCRIPT_DIR}/rollout_engine_ab_benchmark.py" summarize --output-dir "${OUTPUT_DIR}"
