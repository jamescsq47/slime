#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
pd_install_cleanup_traps
cd "${PD_DIR}"

PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
MODEL_PATH="${MODEL_PATH:-/dataset/model/qwen3/Qwen3-8B}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
GPU="${GPU:-0}"
PORT="${PORT:-28400}"
CONCURRENCY="${CONCURRENCY:-32}"
FIRST_TURN_TOKENS="${FIRST_TURN_TOKENS:-256}"
INTER_TURN_DELAY="${INTER_TURN_DELAY:-0}"

mkdir -p "${RUN_DIR}/logs"
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd):${PYTHONPATH:-}"

pd_check_gpu_idle "${GPU}"
pd_check_port_free "${PORT}"

setsid env CUDA_VISIBLE_DEVICES="${GPU}" "${PD_ENV_BIN}/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" --host 0.0.0.0 --port "${PORT}" \
  --context-length 40960 --page-size 64 --mem-fraction-static 0.85 --enable-metrics \
  --enable-deterministic-inference --attention-backend triton --random-seed 2026 \
  >"${RUN_DIR}/logs/server.log" 2>&1 &
server_pid=$!
pd_track_group "${server_pid}"
pd_wait_http session-reference "http://127.0.0.1:${PORT}/health" "${server_pid}" 900

"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/capture_pd_correctness.py" \
  --url "http://127.0.0.1:${PORT}" --model "${MODEL_PATH}" \
  --label session_hbm_reference --session-reference --concurrency "${CONCURRENCY}" \
  --first-turn-tokens "${FIRST_TURN_TOKENS}" \
  --inter-turn-delay "${INTER_TURN_DELAY}" \
  --output "${RUN_DIR}/capture.json" >"${RUN_DIR}/capture.log" 2>&1

echo "session correctness reference complete: ${RUN_DIR}"
