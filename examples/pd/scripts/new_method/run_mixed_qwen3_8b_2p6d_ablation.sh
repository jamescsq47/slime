#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ABLATION="${1:?usage: $0 full|d2p-direct-only|d2p-slow-only|random-routing|p2d-direct-only}"
BASE_RUN_DIR="${PD_DIR}/runs-host/current/ablations/mixed1to1-qwen3-8b-2p6d-c512"

# Each invocation starts from the production behavior.  This prevents an
# inherited shell variable from accidentally composing two ablations.
export SGLANG_AGENTIC_KV_HOST_STAGING=true
export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=false
export SGLANG_PD_ABLATION_RANDOM_ROUTING=false
export SGLANG_PD_ABLATION_RANDOM_SEED=2026
export P2D_HOST_STAGING=true

case "${ABLATION}" in
  full)
    ;;
  d2p-direct-only)
    export SGLANG_AGENTIC_KV_HOST_STAGING=false
    ;;
  d2p-slow-only)
    export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=true
    ;;
  random-routing)
    export SGLANG_PD_ABLATION_RANDOM_ROUTING=true
    export SGLANG_PD_ABLATION_RANDOM_SEED=2026
    ;;
  p2d-direct-only)
    export P2D_HOST_STAGING=false
    ;;
  *)
    echo "unknown ablation: ${ABLATION}" >&2
    exit 2
    ;;
esac

env \
  RUN_DIR="${RUN_DIR:-${BASE_RUN_DIR}/${ABLATION}}" \
  PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd/bin}" \
  MODEL_PATH="${MODEL_PATH:-/dataset/model/qwen3/Qwen3-8B}" \
  MATH_RATIO=0.5 DISPATCH_POLICY=fixed \
  SCHEDULE_FILE="${PD_DIR}/configs/workloads/fixed_random_s2026_n8192.json" \
  REQUESTS="${REQUESTS:-8192}" MAX_INFLIGHT="${MAX_INFLIGHT:-512}" \
  SEED="${SEED:-2026}" TEMPERATURE=0 \
  CLOSED_LOOP=1 WARMUP_REQUESTS=0 \
  WARMUP_SECONDS="${WARMUP_SECONDS:-300}" \
  MAX_WARMUP_SECONDS="${MAX_WARMUP_SECONDS:-420}" \
  MEASURE_SECONDS="${MEASURE_SECONDS:-1200}" \
  SEARCH_GPU="${SEARCH_GPU:-7}" SEARCH_PORT="${SEARCH_PORT:-8730}" \
  timeout --signal=TERM --kill-after=240s 3600s \
  bash "${SCRIPT_DIR}/run_2p6d_numa_case.sh"
