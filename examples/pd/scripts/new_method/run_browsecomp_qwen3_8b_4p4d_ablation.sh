#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ABLATION="${1:?usage: $0 d2p-direct-only|d2p-slow-only|random-routing|p2d-direct-only}"
BASE_RUN_DIR="${PD_DIR}/runs-host/current/ablations/browsecomp-qwen3-8b-4p4d-c512"

# Start every invocation from production behavior so inherited shell variables
# cannot accidentally compose multiple ablations.
export SGLANG_AGENTIC_KV_HOST_STAGING=true
export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=false
export SGLANG_PD_ABLATION_RANDOM_ROUTING=false
export SGLANG_PD_ABLATION_RANDOM_SEED=2026
export P2D_HOST_STAGING=true

case "${ABLATION}" in
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
  EXPERIMENT_CONFIG="${PD_DIR}/configs/profiles/browsecomp_qwen3_8b_tp1_4p4d.yaml" \
  timeout --signal=TERM --kill-after=240s 3600s \
  bash "${SCRIPT_DIR}/run_qwen3_8b_tp1_browsecomp_4p4d.sh"
