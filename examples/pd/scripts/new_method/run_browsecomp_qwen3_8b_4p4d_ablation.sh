#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ABLATION="${1:?usage: $0 full-1s|d2p-slow-only-1s|fast-direct-fail-recompute-1s|direct-only-recompute-1s|d2p-direct-only|d2p-slow-only|random-routing|p2d-direct-only}"
BASE_RUN_DIR="${PD_DIR}/runs-host/current/ablations/browsecomp-qwen3-8b-4p4d-c512"
export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd/bin}"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-/homes/siqic/sglang-h100-integration/python}"
export MODEL_PATH=/dataset/model/qwen3/Qwen3-8B
export PD_DATA_ROOT=/homes/siqic/data
export QA_DATA=/homes/siqic/data/browsecomp/bc_train.jsonl
export HF_HOME=/homes/siqic/.cache/huggingface
export SEARCH_SERVER_EMBEDDING_CACHE=/homes/siqic/.cache/huggingface/hub/datasets--miaolu3--browsecomp-plus/snapshots/9f600f47c5ee9a6251ec5521eb279d8dc5df2966/corpus_embeddings.pkl

# Formal-ablation invariants.  The base launcher also serves historical c384
# runs and therefore has older defaults; pin the aligned comparison here so
# its exported defaults cannot override the experiment profile.
export MAX_INFLIGHT=512
export WARMUP_SECONDS=300
export MAX_WARMUP_SECONDS=420
export MEASURE_SECONDS=1200
export MEM_FRACTION_STATIC=0.80
export DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.60"
export D2P_HOST_ARENA_GIB_PER_P=128
export P2D_HOST_ARENA_GIB_PER_P=32

# Start every invocation from production behavior so inherited shell variables
# cannot accidentally compose multiple ablations.
export SGLANG_AGENTIC_KV_HOST_STAGING=true
export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=false
export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=false
export SGLANG_PD_ABLATION_RANDOM_ROUTING=false
export SGLANG_PD_ABLATION_RANDOM_SEED=2026
export P2D_HOST_STAGING=true

case "${ABLATION}" in
  full-1s)
    export FAST_TOOL_THRESHOLD_SECONDS=1
    export DIRECT_WAIT_SECONDS=1
    ;;
  d2p-slow-only-1s)
    export FAST_TOOL_THRESHOLD_SECONDS=1
    export DIRECT_WAIT_SECONDS=1
    export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=true
    ;;
  fast-direct-fail-recompute-1s)
    export FAST_TOOL_THRESHOLD_SECONDS=1
    export DIRECT_WAIT_SECONDS=1
    export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=true
    ;;
  direct-only-recompute-1s)
    # Effectively unbounded for this benchmark: every valid tool result gets a
    # Direct attempt, and a one-second setup miss safely recomputes. No D->P
    # Shared-Host transition is available in this variant.
    export FAST_TOOL_THRESHOLD_SECONDS=31536000
    export DIRECT_WAIT_SECONDS=1
    export SGLANG_AGENTIC_KV_HOST_STAGING=false
    export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=true
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
  EXPERIMENT_CONFIG="${PD_DIR}/configs/profiles/browsecomp_qwen3_8b_tp1_4p4d.yaml" \
  timeout --signal=TERM --kill-after=240s 3600s \
  bash "${SCRIPT_DIR}/run_qwen3_8b_tp1_browsecomp_4p4d.sh"
