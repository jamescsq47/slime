#!/usr/bin/env bash
set -euo pipefail

# Physical topology: 4 Prefill GPUs + 4 Decode GPUs, represented as two P and
# two D logical TP=2 engines.  Matching ranks keep both Host slow paths local:
#   P [0,4] [2,6]
#   D [1,5] [3,7]
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3-32B}"
export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4;2,6}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-1,5;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-27300 27400}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-28300 28400}"
export DECODE_PORTS="${DECODE_PORTS:-27301 27401}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.85 0.74}"
# GPU7 hosts the retrieval model before the [3,7] Decode group starts.  The
# group already uses GPU7's lower, uniform KV-pool fraction, so the remaining
# free-memory difference is expected rather than evidence of a stray process.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-false}"
export RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/../../runs-host/new-method/qwen3-32b-tp2-4p4d}"

exec bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh"
