#!/usr/bin/env bash
set -euo pipefail

# Physical topology: 2 Prefill GPUs + 6 Decode GPUs.  Each logical engine is
# TP=2 and spans NUMA0/NUMA1 with matching ranks:
#   P  [0,4]
#   D  [1,5] [2,6] [3,7]
# GPU7 also hosts search, so the complete [3,7] TP group uses its lower safe
# mem-fraction; TP peers must never expose different KV-pool capacities.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3-32B}"
export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-1,5;2,6;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-27300}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-28300}"
export DECODE_PORTS="${DECODE_PORTS:-27301 27302 27303}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.85 0.85 0.74}"
export RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/../../runs-host/new-method/qwen3-32b-tp2-2p6d}"

exec bash "${SCRIPT_DIR}/run_2p6d_numa_case.sh"
