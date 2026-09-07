#!/usr/bin/env bash
set -euo pipefail

# Four collocated TP=2 replicas; SWE-bench shell tools use CPU-only Docker
# containers, so all eight GPUs remain available to Qwen3-32B.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MINISWE_ROOT="$(bash "${SCRIPT_DIR}/../tools/prepare_miniswe_agent.sh")"
export PYTHONPATH="${MINISWE_ROOT}/src:${PYTHONPATH:-}"
export MSWEA_SILENT_STARTUP=1
export MSWEA_GLOBAL_CONFIG_DIR="${MSWEA_GLOBAL_CONFIG_DIR:-/tmp/pd-miniswe-config}"

export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3-32B}"
export MODEL_GPU_GROUPS="${MODEL_GPU_GROUPS:-0,4;1,5;2,6;3,7}"
export MODEL_TP_SIZE=2
export MODEL_PORTS="${MODEL_PORTS:-33600 33601 33602 33603}"
export MODEL_MEM_FRACTION_STATICS="${MODEL_MEM_FRACTION_STATICS:-0.80 0.80 0.80 0.80}"
export ROUTER_PORT="${ROUTER_PORT:-33610}"
export START_SEARCH_SERVER=false
export PD_DATA_ROOT="${PD_DATA_ROOT:-/tmp/pd-data}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_full_c128.yaml}"
export DISPATCH_POLICY=random
export MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
export REQUESTS="${REQUESTS:-500}"
export PRESERVE_SOURCE_ORDER=true
export CLOSED_LOOP=false
export POST_ANALYZER=swe_bench
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="qwen3-32b-tp2-swe-bench-verified-colocated-c128-miniswe-${RUN_TAG}"
export RUN_DIR="${RUN_DIR:-/tmp/pd-runs/${RUN_NAME}}"
export PD_INFERENCE_RETURN_LOGPROB=false

status=0
bash "${SCRIPT_DIR}/run_colocated_case.sh" || status=$?

# The serving run stays on local disk; persist the complete trajectories and
# profiles afterwards so NFS latency cannot perturb the measurement.
PERSIST_RUN_DIR="${PERSIST_RUN_DIR:-${PD_DIR}/runs-host/baseline/formal-${RUN_NAME}}"
mkdir -p "${PERSIST_RUN_DIR}"
rsync -a "${RUN_DIR}/" "${PERSIST_RUN_DIR}/"
exit "${status}"
