#!/usr/bin/env bash
set -euo pipefail

# Four collocated TP=2 replicas on eight A100s. SWE-bench environments use
# CPU-only Docker containers, leaving every GPU available to Qwen3.5-27B.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.5-27B}"
export MODEL_GPU_GROUPS="${MODEL_GPU_GROUPS:-0,4;1,5;2,6;3,7}"
export MODEL_TP_SIZE=2
export MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-131072}"
export MODEL_MAX_RESPONSE_LENGTH="${MODEL_MAX_RESPONSE_LENGTH:-81920}"
export MODEL_PAGE_SIZE=1
# The baseline SGLang 0.5.10 environment carries the isolated upstream
# Qwen3Detector tool-boundary backport; all other SGLang code remains pinned.
export MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-qwen3}"
export MODEL_TOOL_CALL_PARSER="${MODEL_TOOL_CALL_PARSER:-qwen3_coder}"
export MODEL_PORTS="${MODEL_PORTS:-33600 33601 33602 33603}"
export MODEL_MEM_FRACTION_STATICS="${MODEL_MEM_FRACTION_STATICS:-0.80 0.80 0.80 0.80}"
export ROUTER_PORT="${ROUTER_PORT:-33610}"
export START_SEARCH_SERVER=false
export PD_DATA_ROOT="${PD_DATA_ROOT:-/tmp/pd-data}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_openenv_qwen35_27b_c128.yaml}"
export DISPATCH_POLICY=random
export MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
export REQUESTS="${REQUESTS:-500}"
export PRESERVE_SOURCE_ORDER=true
export CLOSED_LOOP=false
export POST_ANALYZER=swe_bench
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0}"
export PD_INFERENCE_RETURN_LOGPROB=false
# SWE-bench agents can legitimately spend over an hour in a long trajectory.
# Keep the HTTP transport timeout above all normal episode durations; the
# harness/verifier retain their own explicit safety limits.
export SLIME_HTTP_READ_TIMEOUT_SECONDS="${SLIME_HTTP_READ_TIMEOUT_SECONDS:-86400}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="qwen35-27b-tp2-swe-bench-verified-openenv-colocated-c128-${RUN_TAG}"
export RUN_DIR="${RUN_DIR:-/tmp/pd-runs/${RUN_NAME}}"

status=0
bash "${SCRIPT_DIR}/run_colocated_case.sh" || status=$?

# Keep serving writes on local disk and persist only the finished artifact.
PERSIST_RUN_DIR="${PERSIST_RUN_DIR:-${PD_DIR}/runs-host/baseline/formal-${RUN_NAME}}"
mkdir -p "${PERSIST_RUN_DIR}"
rsync -a "${RUN_DIR}/" "${PERSIST_RUN_DIR}/"
exit "${status}"
