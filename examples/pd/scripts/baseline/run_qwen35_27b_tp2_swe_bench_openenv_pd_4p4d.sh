#!/usr/bin/env bash
set -euo pipefail

# Two TP=2 prefill replicas and two TP=2 decode replicas on eight A100s.
# The SWE-bench harness remains in examples/pd/data/swe_bench_openenv; this
# launcher does not install or modify any code in the pd_baseline environment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.5-27B}"
export CASE_MODE="${CASE_MODE:-no_reverse}"
export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4;1,5}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-2,6;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-36200 36201}"
export PREFILL_BOOTSTRAP_PORTS="${PREFILL_BOOTSTRAP_PORTS:-37200 37201}"
export DECODE_PORTS="${DECODE_PORTS:-36202 36203}"
export PREFILL_MEM_FRACTION_STATICS="${PREFILL_MEM_FRACTION_STATICS:-0.80 0.80}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.80 0.80}"
export MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-131072}"
export MODEL_MAX_RESPONSE_LENGTH="${MODEL_MAX_RESPONSE_LENGTH:-81920}"
export MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-glm45}"
export MODEL_TOOL_CALL_PARSER="${MODEL_TOOL_CALL_PARSER:-qwen3_coder}"
export PAGE_SIZE="${PAGE_SIZE:-1}"
export ROUTER_PORT="${ROUTER_PORT:-36210}"
export ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-36220}"
export START_SEARCH_SERVER=false
export PD_DATA_ROOT="${PD_DATA_ROOT:-/tmp/pd-data}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml}"
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
export SLIME_HTTP_READ_TIMEOUT_SECONDS="${SLIME_HTTP_READ_TIMEOUT_SECONDS:-86400}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="qwen35-27b-tp2-swe-bench-openenv-no-reverse-4p4d-c128-${RUN_TAG}"
export RUN_DIR="${RUN_DIR:-/tmp/pd-runs/${RUN_NAME}}"

status=0
bash "${SCRIPT_DIR}/run_pd_case.sh" || status=$?

PERSIST_RUN_DIR="${PERSIST_RUN_DIR:-/tmp/pd-persist/${RUN_NAME}}"
mkdir -p "${PERSIST_RUN_DIR}"
rsync -a "${RUN_DIR}/" "${PERSIST_RUN_DIR}/"
exit "${status}"
