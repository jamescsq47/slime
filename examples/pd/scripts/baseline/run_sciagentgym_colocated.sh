#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM=/homes/siqic/data/SciAgentGYM
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == e9dbbea4369d67694e38bf8be67bedbcaf9e9300 ]]
[[ -z "$(git -C "$UPSTREAM" status --porcelain)" ]]
unset MODEL_GPU_GROUPS MODEL_MEM_FRACTION_STATICS
export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd_baseline/bin}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/sciagentgym_offline.yaml}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd):${PYTHONPATH:-}"
export START_SEARCH_SERVER=false
export MODEL_PATH=/dataset/model/qwen3/Qwen3-8B
export MODEL_GPUS="${MODEL_GPUS:-0}" MODEL_PORTS=27700 ROUTER_PORT=27710
export MEM_FRACTION_STATIC=0.80 MODEL_TP_SIZE=1
export MAX_INFLIGHT=4 REQUESTS="${REQUESTS:-10}" CLOSED_LOOP=false
export TEMPERATURE=0 TOP_P=1 TOP_K=-1
export DISPATCH_POLICY=random PRESERVE_SOURCE_ORDER=true POST_ANALYZER=none
export MODEL_MAX_RESPONSE_LENGTH=32768 MODEL_CONTEXT_LENGTH=40960
export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline/sciagentgym-qwen3-8b-c4-n10-r1}"
exec bash "${SCRIPT_DIR}/run_colocated_case.sh"
