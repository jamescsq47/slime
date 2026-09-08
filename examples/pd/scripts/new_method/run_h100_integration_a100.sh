#!/usr/bin/env bash
set -euo pipefail

# H100 transport on A100, keeping the r7 workload, GPU KV fractions and Host
# capacities for comparison. The H100 wrapper supplies its DMA/control knobs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PD_ENV_BIN="${PD_ENV_BIN:-/homes/siqic/anaconda3/envs/pd/bin}"
export MODEL_PATH="${MODEL_PATH:-/dataset/model/qwen3/Qwen3-8B}"
export QA_DATA="${QA_DATA:-/homes/siqic/data/browsecomp/bc_train.jsonl}"
export HF_HOME="${HF_HOME:-/homes/siqic/.cache/huggingface}"
export SEARCH_SERVER_EMBEDDING_CACHE="${SEARCH_SERVER_EMBEDDING_CACHE:-${HF_HOME}/hub/datasets--miaolu3--browsecomp-plus/snapshots/9f600f47c5ee9a6251ec5521eb279d8dc5df2966/corpus_embeddings.pkl}"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-/homes/siqic/sglang-h100-integration/python}"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_DIR}/../..:${PYTHONPATH:-}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.60"
export D2P_HOST_ARENA_GIB_PER_P=128
export P2D_HOST_ARENA_GIB_PER_P=32
export SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB=640
export WARMUP_SECONDS=300
export MAX_WARMUP_SECONDS=420
export MEASURE_SECONDS=1200
export MAX_INFLIGHT="${MAX_INFLIGHT:-512}"
export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/current/h100-a100-integration/browsecomp-qwen3-8b-tp1-4p4d-c${MAX_INFLIGHT}-w300-m1200-r1}"
exec bash "${SCRIPT_DIR}/run_qwen3_8b_tp1_browsecomp_4p4d.sh"
