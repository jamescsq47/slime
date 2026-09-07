#!/usr/bin/env bash
set -euo pipefail

# Formal pure-BrowseComp run: four TP=1 Prefill workers and four TP=1 Decode
# workers.  The closed loop keeps exactly 384 end-to-end agents in flight and
# cycles through the canonical 680-row source order after the first pass.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PD_DIR}/../../.." && pwd)"

export PD_ENV_BIN="${PD_ENV_BIN:-/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin}"
export MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen3-8B}"
export QA_DATA="${QA_DATA:-${WORKSPACE_ROOT}/datasets/browsecomp/bc_train.jsonl}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/datasets/huggingface-cache}"
export SEARCH_SERVER_EMBEDDING_CACHE="${SEARCH_SERVER_EMBEDDING_CACHE:-${HF_HOME}/hub/datasets--miaolu3--browsecomp-plus/snapshots/9f600f47c5ee9a6251ec5521eb279d8dc5df2966/corpus_embeddings.pkl}"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-${WORKSPACE_ROOT}/sglang-src/pd/python}"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_DIR}:${WORKSPACE_ROOT}/slime:${PYTHONPATH:-}"
# Songhan's GCC comes from Spack while glibc headers/startup objects use the
# Ubuntu multiarch paths.  SGLang's first Decode CUDA-graph capture JITs fused
# RoPE, so make both paths explicit for nvcc and the host linker.
export CPATH="/usr/include/x86_64-linux-gnu:${CPATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

export PREFILL_GPUS="${PREFILL_GPUS:-0 2 4 6}"
export DECODE_GPUS="${DECODE_GPUS:-1 3 5 7}"
export PREFILL_TP_SIZE=1
export DECODE_TP_SIZE=1
export SEARCH_GPU="${SEARCH_GPU:-7}"
# The retrieval service occupies about 16 GiB on GPU 7.  Full registered-Host
# coverage also consumes CUDA mapping/page-table memory, so reserve explicit
# HBM headroom on the co-located fourth Decode worker.
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.85 0.85 0.85 0.68}"
export SEARCH_START_AFTER_MODELS=true

export MATH_RATIO=0
export PRESERVE_SOURCE_ORDER=true
export SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_browsecomp_source_order_n680.json}"
export SEED=2026
export REQUESTS=680
export WARMUP_REQUESTS=0
export MAX_INFLIGHT="${MAX_INFLIGHT:-384}"
export CLOSED_LOOP=1
# These open-loop settings are recorded for reproducibility but do not pace a
# closed-loop run.
export ARRIVAL_RATE=100
export ARRIVAL_DISTRIBUTION=fixed

export TEMPERATURE=0
export TOP_P=1
export TOP_K=-1
export MAX_CONTEXT_LENGTH=40960
export MAX_RESPONSE_LENGTH=36864
export PREFILL_CHUNKED_PREFILL_SIZE=8192
export PREFILL_MAX_PREFILL_TOKENS=8192

export WARMUP_SECONDS="${WARMUP_SECONDS:-1200}"
export MAX_WARMUP_SECONDS="${MAX_WARMUP_SECONDS:-1320}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"

# One tool-return-relative two-second Direct setup window.  Slow recovery uses
# max-free Host for D2H, then max-free P KV for H2D, as enforced by the router.
export FAST_TOOL_THRESHOLD_SECONDS=2
export DIRECT_WAIT_SECONDS=2
export P_ACCEPT_TIMEOUT_SECONDS=600
export P_QUEUE_TIMEOUT_SECONDS=3600
export P_READY_TIMEOUT_SECONDS=600
export D_TARGET_KV_FRACTION=1.0
export P_READY_BACKPRESSURE_MODE=disabled
# Both Slow directions use ordinary CPU DRAM through memfd rather than the
# 202 GiB /dev/shm mount.  Each P owns independent 256 GiB D->P and 64 GiB
# P->D arenas; first touch follows that P process' NUMA binding.
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND="${SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND:-memfd}"
export SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND="${SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND:-memfd}"
export SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA="${SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA:-1}"
export SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB="${SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB:-8}"
export SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB="${SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB:-1280}"
export SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA="${SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA:-1}"
# Two copy-engine-only lanes absorb bursty snapshot arrivals without launching
# gather/scatter kernels.  The production page-layout isolation gate sustains
# >24 GiB/s per lane in both directions with no Forward p50/p95 regression.
export SGLANG_AGENTIC_KV_D2H_INFLIGHT="${SGLANG_AGENTIC_KV_D2H_INFLIGHT:-2}"
# run_4p4d_numa_case translates this public case knob to the SGLang variable.
# Set both names so neither a wrapper default nor direct invocation can widen
# the physical lane count behind our back.
export P_H2D_MAX_INFLIGHT="${P_H2D_MAX_INFLIGHT:-2}"
export SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT="${SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT:-${P_H2D_MAX_INFLIGHT}}"
export SGLANG_AGENTIC_KV_P2D_D2H_WORKERS="${SGLANG_AGENTIC_KV_P2D_D2H_WORKERS:-2}"
export SGLANG_AGENTIC_KV_P2D_H2D_WORKERS="${SGLANG_AGENTIC_KV_P2D_H2D_WORKERS:-2}"
# CUDA 13 batches all coalesced allocator pages into one copy-engine submit.
# A 4096-token batch amortizes descriptor/control cost while remaining only
# 576 MiB for Qwen3-8B; it does not launch a gather/scatter kernel on H100.
export SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS="${SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS:-4096}"
export SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS="${SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS:-4096}"
export SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS="${SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS:-4096}"
export SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS="${SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS:-4096}"
export D2P_HOST_ARENA_GIB_PER_P="${D2P_HOST_ARENA_GIB_PER_P:-256}"
export P2D_HOST_STAGING="${P2D_HOST_STAGING:-true}"
export P2D_HOST_ARENA_GIB_PER_P="${P2D_HOST_ARENA_GIB_PER_P:-64}"

export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/new-method/formal-qwen3-8b-tp1-browsecomp-4p4d-c384-w1200-m1200-20260906-async-prewarm-v8}"

exec bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh"
