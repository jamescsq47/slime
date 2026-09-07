#!/usr/bin/env bash
set -euo pipefail

# Qwen3-32B Terminal-Bench comparison case.
#
# Keep every serving/workload parameter aligned with the colocated TP=2
# baseline.  The intentional differences are only the physical PD topology
# and the agentic bidirectional KV transport:
#   P: [0,4]
#   D: [1,5] [2,6] [3,7]
# Terminal-Bench uses OpenEnv rather than the BrowseComp search service, so
# GPU7 is fully available to the last Decode group.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3-32B}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${SCRIPT_DIR}/../../configs/experiments/terminal_bench.yaml}"

export PREFILL_GPU_GROUPS="${PREFILL_GPU_GROUPS:-0,4}"
export DECODE_GPU_GROUPS="${DECODE_GPU_GROUPS:-1,5;2,6;3,7}"
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_PORTS="${PREFILL_PORTS:-27300}"
export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-28300}"
export DECODE_PORTS="${DECODE_PORTS:-27301 27302 27303}"

export PD_SKIP_SEARCH=1
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export DECODE_MEM_FRACTION_STATICS="${DECODE_MEM_FRACTION_STATICS:-0.80 0.80 0.80}"
export PD_PAGE_SIZE="${PD_PAGE_SIZE:-64}"
export MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-40960}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-36864}"
export PREFILL_CHUNKED_PREFILL_SIZE="${PREFILL_CHUNKED_PREFILL_SIZE:-8192}"
export PREFILL_MAX_PREFILL_TOKENS="${PREFILL_MAX_PREFILL_TOKENS:-16384}"

export MAX_INFLIGHT="${MAX_INFLIGHT:-256}"
export REQUESTS="${REQUESTS:-8192}"
export SEED="${SEED:-2026}"
export TEMPERATURE="${TEMPERATURE:-0}"
export TOP_P="${TOP_P:-1}"
export TOP_K="${TOP_K:--1}"
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export MAX_WARMUP_SECONDS="${MAX_WARMUP_SECONDS:-420}"

# Preserve the validated agentic transport settings.  A two-second threshold
# covers most Terminal-Bench calls while keeping genuinely slow tools out of
# Decode HBM. Direct and Slow both reserve their exact complete workset from
# ordinary P HBM; there is no permanent Direct-only receive pool.
export FAST_TOOL_THRESHOLD_SECONDS="${FAST_TOOL_THRESHOLD_SECONDS:-2}"
export DIRECT_WAIT_SECONDS="${DIRECT_WAIT_SECONDS:-2}"
export EARLY_CLAIM_POST_TIMEOUT_SECONDS="${EARLY_CLAIM_POST_TIMEOUT_SECONDS:-2}"
export DIRECT_IO_CAP="${DIRECT_IO_CAP:-8}"
export SELECTED_IO_CAP="${SELECTED_IO_CAP:-4}"
export P_H2D_MAX_INFLIGHT="${P_H2D_MAX_INFLIGHT:-4}"
export SGLANG_AGENTIC_KV_TP_HOST_PIPELINE_DEPTH="${SGLANG_AGENTIC_KV_TP_HOST_PIPELINE_DEPTH:-1}"
export MAX_PREFILL_INFLIGHT="${MAX_PREFILL_INFLIGHT:-12}"
export D_TARGET_KV_FRACTION="${D_TARGET_KV_FRACTION:-1.0}"
export P_ACCEPT_TIMEOUT_SECONDS="${P_ACCEPT_TIMEOUT_SECONDS:-600}"
export P2D_HOST_STAGING="${P2D_HOST_STAGING:-true}"
export P2D_HOST_ARENA_GIB_PER_P="${P2D_HOST_ARENA_GIB_PER_P:-64}"

# The new method intentionally has no native HiCache storage prefetch.  Its
# request-generation snapshots use Direct and the two Shared Host Arenas.

# There is one logical Prefill worker: GPUs 0 and 4 are its TP ranks, not two
# independently selectable P workers.  Keep all Decode groups globally
# eligible, exactly as in the validated Qwen3-32B 2P:6D TP=2 launcher.
export PD_LATE_BIND_NUMA_DOMAINS=0
export SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=0
export SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1

export RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/../../runs-host/new-method/qwen3-32b-tp2-terminal-bench-2p6d-c${MAX_INFLIGHT}}"

exec bash "${SCRIPT_DIR}/run_2p6d_numa_case.sh"
