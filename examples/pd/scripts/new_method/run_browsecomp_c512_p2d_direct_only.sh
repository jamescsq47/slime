#!/usr/bin/env bash
set -euo pipefail

# c512 ablation: disable only P->D Host, retain late binding and D->P paths.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-/homes/siqic/sglang-codex-c512/python}"
unset SGLANG_AGENTIC_KV_NUMA_HOST_POOL_DIR SGLANG_AGENTIC_KV_NUMA_HOST_POOL
export MAX_INFLIGHT=512
export FAST_TOOL_THRESHOLD_SECONDS=1 DIRECT_WAIT_SECONDS=1 SEARCH_PORT=8750
export EXPERIMENT_CONFIG="${PD_DIR}/configs/profiles/browsecomp_qwen3_8b_tp1_4p4d.yaml"
export SGLANG_AGENTIC_KV_HOST_STAGING=true
export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=false
export SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=false
export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=true
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=true
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH=32
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW=8
export SGLANG_PD_ABLATION_RANDOM_ROUTING=false
export SGLANG_PD_ABLATION_P2D_PREBIND=false
export P2D_HOST_STAGING=false
export SGLANG_AGENTIC_KV_P2D_HOST_STAGING=false
export RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-host-disabled-20260910-r1}"
exec bash "${SCRIPT_DIR}/run_h100_integration_a100.sh"
