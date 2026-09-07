#!/usr/bin/env bash
set -euo pipefail

# Full SWE-bench Verified pass with Qwen3's recommended thinking-mode sampler.
# SGLang's default min_p is 0; MIN_P is exported as an explicit run manifest
# value even though a zero threshold is behaviorally equivalent to omission.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_full_c128_qwen_recommended.yaml}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export MIN_P="${MIN_P:-0}"
export RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-qwen-sampling}"

exec bash "${SCRIPT_DIR}/run_qwen3_32b_tp2_swe_bench_colocated.sh"
