#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${SCRIPT_DIR}/../../configs/profiles/browsecomp_qwen3_8b_tp1_4p4d.yaml}"
exec bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh"
