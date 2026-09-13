#!/usr/bin/env bash
set -euo pipefail

# First 100 Verified tasks exactly once; retain the previous inference settings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REQUESTS=100
# Upper bound remains 128, but a finite 100-task cohort cannot have >100 agents.
export MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/qwen35-9b-tp1-swe-openenv-4p4d-first100once-$(date -u +%Y%m%dT%H%M%S)}"
exec bash "${SCRIPT_DIR}/run_qwen35_9b_tp1_swe_openenv_4p4d_full.sh"
