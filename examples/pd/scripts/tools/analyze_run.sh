#!/usr/bin/env bash
set -euo pipefail
PD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 1 ]] || { echo "usage: $0 RUN_DIR" >&2; exit 2; }
exec /homes/siqic/anaconda3/envs/pd/bin/python "${PD_DIR}/scripts/tools/analyze_pd_offload.py" --run-dir "$1"
