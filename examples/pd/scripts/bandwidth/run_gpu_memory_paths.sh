#!/usr/bin/env bash
set -euo pipefail
PD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec /homes/siqic/anaconda3/envs/pd/bin/python "${PD_DIR}/scripts/bandwidth/benchmark_gpu_memory_paths.py" "$@"
