#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_ENV="${TERMINAL_BENCH_SERVICE_ENV:-${PD_DIR}/configs/services/terminal_bench.env}"

set -a
# shellcheck disable=SC1090
source "${SERVICE_ENV}"
set +a

# Keep the environment-service admission limit aligned with the formal c256
# serving workload.  Callers may still override it explicitly for another
# concurrency sweep.
export MAX_CONCURRENT_ENVS="${MAX_CONCURRENT_ENVS:-256}"

OPENENV_ROOT="${OPENENV_ROOT:-/homes/siqic/openenv}"
TB2_PYTHON="${TB2_PYTHON:-${OPENENV_ROOT}/envs/tbench2_env/.venv/bin/python}"
export PYTHONPATH="${OPENENV_ROOT}/src:${OPENENV_ROOT}/envs${PYTHONPATH:+:${PYTHONPATH}}"

if "${TB2_PYTHON}" -c 'import tbench2_env.server.app' >/dev/null 2>&1; then
  exec "${TB2_PYTHON}" -m tbench2_env.server.app --port "${TB2_PORT:-8003}"
fi

exec "${TB2_PYTHON}" "${SCRIPT_DIR}/run_terminal_bench_env_from_pyc.py" \
  --root "${OPENENV_ROOT}/envs/tbench2_env" --port "${TB2_PORT:-8003}"
