#!/usr/bin/env bash
set -euo pipefail
# No SSH fan-out or GPU work occurs implicitly. Run on the intended node.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${DUALPD_PYTHON:-python3}" "${SCRIPT_DIR}/multinode.py" "$@"
