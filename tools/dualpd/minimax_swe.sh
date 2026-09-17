#!/usr/bin/env bash
# Remote H100 entry point. No auto-install, SSH, GPU reset or global kill.
set -euo pipefail
dualpd_tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${DUALPD_PYTHON:-python3}" "${dualpd_tools}/minimax_swe.py" "$@"
