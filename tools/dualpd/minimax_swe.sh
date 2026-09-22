#!/usr/bin/env bash
# Remote H100 entry point. No auto-install, SSH, GPU reset or global kill.
# SWE-bench fixed sampling: temperature=0.6, top_p=0.95, top_k=20.
# minimax_swe.json is validated before launch; invalid settings fail closed.
set -euo pipefail
dualpd_tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${DUALPD_PYTHON:-python3}" "${dualpd_tools}/minimax_swe.py" "$@"
