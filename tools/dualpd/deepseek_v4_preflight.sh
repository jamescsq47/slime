#!/usr/bin/env bash
# No dependency installation, model launch, or changes to existing PD settings.
set -euo pipefail
dualpd_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dualpd_python="${DUALPD_PYTHON:-python3}"
if [[ "${1:-}" == "protocol-tests" ]]; then
  exec bash "${dualpd_tools_dir}/../../../sglang/validation/check_multinode_cpu.sh"
fi
exec "${dualpd_python}" "${dualpd_tools_dir}/deepseek_v4_preflight.py" "$@"
