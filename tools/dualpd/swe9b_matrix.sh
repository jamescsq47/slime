#!/usr/bin/env bash
set -euo pipefail
# SWE Verified500: openai_tools; temperature=.6 top_p=.95 top_k=20.
# Each of four cases runs all 500 tasks once, including inline verifier.
# systemd owns this queue, not the initiating terminal or Codex exec session.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/homes/siqic/anaconda3/envs/pd_multi_node/bin/python
ENTRY="${ROOT}/examples/pd/scripts/tools/swe9b_structured_matrix.py"
ACTION="${1:-status}"
QUEUE="${2:-${ROOT}/runs/dualpd/swe9b-openai-matrix-20260918-r3}"
UNIT=dualpd-swe9b-openai-20260918-r3
case "${ACTION}" in
  plan)
    exec "${PY}" "${ENTRY}" plan --root "${QUEUE}" --allow-gpu-process 7:1868643
    ;;
  start)
    for executable in rg docker rsync nvidia-smi setsid curl ss timeout; do
      command -v "${executable}" >/dev/null || { echo "Missing dependency: ${executable}" >&2; exit 2; }
    done
    [[ ! -e "${QUEUE}/sequence_status.json" ]] || { echo 'Queue already started; inspect status, do not rerun it.' >&2; exit 2; }
    "${PY}" "${ENTRY}" plan --root "${QUEUE}" --allow-gpu-process 7:1868643 >/dev/null
    mkdir -p "${QUEUE}"
    systemd-run --user --unit="${UNIT}" --description='DualPD SWE9B openai_tools four-case evaluation' \
      --property=Type=exec --property=KillMode=mixed --property=TimeoutStopSec=900 \
      --property=Restart=no --property=RemainAfterExit=yes \
      --property="WorkingDirectory=${ROOT}" \
      --property="StandardOutput=append:${QUEUE}/controller.log" \
      --property="StandardError=append:${QUEUE}/controller.log" \
      --setenv=PYTHONDONTWRITEBYTECODE=1 \
      --setenv="PATH=${PATH}" \
      "${PY}" "${ENTRY}" run --root "${QUEUE}" --allow-gpu-process 7:1868643
    ;;
  status)
    systemctl --user status "${UNIT}" --no-pager || true
    if [[ -f "${QUEUE}/sequence_status.json" ]]; then
      "${PY}" -m json.tool "${QUEUE}/sequence_status.json"
    fi
    ;;
  stop)
    exec systemctl --user stop "${UNIT}"
    ;;
  *) echo 'Usage: bash tools/dualpd/swe9b_matrix.sh plan|start|status|stop [queue_dir]' >&2; exit 2 ;;
esac
