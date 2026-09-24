#!/usr/bin/env bash
set -euo pipefail
# No SSH fan-out or GPU work occurs implicitly. Start the TCP control relay on
# group_control.node_id first, then start one worker on each configured node:
#   multinode.sh start-control --config CONFIG
#   multinode.sh control-ready --config CONFIG
#   multinode.sh start-worker --config CONFIG --node-id NODE
# Shared control_root/fs-publish actions are intentionally unsupported in V2.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${DUALPD_PYTHON:-python3}" "${SCRIPT_DIR}/multinode.py" "$@"
