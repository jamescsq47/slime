#!/usr/bin/env bash
# a10=P (also Docker/verifier), a11=D; each TP8/EP1, memory .8.
# preflight has no model launch. run requires the independent audit gate first.
set -euo pipefail
dualpd_tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dualpd_py="${DUALPD_PYTHON:-/homes/siqic/anaconda3/envs/pd_multi_node/bin/python}"
exec "$dualpd_py" "$dualpd_tools/qwen35_multinode.py" "$@"
