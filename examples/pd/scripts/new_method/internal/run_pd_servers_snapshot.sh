#!/usr/bin/env bash
set -euo pipefail

# Bash normally parses a script incrementally.  Another experiment may update
# the shared launcher after this process starts, which can expose a transient
# half-written compound command to the running shell.  Read one complete file
# version first and execute that immutable in-memory snapshot.  The target is
# passed as $0 so the launcher's SCRIPT_DIR resolution remains unchanged.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/run_pd_servers.sh"
SCRIPT_CONTENT="$(<"${TARGET_SCRIPT}")"
exec bash -c "${SCRIPT_CONTENT}" "${TARGET_SCRIPT}"
