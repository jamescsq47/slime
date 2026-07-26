#!/usr/bin/env bash
set -euo pipefail

source_pid=${1:?usage: run_browsecomp_followup.sh SOURCE_PID}
for variable in LOCAL_SEARCH_URL GRADER_API_KEY GRADER_BASE_URL GRADER_MODEL; do
  value=$(tr '\0' '\n' < "/proc/${source_pid}/environ" | sed -n "s/^${variable}=//p")
  if [ -z "${value}" ]; then
    echo "Missing ${variable} in source process ${source_pid}" >&2
    exit 1
  fi
  export "${variable}=${value}"
done

while tmux has-session -t '=browsecomp_eval' 2>/dev/null; do
  sleep 30
done

exec bash examples/mixed/run_browsecomp_downstream_suite.sh
