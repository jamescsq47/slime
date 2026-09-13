#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
BASELINE=/homes/siqic/anaconda3/envs/pd_baseline/bin/python
export PATH="$(dirname "$BASELINE"):$PATH"
PROBE=/homes/siqic/.venvs/dabstep_smoke/bin/python
CONFIG_PATH="${CONFIG_PATH:-${PD_DIR}/configs/experiments/scienceagentbench_probe.json}"
mapfile -t settings < <("$BASELINE" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["model"]); print(c["context_length"]); print(c["mem_fraction_static"]); print(int(c.get("language_only",False))); print(c.get("page_size",64))' "$CONFIG_PATH")
[[ ${#settings[@]} == 5 ]] || { echo "Invalid probe config" >&2; exit 1; }
extra_model_args=()
if [[ "${settings[3]}" == 1 ]]; then extra_model_args+=(--language-only); fi
# Reuse the existing isolated executor and its per-run cleanup label.
export DABSTEP_RUN_ID="$("$BASELINE" -c 'import uuid; print(uuid.uuid4().hex)')"
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/baseline/scienceagentbench-qwen3-8b-c2-n6-r1}"
[[ ! -e "$RUN_DIR" ]] || { echo "Use a fresh RUN_DIR" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
cleanup() {
  pd_cleanup_all
  local ids
  ids="$(timeout 30 docker ps -aq --filter "label=dabstep.probe_run=${DABSTEP_RUN_ID}")" || {
    echo "Cannot verify experiment container cleanup" >&2; return 1;
  }
  if [[ -n "$ids" ]]; then
    while IFS= read -r cid; do
      timeout 30 docker rm -f "$cid" || echo "Could not remove experiment container $cid" >&2
    done <<< "$ids"
  fi
}
trap cleanup EXIT
trap pd_signal_exit INT TERM
pd_check_gpu_idle 0
pd_check_port_free 27700
unset SGLANG_AGENTIC_KV_LIFECYCLE SGLANG_AGENTIC_KV_HOST_STAGING SGLANG_AGENTIC_KV_LEDGER_PATH
"$BASELINE" "$SCRIPT_DIR/../tools/check_environments.py" --expect baseline --output "$RUN_DIR/environment.json"
setsid env CUDA_VISIBLE_DEVICES=0 SGLANG_ENABLE_METRICS_DEVICE_TIMER=true "$BASELINE" -m sglang.launch_server \
  --model-path "${settings[0]}" --host 127.0.0.1 --port 27700 \
  --context-length "${settings[1]}" --page-size "${settings[4]}" --mem-fraction-static "${settings[2]}" --enable-metrics \
  "${extra_model_args[@]}" \
  >"$RUN_DIR/logs/model.log" 2>&1 &
model_pid=$!
pd_track_group "$model_pid"
pd_wait_http model http://127.0.0.1:27700/health "$model_pid" 900
setsid "$PROBE" "$PD_DIR/data/scienceagentbench/probe.py" \
  --config "$CONFIG_PATH" --output "$RUN_DIR" \
  >"$RUN_DIR/probe.log" 2>&1 &
probe_pid=$!
pd_track_group "$probe_pid"
wait "$probe_pid"
echo "ScienceAgentBench probe complete: $RUN_DIR"
