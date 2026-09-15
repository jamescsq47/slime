#!/usr/bin/env bash
set -euo pipefail

# Isolated fused engine; no pip install and no writes to either shared env.
# Finite SWE Verified500 evaluation, NOT a 300+1200 steady-state benchmark.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PD_DIR}"
source "${SCRIPT_DIR}/../common/runtime.sh"
export PD_ENV_BIN="${PD_FUSED_ENV_BIN:-/homes/siqic/anaconda3/envs/pd/bin}"
export SGLANG_OVERLAY_ROOT=/homes/siqic/sglang-qwen35-integration/python
export PD_RUN_QWEN_SCRIPT=/homes/siqic/sglang-qwen35-integration/validation/run_pd_servers.sh
export PATH="${PD_ENV_BIN}:${PATH}" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_DIR}:${PD_DIR}/../.."
# Avoid accidentally inheriting an ablation or the old Mamba router.
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_AGENTIC_)
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_PD_)
unset SCHEDULE_FILE EXPERIMENT_CONFIG LOCAL_GPUS LOCAL_PORTS LOCAL_ROUTER_PORT MEASUREMENT_DURATION_SECONDS
export PD_LATE_BIND_ROUTER_ENTRY="${PD_DIR}/launch_late_binding_router.py"
export MODEL_PATH=/homes/siqic/Qwen3.5-9B PD_DATA_ROOT=/tmp/pd-data
export WORKLOAD_CONFIG="${PD_DIR}/configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml"
export MODEL_REASONING_PARSER=qwen3 MODEL_TOOL_CALL_PARSER=qwen3_coder
export PREFILL_GPU_GROUPS='0;4' DECODE_GPU_GROUPS='1;2;3;5;6;7'
export PREFILL_GPUS='0 4' DECODE_GPUS='1 2 3 5 6 7'
export PREFILL_TP_SIZE=1 DECODE_TP_SIZE=1
# Below this host's ephemeral range32768..60999: otherwise unrelated outbound
# connections can acquire a checked-free server port while models are loading.
export PREFILL_PORT=23700 PREFILL_PORTS='23700 23720'
export DECODE_PORT=23701 DECODE_PORTS='23701 23702 23703 23721 23722 23723'
export BOOTSTRAP_PORT=23800 BOOTSTRAP_PORTS='23800 23820'
export ROUTER_PORT=23750 ROUTER_PROMETHEUS_PORT=23751 AGENTIC_DIRECT_BASE_PORT=23900
export PD_PREFILL_NCCL_PORT_BASE=23910 PD_DECODE_NCCL_PORT_BASE=23912
export PD_SKIP_SEARCH=1
export MEM_FRACTION_STATIC=0.80 DECODE_MEM_FRACTION_STATICS='0.80 0.80 0.80 0.80 0.80 0.80'
if [[ "${PD_FUSED_4P4D:-0}" == 1 ]]; then
  [[ "${PD_FUSED_2P2D:-0}" != 1 ]] || { echo 'Choose only one GPU topology' >&2; exit 2; }
  export PREFILL_GPU_GROUPS='0;1;4;5' DECODE_GPU_GROUPS='2;3;6;7'
  export PREFILL_GPUS='0 1 4 5' DECODE_GPUS='2 3 6 7'
  export PREFILL_PORTS='23700 23710 23720 23730'
  export DECODE_PORTS='23701 23711 23721 23731'
  export BOOTSTRAP_PORTS='23800 23810 23820 23830'
  export PD_DECODE_NCCL_PORT_BASE=23914
  export DECODE_MEM_FRACTION_STATICS='0.80 0.80 0.80 0.80'
fi
# Isolated four-card real-workload validation; default eight-card run unchanged.
if [[ "${PD_FUSED_2P2D:-0}" == 1 ]]; then
  export PREFILL_GPU_GROUPS='0;6' DECODE_GPU_GROUPS='1;7'
  export PREFILL_GPUS='0 6' DECODE_GPUS='1 7'
  export DECODE_PORTS='23701 23721'
  export DECODE_MEM_FRACTION_STATICS='0.80 0.80'
fi
export PD_PAGE_SIZE=64 MAMBA_TRACK_INTERVAL=64
export MAMBA_FULL_MEMORY_RATIO=0.5
export MAX_CONTEXT_LENGTH=131072 MAX_RESPONSE_LENGTH=81920
export PREFILL_CHUNKED_PREFILL_SIZE="${PD_DIAGNOSTIC_PREFILL_CHUNK_SIZE:-8192}" PREFILL_MAX_PREFILL_TOKENS=8192
if [[ "${PREFILL_CHUNKED_PREFILL_SIZE}" != 8192 && "${PD_SERVE_ONLY:-0}" != 1 ]]; then
  echo 'Diagnostic Prefill chunk override requires PD_SERVE_ONLY=1' >&2
  exit 2
fi
unset SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE
if [[ "${PREFILL_CHUNKED_PREFILL_SIZE}" != 8192 ]]; then
  # Native deterministic Triton defaults to4096; smaller diagnostic chunks
  # would otherwise be rejected forever by PrefillAdder's alignment check.
  export SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE="${PREFILL_CHUNKED_PREFILL_SIZE}"
fi
export REQUESTS="${REQUESTS:-500}" WARMUP_REQUESTS=0 MAX_INFLIGHT="${MAX_INFLIGHT:-256}"
export CLOSED_LOOP=0 ARRIVAL_RATES=100 ARRIVAL_DISTRIBUTION=fixed
export DISPATCH_POLICY=random PRESERVE_SOURCE_ORDER=true SEED=2026
export TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0 METRICS_INTERVAL=2
export PD_DETERMINISTIC_INFERENCE=1 PD_SERVER_RANDOM_SEED=2026
export PD_INFERENCE_RETURN_LOGPROB=false SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
export PD_LATE_BINDING=1 PD_LATE_BIND_NUMA_DOMAINS=1
export SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS=1 SGLANG_PD_LATE_BIND_GLOBAL_DECODE=1
export SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT=32 SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION=0.90
export SGLANG_PD_LATE_BIND_ACCEPT_TIMEOUT_S=600 PD_LATE_BIND_READY_TIMEOUT_S=600
export SGLANG_PD_LATE_BIND_QUEUE_TIMEOUT_S=3600
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB=128 SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB=32
if [[ "${PD_FUSED_4P4D:-0}" == 1 ]]; then
  # Keep total Host capacity identical to R19: 4*(64+16)=320 GiB.
  export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB=64 SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB=16
fi
export SGLANG_AGENTIC_KV_P2D_HOST_STAGING=true
export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND=memfd SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND=memfd
export SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB=8 SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB=320
# Register every reachable arena before ANY workload/measurement traffic.
# The existing server launcher waits for all 8 CUDA contexts and fails closed.
# Physical backing is 320 GiB for both eight-card topologies.
# Each P registers all D2P arenas plus its own P2D arena: 288 GiB (2P),
# 272 GiB (4P). Each D registers all 320 GiB; the cache covers every mapping.
export SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA=1
export SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER=1
export SGLANG_AGENTIC_KV_REGISTER_PREWARM_TIMEOUT_SECONDS=1800
export SGLANG_AGENTIC_KV_RELAY_ENABLED=false
# Fixed, previously validated harness: skip per-token hashes at both ends.
# Identity/length/layout/checkpoint/fence/ownership validation stays enabled.
export SGLANG_AGENTIC_KV_TOKEN_CONTENT_HASH=false
# Allow tools returning within 2s to try Direct. This does NOT extend the
# subsequent admission/handshake deadline or change ownership/fence handling.
export SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD="${PD_FUSED_FAST_TOOL_THRESHOLD:-2}" SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT=1
export SGLANG_AGENTIC_KV_EARLY_CLAIM_POST_TIMEOUT=1
# These PD_* overrides survive the SGLANG_* environment reset above. Disabling
# congestion recompute sends failed Direct offers to Host; HIGH/LOW are then
# inactive metadata (equal thresholds are not valid for enabled hysteresis).
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE="${PD_FUSED_CONGESTION_RECOMPUTE:-true}"
export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=false
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH="${PD_FUSED_CONGESTION_HIGH:-32}"
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW="${PD_FUSED_CONGESTION_LOW:-8}"
# The unchanged SWE harness already publishes authoritative tool/final ACKs.
# Shell commands mentioning TASK_COMPLETE must not discard the parent snapshot.
export SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION=true
export SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT="${PD_FUSED_P_H2D_MAX_INFLIGHT:-2}"
export SGLANG_AGENTIC_KV_P_H2D_DECOUPLED="${PD_FUSED_P_H2D_DECOUPLED:-false}"
export SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS="${PD_FUSED_P_HOST_EVENT_PROGRESS:-false}"
export SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE="${PD_FUSED_P_HOST_ASYNC_PREPARE:-false}"
export SGLANG_AGENTIC_KV_D2H_ACTIVE_SNAPSHOTS=4
[[ "${SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT}" =~ ^[1-9][0-9]*$ ]] || {
  echo 'PD_FUSED_P_H2D_MAX_INFLIGHT must be a positive integer' >&2
  exit 2
}
export SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS=4096 SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS=4096
export SGLANG_AGENTIC_KV_D_CONTROL_POLL_SECONDS=0.05
export SGLANG_PD_P_READY_BACKPRESSURE_MODE=disabled PD_MAX_TRANSFER_INFLIGHT=8
export SGLANG_PREFILL_TRANSFER_CONSUMERS=24
# Preserve the actual unchanged SWE chat history. Its template removes prior
# reasoning, so reuse the verified stable prompt checkpoint, not stale full-tail
# Mamba state. Account separately for the resulting suffix recomputation.
export SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT=true
export SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED=true
export SGLANG_AGENTIC_KV_DEBUG_DIGEST="${PD_FUSED_DEBUG_DIGEST:-0}"
# Allow model bootstrap plus the mandatory Host-registration barrier.
export SERVICE_STARTUP_TIMEOUT_SECONDS=3600
export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c${MAX_INFLIGHT}-$(date -u +%Y%m%dT%H%M%S)}"
export PD_SWE_RUN_ID="$(basename -- "${RUN_DIR}")" PD_SWE_PROGRESS_FILE="${RUN_DIR}/episode_progress.jsonl"
export PD_RAW_REQUEST_LOG_DIR="${RUN_DIR}/raw"
export INFERENCE_ENTRY="${SCRIPT_DIR}/internal/inference_checkpointed.py"
mkdir -p "${RUN_DIR}/logs"
[[ ! -e "${RUN_DIR}/environment.json" ]] || { echo 'Refusing reused run directory' >&2; exit 2; }
[[ -z "$(docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")" ]] || { echo 'Run label already owns containers' >&2; exit 2; }
for gpu in ${PREFILL_GPUS} ${DECODE_GPUS}; do pd_check_gpu_idle "${gpu}"; done
for port in ${PREFILL_PORTS} ${DECODE_PORTS} ${BOOTSTRAP_PORTS} 23750 23751 23900 23901 23902 23903 23904 23905 23910 23911 23912 23913 23914 23915 23916 23917; do pd_check_port_free "${port}"; done
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_environments.py" --expect modified --output "${RUN_DIR}/environment.json"
"${PD_ENV_BIN}/python" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
import sglang
root = Path(os.environ['RUN_DIR'])
assert str(Path(sglang.__file__).resolve()).startswith(os.environ['SGLANG_OVERLAY_ROOT'] + '/')
source = Path(os.environ['SGLANG_OVERLAY_ROOT']).parent
revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert subprocess.call(['git','-C',str(source),'merge-base','--is-ancestor',
    '9262db4fa3', revision]) == 0, f'Engine does not contain published R9 checkpoint: {revision}'
diff = subprocess.check_output(['git','-C',str(source),'diff','HEAD'],text=True)
(root/'engine.patch').write_text(diff)
dataset = Path(os.environ['PD_DATA_ROOT'])/'swe-bench-verified/test.jsonl'
rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
assert len(rows) == len({row['instance_id'] for row in rows}) == 500
assert 0 < int(os.environ['REQUESTS']) <= 500
images = set(subprocess.check_output(['docker','image','ls','--format','{{.Repository}}:{{.Tag}}'],text=True,timeout=90).splitlines())
missing = [row['instance_id'] for row in rows if (row.get('image_name') or 'swebench/sweb.eval.x86_64.'+row['instance_id'].lower().replace('__','_1776_')+':latest').removeprefix('docker.io/') not in images]
assert not missing, missing[:8]
baseline = Path('/tmp/pd-persist/baseline-qwen35-9b-tp1-swe-verified500-colocated-c256-20260910-r2')
harness_hashes = {}
for old in sorted((baseline/'source-snapshot/data/swe_bench_openenv').rglob('*.py')):
    relative = old.relative_to(baseline/'source-snapshot')
    current = Path(relative)
    assert old.read_bytes() == current.read_bytes(), f'Baseline harness changed: {relative}'
    harness_hashes[str(relative)] = hashlib.sha256(current.read_bytes()).hexdigest()
assert harness_hashes, 'Missing baseline harness snapshot'
assert (baseline/'workload.yaml').read_bytes() == Path(os.environ['WORKLOAD_CONFIG']).read_bytes()
record = dict(engine_revision=revision, source=str(source), host=os.uname().nodename,
    dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), tasks=int(os.environ['REQUESTS']),
    concurrency=int(os.environ['MAX_INFLIGHT']),
    prefill_gpus=list(map(int, os.environ['PREFILL_GPUS'].split())),
    decode_gpus=list(map(int, os.environ['DECODE_GPUS'].split())),
    tp=1, mem_fraction_static=0.8, mamba_full_memory_ratio=0.5,
    request_owned_mamba=True, prefill_chunk_size=int(os.environ['PREFILL_CHUNKED_PREFILL_SIZE']),
    app_owns_termination=True,
    slow_congestion_recompute=os.environ['SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE'].lower() in {'1','true'},
    fast_direct_failure_recompute=False,
    fast_tool_threshold_seconds=float(os.environ['SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD']),
    direct_handshake_timeout_seconds=float(os.environ['SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT']),
    slow_congestion_high=int(os.environ['SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH']),
    slow_congestion_low=int(os.environ['SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW']),
    p_h2d_max_inflight_per_worker=int(os.environ['SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT']),
    p_h2d_decoupled=os.environ['SGLANG_AGENTIC_KV_P_H2D_DECOUPLED'].lower() in {'1','true'},
    p_host_event_progress=os.environ['SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS'].lower() in {'1','true'},
    p_host_async_prepare=os.environ['SGLANG_AGENTIC_KV_P_HOST_ASYNC_PREPARE'].lower() in {'1','true'},
    host_register_startup_barrier=True, host_register_eager_arena=True,
    host_register_cache_gib=320,
    host_physical_arena_gib=len(os.environ['PREFILL_GPUS'].split()) * (
        int(os.environ['SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB']) +
        int(os.environ['SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB'])),
    host_register_expected_contexts=len(os.environ['PREFILL_GPUS'].split())+len(os.environ['DECODE_GPUS'].split()),
    host_register_expected_gib_per_p=len(os.environ['PREFILL_GPUS'].split()) *
        int(os.environ['SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB']) +
        int(os.environ['SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB']),
    host_register_expected_gib_per_d=len(os.environ['PREFILL_GPUS'].split()) * (
        int(os.environ['SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB']) +
        int(os.environ['SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB'])),
    host_register_completion_report='host_register_prewarm.json',
    token_content_hash=False,
    engine_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
    native_hicache=False, mooncake=False, harness_matches_baseline=True,
    harness_sha256=harness_hashes,
    reverse_kv='Attention+Mamba stable prompt checkpoint; unchanged SWE serialization',
    full_dataset_evaluation=int(os.environ['REQUESTS'])==500,
    caveat='Fusion engine version differs from pd_mamba_baseline; not a pure same-engine PD ablation')
(root/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
mkdir -p "${RUN_DIR}/source-snapshot"
while IFS= read -r source_file; do cp --parents "${source_file}" "${RUN_DIR}/source-snapshot/"; done < <(rg --files -g '*.py' data)
cp inference.py agentic_kv_request.py pd_metrics.py late_binding_router.py launch_late_binding_router.py "${RUN_DIR}/source-snapshot/"
cp "${BASH_SOURCE[0]}" "${RUN_DIR}/source-snapshot/launcher.sh"
cp "${WORKLOAD_CONFIG}" "${RUN_DIR}/workload.yaml"
cp "${PD_DATA_ROOT}/swe-bench-verified/test.jsonl" "${RUN_DIR}/dataset.jsonl"
export WORKLOAD_CONFIG="${RUN_DIR}/workload.yaml"
CONTROL_DIR="$(mktemp -d /dev/shm/pd-q35-fused-swe.XXXXXX)"
export PD_P_READY_DIR="${CONTROL_DIR}/ready"
mkdir -p "${PD_P_READY_DIR}"
ln -s "${PD_P_READY_DIR}" "${RUN_DIR}/ready"
export SGLANG_AGENTIC_KV_LEDGER_PATH="${CONTROL_DIR}/ledger.json"
export SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH="${CONTROL_DIR}/host.json"
export SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH="${CONTROL_DIR}/p2d-host.json"
export SGLANG_AGENTIC_KV_METADATA_DIR="${PD_P_READY_DIR}/snapshot-metadata"
export SGLANG_AGENTIC_KV_REGISTER_PREWARM_DIR="${CONTROL_DIR}/host-register-prewarm"
env | rg '^(PD_|SGLANG_|PREFILL_|DECODE_|MODEL_|MAX_|MAMBA_|REQUESTS=|TEMPERATURE=|TOP_|MIN_P=|SEED=|WORKLOAD_|CLOSED_LOOP=|ARRIVAL_|DISPATCH_|PRESERVE_|MEM_FRACTION_|PYTHONPATH=)' | LC_ALL=C sort >"${RUN_DIR}/launch-environment.txt"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv >"${RUN_DIR}/gpu-processes-before.csv"
case_pid=''
cleanup() {
  local status=$? owned=()
  trap - INT TERM
  if [[ -n "${case_pid}" ]]; then
    kill -TERM -- "-${case_pid}" 2>/dev/null || true
    wait "${case_pid}" 2>/dev/null || true
  fi
  mapfile -t owned < <(timeout 15 docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")
  if (( ${#owned[@]} )); then timeout 30 docker rm -f "${owned[@]}" || true; fi
  # Payload arenas are cleaned by their owning supervisor after all DMA users
  # exit. Retain the small control directory for ownership/fault accounting.
  cp -a "${CONTROL_DIR}" "${RUN_DIR}/control-final" || true
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
setsid bash "${SCRIPT_DIR}/internal/run_agentic_pipeline.sh" >"${RUN_DIR}/inference.log" 2>&1 &
case_pid=$!
printf '%s\n' "${case_pid}" >"${RUN_DIR}/pipeline.pid"
wait "${case_pid}"
case_pid=''
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/internal/analyze_mamba_swe_run.py" "${RUN_DIR}"
