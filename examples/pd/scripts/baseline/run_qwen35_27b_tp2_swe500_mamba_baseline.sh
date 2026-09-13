#!/usr/bin/env bash
set -euo pipefail

# Isolated native baseline for the current 27B SWE500 PD comparison.
# No modification/overlay of either SGLang environment or the external harness.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
pd_install_cleanup_traps
cd "${PD_DIR}"
export PD_ENV_BIN=/homes/siqic/anaconda3/envs/pd_mamba_baseline/bin
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
unset SGLANG_OVERLAY_ROOT PD_P_READY_DIR
unset SGLANG_ENABLE_DETERMINISTIC_INFERENCE NCCL_ALGO
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_AGENTIC_)
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_PD_)
export PD_DATA_ROOT=/tmp/pd-data
export MODEL_PATH=/homes/siqic/Qwen3.5-27B
export WORKLOAD_CONFIG="${PD_DIR}/configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml"
export PD_INFERENCE_RETURN_LOGPROB=false SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
export MIN_P=0
export MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
[[ "${MAX_INFLIGHT}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency' >&2; exit 2; }
export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/baseline-qwen35-27b-tp2-swe500-colocated-c${MAX_INFLIGHT}-$(date -u +%Y%m%dT%H%M%S)}"
export RESULTS_DIR="${RESULTS_DIR:-${PD_DIR}/runs-host/baseline/$(basename -- "${RUN_DIR}")}"
export PD_SWE_RUN_ID="$(basename -- "${RUN_DIR}")"
export PD_SWE_PROGRESS_FILE="${RUN_DIR}/episode_progress.jsonl"
export PD_CLEANUP_GRACE_SECONDS=120
mkdir -p "${RUN_DIR}/logs"
[[ ! -e "${RUN_DIR}/environment.json" && ! -e "${RUN_DIR}/requests.completed.jsonl" ]] || { echo 'Refusing reused run' >&2; exit 2; }
[[ -z "$(docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")" ]] || { echo 'Run already owns containers' >&2; exit 2; }
baseline_cleanup() {
  local status=$? owned=()
  pd_cleanup_all
  mapfile -t owned < <(timeout 15 docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")
  if (( ${#owned[@]} )); then timeout 30 docker rm -f "${owned[@]}" || true; fi
  return "${status}"
}
trap baseline_cleanup EXIT
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_environments.py" --expect baseline --output "${RUN_DIR}/environment.json"
"${PD_ENV_BIN}/python" - <<'PY'
import hashlib, json, os, subprocess, importlib.metadata
from pathlib import Path
import sglang
root=Path(os.environ['RUN_DIR'])
package=Path(sglang.__file__).resolve()
assert package.is_relative_to(Path(os.environ['PD_ENV_BIN']).parent), package
dataset=Path(os.environ['PD_DATA_ROOT'])/'swe-bench-verified/test.jsonl'
rows=[json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
assert len(rows)==len({r['instance_id'] for r in rows})==500
images=set(subprocess.check_output(['docker','image','ls','--format','{{.Repository}}:{{.Tag}}'],text=True,timeout=90).splitlines())
missing=[]
for r in rows:
    tag=r.get('image_name') or 'swebench/sweb.eval.x86_64.'+r['instance_id'].lower().replace('__','_1776_')+':latest'
    if tag.removeprefix('docker.io/') not in images: missing.append(tag)
assert not missing, missing[:8]
record=dict(host=os.uname().nodename, dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
    tasks=500, model=os.environ['MODEL_PATH'], tp=2, gpu_groups=[[0,4],[1,5],[2,6],[3,7]],
    concurrency=int(os.environ['MAX_INFLIGHT']), mem_fraction_static=.8, mamba_full_memory_ratio=.9,
    page_size=64, mamba_track_interval=64, pd=False, hicache=False, mooncake=False,
    attention_backend='triton', sampling_backend='flashinfer', deterministic_inference=False,
    disable_custom_all_reduce=False, numa_nodes_per_tp_group=[0,1], random_seed=2026,
    chunked_prefill_size=8192, max_prefill_tokens=8192, mamba_radix_cache_strategy='extra_buffer',
    runtime_versions={p:importlib.metadata.version(p) for p in ('sglang','torch','triton','flashinfer-python')},
    full_dataset_evaluation=True, warmup_requests=0, reasoning_parser='glm45', tool_parser='qwen3_coder',
    engine=str(package), harness_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data').rglob('*.py')})
(root/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
mkdir -p "${RUN_DIR}/source-snapshot"
while IFS= read -r source_file; do cp --parents "${source_file}" "${RUN_DIR}/source-snapshot/"; done < <(rg --files -g '*.py' data)
cp inference.py agentic_kv_request.py pd_metrics.py "${RUN_DIR}/source-snapshot/"
cp --parents scripts/common/runtime.sh scripts/new_method/internal/inference_checkpointed.py \
  scripts/new_method/internal/analyze_mamba_swe_run.py scripts/tools/analyze_swe_bench_run.py "${RUN_DIR}/source-snapshot/"
cp "${BASH_SOURCE[0]}" "${RUN_DIR}/source-snapshot/launcher.sh"
cp patches/sglang_0_5_14_mamba_resumed_chunk_{alignment,capacity}.patch "${RUN_DIR}/source-snapshot/"
for engine_file in schedule_policy.py scheduler.py; do
  cp "${PD_ENV_BIN}/../lib/python3.12/site-packages/sglang/srt/managers/${engine_file}" "${RUN_DIR}/source-snapshot/"
done
cp "${WORKLOAD_CONFIG}" "${RUN_DIR}/workload.yaml"
cp "${PD_DATA_ROOT}/swe-bench-verified/test.jsonl" "${RUN_DIR}/dataset.jsonl"
export WORKLOAD_CONFIG="${RUN_DIR}/workload.yaml"
"${PD_ENV_BIN}/python" -m pip freeze >"${RUN_DIR}/packages.txt"
groups=('0,4' '1,5' '2,6' '3,7')
ports=(33700 33701 33702 33703)
for gpu in 0 1 2 3 4 5 6 7; do pd_check_gpu_idle "${gpu}"; done
for port in "${ports[@]}" 33710; do pd_check_port_free "${port}"; done
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv >"${RUN_DIR}/gpu-processes-before.csv"
worker_urls=()
for index in "${!ports[@]}"; do
  mkdir -p "${RUN_DIR}/raw-${index}"
  setsid env CUDA_VISIBLE_DEVICES="${groups[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${ports[index]}" \
    --tp-size 2 --context-length 131072 --page-size 64 --mamba-track-interval 64 \
    --mamba-full-memory-ratio 0.9 --mem-fraction-static 0.80 \
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --attention-backend triton --sampling-backend flashinfer --random-seed 2026 \
    --numa-node 0 1 --mamba-radix-cache-strategy extra_buffer \
    --reasoning-parser glm45 --tool-call-parser qwen3_coder --enable-metrics --skip-server-warmup \
    --log-requests --log-requests-level 3 --log-requests-format json \
    --log-requests-target "${RUN_DIR}/raw-${index}" \
    --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
    >"${RUN_DIR}/logs/model-${index}.log" 2>&1 &
  worker_pid=$!; pd_track_group "${worker_pid}"
  worker_urls+=("http://127.0.0.1:${ports[index]}")
  pd_wait_http "model-${index}" "${worker_urls[index]}/model_info" "${worker_pid}" 1200
done
setsid "${PD_ENV_BIN}/python" -m sglang_router.launch_router \
  --worker-urls "${worker_urls[@]}" --policy cache_aware \
  --cache-threshold 0.3 --balance-abs-threshold 8 --balance-rel-threshold 1.2 \
  --host 127.0.0.1 --port 33710 >"${RUN_DIR}/logs/router.log" 2>&1 &
router_pid=$!; pd_track_group "${router_pid}"
pd_wait_http router http://127.0.0.1:33710/health "${router_pid}" 300
ports_csv="$(IFS=,; echo "${ports[*]}")"
setsid "${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../new_method/internal/inference_checkpointed.py" \
  --model "${MODEL_PATH}" --workload-config "${WORKLOAD_CONFIG}" \
  --router-port 33710 --prefill-port 33700 --prefill-ports "${ports_csv}" \
  --decode-port 33700 --decode-ports "${ports_csv}" \
  --requests 500 --warmup-requests 0 --dispatch-policy random --preserve-source-order \
  --request-rate 100 --arrival-distribution fixed --max-inflight "${MAX_INFLIGHT}" \
  --metrics-interval 2 --seed 2026 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --max-context-length 131072 --max-response-length 81920 --output-dir "${RUN_DIR}" \
  >"${RUN_DIR}/inference.log" 2>&1 &
inference_pid=$!; pd_track_group "${inference_pid}"
wait "${inference_pid}"
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../new_method/internal/analyze_mamba_swe_run.py" "${RUN_DIR}"
mkdir -p "${RESULTS_DIR}"
rsync -a "${RUN_DIR}/" "${RESULTS_DIR}/"
