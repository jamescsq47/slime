#!/usr/bin/env bash
set -euo pipefail

# Native Qwen3.5 baseline: all 500 distinct Verified tasks, each exactly once.
# No lifecycle, PD, reverse transfer, HiCache or Mooncake. Do not install or
# overlay the concurrently developed SGLang code into this baseline environment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../common/runtime.sh"
pd_install_cleanup_traps
cd "${PD_DIR}"
export PD_ENV_BIN=/homes/siqic/anaconda3/envs/pd_mamba_baseline/bin
unset SGLANG_OVERLAY_ROOT
export PATH="${PD_ENV_BIN}:${PATH}"
export PYTHONPATH="${PD_DIR}:$(cd -- "${PD_DIR}/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PD_DATA_ROOT="${PD_DATA_ROOT:-/tmp/pd-data}"
export PD_INFERENCE_RETURN_LOGPROB=false SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
export PD_MODEL_HTTP_TRANSPORT="${PD_MODEL_HTTP_TRANSPORT:-aiohttp}"
export MODEL_PATH="${MODEL_PATH:-/homes/siqic/Qwen3.5-9B}"
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml}"
export MODEL_REASONING_PARSER="${MODEL_REASONING_PARSER:-qwen3}"
export MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
export PD_COLOCATED_GPU_IDS="${PD_COLOCATED_GPU_IDS:-0,1,2,3,4,5,6,7}"
# Optional override; unset preserves the baseline engine's native default.
export PD_COLOCATED_MAMBA_RATIO="${PD_COLOCATED_MAMBA_RATIO:-}"
[[ "${PD_COLOCATED_GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU list' >&2; exit 2; }
IFS=, read -r -a gpu_ids <<< "${PD_COLOCATED_GPU_IDS}"
[[ "${MAX_INFLIGHT}" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_INFLIGHT must be a positive integer' >&2; exit 2; }
export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/baseline-qwen35-9b-tp1-swe-verified500-colocated-c${MAX_INFLIGHT}-$(date -u +%Y%m%dT%H%M%S)}"
export RESULTS_DIR="${RESULTS_DIR:-${PD_DIR}/runs-host/baseline/$(basename -- "${RUN_DIR}")}"
export PD_CLEANUP_GRACE_SECONDS=120
export PD_SWE_RUN_ID="${PD_SWE_RUN_ID:-$(basename -- "${RUN_DIR}")}"
export PD_SWE_PROGRESS_FILE="${RUN_DIR}/episode_progress.jsonl"
export MIN_P=0
# Do not inherit any custom PD activation from an interactive shell.
while IFS= read -r variable; do unset "${variable}"; done < <(compgen -v SGLANG_AGENTIC_)
unset PD_P_READY_DIR
mkdir -p "${RUN_DIR}/logs"
[[ ! -e "${RUN_DIR}/requests.completed.jsonl" ]] || { echo 'Run already contains episodes' >&2; exit 2; }
[[ ! -e "${RUN_DIR}/environment.json" ]] || { echo 'Refusing reused run directory' >&2; exit 2; }
# Docker containers outlive their host process group if Python receives TERM.
# Refuse a reused label before installing cleanup, so pre-existing containers
# can never be mistaken for this launch's resources.
existing_containers="$(docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")"
[[ -z "${existing_containers}" ]] || { echo 'Run label already owns containers' >&2; exit 2; }
pd_colocated_cleanup() {
  local status=$? owned_containers=()
  pd_cleanup_all
  mapfile -t owned_containers < <(timeout 15 docker ps -aq --filter "label=pd.swe.run_id=${PD_SWE_RUN_ID}")
  if (( ${#owned_containers[@]} )); then
    timeout 30 docker rm -f "${owned_containers[@]}" || true
  fi
  return "${status}"
}
trap pd_colocated_cleanup EXIT
"${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../tools/check_environments.py" \
  --expect baseline --output "${RUN_DIR}/environment.json"
"${PD_ENV_BIN}/python" -c 'import inference; import data.dispatch; import data.swe_bench_openenv.harness'
# Read-only preflight: no pulling, retagging, or pruning images here.
"${PD_ENV_BIN}/python" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
import sglang
root = Path(os.environ['RUN_DIR'])
dataset = Path(os.environ['PD_DATA_ROOT']) / 'swe-bench-verified/test.jsonl'
rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
ids = [r['instance_id'] for r in rows]
assert len(ids) == len(set(ids)) == 500, 'Expected 500 distinct Verified instances'
assert str(Path(sglang.__file__).resolve()).startswith(os.environ['PD_ENV_BIN'].removesuffix('/bin') + '/'), 'Non-baseline SGLang source'
images = set(subprocess.check_output(['docker', 'image', 'ls', '--format', '{{.Repository}}:{{.Tag}}'], text=True, timeout=90).splitlines())
missing = []
for r in rows:
    tag = r.get('image_name') or 'swebench/sweb.eval.x86_64.' + r['instance_id'].lower().replace('__', '_1776_') + ':latest'
    if tag.removeprefix('docker.io/') not in images:
        missing.append(tag)
assert not missing, f'Missing images: {missing[:8]}'
gpu_ids = [int(value) for value in os.environ['PD_COLOCATED_GPU_IDS'].split(',')]
assert len(gpu_ids) == len(set(gpu_ids)) and len(gpu_ids) <= 8, 'Expected 1-8 distinct GPUs'
ratio = os.environ.get('PD_COLOCATED_MAMBA_RATIO')
if ratio:
    assert 0 < float(ratio) <= 1, 'Invalid Mamba ratio'
record = dict(host=os.uname().nodename, dataset=str(dataset), tasks=len(ids),
              dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
              local_images_present=len(ids), mode='native_colocated',
              model=os.environ['MODEL_PATH'], tp=1, gpu_ids=gpu_ids,
              mamba_full_memory_ratio=float(ratio) if ratio else 'engine_default',
              max_inflight=int(os.environ['MAX_INFLIGHT']), mem_fraction_static=0.8, pd=False, hicache=False,
              mooncake=False, warmup_requests=0, full_dataset_evaluation=True,
              gpu7_shared_with_existing_user_process=7 in gpu_ids)
(root/'preflight.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
PY
# Preserve exact input/config/code for diagnosis without changing the harness.
mkdir -p "${RUN_DIR}/source-snapshot"
while IFS= read -r source_file; do
  cp --parents "${source_file}" "${RUN_DIR}/source-snapshot/"
done < <(rg --files -g '*.py' data)
cp inference.py model_http_transport.py agentic_kv_request.py pd_metrics.py "${RUN_DIR}/source-snapshot/"
cp --parents scripts/common/runtime.sh scripts/new_method/internal/inference_checkpointed.py \
  scripts/new_method/internal/analyze_mamba_swe_run.py scripts/tools/analyze_swe_bench_run.py \
  "${RUN_DIR}/source-snapshot/"
cp "${BASH_SOURCE[0]}" "${RUN_DIR}/source-snapshot/launcher.sh"
cp "${WORKLOAD_CONFIG}" "${RUN_DIR}/workload.yaml"
cp "${PD_DATA_ROOT}/swe-bench-verified/test.jsonl" "${RUN_DIR}/dataset.jsonl"
export WORKLOAD_CONFIG="${RUN_DIR}/workload.yaml"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv >"${RUN_DIR}/gpu-processes-before.csv"
"${PD_ENV_BIN}/python" -m pip freeze >"${RUN_DIR}/packages.txt"

ports=()
for index in "${!gpu_ids[@]}"; do ports+=("$((33600 + index))"); done
mamba_args=()
if [[ -n "${PD_COLOCATED_MAMBA_RATIO}" ]]; then
  mamba_args+=(--mamba-full-memory-ratio "${PD_COLOCATED_MAMBA_RATIO}")
fi
for index in "${!ports[@]}"; do
  pd_check_gpu_idle "${gpu_ids[index]}"
  pd_check_port_free "${ports[index]}"
done
pd_check_port_free 33610

worker_pids=()
worker_urls=()
for index in "${!ports[@]}"; do
  mkdir -p "${RUN_DIR}/raw-${index}"
  # Level3 JSON logging retains the raw pre-parser output without truncation.
  # Explicit input_ids remain authoritative; logging does not re-tokenize them.
  setsid env CUDA_VISIBLE_DEVICES="${gpu_ids[index]}" SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
    "${PD_ENV_BIN}/python" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${ports[index]}" \
    --tp-size 1 --context-length 131072 --page-size 64 --mamba-track-interval 64 \
    "${mamba_args[@]}" \
    --mem-fraction-static 0.80 --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --enable-deterministic-inference --attention-backend triton --random-seed 2026 \
    --reasoning-parser "${MODEL_REASONING_PARSER}" --tool-call-parser qwen3_coder \
    --enable-metrics --skip-server-warmup \
    --log-requests --log-requests-level 3 --log-requests-format json \
    --log-requests-target "${RUN_DIR}/raw-${index}" \
    --uvicorn-access-log-exclude-prefixes /get_load /metrics /health \
    >"${RUN_DIR}/logs/model-${index}.log" 2>&1 &
  worker_pids+=("$!"); pd_track_group "$!"
  worker_urls+=("http://127.0.0.1:${ports[index]}")
done
for index in "${!ports[@]}"; do
  pd_wait_http "model-${index}" "${worker_urls[index]}/model_info" "${worker_pids[index]}" 1200
  echo "model-${index} ready"
done
setsid "${PD_ENV_BIN}/python" -m sglang_router.launch_router \
  --worker-urls "${worker_urls[@]}" --policy cache_aware \
  --cache-threshold 0.3 --balance-abs-threshold 8 --balance-rel-threshold 1.2 \
  --host 127.0.0.1 --port 33610 >"${RUN_DIR}/logs/router.log" 2>&1 &
router_pid=$!; pd_track_group "${router_pid}"
pd_wait_http router http://127.0.0.1:33610/health "${router_pid}" 300
ports_csv="$(IFS=,; echo "${ports[*]}")"
setsid "${PD_ENV_BIN}/python" "${SCRIPT_DIR}/../new_method/internal/inference_checkpointed.py" \
  --model "${MODEL_PATH}" \
  --workload-config "${WORKLOAD_CONFIG}" \
  --router-port 33610 --prefill-port 33600 --prefill-ports "${ports_csv}" \
  --decode-port 33600 --decode-ports "${ports_csv}" \
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
