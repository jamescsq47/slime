#!/usr/bin/env bash
set -euo pipefail

# Two-node Qwen3.5-27B validation: one global 4P x 4D fabric, TP=2 per
# endpoint. Runtime ownership/control is TCP-only; Host KV is source-local.

SLIME_ROOT="/homes/siqic/dualpd/pd_multi_node_v3/slime"
MULTINODE="${SLIME_ROOT}/tools/dualpd/multinode.py"
PYTHON="/homes/siqic/anaconda3/envs/pd_multi_node_v3/bin/python"
MODEL="${MODEL_PATH:-/homes/siqic/Qwen3.5-27B}"
REMOTE_NODE="${REMOTE_NODE:-a11}"
P_HOST="${P_HOST:-10.0.1.170}"
D_HOST="${D_HOST:-10.0.1.171}"
RUN_ID="${RUN_ID:-qwen35-27b-tp2-global-pd-c128-$(date -u +%Y%m%dT%H%M%SZ)}"
D2P_HOST_GIB_PER_RANK="${D2P_HOST_GIB_PER_RANK:-32}"
P2D_HOST_GIB_PER_RANK="${P2D_HOST_GIB_PER_RANK:-16}"
REQUESTS="${REQUESTS:-500}"
MAX_INFLIGHT="${MAX_INFLIGHT:-128}"
RUN_ROOT="/tmp/dualpd-q35/${RUN_ID}"
RESULT_ROOT="${SLIME_ROOT}/runs/dualpd/${RUN_ID}"
CONFIG="${RUN_ROOT}/global.json"
WORKLOAD_CONFIG="${SLIME_ROOT}/examples/pd/configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml"
DATA_ROOT="/homes/siqic/dualpd/slime/downloads/pd-data"
SUPERVISORS=()
TUNNEL_PID=""

mkdir -p "${RUN_ROOT}/logs" "${RESULT_ROOT}"
export RUN_ID P_HOST D_HOST MODEL D2P_HOST_GIB_PER_RANK P2D_HOST_GIB_PER_RANK
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Qwen3.5 model config not found: ${MODEL}/config.json" >&2
  exit 2
fi
echo "Model: ${MODEL}"

wait_http() {
  local name="$1" url="$2" timeout_seconds="$3" start now
  start="$(date +%s)"
  while ! curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; do
    now="$(date +%s)"
    if (( now - start >= timeout_seconds )); then
      echo "Timed out waiting for ${name}: ${url}" >&2
      return 1
    fi
    sleep 2
  done
  echo "Ready: ${name}"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  "${PYTHON}" "${MULTINODE}" stop --config "${CONFIG}" --component router >/dev/null 2>&1 || true
  for i in 0 1 2 3; do
    "${PYTHON}" "${MULTINODE}" stop --config "${CONFIG}" --component worker --node-id "p${i}" >/dev/null 2>&1 || true
    ssh -o BatchMode=yes "${REMOTE_NODE}" "${PYTHON} '${MULTINODE}' stop --config '${CONFIG}' --component worker --node-id 'd${i}'" >/dev/null 2>&1 || true
  done
  "${PYTHON}" "${MULTINODE}" stop --config "${CONFIG}" --component control >/dev/null 2>&1 || true
  [[ -n "${TUNNEL_PID}" ]] && kill -TERM "${TUNNEL_PID}" 2>/dev/null || true
  for pid in "${SUPERVISORS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
  printf '%s\n' "${rc}" >"${RESULT_ROOT}/launcher_exit_code"
  exit "${rc}"
}
trap cleanup EXIT INT TERM

"${PYTHON}" - "${CONFIG}" <<'PY'
import json, os, sys
path = sys.argv[1]
p_host, d_host = os.environ["P_HOST"], os.environ["D_HOST"]
# mlx5_1 is the only active data rail on a10/a11 and is attached to NUMA 1.
# Keep TP ranks NUMA-local instead of making every logical replica wait for
# one remote-socket shard.  The global router is still free to balance over
# all four replicas; source-local Host staging remains valid for both NUMAs.
groups = [(4, 5), (6, 7), (0, 1), (2, 3)]
nodes = []
for role, host, start in (("prefill", p_host, 25100), ("decode", d_host, 25200)):
    prefix = "p" if role == "prefill" else "d"
    for i, pair in enumerate(groups):
        base = start + i * 10
        nodes.append({
            "node_id": f"{prefix}{i}", "engine_id": f"{role}-{i}",
            "role": role, "host_ip": host, "gpus": list(pair),
            "numa_nodes": [1 if gpu >= 4 else 0 for gpu in pair],
            "mem_fraction_static": 0.8,
            "port": base, "bootstrap_port": base + 1,
            "reverse_bootstrap_port": base + 2,
            "ib_device": "mlx5_1", "ucx_net_devices": "mlx5_1:1",
        })
cfg = {
    "run_id": os.environ["RUN_ID"], "group_id": "global-pd-fabric",
    "local_root": "/tmp/dualpd-q35",
    "sglang_root": "/homes/siqic/dualpd/pd_multi_node_v3/sglang",
    "slime_root": "/homes/siqic/dualpd/pd_multi_node_v3/slime",
    "python": "/homes/siqic/anaconda3/envs/pd_multi_node_v3/bin/python",
    "model_path": os.environ["MODEL"], "model_family": "qwen35_moe",
    "cuda_home": "/homes/siqic/cuda-12.8", "local_triton_cache": True,
    "tp_size": 2, "page_size": 64, "context_length": 131072,
    "chunked_prefill_size": 8192, "max_prefill_tokens": 8192,
    # SWE-bench permits 8192 generated tokens in one model turn.  Keep one
    # allocator-wide Decode growth floor per D rank; do not multiply this
    # budget by the number of admitted requests.
    "decode_growth_tokens": 8192,
    "d2p_host_gib_per_rank": int(os.environ["D2P_HOST_GIB_PER_RANK"]),
    "p2d_host_gib_per_rank": int(os.environ["P2D_HOST_GIB_PER_RANK"]),
    "fast_tool_seconds": 1, "direct_admission_seconds": 1,
    # P->D has no tool-time deadline.  Give the Router enough time to return
    # the late-bound D before spilling an otherwise directly deliverable
    # Prefill result to Host.  A truly capacity-blocked request still spills
    # after this bounded grace and releases P HBM once Host is durable.
    "p2d_late_bind_grace_seconds": 2.0,
    # a10/a11 expose one active mlx5 rail for all four logical TP groups.
    # Keep D->P Direct bounded so congestion falls back to source-local Host
    # within the one-second admission window and releases D HBM promptly.
    # P->D can use all eight lanes because completed Prefill should refill D.
    "direct_lanes": 8,
    "d2p_direct_lanes": 8,
    "p2d_direct_lanes": 8,
    "host_lanes": 4,
    "shared_network_lanes": 16,
    "d2p_shared_network_lanes": 16,
    "p2d_shared_network_lanes": 16,
    # Under D->P Host backlog run at most four Direct plus twelve Host lanes.
    # With no Host waiter, Direct may still consume all 16 slots.
    "d2p_direct_network_reserve": 4,
    "p2d_direct_network_reserve": 0,
    # Demand-driven reservation: when durable D->P Host snapshots are
    # waiting, drain them on at least half of the shared rail.  With no Host
    # waiter, Direct remains free to consume every lane.
    "d2p_host_network_reserve": 12,
    "p2d_host_network_reserve": 0,
    # Each TP-rank agent can own up to 12 local Direct/Host operations.  Keep
    # enough native NIXL progress workers to advance all of them without
    # relying on sparse Python polling.
    "nixl_progress_threads": 16,
    "prefill_router_reservation_tokens": 16384,
    # Crash-only bound.  Normal Router shadow credit is released causally
    # after all D TP ranks finish DMA and /get_load observes the physical lease.
    "decode_router_reservation_seconds": 3600.0,
    "mamba_full_memory_ratio": 0.9, "prefill_mamba_full_memory_ratio": 0.9,
    "decode_mamba_full_memory_ratio": 0.9, "seed": 2026,
    "group_control": {
        "enabled": True, "node_id": "p0", "listen_host": "0.0.0.0",
        "advertise_host": p_host, "port": 25009,
        "token": "dualpd-q35-tp2-global-control-2026",
    },
    "router": {"node_id": "p0", "engine_id": "router-0", "port": 25000,
               "metrics_port": 25001},
    "workload_command": [], "nodes": nodes,
}
with open(path, "w") as out:
    json.dump(cfg, out, indent=2)
PY

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  "${PYTHON}" "${MULTINODE}" plan --config "${CONFIG}"
  trap - EXIT INT TERM
  exit 0
fi

# Immutable launch metadata is copied once; workers never poll it.
ssh -o BatchMode=yes "${REMOTE_NODE}" "mkdir -p '${RUN_ROOT}'"
scp -q "${CONFIG}" "${REMOTE_NODE}:${CONFIG}"

"${PYTHON}" "${MULTINODE}" start-control --config "${CONFIG}" >"${RUN_ROOT}/logs/control.log" 2>&1 &
SUPERVISORS+=("$!")
"${PYTHON}" "${MULTINODE}" control-ready --config "${CONFIG}" --timeout 120

for i in 0 1 2 3; do
  pport=$((25100 + i * 10)); dport=$((25200 + i * 10))
  "${PYTHON}" "${MULTINODE}" start-worker --config "${CONFIG}" --node-id "p${i}" >"${RUN_ROOT}/logs/p-${i}.log" 2>&1 &
  SUPERVISORS+=("$!")
  wait_http "p-${i}" "http://${P_HOST}:${pport}/model_info" 1200
  ssh -o BatchMode=yes "${REMOTE_NODE}" "${PYTHON} '${MULTINODE}' start-worker --config '${CONFIG}' --node-id 'd${i}'" >"${RUN_ROOT}/logs/d-${i}.supervisor.log" 2>&1 &
  SUPERVISORS+=("$!")
  wait_http "d-${i}" "http://${D_HOST}:${dport}/model_info" 1200
done

"${PYTHON}" "${MULTINODE}" start-router --config "${CONFIG}" --timeout 120 >"${RUN_ROOT}/logs/router.log" 2>&1 &
SUPERVISORS+=("$!")
wait_http router "http://${P_HOST}:25000/health" 300

tunnel_args=(); prefill_ports=(); decode_ports=()
for i in 0 1 2 3; do
  pport=$((25100 + i * 10)); dport=$((25200 + i * 10))
  prefill_ports+=("${pport}"); decode_ports+=("${dport}")
  tunnel_args+=(-L "${dport}:${D_HOST}:${dport}")
done
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  "${tunnel_args[@]}" "${REMOTE_NODE}" >"${RUN_ROOT}/logs/metrics-tunnel.log" 2>&1 &
TUNNEL_PID="$!"
sleep 2

export PD_DATA_ROOT="${DATA_ROOT}" PD_SWE_RUN_ID="${RUN_ID}"
export PD_AGENTIC_FINAL_CALLBACK_URL="http://${P_HOST}:25000/dualpd/application_final"
export PD_SWE_PROGRESS_FILE="${RESULT_ROOT}/episode_progress.jsonl"
export PD_INFERENCE_RETURN_LOGPROB=false SGLANG_AGENTIC_KV_LIFECYCLE=true
export SLIME_HTTP_READ_TIMEOUT_SECONDS=86400
export PYTHONPATH="${SLIME_ROOT}:/homes/siqic/dualpd/pd_multi_node_v3/sglang/python:${PYTHONPATH:-}"
prefill_csv="$(IFS=,; echo "${prefill_ports[*]}")"
decode_csv="$(IFS=,; echo "${decode_ports[*]}")"
cd "${SLIME_ROOT}/examples/pd"
"${PYTHON}" "${SLIME_ROOT}/examples/pd/scripts/new_method/internal/inference_checkpointed.py" \
  --model "${MODEL}" --workload-config "${WORKLOAD_CONFIG}" \
  --router-host "${P_HOST}" --router-port 25000 \
  --prefill-host 127.0.0.1 --prefill-port "${prefill_ports[0]}" --prefill-ports "${prefill_csv}" \
  --decode-host 127.0.0.1 --decode-port "${decode_ports[0]}" --decode-ports "${decode_csv}" \
  --requests "${REQUESTS}" --warmup-requests 0 --dispatch-policy random --preserve-source-order \
  --request-rate 100 --arrival-distribution fixed --max-inflight "${MAX_INFLIGHT}" --metrics-interval 2 --seed 2026 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --max-context-length 131072 \
  --max-response-length 81920 --output-dir "${RESULT_ROOT}/workload" \
  >"${RUN_ROOT}/logs/inference.log" 2>&1

"${PYTHON}" "${SLIME_ROOT}/examples/pd/scripts/tools/analyze_swe_bench_run.py" \
  "${RESULT_ROOT}/workload" >"${RUN_ROOT}/logs/analyzer.log" 2>&1
