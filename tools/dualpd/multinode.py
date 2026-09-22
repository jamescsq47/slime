#!/usr/bin/env python3
"""Explicit, fail-closed remote entry point for the multi-node development V1.

Plan/preflight use only stdlib. Engine start is deliberately capability-gated:
source-local Host extents must have a remotely usable data-plane implementation,
not merely a shared pathname. Existing single-node launchers are untouched.
"""
import argparse
import fcntl
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid
import urllib.request
import urllib.error

from process_supervisor import control as process_control, supervise


NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
REQUIRED_CAPABILITIES = {
    "source_local_host_rdma", "shared_control_polling", "tp_shard_atomicity",
    "remote_host_fence_release", "p2d_d2p_integration",
}


def absolute(value, field):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(field + " must be an absolute path")
    if ".." in Path(value).parts or value in {"/", "/dev/shm", "/tmp"}:
        raise ValueError(field + " needs a dedicated directory/path")
    return value


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("model_family", "qwen3") not in {"qwen3", "minimax_m2", "qwen35_moe"}:
        raise ValueError("unsupported multi-node model family")
    for key in ("run_id",):
        if not NAME.fullmatch(cfg.get(key, "")):
            raise ValueError("invalid " + key)
    for key in ("control_root", "local_root", "sglang_root", "slime_root", "model_path", "python"):
        absolute(cfg.get(key), key)
    if cfg.get("control_backend", "files") not in {"files", "tcp"}:
        raise ValueError("control_backend must be files or tcp")
    event_control = cfg.get("control_backend") == "tcp"
    if not event_control and cfg["control_root"].startswith(("/dev/shm/", "/tmp/")):
        raise ValueError("control_root must name the explicitly shared POSIX mount")
    if cfg["local_root"] == cfg["control_root"]:
        raise ValueError("Host data/local process state must not use the shared control directory")
    tp = cfg.get("tp_size")
    if type(tp) is not int or tp not in (1, 2, 4, 8):
        raise ValueError("V1 supports matching TP=1/2/4/8; a whole group stays on one host")
    nodes = cfg.get("nodes", [])
    if len(nodes) < 2 or {n.get("role") for n in nodes} != {"prefill", "decode"}:
        raise ValueError("at least one P and one D node required")
    if sum(n["role"] == "prefill" for n in nodes) != 1:
        raise ValueError("V1 requires exactly one logical P group; multi-P routing is not validated")
    for field in ("node_id", "engine_id", "host_ip"):
        vals = [n.get(field) for n in nodes]
        if len(set(vals)) != len(vals):
            raise ValueError("one engine group per host; duplicate " + field)
    ports = ("port", "bootstrap_port", "reverse_bootstrap_port")
    for n in nodes:
        for field in ("node_id", "engine_id"):
            if not NAME.fullmatch(n.get(field, "")):
                raise ValueError("invalid " + field)
        ip = ipaddress.ip_address(n["host_ip"])
        if ip.version != 4 or ip.is_loopback or ip.is_unspecified or ip.is_multicast or ip.is_link_local:
            raise ValueError("V1 host_ip must be an explicit routable IPv4 address")
        gpus = n.get("gpus", [])
        if len(gpus) != tp or len(set(gpus)) != tp or any(type(g) is not int or g < 0 for g in gpus):
            raise ValueError("each node must list exactly TP distinct physical GPUs")
        numas = n.get("numa_nodes", [])
        if numas and (len(numas) != tp or any(type(x) is not int or x < 0 for x in numas)):
            raise ValueError("numa_nodes must be empty or exactly one NUMA ID per TP rank")
        if not 0 < n.get("mem_fraction_static", 0) < 1:
            raise ValueError("explicit group-wide mem_fraction_static required")
        if any(type(n.get(p)) is not int or not 1024 <= n[p] <= 65535 for p in ports):
            raise ValueError("explicit unprivileged server/bootstrap ports required")
        if len({n[p] for p in ports}) != len(ports):
            raise ValueError("listener ports must be distinct on each host")
    for key in ("d2p_host_gib_per_rank", "p2d_host_gib_per_rank"):
        if type(cfg.get(key)) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("positive " + key + " required")
    for key in ("page_size", "context_length", "chunked_prefill_size", "max_prefill_tokens"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError("positive integer " + key + " required")
    for key in ("fast_tool_seconds", "direct_admission_seconds"):
        if type(cfg.get(key)) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("positive " + key + " required")
    transfer_limit = cfg.get("pd_max_transfer_inflight", 0)
    if type(transfer_limit) is not int or transfer_limit < 0:
        raise ValueError("pd_max_transfer_inflight must be a nonnegative integer")
    if type(cfg.get("slow_congestion_recompute", False)) is not bool:
        raise ValueError("slow_congestion_recompute must be boolean")
    if type(cfg.get("d2p_host_staging", True)) is not bool:
        raise ValueError("d2p_host_staging must be boolean")
    if type(cfg.get("d2p_direct_wait_only", False)) is not bool:
        raise ValueError("d2p_direct_wait_only must be boolean")
    if cfg.get("d2p_direct_wait_only", False) and (
        cfg.get("d2p_host_staging", True) or cfg.get("slow_congestion_recompute", False)
        or not event_control
    ):
        raise ValueError("d2p_direct_wait_only requires socket control, no D->P Host and no recompute")
    if type(cfg.get("tp_host_async_prepare", False)) is not bool:
        raise ValueError("tp_host_async_prepare must be boolean")
    if type(cfg.get("p_workset_controller", False)) is not bool:
        raise ValueError("p_workset_controller must be boolean")
    if cfg.get("p_workset_controller", False) and not event_control:
        raise ValueError("p_workset_controller requires socket control")
    if type(cfg.get("local_triton_cache", False)) is not bool:
        raise ValueError("local_triton_cache must be boolean")
    if type(cfg.get("router_profile_imports", False)) is not bool:
        raise ValueError("router_profile_imports must be boolean")
    timeout = cfg.get("router_startup_timeout_seconds", 1800)
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("router_startup_timeout_seconds must be positive and finite")
    high, low = cfg.get("slow_congestion_high", 32), cfg.get("slow_congestion_low", 8)
    if type(high) is not int or type(low) is not int or not 0 <= low < high:
        raise ValueError("Slow congestion thresholds require integer 0 <= low < high")
    router = cfg.get("router", {})
    if router.get("node_id") not in {n["node_id"] for n in nodes}:
        raise ValueError("router.node_id must identify one configured node")
    if not NAME.fullmatch(router.get("engine_id", "")) or router["engine_id"] in {n["engine_id"] for n in nodes}:
        raise ValueError("router.engine_id must be unique")
    rnode = node_for(cfg, router["node_id"])
    occupied = {rnode[k] for k in ports}
    for key in ("port", "metrics_port"):
        port = router.get(key)
        if type(port) is not int or not 1024 <= port <= 65535 or port in occupied:
            raise ValueError("router port invalid or collides with a local listener")
        occupied.add(port)
    if event_control:
        control = cfg.get("control", {})
        if control.get("node_id") != router["node_id"]:
            raise ValueError("control broker must be on the Router node")
        if not isinstance(control.get("token"), str) or len(control["token"]) < 32:
            raise ValueError("control requires a unique run-scoped token of at least 32 characters")
        if sum(n["role"] == "decode" for n in nodes) != 1:
            raise ValueError("socket receipt mapping currently requires one logical D TP group")
        for key in ("port", "tp_port"):
            value = control.get(key)
            if type(value) is not int or not 1024 <= value <= 65535 or value in occupied:
                raise ValueError("control port invalid or collides with another listener")
            occupied.add(value)
    workload = cfg.get("workload_command", [])
    if not isinstance(workload, list) or any(not isinstance(x, str) or not x for x in workload):
        raise ValueError("workload_command must be an argv list, not a shell string")
    return cfg


def node_for(cfg, node_id):
    return next(n for n in cfg["nodes"] if n["node_id"] == node_id)


def control_dir(cfg):
    return Path(cfg["control_root"]) / cfg["run_id"]


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def common_env(cfg):
    root = control_dir(cfg)
    env = {
        "SGLANG_AGENTIC_MULTINODE_ENABLED": "1",
        "SGLANG_AGENTIC_MULTINODE_RUN_ID": cfg["run_id"],
        "SGLANG_AGENTIC_MULTINODE_CONTROL_ROOT": cfg["control_root"],
        "SGLANG_AGENTIC_MULTINODE_LOCAL_ROOT": cfg["local_root"],
        "SGLANG_AGENTIC_MULTINODE_TP_SIZE": str(cfg["tp_size"]),
        "SGLANG_AGENTIC_MULTINODE_PEER_TP_SIZE": str(cfg["tp_size"]),
        "SGLANG_AGENTIC_MULTINODE_CONTROL_POLL_INTERVAL": "0.1",
        "SGLANG_PD_P_READY_DIR": str(root),
        "PD_P_READY_DIR": str(root),
        "PD_INFERENCE_RETURN_LOGPROB": "false",
        # Router retains backend idle sockets for 30s. Match the existing
        # single-node launcher: SGLang's 5s default can close a reused POST
        # connection before the Router's pool retires it.
        "SGLANG_TIMEOUT_KEEP_ALIVE": "120",
        "SGLANG_AGENTIC_KV_LEDGER_PATH": str(root / "lifecycle.json"),
        "SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH": str(root / "d2p.json"),
        "SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH": str(root / "p2d.json"),
        "SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR": str(root / "early-claims"),
        "SGLANG_AGENTIC_KV_METADATA_DIR": str(root / "snapshot-metadata"),
        "SGLANG_AGENTIC_KV_PREFILL_LOAD_PATH": str(root / "early-claims/prefill-loads.json"),
        "SGLANG_AGENTIC_KV_REGISTER_PREWARM_DIR": str(root / "host-register-prewarm"),
        "SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER": "1",
        "SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA": "1",
        # One new chunk per progress visit, with the existing four DMA lanes.
        # Keep single-node defaults unchanged.
        "SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS": "1024",
        "SGLANG_AGENTIC_KV_D2H_INFLIGHT": "4",
        "SGLANG_AGENTIC_KV_LIFECYCLE": "true",
        "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY": "true",
        "SGLANG_AGENTIC_KV_D_HOSTLESS": "true",
        "SGLANG_AGENTIC_KV_HOST_STAGING": str(cfg.get("d2p_host_staging", True)).lower(),
        "SGLANG_AGENTIC_KV_DIRECT_WAIT_ONLY": str(cfg.get("d2p_direct_wait_only", False)).lower(),
        "SGLANG_AGENTIC_KV_P2D_HOST_STAGING": "true",
        "SGLANG_AGENTIC_KV_EARLY_CLAIM": "1",
        "SGLANG_AGENTIC_KV_RELAY_ENABLED": "false",
        "SGLANG_PD_DECODE_ENABLE_RADIX_CACHE": "true",
        "SGLANG_PD_P_READY_BACKPRESSURE_MODE": "disabled",
        "SGLANG_PD_P_READY_REQUEST_CAP": "0",
        # Zero removes the unrelated eight-request default in Decode. The
        # Decode page/metadata/Mamba admission checks still bound real usage.
        "SGLANG_PD_MAX_TRANSFER_INFLIGHT": str(cfg.get("pd_max_transfer_inflight", 0)),
        "SGLANG_PD_LATE_BIND_FORCE_LEGACY_LOADS": "1",
        "SGLANG_AGENTIC_KV_P2D_SPILL_DELAY_SECONDS": "0.5",
        "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "true",
        "SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB": str(cfg["d2p_host_gib_per_rank"]),
        "SGLANG_AGENTIC_MULTINODE_P2D_HOST_GIB": str(cfg["p2d_host_gib_per_rank"]),
        "SGLANG_AGENTIC_KV_TP_SIZE": str(cfg["tp_size"]),
        "SGLANG_AGENTIC_KV_TP_HOST_ASYNC_PREPARE": str(cfg.get("tp_host_async_prepare", False)).lower(),
        "SGLANG_AGENTIC_P_WORKSET_CONTROLLER": str(cfg.get("p_workset_controller", False)).lower(),
        "SGLANG_AGENTIC_KV_PREFILL_DOMAIN_COUNT": str(sum(n["role"] == "prefill" for n in cfg["nodes"])),
        "SGLANG_PD_LATE_BIND_NUMA_DOMAINS": "0",
        "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS": "0",
        "SGLANG_PD_LATE_BIND_GLOBAL_DECODE": "1",
        "SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION": "1.0",
        "SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD": str(cfg["fast_tool_seconds"]),
        "SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT": str(cfg["direct_admission_seconds"]),
        "SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE": str(cfg.get("slow_congestion_recompute", False)).lower(),
        "SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE": "false",
        "SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE": "false",
        "SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH": str(cfg.get("slow_congestion_high", 32)),
        "SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW": str(cfg.get("slow_congestion_low", 8)),
        "PYTHONPATH": str(Path(cfg["sglang_root"]) / "python") + ":" + cfg["slime_root"],
    }
    if cfg.get("model_family") == "qwen35_moe":
        env.update({"SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID": "1",
                    "SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT": "true",
                    "SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED": "true",
                    "SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION": "true"})
    if cfg.get("cuda_home"):
        env["CUDA_HOME"] = absolute(cfg["cuda_home"], "cuda_home")
    if cfg.get("debug_kv_digest", False):
        env["SGLANG_AGENTIC_KV_DEBUG_DIGEST"] = "1"
    if cfg.get("control_backend") == "tcp":
        broker = cfg["control"]
        host = node_for(cfg, broker["node_id"])["host_ip"]
        env.update({
            "SGLANG_AGENTIC_CONTROL_ENDPOINT": f'{host}:{broker["port"]}',
            "SGLANG_AGENTIC_TP_EVENT_ENDPOINT": f'{host}:{broker["tp_port"]}',
            "SGLANG_AGENTIC_CONTROL_RUN_ID": cfg["run_id"],
            "SGLANG_AGENTIC_CONTROL_TOKEN": broker["token"],
            "SGLANG_AGENTIC_KV_TP_HOST_ASYNC_PREPARE": "true",
            "SGLANG_AGENTIC_KV_P_ASYNC_CONTROL": "1",
        })
    return env


def worker_plan(cfg, n):
    env = common_env(cfg)
    local = Path(cfg["local_root"]) / cfg["run_id"] / n["engine_id"]
    if cfg.get("local_triton_cache", False):
        # Runtime compilation must not write to the default NFS home cache.
        # Keep content-addressed kernels across runs, separately per engine;
        # this is compiler data, never a lifecycle/TP coordination channel.
        env["TRITON_CACHE_DIR"] = str(
            Path(cfg["local_root"]) / "compiler-cache" / n["engine_id"] / "triton"
        )
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, n["gpus"])),
        "SGLANG_HOST_IP": n["host_ip"],
        "SGLANG_AGENTIC_MULTINODE_NODE_ID": n["node_id"],
        "SGLANG_AGENTIC_MULTINODE_ENGINE_ID": n["engine_id"],
        "SGLANG_AGENTIC_MULTINODE_ROLE": n["role"],
        "SGLANG_AGENTIC_MULTINODE_HOST_IP": n["host_ip"],
        "SGLANG_AGENTIC_KV_ENGINE_ID": n["engine_id"],
        "SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT": str(n["reverse_bootstrap_port"]),
        "SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_DIR": str(local / "d2p-arena"),
        "SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_DIR": str(local / "p2d-arena"),
        "SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND": "memfd",
        "SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND": "memfd",
        "SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB": str(cfg["d2p_host_gib_per_rank"]),
        "SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB": str(cfg["p2d_host_gib_per_rank"]),
    })
    if cfg.get("control_backend") == "tcp":
        env["SGLANG_AGENTIC_CONTROL_GROUP_ID"] = n["engine_id"]
        env["SGLANG_AGENTIC_TP_RECEIPT_SOURCE_GROUP"] = next(
            node["engine_id"] for node in cfg["nodes"] if node["role"] == "decode"
        )
    elif cfg["tp_size"] > 1:
        # V1 keeps each whole TP group on one host. Internal rank reports use
        # run/engine-scoped tmpfs; cross-engine receipts and ownership ledgers
        # remain shared. The engine enforces the namespace allowlist.
        scope = hashlib.sha256(str(control_dir(cfg)).encode()).hexdigest()[:16]
        env["SGLANG_AGENTIC_KV_TP_CONTROL_DIR"] = str(
            Path("/dev/shm/dualpd-tp") / scope / n["engine_id"]
        )
        if n["role"] == "decode":
            env["SGLANG_AGENTIC_KV_D_TP_CONTROL_DIR"] = env["SGLANG_AGENTIC_KV_TP_CONTROL_DIR"]
    if n.get("numa_nodes"):
        env["SGLANG_AGENTIC_KV_TP_NUMA_NODES"] = ",".join(map(str, n["numa_nodes"]))
    ps = [x for x in cfg["nodes"] if x["role"] == "prefill"]
    env["SGLANG_AGENTIC_KV_PREFILL_DOMAIN"] = str(ps.index(n) if n["role"] == "prefill" else 0)
    command = [cfg["python"], "-m", "sglang.launch_server", "--model-path", cfg["model_path"],
               "--host", "0.0.0.0", "--port", str(n["port"]), "--tp-size", str(cfg["tp_size"]),
               "--disaggregation-mode", n["role"], "--disaggregation-transfer-backend", "nixl",
               "--disaggregation-bootstrap-port", str(n["bootstrap_port"]),
               "--mem-fraction-static", str(n["mem_fraction_static"]),
               "--page-size", str(cfg["page_size"]), "--context-length", str(cfg["context_length"]),
               "--enable-metrics", "--skip-server-warmup"]
    if n["role"] == "prefill":
        command += ["--chunked-prefill-size", str(cfg["chunked_prefill_size"]),
                    "--max-prefill-tokens", str(cfg["max_prefill_tokens"])]
    if cfg.get("p2d_host_probe") and n["role"] == "decode":
        # Engineering-only capacity test: one 8k request fits, two do not.
        # No fake load samples or changes to the production router/state machine.
        command += ["--max-total-tokens", "12288"]
    if cfg.get("model_family") == "minimax_m2":
        command += ["--trust-remote-code", "--reasoning-parser", "minimax-append-think",
                    "--tool-call-parser", "minimax-m2", "--kv-cache-dtype", "bfloat16",
                    "--ep-size", str(cfg["tp_size"])]
    if cfg.get("model_family") == "qwen35_moe":
        command += ["--trust-remote-code", "--dtype", "bfloat16",
                    "--kv-cache-dtype", "bfloat16", "--ep-size", "1",
                    "--reasoning-parser", "glm45", "--tool-call-parser", "qwen3_coder",
                    "--attention-backend", "triton", "--linear-attn-backend", "triton",
                    "--moe-runner-backend", "triton", "--sampling-backend", "flashinfer",
                    "--mamba-scheduler-strategy", "extra_buffer", "--mamba-track-interval", "64",
                    "--mamba-full-memory-ratio", str(cfg.get("mamba_full_memory_ratio", 0.5)),
                    "--random-seed", str(cfg.get("seed", 2026))]
    if n.get("numa_nodes"):
        command += ["--numa-node"] + list(map(str, n["numa_nodes"]))
    if n.get("ib_device"):
        command += ["--disaggregation-ib-device", n["ib_device"]]
    if n.get("ucx_net_devices"):
        env["UCX_NET_DEVICES"] = n["ucx_net_devices"]
    return {"node_id": n["node_id"], "engine_id": n["engine_id"], "environment": env,
            "command": command, "local_run_dir": str(local),
            "host_capacity_gib_per_direction": {
                "d2p_source": cfg["d2p_host_gib_per_rank"] * cfg["tp_size"] if n["role"] == "decode" and cfg.get("d2p_host_staging", True) else 0,
                "p2d_source": cfg["p2d_host_gib_per_rank"] * cfg["tp_size"] if n["role"] == "prefill" else 0}}


def router_plan(cfg):
    router = cfg["router"]
    n = node_for(cfg, router["node_id"])
    env = common_env(cfg)
    env.update({
        "SGLANG_AGENTIC_MULTINODE_NODE_ID": n["node_id"],
        "SGLANG_AGENTIC_MULTINODE_ENGINE_ID": router["engine_id"],
        "SGLANG_AGENTIC_MULTINODE_ROLE": "router",
        "SGLANG_AGENTIC_MULTINODE_HOST_IP": n["host_ip"],
        "SGLANG_HOST_IP": n["host_ip"],
        "SGLANG_AGENTIC_KV_ENGINE_ID": router["engine_id"],
        "CUDA_VISIBLE_DEVICES": "",
    })
    command = [cfg["python"], "-u"]
    if cfg.get("router_profile_imports", False):
        command += ["-X", "importtime"]
    command += [str(Path(cfg["slime_root"]) / "examples/pd/launch_late_binding_router.py"),
               "--pd-disaggregation", "--policy", "random", "--host", "0.0.0.0",
               "--port", str(router["port"]), "--prometheus-port", str(router["metrics_port"]),
               "--health-check-timeout-secs", "60", "--health-failure-threshold", "10"]
    for worker in cfg["nodes"]:
        url = "http://{}:{}".format(worker["host_ip"], worker["port"])
        if worker["role"] == "prefill":
            command += ["--prefill", url, str(worker["bootstrap_port"])]
        else:
            command += ["--decode", url]
    return {"node_id": n["node_id"], "engine_id": router["engine_id"], "environment": env,
            "command": command, "local_run_dir": str(Path(cfg["local_root"]) / cfg["run_id"] / router["engine_id"])}


def control_plan(cfg):
    if cfg.get("control_backend") != "tcp":
        raise ValueError("start-control requires control_backend=tcp")
    broker = cfg["control"]
    host = node_for(cfg, broker["node_id"])["host_ip"]
    env = common_env(cfg)
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [cfg["python"], "-m", "sglang.srt.disaggregation.agentic_control_server",
               "--listen", host, "--port", str(broker["port"]),
               "--tp-port", str(broker["tp_port"]), "--run-id", cfg["run_id"]]
    for node in cfg["nodes"]:
        command += ["--" + node["role"] + "-group", node["engine_id"]]
    return {"node_id": broker["node_id"], "engine_id": "control-broker", "environment": env,
            "command": command,
            "local_run_dir": str(Path(cfg["local_root"]) / cfg["run_id"] / "control-broker")}


_control_clients = {}
_prewarm_controls = {}


def control_client(cfg):
    """Launcher-only persistent client; startup/readiness never touches NFS."""
    key = fingerprint(cfg)
    if key not in _control_clients:
        sys.path.insert(0, str(Path(cfg["sglang_root"]) / "python"))
        from sglang.srt.disaggregation.agentic_control_rpc import ControlRPCClient
        broker = cfg["control"]
        client = ControlRPCClient(
            (node_for(cfg, broker["node_id"])["host_ip"], broker["port"]),
            run_id=cfg["run_id"], token=broker["token"],
        )
        client.wait_ready()
        _control_clients[key] = client
    return _control_clients[key]


def plan(cfg):
    return {"status": "DEVELOPMENT_PLAN_NOT_RUNTIME_ACCEPTANCE", "config_sha256": fingerprint(cfg),
            "workers": [worker_plan(cfg, n) for n in cfg["nodes"]],
            "router": router_plan(cfg),
            "requirements": ["same model/revision/dtype/page layout on all ranks",
                ("persistent TCP control broker; no NFS or TP mailbox files"
                 if cfg.get("control_backend") == "tcp"
                 else "shared POSIX metadata, coherent hardlink/rename/O_EXCL/distributed flock"),
                "source-local DRAM extents exported by remote RDMA Host backend",
                "no NFS KV payloads; registered Host owner stays alive until remote fence",
                "TP groups must stay within a node; equal TP across P and D",
                "NIXL/UCX GPU and Host RDMA verified separately; dynamic transport ports reachable"]}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def fs_publish(cfg, n):
    # A tiny artifact only, no GPU initialization and no capacity-sized allocation.
    event_control = cfg.get("control_backend") == "tcp"
    root = ((Path(cfg["local_root"]) / cfg["run_id"]) if event_control else control_dir(cfg)) / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    record = {"node_id": n["node_id"], "config_sha256": fingerprint(cfg),
              "hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "nonce": uuid.uuid4().hex, "time": time.time()}
    if event_control:
        # Node-local launcher identity, not TP/control communication.
        atomic_json(root / (n["node_id"] + ".json"), record)
        return record
    probe = root / (n["node_id"] + ".exclusive")
    fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(record["nonce"])
        out.flush()
        os.fsync(out.fileno())
    os.link(probe, root / (n["node_id"] + ".hardlink"))
    atomic_json(root / (n["node_id"] + ".json"), record)
    return record


def fs_verify(cfg):
    if cfg.get("control_backend") == "tcp":
        raise ValueError("TCP mode never performs a shared-filesystem verification")
    records = [json.loads((control_dir(cfg) / "preflight" / (n["node_id"] + ".json")).read_text())
               for n in cfg["nodes"]]
    if any(r["config_sha256"] != fingerprint(cfg) for r in records):
        raise ValueError("nodes published different configuration hashes")
    if len({r["boot_id"] for r in records}) != len(records):
        raise ValueError("expected one TP group per distinct host; duplicate host boot ID")
    for r in records:
        probe = control_dir(cfg) / "preflight" / (r["node_id"] + ".exclusive")
        if probe.read_text() != r["nonce"]:
            raise ValueError("inconsistent remote create/rename visibility")
        link = probe.with_suffix(".hardlink")
        if link.read_text() != r["nonce"]:
            raise ValueError("inconsistent remote hardlink visibility")
        try:
            os.link(probe, link)
        except FileExistsError:
            pass
        else:
            raise RuntimeError("remote hardlink election overwrote an existing owner")
        try:
            fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
            raise RuntimeError("remote O_EXCL fence failed; metadata filesystem unsafe")
    return {"metadata_visible": True, "records": records,
            "note": "Visibility only, not proof of remote flock or data-plane correctness; run lock hold/probe across nodes."}


def lock_check(cfg, hold):
    if cfg.get("control_backend") == "tcp":
        raise ValueError("TCP mode has no distributed file locks")
    path = control_dir(cfg) / "preflight/distributed.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if hold:
                raise RuntimeError("another lock holder exists")
            return {"remote_lock_observed": True}
        if not hold:
            raise RuntimeError("lock unexpectedly acquirable: holder absent or filesystem locking unsafe")
        print("LOCK_HELD: run fs-lock-probe on a DIFFERENT host within 30 seconds", flush=True)
        time.sleep(30)
    return {"hold_completed": True}


def capability_check(cfg, env):
    code = ("import json; from sglang.srt.disaggregation.agentic_multinode import capabilities, load_multinode_config; "
            "load_multinode_config(); print(json.dumps(capabilities()))")
    result = subprocess.run([cfg["python"], "-c", code], env=env, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise RuntimeError("Multi-node runtime gate unavailable; no GPU launched. " + result.stderr[-2000:])
    caps = json.loads(result.stdout)
    if cfg.get("control_backend") == "tcp" and not caps.get("socket_control_engine", False):
        raise RuntimeError("Socket control engine integration/audit is incomplete; no GPU launched")
    missing = sorted(REQUIRED_CAPABILITIES - set(caps.get("features", [])))
    if not caps.get("integrated") or missing:
        raise RuntimeError("Multi-node engine not integrated/audited; no GPU launched. Missing: " + repr(missing))


def runtime_env(p):
    # Do not inherit another experiment's engine/arena/HiCache settings. Network
    # library settings such as UCX/NCCL remain explicit operator responsibility.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SGLANG_", "PD_")) and k != "HOST_IP"}
    env.update(p["environment"])
    # SSH does not activate Conda. The correct interpreter alone is not enough:
    # FlashInfer launches ninja and other subprocesses by executable name.
    env["PATH"] = str(Path(p["command"][0]).parent) + os.pathsep + env.get("PATH", "")
    return env


def process_identity(cfg, component):
    return fingerprint(cfg) + ":" + component


def local_node_check(cfg, node_id):
    root = (Path(cfg["local_root"]) / cfg["run_id"] if cfg.get("control_backend") == "tcp"
            else control_dir(cfg))
    manifest = json.loads((root / "preflight" / (node_id + ".json")).read_text())
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if manifest["boot_id"] != current_boot or manifest["config_sha256"] != fingerprint(cfg):
        raise RuntimeError("wrong node/configuration or host rebooted; republish under a new run ID")


def save_launch(cfg, p):
    directory = Path(p["local_run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    # Engine epochs are not restart-transparent: old ledgers can reference the
    # previous process's DRAM/GPU registrations. Never silently reuse that run.
    fd = os.open(directory / "launch.once", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(fingerprint(cfg))
    atomic_json(directory / "config.json", cfg)
    atomic_json(directory / "launch.json", p)
    revisions = {}
    for key in ("sglang_root", "slime_root"):
        result = subprocess.run(["git", "-C", cfg[key], "rev-parse", "HEAD"],
                                universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=10)
        revisions[key] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    atomic_json(directory / "revisions.json", revisions)


def wait_ready(cfg, timeout=1800):
    """Non-generative model_info barrier, never /health-generated Prefill."""
    deadline = time.monotonic() + timeout
    pending = {n["engine_id"]: "http://{}:{}/model_info".format(n["host_ip"], n["port"])
               for n in cfg["nodes"]}
    errors = {}
    # Never inherit HTTP proxies for private worker probes.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while pending and time.monotonic() < deadline:
        for engine, url in list(pending.items()):
            try:
                with opener.open(url, timeout=2) as response:
                    if response.status != 200:
                        raise RuntimeError("HTTP " + str(response.status))
                    json.loads(response.read())
                del pending[engine]
                errors.pop(engine, None)
            except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
                errors[engine] = str(exc)
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError("model_info barrier timed out: " + repr(errors))
    return {"workers_ready": True, "note": "HTTP/model readiness, not KV transfer correctness or throughput acceptance"}


def host_prewarm_status(cfg):
    """Start/wait the existing rank-local CUDA registration barrier, no KV I/O."""
    root = control_dir(cfg) / "host-register-prewarm"
    controls = None
    if cfg.get("control_backend") == "tcp":
        key = fingerprint(cfg)
        if key not in _prewarm_controls:
            client = control_client(cfg)
            from sglang.srt.disaggregation.agentic_control_store import ControlKV, record_namespace
            controls = ControlKV(record_namespace("prewarm", str(root)), client=client)
            controls.call("put", "start", {"run_id": cfg["run_id"]})
            _prewarm_controls[key] = controls
        controls = _prewarm_controls[key]
    else:
        root.mkdir(parents=True, exist_ok=True)
        (root / "start").touch(exist_ok=True)
    expected = {
        f'{n["role"]}-{n["engine_id"]}-rank-{rank}'
        for n in cfg["nodes"] for rank in range(cfg["tp_size"])
    }
    completed = set()
    def valid_record(record, participant):
        identity = f'{record["role"]}-{record["engine_id"]}-rank-{record["tp_rank"]}'
        no_source = record['role'] == 'decode' and not cfg.get('d2p_host_staging', True)
        size_ok = (record['registered_bytes'] == 0 and record.get('arena_count') == 0
                   if no_source else record['registered_bytes'] > 0)
        return identity == participant and size_ok
    for participant in expected:
        if controls is not None:
            failure = controls.get("failed/" + participant + ".json")
            if failure is not None:
                raise RuntimeError("Host prewarm failed: " + repr(failure))
            record = controls.get("complete/" + participant + ".json")
            if record is not None:
                if not valid_record(record, participant):
                    raise RuntimeError("invalid Host prewarm completion: " + participant)
                completed.add(participant)
            continue
        failure = root / "failed" / (participant + ".json")
        if failure.exists():
            raise RuntimeError("Host prewarm failed: " + failure.read_text())
        path = root / "complete" / (participant + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if not valid_record(record, participant):
                raise RuntimeError("invalid Host prewarm completion: " + str(path))
            completed.add(participant)
    return {"ready": completed == expected, "completed": len(completed),
            "expected": len(expected), "pending": sorted(expected - completed)}


def wait_host_prewarm(cfg, timeout=1800):
    deadline = time.monotonic() + timeout
    while True:
        status = host_prewarm_status(cfg)
        if status["ready"]:
            return status
        if time.monotonic() >= deadline:
            raise RuntimeError("Host prewarm timed out: " + repr(status))
        time.sleep(1)


def check_listen_ports(cfg, p, is_worker):
    """Fail before launch if our listening ports already belong to a service.

    Probes are closed before model initialization, so this is a conflict check,
    not a reservation against another process concurrently starting up.
    """
    if is_worker:
        n = next(n for n in cfg["nodes"] if n["engine_id"] == p["engine_id"])
        ports = [n["port"], n["bootstrap_port"], n["reverse_bootstrap_port"]]
    elif p["engine_id"] == cfg["router"]["engine_id"]:
        ports = [cfg["router"]["port"], cfg["router"]["metrics_port"]]
    elif p["engine_id"] == "control-broker":
        ports = [cfg["control"]["port"], cfg["control"]["tp_port"]]
    else:
        return
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError("local TCP port {} is unavailable; refusing to reuse another experiment's service".format(port)) from exc


def start_component(cfg, p, *, is_worker=False):
    env = runtime_env(p)
    if p["engine_id"] != "control-broker":
        capability_check(cfg, env)
        if cfg.get("control_backend") == "tcp":
            control_client(cfg).call("system", "describe")
        else:
            fs_verify(cfg)
    local_node_check(cfg, p["node_id"])
    check_listen_ports(cfg, p, is_worker)
    if is_worker:
        model = json.loads((Path(cfg["model_path"]) / "config.json").read_text())
        validate_launch_model(cfg, model)
    save_launch(cfg, p)
    print("Starting {} on {}; log={}".format(p["engine_id"], p["node_id"], Path(p["local_run_dir"]) / "service.log"), flush=True)
    return supervise(p["command"], env, p["local_run_dir"], process_identity(cfg, p["engine_id"]))


def validate_launch_model(cfg, model):
    """Explicit model layout validation; hybrid transfer is separately gated."""
    family = cfg.get("model_family", "qwen3")
    if family == "qwen35_moe":
        from qwen35_swe import validate_model
        validate_model(model)
        return
    if model.get("model_type") != family:
        raise ValueError("model config disagrees with explicit model_family")
    if family == "qwen3":
        return
    if family != "minimax_m2" or model.get("architectures") != ["MiniMaxM2ForCausalLM"]:
        raise ValueError("unsupported model architecture")
    layers = model.get("num_hidden_layers")
    if type(layers) is not int or layers <= 0 or model.get("attn_type_list") != [1] * layers:
        raise ValueError("MiniMax adapter requires full Attention in every layer")
    tp = cfg["tp_size"]
    for field in ("num_attention_heads", "num_key_value_heads", "num_local_experts"):
        value = model.get(field)
        if type(value) is not int or value <= 0 or value % tp:
            raise ValueError("MiniMax {} must partition evenly across TP".format(field))
    if model.get("quantization_config", {}).get("quant_method") != "fp8":
        raise ValueError("MiniMax-M2.7 entry point expects the official FP8 checkpoint")
    block = model["quantization_config"].get("weight_block_size")
    # EP=TP: each rank holds whole experts, so MoE TP=1, not 1536/8=192.
    if block != [128, 128] or any(type(model.get(k)) is not int or model[k] <= 0
                                or model[k] % 128 for k in ("intermediate_size", "hidden_size")):
        raise ValueError("MiniMax FP8 expert dimensions must align to 128x128 blocks")


def start_worker(cfg, n):
    return start_component(cfg, worker_plan(cfg, n), is_worker=True)


def workload_plan(cfg):
    if not cfg.get("workload_command"):
        raise ValueError("set workload_command argv explicitly; launcher does not invent datasets or sampling settings")
    p = router_plan(cfg)
    p["engine_id"] = "workload"
    p["command"] = cfg["workload_command"]
    p["local_run_dir"] = str(Path(cfg["local_root"]) / cfg["run_id"] / "workload")
    allowed = {"PD_DATA_ROOT", "PD_SWE_RUN_ID", "PD_SWE_PROGRESS_FILE",
               "PD_MODEL_HTTP_TRANSPORT", "SLIME_HTTP_READ_TIMEOUT_SECONDS", "MIN_P"}
    for key, value in cfg.get("workload_environment", {}).items():
        if key not in allowed or not isinstance(value, str):
            raise ValueError("unsupported workload environment setting: " + key)
        p["environment"][key] = value
    return p


def smoke_plan(cfg, config_path):
    p = router_plan(cfg)
    p["engine_id"] = "smoke"
    p["local_run_dir"] = str(Path(cfg["local_root"]) / cfg["run_id"] / "smoke")
    p["command"] = [cfg["python"], str(Path(__file__).with_name("multinode_smoke.py")),
                    "--config", str(Path(config_path).resolve()), "--output", p["local_run_dir"]]
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "fs-publish", "fs-verify", "fs-lock-hold", "fs-lock-probe",
                                         "start-control", "start-worker", "start-router", "run-workload", "smoke", "wait-ready", "stop", "status"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-id")
    parser.add_argument("--component", choices=("worker", "router", "workload", "smoke", "control"), default="worker")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if (args.action in {"fs-publish", "start-worker"}
            or args.action in {"stop", "status"} and args.component == "worker") and not args.node_id:
        parser.error("this action requires --node-id")
    n = node_for(cfg, args.node_id) if args.node_id else None
    if args.action == "plan":
        result = plan(cfg)
    elif args.action == "fs-publish":
        result = fs_publish(cfg, n)
    elif args.action == "fs-verify":
        result = fs_verify(cfg)
    elif args.action.startswith("fs-lock-"):
        result = lock_check(cfg, args.action == "fs-lock-hold")
    elif args.action == "start-worker":
        return start_worker(cfg, n)
    elif args.action == "start-control":
        return start_component(cfg, control_plan(cfg))
    elif args.action == "start-router":
        capability_check(cfg, runtime_env(router_plan(cfg)))
        wait_ready(cfg, args.timeout)
        wait_host_prewarm(cfg, args.timeout)
        return start_component(cfg, router_plan(cfg))
    elif args.action == "run-workload":
        capability_check(cfg, runtime_env(workload_plan(cfg)))
        wait_ready(cfg, args.timeout)
        wait_host_prewarm(cfg, args.timeout)
        return start_component(cfg, workload_plan(cfg))
    elif args.action == "smoke":
        capability_check(cfg, runtime_env(smoke_plan(cfg, args.config)))
        wait_ready(cfg, args.timeout)
        wait_host_prewarm(cfg, args.timeout)
        return start_component(cfg, smoke_plan(cfg, args.config))
    elif args.action == "wait-ready":
        result = wait_ready(cfg, args.timeout)
    elif args.action in {"stop", "status"}:
        p = control_plan(cfg) if args.component == "control" else worker_plan(cfg, n) if args.component == "worker" else router_plan(cfg) if args.component == "router" else smoke_plan(cfg, args.config) if args.component == "smoke" else workload_plan(cfg)
        local_node_check(cfg, p["node_id"])
        result = process_control(p["local_run_dir"], process_identity(cfg, p["engine_id"]), args.action)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
