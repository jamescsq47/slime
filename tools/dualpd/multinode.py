#!/usr/bin/env python3
"""Explicit, fail-closed remote entry point for the multi-node development V1.

Plan/preflight use only stdlib. Engine start is deliberately capability-gated:
source-local Host extents must have a remotely usable data-plane implementation,
not merely a shared pathname. Existing single-node launchers are untouched.
"""
import argparse
import fcntl
import hashlib
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
    for key in ("run_id",):
        if not NAME.fullmatch(cfg.get(key, "")):
            raise ValueError("invalid " + key)
    for key in ("control_root", "local_root", "sglang_root", "slime_root", "model_path", "python"):
        absolute(cfg.get(key), key)
    if cfg["control_root"].startswith(("/dev/shm/", "/tmp/")):
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
    if type(cfg.get("slow_congestion_recompute", True)) is not bool:
        raise ValueError("slow_congestion_recompute must be boolean")
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
    return {
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
        "SGLANG_AGENTIC_KV_LEDGER_PATH": str(root / "lifecycle.json"),
        "SGLANG_AGENTIC_KV_STAGING_LEDGER_PATH": str(root / "d2p.json"),
        "SGLANG_AGENTIC_KV_P2D_STAGING_LEDGER_PATH": str(root / "p2d.json"),
        "SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR": str(root / "early-claims"),
        "SGLANG_AGENTIC_KV_METADATA_DIR": str(root / "snapshot-metadata"),
        "SGLANG_AGENTIC_KV_PREFILL_LOAD_PATH": str(root / "early-claims/prefill-loads.json"),
        "SGLANG_AGENTIC_KV_REGISTER_PREWARM_DIR": str(root / "host-register-prewarm"),
        "SGLANG_AGENTIC_KV_LIFECYCLE": "true",
        "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY": "true",
        "SGLANG_AGENTIC_KV_D_HOSTLESS": "true",
        "SGLANG_AGENTIC_KV_HOST_STAGING": "true",
        "SGLANG_AGENTIC_KV_P2D_HOST_STAGING": "true",
        "SGLANG_AGENTIC_KV_EARLY_CLAIM": "1",
        "SGLANG_AGENTIC_KV_RELAY_ENABLED": "false",
        "SGLANG_PD_DECODE_ENABLE_RADIX_CACHE": "true",
        "SGLANG_PD_P_READY_BACKPRESSURE_MODE": "disabled",
        "SGLANG_PD_P_READY_REQUEST_CAP": "0",
        "SGLANG_PD_LATE_BIND_FORCE_LEGACY_LOADS": "1",
        "SGLANG_AGENTIC_KV_P2D_SPILL_DELAY_SECONDS": "0.5",
        "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "true",
        "SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB": str(cfg["d2p_host_gib_per_rank"]),
        "SGLANG_AGENTIC_MULTINODE_P2D_HOST_GIB": str(cfg["p2d_host_gib_per_rank"]),
        "SGLANG_AGENTIC_KV_TP_SIZE": str(cfg["tp_size"]),
        "SGLANG_AGENTIC_KV_PREFILL_DOMAIN_COUNT": str(sum(n["role"] == "prefill" for n in cfg["nodes"])),
        "SGLANG_PD_LATE_BIND_NUMA_DOMAINS": "0",
        "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS": "0",
        "SGLANG_PD_LATE_BIND_GLOBAL_DECODE": "1",
        "SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION": "1.0",
        "SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD": str(cfg["fast_tool_seconds"]),
        "SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT": str(cfg["direct_admission_seconds"]),
        "SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE": str(cfg.get("slow_congestion_recompute", True)).lower(),
        "SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH": str(cfg.get("slow_congestion_high", 32)),
        "SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW": str(cfg.get("slow_congestion_low", 8)),
        "PYTHONPATH": str(Path(cfg["sglang_root"]) / "python") + ":" + cfg["slime_root"],
    }


def worker_plan(cfg, n):
    env = common_env(cfg)
    local = Path(cfg["local_root"]) / cfg["run_id"] / n["engine_id"]
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
    if n.get("numa_nodes"):
        command += ["--numa-node"] + list(map(str, n["numa_nodes"]))
    if n.get("ib_device"):
        command += ["--disaggregation-ib-device", n["ib_device"]]
    return {"node_id": n["node_id"], "engine_id": n["engine_id"], "environment": env,
            "command": command, "local_run_dir": str(local),
            "host_capacity_gib_per_direction": {
                "d2p_source": cfg["d2p_host_gib_per_rank"] * cfg["tp_size"] if n["role"] == "decode" else 0,
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
    command = [cfg["python"], str(Path(cfg["slime_root"]) / "examples/pd/launch_late_binding_router.py"),
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


def plan(cfg):
    return {"status": "DEVELOPMENT_PLAN_NOT_RUNTIME_ACCEPTANCE", "config_sha256": fingerprint(cfg),
            "workers": [worker_plan(cfg, n) for n in cfg["nodes"]],
            "router": router_plan(cfg),
            "requirements": ["same model/revision/dtype/page layout on all ranks",
                "shared POSIX metadata, coherent hardlink/rename/O_EXCL/distributed flock",
                "source-local DRAM extents exported by remote RDMA Host backend",
                "no NFS KV payloads; registered Host owner stays alive until remote fence",
                "TP groups must stay within a node; equal TP across P and D",
                "NIXL/UCX GPU and Host RDMA verified separately; dynamic transport ports reachable"]}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with tmp.open("x") as out:
            json.dump(value, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def fs_publish(cfg, n):
    # A tiny artifact only, no GPU initialization and no capacity-sized allocation.
    root = control_dir(cfg) / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    record = {"node_id": n["node_id"], "config_sha256": fingerprint(cfg),
              "hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "nonce": uuid.uuid4().hex, "time": time.time()}
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
    missing = sorted(REQUIRED_CAPABILITIES - set(caps.get("features", [])))
    if not caps.get("integrated") or missing:
        raise RuntimeError("Multi-node engine not integrated/audited; no GPU launched. Missing: " + repr(missing))


def runtime_env(p):
    # Do not inherit another experiment's engine/arena/HiCache settings. Network
    # library settings such as UCX/NCCL remain explicit operator responsibility.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SGLANG_", "PD_")) and k != "HOST_IP"}
    env.update(p["environment"])
    return env


def process_identity(cfg, component):
    return fingerprint(cfg) + ":" + component


def local_node_check(cfg, node_id):
    manifest = json.loads((control_dir(cfg) / "preflight" / (node_id + ".json")).read_text())
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
            except (OSError, ValueError, RuntimeError) as exc:
                errors[engine] = str(exc)
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError("model_info barrier timed out: " + repr(errors))
    return {"workers_ready": True, "note": "HTTP/model readiness, not KV transfer correctness or throughput acceptance"}


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
    capability_check(cfg, env)
    fs_verify(cfg)
    local_node_check(cfg, p["node_id"])
    check_listen_ports(cfg, p, is_worker)
    if is_worker:
        model = json.loads((Path(cfg["model_path"]) / "config.json").read_text())
        if model.get("model_type") != "qwen3":
            raise ValueError("multi-node V1 launcher supports dense Qwen3 only; Mamba/MLA/MoE are not validated")
    save_launch(cfg, p)
    print("Starting {} on {}; log={}".format(p["engine_id"], p["node_id"], Path(p["local_run_dir"]) / "service.log"), flush=True)
    return supervise(p["command"], env, p["local_run_dir"], process_identity(cfg, p["engine_id"]))


def start_worker(cfg, n):
    return start_component(cfg, worker_plan(cfg, n), is_worker=True)


def workload_plan(cfg):
    if not cfg.get("workload_command"):
        raise ValueError("set workload_command argv explicitly; launcher does not invent datasets or sampling settings")
    p = router_plan(cfg)
    p["engine_id"] = "workload"
    p["command"] = cfg["workload_command"]
    p["local_run_dir"] = str(Path(cfg["local_root"]) / cfg["run_id"] / "workload")
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
                                         "start-worker", "start-router", "run-workload", "smoke", "wait-ready", "stop", "status"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-id")
    parser.add_argument("--component", choices=("worker", "router", "workload", "smoke"), default="worker")
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
    elif args.action == "start-router":
        capability_check(cfg, runtime_env(router_plan(cfg)))
        wait_ready(cfg, args.timeout)
        return start_component(cfg, router_plan(cfg))
    elif args.action == "run-workload":
        capability_check(cfg, runtime_env(workload_plan(cfg)))
        wait_ready(cfg, args.timeout)
        return start_component(cfg, workload_plan(cfg))
    elif args.action == "smoke":
        capability_check(cfg, runtime_env(smoke_plan(cfg, args.config)))
        wait_ready(cfg, args.timeout)
        return start_component(cfg, smoke_plan(cfg, args.config))
    elif args.action == "wait-ready":
        result = wait_ready(cfg, args.timeout)
    elif args.action in {"stop", "status"}:
        p = worker_plan(cfg, n) if args.component == "worker" else router_plan(cfg) if args.component == "router" else smoke_plan(cfg, args.config) if args.component == "smoke" else workload_plan(cfg)
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
