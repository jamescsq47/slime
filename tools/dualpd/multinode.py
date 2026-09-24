#!/usr/bin/env python3
"""Fail-closed launcher for DualPD's TCP-controlled multi-node runtime.

Runtime lifecycle state is exchanged through an in-memory TCP relay. Shared
filesystems are never used for control or KV payloads; ``local_root`` stores
only logs and immutable launch records on the node running a component.
"""

import argparse
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
import urllib.request
import uuid

from process_supervisor import control as process_control, supervise


NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def absolute(value, field):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(field + " must be an absolute path")
    if ".." in Path(value).parts or value in {"/", "/dev/shm", "/tmp"}:
        raise ValueError(field + " needs a dedicated directory/path")
    return value


def node_for(cfg, node_id):
    return next(n for n in cfg["nodes"] if n["node_id"] == node_id)


def _port(value, field):
    if type(value) is not int or not 1024 <= value <= 65535:
        raise ValueError(field + " must be an unprivileged TCP port")
    return value


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("model_family", "qwen3") not in {"qwen3", "qwen35_moe"}:
        raise ValueError("unsupported multi-node model family")
    if not NAME.fullmatch(cfg.get("run_id", "")):
        raise ValueError("invalid run_id")
    for key in ("local_root", "sglang_root", "slime_root", "model_path", "python"):
        absolute(cfg.get(key), key)
    # Kept only so an old JSON file gives a precise error instead of silently
    # re-enabling NFS control.
    if cfg.get("legacy_shared_control", False):
        raise ValueError("legacy shared-filesystem control is disabled in V2")

    tp = cfg.get("tp_size")
    if type(tp) is not int or tp not in (1, 2, 4, 8):
        raise ValueError("matching TP=1/2/4/8 is required")
    nodes = cfg.get("nodes", [])
    if len(nodes) < 2 or {n.get("role") for n in nodes} != {"prefill", "decode"}:
        raise ValueError("at least one P and one D node required")
    if len(nodes) != 2:
        raise ValueError("V2 launcher currently requires exactly one P and one D TP group")
    for field in ("node_id", "engine_id", "host_ip"):
        values = [n.get(field) for n in nodes]
        if len(set(values)) != len(values):
            raise ValueError("one engine group per host; duplicate " + field)
    worker_ports = ("port", "bootstrap_port", "reverse_bootstrap_port")
    for n in nodes:
        for field in ("node_id", "engine_id"):
            if not NAME.fullmatch(n.get(field, "")):
                raise ValueError("invalid " + field)
        ip = ipaddress.ip_address(n["host_ip"])
        if ip.version != 4 or ip.is_loopback or ip.is_unspecified or ip.is_multicast or ip.is_link_local:
            raise ValueError("host_ip must be an explicit routable IPv4 address")
        gpus = n.get("gpus", [])
        if len(gpus) != tp or len(set(gpus)) != tp or any(type(g) is not int or g < 0 for g in gpus):
            raise ValueError("each node must list exactly TP distinct physical GPUs")
        numas = n.get("numa_nodes", [])
        if numas and (len(numas) != tp or any(type(x) is not int or x < 0 for x in numas)):
            raise ValueError("numa_nodes must be empty or exactly one NUMA ID per TP rank")
        if not 0 < n.get("mem_fraction_static", 0) < 1:
            raise ValueError("explicit group-wide mem_fraction_static required")
        for field in worker_ports:
            _port(n.get(field), n["node_id"] + "." + field)
        if len({n[p] for p in worker_ports}) != len(worker_ports):
            raise ValueError("listener ports must be distinct on each host")

    for key in ("d2p_host_gib_per_rank", "p2d_host_gib_per_rank"):
        if type(cfg.get(key)) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("positive " + key + " required")
    for key in (
        "page_size",
        "context_length",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "decode_growth_tokens",
    ):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError("positive integer " + key + " required")
    for key in ("fast_tool_seconds", "direct_admission_seconds"):
        if type(cfg.get(key)) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("positive " + key + " required")
    router = cfg.get("router", {})
    if router.get("node_id") not in {n["node_id"] for n in nodes}:
        raise ValueError("router.node_id must identify one configured node")
    if not NAME.fullmatch(router.get("engine_id", "")) or router["engine_id"] in {n["engine_id"] for n in nodes}:
        raise ValueError("router.engine_id must be unique")
    router_node = node_for(cfg, router["node_id"])
    occupied = {router_node[k] for k in worker_ports}
    for key in ("port", "metrics_port"):
        value = _port(router.get(key), "router." + key)
        if value in occupied:
            raise ValueError("router port collides with a local listener")
        occupied.add(value)

    control = cfg.get("group_control")
    if not isinstance(control, dict) or control.get("enabled") is not True:
        raise ValueError("group_control.enabled=true is required for multi-node V2")
    if control.get("node_id") not in {n["node_id"] for n in nodes}:
        raise ValueError("group_control.node_id must identify one configured node")
    control["listen_host"] = control.get("listen_host", "0.0.0.0")
    if control["listen_host"] not in {"0.0.0.0", "::"}:
        ipaddress.ip_address(control["listen_host"])
    control["advertise_host"] = control.get("advertise_host") or node_for(cfg, control["node_id"])["host_ip"]
    address = ipaddress.ip_address(control["advertise_host"])
    if address.is_loopback or address.is_unspecified or address.is_multicast or address.is_link_local:
        raise ValueError("group_control.advertise_host must be remotely reachable")
    control["port"] = _port(control.get("port"), "group_control.port")
    if not isinstance(control.get("token"), str) or len(control["token"]) < 16:
        raise ValueError("group_control.token must contain at least 16 characters")
    relay_node = node_for(cfg, control["node_id"])
    local_ports = {relay_node[k] for k in worker_ports}
    if control["node_id"] == router["node_id"]:
        local_ports.update((router["port"], router["metrics_port"]))
    if control["port"] in local_ports:
        raise ValueError("group_control.port collides with another local listener")

    workload = cfg.get("workload_command", [])
    if not isinstance(workload, list) or any(not isinstance(x, str) or not x for x in workload):
        raise ValueError("workload_command must be an argv list, not a shell string")
    return cfg


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def control_endpoint(cfg):
    control = cfg["group_control"]
    return "tcp://{}:{}".format(control["advertise_host"], control["port"])


def link_id(cfg):
    # V2 currently has one P<->D link. Multiple links can be emitted later
    # without changing rank identity or the relay wire protocol.
    p = next(n for n in cfg["nodes"] if n["role"] == "prefill")
    d = next(n for n in cfg["nodes"] if n["role"] == "decode")
    return p["engine_id"] + "--" + d["engine_id"]


def group_env(cfg, node):
    peer = next(item for item in cfg["nodes"] if item["role"] != node["role"])
    prefill = next(item for item in cfg["nodes"] if item["role"] == "prefill")
    return {
        "SGLANG_AGENTIC_GROUP_ENDPOINT": control_endpoint(cfg),
        "SGLANG_AGENTIC_GROUP_RUN_ID": cfg["run_id"],
        "SGLANG_AGENTIC_GROUP_TOKEN": cfg["group_control"]["token"],
        "SGLANG_AGENTIC_GROUP_GROUP_ID": link_id(cfg),
        "SGLANG_AGENTIC_GROUP_ENDPOINT_GROUP": node["engine_id"],
        "SGLANG_AGENTIC_GROUP_ENDPOINT_ROLE": node["role"],
        "SGLANG_AGENTIC_GROUP_PEER_GROUP": peer["engine_id"],
        "SGLANG_AGENTIC_GROUP_PEER_ROLE": peer["role"],
        "SGLANG_AGENTIC_GROUP_COORDINATOR_GROUP": prefill["engine_id"],
        "SGLANG_AGENTIC_GROUP_SIZE": str(cfg["tp_size"]),
    }


def common_env(cfg):
    # No control_root, marker directory, runtime ledger, or NFS polling knobs.
    env = {
        # Workers are commonly launched through a non-interactive SSH shell.
        # Keep JIT build tools (notably ninja) from the selected environment
        # available even when that shell does not activate conda.
        "PATH": str(Path(cfg["python"]).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
        "SGLANG_AGENTIC_MULTINODE_ENABLED": "1",
        "SGLANG_AGENTIC_MULTINODE_PEER_TP_SIZE": str(cfg["tp_size"]),
        "PD_INFERENCE_RETURN_LOGPROB": "false",
        "SGLANG_AGENTIC_KV_LIFECYCLE": "true",
        "SGLANG_AGENTIC_MEMORY_AUTHORITY_V2": "true",
        # V2 does not set any of the V1 storage/marker/ledger switches.  Its
        # scheduler startup returns before those managers are constructed.
        "SGLANG_PD_DECODE_ENABLE_RADIX_CACHE": "true",
        "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "true",
        "SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB": str(cfg["d2p_host_gib_per_rank"]),
        "SGLANG_AGENTIC_MULTINODE_P2D_HOST_GIB": str(cfg["p2d_host_gib_per_rank"]),
        "SGLANG_AGENTIC_MULTINODE_HOST_BACKEND": "memfd",
        "SGLANG_AGENTIC_KV_TP_SIZE": str(cfg["tp_size"]),
        "SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD": str(cfg["fast_tool_seconds"]),
        "SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT": str(cfg["direct_admission_seconds"]),
        "SGLANG_AGENTIC_MULTINODE_DIRECT_WINDOW_SECONDS": str(cfg["fast_tool_seconds"]),
        "SGLANG_AGENTIC_MULTINODE_DIRECT_ADMISSION_SECONDS": str(cfg["direct_admission_seconds"]),
        "SGLANG_AGENTIC_MULTINODE_DECODE_GROWTH_TOKENS": str(cfg["decode_growth_tokens"]),
        "PYTHONPATH": str(Path(cfg["sglang_root"]) / "python") + ":" + cfg["slime_root"],
    }
    if cfg.get("model_family") == "qwen35_moe":
        env.update(
            {
                "SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID": "1",
                "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY": "true",
                "SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT": "true",
                "SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED": "true",
                "SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION": "true",
            }
        )
    if cfg.get("cuda_home"):
        env["CUDA_HOME"] = str(Path(cfg["cuda_home"]))
    return env


def worker_plan(cfg, node):
    env = common_env(cfg)
    env.update(group_env(cfg, node))
    local = Path(cfg["local_root"]) / cfg["run_id"] / node["engine_id"]
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, node["gpus"])),
        "SGLANG_HOST_IP": node["host_ip"],
        "SGLANG_AGENTIC_MULTINODE_NODE_ID": node["node_id"],
        "SGLANG_AGENTIC_MULTINODE_ENGINE_ID": node["engine_id"],
        "SGLANG_AGENTIC_MULTINODE_ROLE": node["role"],
        "SGLANG_AGENTIC_MULTINODE_HOST_IP": node["host_ip"],
        "SGLANG_AGENTIC_KV_ENGINE_ID": node["engine_id"],
        "SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_PORT": str(node["reverse_bootstrap_port"]),
    })
    if node.get("ucx_net_devices"):
        env["UCX_NET_DEVICES"] = str(node["ucx_net_devices"])
    if cfg.get("local_triton_cache", False):
        env["TRITON_CACHE_DIR"] = str(
            Path(cfg["local_root"]) / "compiler-cache" / node["engine_id"] / "triton"
        )
    if node.get("numa_nodes"):
        env["SGLANG_AGENTIC_KV_TP_NUMA_NODES"] = ",".join(map(str, node["numa_nodes"]))
    prefill_nodes = [n for n in cfg["nodes"] if n["role"] == "prefill"]
    env["SGLANG_AGENTIC_KV_PREFILL_DOMAIN"] = str(prefill_nodes.index(node) if node["role"] == "prefill" else 0)
    command = [cfg["python"], "-m", "sglang.launch_server", "--model-path", cfg["model_path"],
               "--host", "0.0.0.0", "--port", str(node["port"]), "--tp-size", str(cfg["tp_size"]),
               "--disaggregation-mode", node["role"], "--disaggregation-transfer-backend", "nixl",
               "--disaggregation-bootstrap-port", str(node["bootstrap_port"]),
               "--mem-fraction-static", str(node["mem_fraction_static"]),
               "--page-size", str(cfg["page_size"]), "--context-length", str(cfg["context_length"]),
               "--enable-metrics", "--skip-server-warmup"]
    if node["role"] == "prefill":
        command += ["--chunked-prefill-size", str(cfg["chunked_prefill_size"]),
                    "--max-prefill-tokens", str(cfg["max_prefill_tokens"])]
    if cfg.get("model_family") == "qwen35_moe":
        command += [
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--kv-cache-dtype", "bfloat16",
            "--ep-size", "1",
            "--reasoning-parser", "glm45",
            "--tool-call-parser", "qwen3_coder",
            "--attention-backend", "triton",
            "--linear-attn-backend", "triton",
            "--moe-runner-backend", "triton",
            "--sampling-backend", "flashinfer",
            "--mamba-scheduler-strategy", "extra_buffer",
            "--mamba-track-interval", "64",
            "--mamba-full-memory-ratio",
            str(
                cfg.get(
                    f"{node['role']}_mamba_full_memory_ratio",
                    cfg.get("mamba_full_memory_ratio", 0.5),
                )
            ),
            "--random-seed", str(cfg.get("seed", 2026)),
        ]
    if node.get("numa_nodes"):
        command += ["--numa-node"] + list(map(str, node["numa_nodes"]))
    if node.get("ib_device"):
        command += ["--disaggregation-ib-device", node["ib_device"]]

    # launch_server creates scheduler children. It must pass each child's real
    # tp_rank to TCPRankAgent.from_env(rank=tp_rank); setting one scalar RANK in
    # this parent would incorrectly identify every child as rank zero.
    rank_envs = []
    for rank in range(cfg["tp_size"]):
        rank_env = dict(group_env(cfg, node))
        rank_env["SGLANG_AGENTIC_GROUP_RANK"] = str(rank)
        rank_envs.append(rank_env)
    return {
        "node_id": node["node_id"], "engine_id": node["engine_id"],
        "environment": env, "rank_environments": rank_envs,
        "command": command, "local_run_dir": str(local),
        "host_capacity_gib_per_direction": {
            "d2p_source": cfg["d2p_host_gib_per_rank"] * cfg["tp_size"] if node["role"] == "decode" else 0,
            "p2d_source": cfg["p2d_host_gib_per_rank"] * cfg["tp_size"] if node["role"] == "prefill" else 0,
        },
    }


def router_plan(cfg):
    router = cfg["router"]
    node = node_for(cfg, router["node_id"])
    env = common_env(cfg)
    env.update({"SGLANG_AGENTIC_MULTINODE_NODE_ID": node["node_id"],
                "SGLANG_AGENTIC_MULTINODE_ENGINE_ID": router["engine_id"],
                "SGLANG_AGENTIC_MULTINODE_ROLE": "router",
                "SGLANG_AGENTIC_MULTINODE_HOST_IP": node["host_ip"],
                "SGLANG_HOST_IP": node["host_ip"], "CUDA_VISIBLE_DEVICES": ""})
    # V2 currently has exactly one P group and one D group.  Use the stock
    # Python MiniLB as a transparent PD HTTP relay: unlike the Rust router's
    # strict OpenAI normalization it preserves the immutable lifecycle
    # envelope.  All ownership, admission and transfer decisions remain in
    # the TCP V2 controller; MiniLB keeps no filesystem control state.
    command = [cfg["python"], "-m", "sglang_router.launch_router",
               "--mini-lb", "--pd-disaggregation", "--policy", "random", "--host", "0.0.0.0",
               "--port", str(router["port"]), "--prometheus-port", str(router["metrics_port"]),
               "--health-check-timeout-secs", "60", "--health-failure-threshold", "10"]
    for worker in cfg["nodes"]:
        url = "http://{}:{}".format(worker["host_ip"], worker["port"])
        command += ["--prefill", url, str(worker["bootstrap_port"])] if worker["role"] == "prefill" else ["--decode", url]
    return {"node_id": node["node_id"], "engine_id": router["engine_id"], "environment": env,
            "command": command,
            "local_run_dir": str(Path(cfg["local_root"]) / cfg["run_id"] / router["engine_id"])}


def control_links(cfg):
    p = next(n for n in cfg["nodes"] if n["role"] == "prefill")
    d = next(n for n in cfg["nodes"] if n["role"] == "decode")
    return {
        link_id(cfg): {
            "coordinator": {"endpoint_group": p["engine_id"], "rank": 0},
            "endpoints": [
                {"endpoint_group": p["engine_id"], "role": "prefill", "size": cfg["tp_size"]},
                {"endpoint_group": d["engine_id"], "role": "decode", "size": cfg["tp_size"]},
            ],
        }
    }


def control_plan(cfg):
    control = cfg["group_control"]
    env = {"PYTHONPATH": str(Path(cfg["sglang_root"]) / "python"), "CUDA_VISIBLE_DEVICES": ""}
    command = [cfg["python"], "-m", "sglang.srt.disaggregation.agentic_group_protocol",
               "--listen", "{}:{}".format(control["listen_host"], control["port"]),
               "--run-id", cfg["run_id"], "--token", control["token"],
               "--links", json.dumps(control_links(cfg), sort_keys=True, separators=(",", ":"))]
    return {"node_id": control["node_id"], "engine_id": "agentic-group-control",
            "environment": env, "command": command,
            "local_run_dir": str(Path(cfg["local_root"]) / cfg["run_id"] / "agentic-group-control")}


def plan(cfg):
    return {"status": "DEVELOPMENT_PLAN_NOT_RUNTIME_ACCEPTANCE", "config_sha256": fingerprint(cfg),
            "control": control_plan(cfg), "workers": [worker_plan(cfg, n) for n in cfg["nodes"]],
            "router": router_plan(cfg),
            "requirements": ["rank0 is the only path/owner decision maker",
                             "one cross-endpoint transaction includes all source and target TP ranks",
                             "TCP control is reachable from every node",
                             "no NFS/shared-filesystem runtime state or KV payload",
                             "source-local Host DRAM remains registered until the remote fence"]}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x") as out:
            json.dump(value, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def protocol_check(cfg, env):
    code = "from sglang.srt.disaggregation.agentic_group_protocol import TCPGroupRelayServer, TCPRankAgent; print('ok')"
    result = subprocess.run([cfg["python"], "-c", code], env=env, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode or result.stdout.strip() != "ok":
        raise RuntimeError("TCP group protocol is unavailable; no component launched. " + result.stderr[-2000:])


def runtime_env(component):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SGLANG_", "PD_")) and key != "HOST_IP"}
    env.update(component["environment"])
    return env


def process_identity(cfg, component):
    return fingerprint(cfg) + ":" + component


def save_launch(cfg, component):
    directory = Path(component["local_run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / "launch.once", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(fingerprint(cfg))
    atomic_json(directory / "config.json", cfg)
    atomic_json(directory / "launch.json", component)


def wait_tcp(host, port, timeout=60):
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return {"control_ready": True, "endpoint": "tcp://{}:{}".format(host, port)}
        except OSError as exc:
            error = str(exc)
            time.sleep(0.2)
    raise RuntimeError("TCP group relay readiness timed out: " + str(error))


def wait_ready(cfg, timeout=1800):
    deadline = time.monotonic() + timeout
    pending = {n["engine_id"]: "http://{}:{}/model_info".format(n["host_ip"], n["port"])
               for n in cfg["nodes"]}
    errors = {}
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
    return {"workers_ready": True}


def check_listen_ports(cfg, component, kind):
    if kind == "worker":
        node = next(n for n in cfg["nodes"] if n["engine_id"] == component["engine_id"])
        ports = [node["port"], node["bootstrap_port"], node["reverse_bootstrap_port"]]
    elif kind == "router":
        ports = [cfg["router"]["port"], cfg["router"]["metrics_port"]]
    elif kind == "control":
        ports = [cfg["group_control"]["port"]]
    else:
        ports = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError("local TCP port {} is unavailable".format(port)) from exc


def start_component(cfg, component, kind):
    env = runtime_env(component)
    protocol_check(cfg, env)
    check_listen_ports(cfg, component, kind)
    if kind != "control":
        wait_tcp(cfg["group_control"]["advertise_host"], cfg["group_control"]["port"], 60)
    save_launch(cfg, component)
    print("Starting {} on {}; log={}".format(component["engine_id"], component["node_id"],
                                              Path(component["local_run_dir"]) / "service.log"), flush=True)
    return supervise(component["command"], env, component["local_run_dir"],
                     process_identity(cfg, component["engine_id"]))


def workload_plan(cfg):
    if not cfg.get("workload_command"):
        raise ValueError("set workload_command argv explicitly")
    result = router_plan(cfg)
    result["engine_id"] = "workload"
    result["command"] = cfg["workload_command"]
    result["local_run_dir"] = str(Path(cfg["local_root"]) / cfg["run_id"] / "workload")
    return result


def component_plan(cfg, component, node_id=None):
    if component == "worker":
        return worker_plan(cfg, node_for(cfg, node_id))
    if component == "router":
        return router_plan(cfg)
    if component == "control":
        return control_plan(cfg)
    if component == "workload":
        return workload_plan(cfg)
    raise ValueError("unknown component")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "start-control", "control-ready", "start-worker",
                                           "start-router", "run-workload", "wait-ready", "stop", "status"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-id")
    parser.add_argument("--component", choices=("worker", "control", "router", "workload"), default="worker")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if (args.action == "start-worker" or args.action in {"stop", "status"} and args.component == "worker") and not args.node_id:
        parser.error("this action requires --node-id")
    if args.action == "plan":
        result = plan(cfg)
    elif args.action == "start-control":
        return start_component(cfg, control_plan(cfg), "control")
    elif args.action == "control-ready":
        result = wait_tcp(cfg["group_control"]["advertise_host"], cfg["group_control"]["port"], args.timeout)
    elif args.action == "start-worker":
        return start_component(cfg, worker_plan(cfg, node_for(cfg, args.node_id)), "worker")
    elif args.action == "start-router":
        wait_ready(cfg, args.timeout)
        return start_component(cfg, router_plan(cfg), "router")
    elif args.action == "run-workload":
        wait_ready(cfg, args.timeout)
        return start_component(cfg, workload_plan(cfg), "workload")
    elif args.action == "wait-ready":
        result = wait_ready(cfg, args.timeout)
    else:
        component = component_plan(cfg, args.component, args.node_id)
        result = process_control(component["local_run_dir"], process_identity(cfg, component["engine_id"]), args.action)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
