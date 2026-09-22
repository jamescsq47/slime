"""Owned a10/a11 Qwen3.5 TP8 experiment; run from a10 only.

Reuse the existing multi-node launcher and process supervisors. Model data uses
NIXL; runtime lifecycle/control records use persistent TCP messages, never NFS.
No remote Docker: tools and verifier run on a10 with the baseline workload.
"""
import argparse
import http.client
import json
import math
import os
from pathlib import Path
import shlex
import signal
import secrets
import socket
import subprocess
import sys
import time
import urllib.request

import multinode as m
import minimax_swe as swe

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
ENGINE = ROOT.parent / "sglang"
PYTHON = "/homes/siqic/anaconda3/envs/pd_multi_node/bin/python"
REFERENCE = ROOT / "runs/dualpd/qwen35-122b-swe500-tp8-c64-openai27b-r1"


def config(run, concurrency=None):
    cfg = json.loads((TOOLS / "multinode.example.json").read_text())
    cfg.update(run_id=run.name, model_family="qwen35_moe",
               # Identity labels only: message mode never opens these paths.
               control_backend="tcp", control_root="/run/dualpd-control",
               control={"node_id": "a10", "port": 23904, "tp_port": 23905,
                        "token": secrets.token_urlsafe(32)},
               local_root="/tmp/dualpd-multinode", sglang_root=str(ENGINE),
               slime_root=str(ROOT), python=PYTHON,
               cuda_home="/homes/siqic/cuda-12.8",
               model_path=str(ROOT / "downloads/Qwen3.5-122B-A10B"),
               context_length=131072, max_prefill_tokens=8192,
               mamba_full_memory_ratio=0.5, seed=2026, fast_tool_seconds=1,
               direct_admission_seconds=1,
               pd_max_transfer_inflight=0,
               # SWE-bench keeps Host recovery instead of congestion recompute.
               slow_congestion_recompute=False,
               local_triton_cache=True,
               tp_host_async_prepare=True,
               router_startup_timeout_seconds=1800,
               router_profile_imports=True,
               d2p_host_gib_per_rank=32, p2d_host_gib_per_rank=16)
    for n, node, ip in zip(cfg["nodes"], ["a10", "a11"], ["10.0.1.170", "10.0.1.171"]):
        # Keep listeners outside both nodes' ephemeral range (32768-60999).
        n.update(node_id=node, host_ip=ip, port=23900,
                 bootstrap_port=23901, reverse_bootstrap_port=61900,
                 numa_nodes=[0, 0, 0, 0, 1, 1, 1, 1], ib_device="mlx5_1",
                 ucx_net_devices="mlx5_1:1")
    cfg["router"].update(node_id="a10", port=23902, metrics_port=23903)
    baseline = swe.read_config(TOOLS / "qwen35_swe.json")
    infer = swe.commands(baseline, Path(cfg["model_path"]), run,
                         ROOT / baseline["workload_config"])["inference"]
    if concurrency is not None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        infer[infer.index("--max-inflight") + 1] = str(concurrency)
    for flag, value in [("--router-port", cfg["router"]["port"]),
                        ("--prefill-port", cfg["nodes"][0]["port"]),
                        ("--decode-port", cfg["nodes"][1]["port"])]:
        infer[infer.index(flag) + 1] = str(value)
    infer += ["--router-host", "10.0.1.170", "--prefill-host", "10.0.1.170",
              "--decode-host", "10.0.1.171"]
    cfg["workload_command"] = infer
    cfg["workload_environment"] = {
        "PD_DATA_ROOT": str(run / "data"), "PD_SWE_RUN_ID": run.name,
        "PD_SWE_PROGRESS_FILE": str(run / "episode_progress.jsonl"),
        "PD_MODEL_HTTP_TRANSPORT": "aiohttp", "SLIME_HTTP_READ_TIMEOUT_SECONDS": "86400",
        "MIN_P": "0"}
    return cfg


def remote(node, argv):
    if node == "a10":
        return argv
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            node, shlex.join(argv)]


def command(path, action, node="a10", component=None):
    args = [PYTHON, str(TOOLS / "multinode.py"), action, "--config", str(path)]
    if action in {"fs-publish", "start-worker"} or component == "worker":
        args += ["--node-id", node]
    if component:
        args += ["--component", component]
    return remote(node, args)


def call(argv, timeout=120):
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {shlex.join(argv)}\n{proc.stderr[-3000:]}")
    return proc.stdout


def prepare(run, *, diagnostic_digests=False, p2d_host_probe=False, concurrency=None, d2p_direct_only=False, d2p_direct_wait=False, p_workset_controller=False):
    if socket.gethostname().split('.')[0] != "a10":
        raise RuntimeError("run this coordinator on a10; Docker images are on a10")
    run.mkdir(parents=True, exist_ok=False)
    cfg = config(run, concurrency=concurrency)
    cfg["p_workset_controller"] = bool(p_workset_controller)
    if d2p_direct_only or d2p_direct_wait:
        cfg["d2p_host_staging"] = False
    if d2p_direct_wait:
        cfg["d2p_direct_wait_only"] = True
    cfg["debug_kv_digest"] = bool(diagnostic_digests)
    cfg["p2d_host_probe"] = bool(p2d_host_probe)
    path = run / "multinode.json"
    m.atomic_json(path, cfg)
    m.load_config(path)
    # Immutable reference data/order; never fetch a different dataset revision.
    baseline = swe.read_config(TOOLS / "qwen35_swe.json")
    if swe.sha(REFERENCE / "data/swe-bench-verified/test.jsonl") != baseline["dataset_sha256"]:
        raise RuntimeError("reference dataset/order changed")
    (run / "data").symlink_to(REFERENCE / "data", target_is_directory=True)
    baseline["port"] = cfg["nodes"][0]["port"]
    baseline["allow_gpu_pids"] = {"7": [1868643]}  # user-approved ComfyUI only
    os.environ["PYTHONPATH"] = f"{ENGINE / 'python'}:{ROOT / 'examples/pd'}:{ROOT}"
    sys.path.insert(0, str(ROOT / "examples/pd"))
    checks = {"a10": swe.preflight(baseline, Path(cfg["model_path"]), run / "data", run)}
    # Exact environment/model paths are shared; source hashes catch stale mounts.
    probe = """import json, pathlib, subprocess, sglang, torch, os, shutil, sys
os.environ['PATH']=str(pathlib.Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
assert shutil.which('ninja'), 'ninja missing from worker environment'
assert pathlib.Path('/homes/siqic/cuda-12.8/bin/nvcc').is_file(), 'CUDA toolkit missing'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not apps, 'a11 has other GPU processes: '+apps
root=pathlib.Path('/homes/siqic/dualpd/slime/downloads/Qwen3.5-122B-A10B')
idx=json.loads((root/'model.safetensors.index.json').read_text())
assert all((root/f).is_file() for f in set(idx['weight_map'].values()))
print(json.dumps({'sglang':sglang.__file__,'torch':torch.__version__,'weights_present':True,'gpu_processes':apps}))
"""
    checks["a11"] = json.loads(call(remote("a11", [PYTHON, "-c", probe])))
    if not checks["a11"]["sglang"].startswith(str(ENGINE) + "/"):
        raise RuntimeError("a11 uses a different engine")
    checks["node_identity"] = {
        node: json.loads(call(command(path, "fs-publish", node)))
        for node in ["a10", "a11"]
    }
    identities = list(checks["node_identity"].values())
    if len({item["boot_id"] for item in identities}) != 2:
        raise RuntimeError("expected two distinct physical node boots")
    checks["control_backend"] = "tcp-memory; no shared filesystem preflight or locking"
    m.atomic_json(run / "preflight.json", checks)
    m.atomic_json(run / "plan.json", m.plan(cfg))
    for repo in [ROOT, ENGINE]:
        (run / (repo.name + ".diff")).write_text(call(["git", "-C", str(repo), "diff"]))
    return cfg, path


def stop(path):
    results = []
    for node, component in [("a10", "workload"), ("a10", "smoke"), ("a10", "router"),
                             ("a11", "worker"), ("a10", "worker")]:
        try:
            results.append(call(command(path, "stop", node, component), timeout=40))
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            results.append(str(exc))
    return results


def check_gpu_owners():
    # Recheck even if preflight was done earlier; a free GPU is not a lease.
    for node in ['a10', 'a11']:
        probe = """import json, subprocess
allowed=json.loads(%r)
for gpu in range(8):
 text=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
 owners={int(x) for x in text.splitlines() if x.strip()}
 assert owners <= set(allowed.get(str(gpu), [])), 'GPU '+str(gpu)+' now occupied: '+str(owners)
print('GPU ownership checked')
""" % json.dumps({'7': [1868643]} if node == 'a10' else {})
        call(remote(node, [PYTHON, '-c', probe]), timeout=150)


def wait_router_ready(cfg, children):
    """Bound startup by elapsed time, including slow imports on shared storage.

    Readiness remains mandatory; a live supervisor alone is not readiness.
    Model/supervisor failure still aborts immediately through owned cleanup.
    """
    timeout = cfg.get("router_startup_timeout_seconds", 1800)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("router_startup_timeout_seconds must be positive and finite")
    router = cfg["router"]
    node = m.node_for(cfg, router["node_id"])
    url = f'http://{node["host_ip"]}:{router["port"]}/health'
    log = Path(m.router_plan(cfg)["local_run_dir"]) / "service.log"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    deadline, next_report = started + timeout, started
    last_error = "no health response"
    while True:
        for name, child in children.items():
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"{name} exited during Router startup (code={code}); log={log}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Router startup timed out after {timeout}s: {last_error}; log={log}")
        try:
            with opener.open(url, timeout=min(2, remaining)) as response:
                if response.status == 200:
                    elapsed = time.monotonic() - started
                    print(f"Router ready after {elapsed:.1f}s", flush=True)
                    return {"ready": True, "elapsed_seconds": elapsed, "url": url}
                last_error = f"HTTP {response.status}"
        except (OSError, http.client.HTTPException) as exc:
            last_error = str(exc)
        now = time.monotonic()
        if now >= next_report:
            print(f"Waiting for Router: elapsed={now-started:.1f}s limit={timeout}s "
                  f"last_error={last_error}; log={log}", flush=True)
            next_report = now + 30
        time.sleep(min(1, max(0, deadline - now)))


def run_experiment(run, cfg, path, smoke_only):
    if cfg.get("control_backend") != "tcp":
        raise RuntimeError("old file-control runs cannot restart; create a new TCP-control run")
    # Run the independent integration gate BEFORE allocating a GPU or starting
    # any run-owned process, not halfway through model startup.
    m.capability_check(cfg, m.runtime_env(m.worker_plan(cfg, cfg["nodes"][0])))
    check_gpu_owners()
    children = {}
    logs = []
    def launch(node, action, key):
        log = (run / (key + ".log")).open("ab", buffering=0)
        logs.append(log)
        child = subprocess.Popen(command(path, action, node), stdout=log,
                                 stderr=subprocess.STDOUT, env=env, start_new_session=True)
        children[key] = child
        return child
    baseline = swe.read_config(TOOLS / "qwen35_swe.json")
    env = swe.environment(run, run / "data", baseline)
    env.update(PYTHONDONTWRITEBYTECODE="1")
    def terminate(signum, frame):
        raise InterruptedError(f"coordinator received signal {signum}")
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, terminate)
    try:
        broker = launch("a10", "start-control", "control")
        deadline = time.monotonic() + 60
        while True:
            if broker.poll() is not None:
                raise RuntimeError("control broker exited during startup")
            try:
                if m.control_client(cfg).call("system", "describe")["run_id"] == cfg["run_id"]:
                    break
                raise RuntimeError("control broker returned a different run")
            except (OSError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        launch("a10", "start-worker", "prefill")
        launch("a11", "start-worker", "decode")
        deadline = time.monotonic() + 3600
        while True:
            for key, child in children.items():
                if child.poll() is not None:
                    raise RuntimeError(f"{key} exited during startup; inspect {key}.log and local service.log")
            try:
                m.wait_ready(cfg, timeout=3)
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
        deadline = time.monotonic() + 1800
        while True:
            for key, child in children.items():
                if child.poll() is not None:
                    raise RuntimeError(f"{key} exited during Host prewarm")
            prewarm = m.host_prewarm_status(cfg)
            if prewarm["ready"]:
                m.atomic_json(run / "host-prewarm.json", prewarm)
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Host prewarm timed out: " + repr(prewarm))
            time.sleep(1)
        launch("a10", "start-router", "router")
        m.atomic_json(run / "router-ready.json", wait_router_ready(cfg, children))
        smoke = launch("a10", "smoke", "smoke")
        if smoke.wait() != 0:
            raise RuntimeError("multi-node Direct/Slow correctness smoke failed; refusing SWE run")
        if not smoke_only:
            workload = launch("a10", "run-workload", "workload")
            while workload.poll() is None:
                for key in ["prefill", "decode", "router", "control"]:
                    if children[key].poll() is not None:
                        raise RuntimeError(f"{key} exited during SWE evaluation")
                time.sleep(5)
            if workload.returncode:
                raise RuntimeError("SWE evaluation failed")
        m.atomic_json(run / "completion.json", {"ok": True, "smoke_only": smoke_only, "time": time.time()})
    except BaseException as exc:
        m.atomic_json(run / "failure.json", {"error": str(exc), "time": time.time()})
        raise
    finally:
        m.atomic_json(run / "cleanup.json", {"stops": stop(path)})
        # Never kill an SSH PID as a substitute for remote ownership cleanup.
        for key, child in children.items():
            if key == "control":
                continue
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass
        if "control" in children:
            if all(children[key].poll() is not None for key in ("prefill", "decode") if key in children):
                try:
                    call(command(path, "stop", "a10", "control"), timeout=10)
                    children["control"].wait(timeout=45)
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                    print(f"Control broker cleanup needs inspection: {exc}", flush=True)
            else:
                print("Model worker has not stopped; retaining control broker and physical ownership.", flush=True)
        ids = call(["docker", "ps", "-aq", "--filter", f"label=pd.swe.run_id={run.name}"]).split()
        if ids:
            call(["docker", "rm", "-f", *ids])
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "run", "status", "stop"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--concurrency", type=int, help="SWE concurrency; omitted preserves the reference default")
    parser.add_argument("--p-workset-controller", action="store_true", help="Opt-in single-authority P workset controller; requires its independent model audit")
    parser.add_argument("--d2p-direct-only", action="store_true",
                        help="Disable only D->P Host staging; tool/Direct timeouts recompute. P->D remains unchanged.")
    parser.add_argument("--d2p-direct-wait", action="store_true",
                        help="Strict D->P Direct: retain parent through tool/capacity/setup waits; no Host or timeout recompute.")
    parser.add_argument("--diagnostic-digests", action="store_true",
                        help="Engineering smoke only; hashes synchronize CUDA and must not measure performance")
    parser.add_argument("--p2d-host-probe", action="store_true",
                        help="Smoke only: cap D at12288tokens to exercise real P2D Host backpressure")
    args = parser.parse_args()
    if args.d2p_direct_only and args.d2p_direct_wait:
        parser.error("choose either timeout-recompute or strict Direct wait, not both")
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if (args.diagnostic_digests or args.p2d_host_probe) and args.action == "run" and not args.smoke_only:
        parser.error("diagnostic settings are limited to --smoke-only")
    run = args.run_dir.resolve()
    path = run / "multinode.json"
    if args.action in {"status", "stop"}:
        if args.action == "stop":
            print(json.dumps(stop(path), indent=2))
        else:
            for node, component in [("a10", "worker"), ("a11", "worker"), ("a10", "router"), ("a10", "workload")]:
                try:
                    print(node, component, call(command(path, "status", node, component)))
                except RuntimeError as exc:
                    print(exc)
        return
    if path.exists():
        if args.action == "preflight":
            raise RuntimeError("preflight already exists; use a fresh run directory")
        cfg = m.load_config(path)
        if args.p_workset_controller and not cfg.get("p_workset_controller", False):
            raise RuntimeError("saved run has no P workset controller; use a fresh run directory")
        if args.d2p_direct_only and cfg.get("d2p_host_staging", True):
            raise RuntimeError("saved run enables D->P Host staging; use a fresh run directory")
        if args.d2p_direct_wait and not cfg.get("d2p_direct_wait_only", False):
            raise RuntimeError("saved run is not strict Direct wait; use a fresh run directory")
        if args.concurrency is not None:
            workload = cfg["workload_command"]
            if int(workload[workload.index("--max-inflight") + 1]) != args.concurrency:
                raise RuntimeError("saved run has a different concurrency; use a fresh run directory")
    else:
        cfg, path = prepare(run, diagnostic_digests=args.diagnostic_digests,
                            p2d_host_probe=args.p2d_host_probe, concurrency=args.concurrency,
                            d2p_direct_only=args.d2p_direct_only,
                            d2p_direct_wait=args.d2p_direct_wait,
                            p_workset_controller=args.p_workset_controller)
    if (cfg.get('debug_kv_digest') or cfg.get('p2d_host_probe')) and args.action == 'run' and not args.smoke_only:
        raise RuntimeError('diagnostic preflight config cannot run a performance/evaluation workload')
    print(f"Preflight complete: {run}", flush=True)
    if args.action == "run":
        run_experiment(run, cfg, path, args.smoke_only)


if __name__ == "__main__":
    main()
