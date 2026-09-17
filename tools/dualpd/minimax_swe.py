"""MiniMax-M2.7 TP8/EP8 native colocated SWE-bench Verified evaluation.

plan/check-model are CPU-only. run is explicitly remote-operated and owns only
its supervisor sessions and uniquely labelled Docker containers. This is a
finite 500-task quality evaluation, not a 300+1200s throughput acceptance run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from multinode import validate_launch_model
from process_supervisor import atomic_record, supervise

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT.parent / "sglang"
PD = ROOT / "examples/pd"
DEFAULT_CONFIG = Path(__file__).with_name("minimax_swe.json")


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg["tp_size"], cfg["ep_size"], cfg["mem_fraction_static"], cfg["max_inflight"]) != (8, 8, 0.8, 64):
        raise ValueError("this acceptance configuration is TP8/EP8, memory .8, c64")
    if len(cfg["gpus"]) != 8 or len(set(cfg["gpus"])) != 8 or any(type(g) is not int or g < 0 for g in cfg["gpus"]):
        raise ValueError("exactly eight distinct physical GPUs required")
    if cfg["requests"] != 500:
        raise ValueError("full Verified evaluation requires 500 unique instances")
    if not 1024 <= cfg["port"] <= 65535:
        raise ValueError("invalid port")
    return cfg


def environment(run, data_root, cfg):
    # Remove inherited custom PD/Mamba/allocator switches. Preserve NIC/CUDA
    # library configuration, but never inherit another source overlay.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SGLANG_", "PD_")) and k not in {"PYTHONPATH", "HOST_IP"}}
    env.update(PYTHONPATH=f"{ENGINE / 'python'}:{PD}:{ROOT}",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=",".join(map(str, cfg["gpus"])),
               SGLANG_ENABLE_METRICS_DEVICE_TIMER="true",
               PD_DATA_ROOT=str(data_root), PD_INFERENCE_RETURN_LOGPROB="false",
               PD_MODEL_HTTP_TRANSPORT="aiohttp", SLIME_HTTP_READ_TIMEOUT_SECONDS="86400",
               PD_SWE_RUN_ID=run.name, PD_SWE_PROGRESS_FILE=str(run / "episode_progress.jsonl"), MIN_P="0")
    return env


def commands(cfg, model, run, workload):
    py, port = sys.executable, str(cfg["port"])
    server = [py, "-m", "sglang.launch_server", "--model-path", str(model),
              "--trust-remote-code", "--host", "127.0.0.1", "--port", port,
              "--tp-size", "8", "--ep-size", "8", "--mem-fraction-static", "0.8",
              "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
              "--context-length", str(cfg["context_length"]), "--page-size", str(cfg["page_size"]),
              "--chunked-prefill-size", str(cfg["chunked_prefill_size"]),
              "--max-prefill-tokens", str(cfg["max_prefill_tokens"]),
              "--reasoning-parser", "minimax-append-think", "--tool-call-parser", "minimax-m2",
              "--random-seed", str(cfg["seed"]), "--enable-metrics", "--skip-server-warmup"]
    infer = [py, str(PD / "scripts/new_method/internal/inference_checkpointed.py"),
             "--model", str(model), "--workload-config", str(workload),
             "--router-port", port, "--prefill-port", port, "--decode-port", port,
             "--requests", "500", "--warmup-requests", "0", "--max-inflight", "64",
             "--dispatch-policy", "random", "--preserve-source-order",
             "--request-rate", "100", "--arrival-distribution", "fixed",
             "--metrics-interval", "2", "--seed", str(cfg["seed"]),
             "--temperature", str(cfg["temperature"]), "--top-p", str(cfg["top_p"]),
             "--top-k", str(cfg["top_k"]), "--max-context-length", str(cfg["context_length"]),
             "--max-response-length", "524288", "--output-dir", str(run)]
    return {"model": server, "inference": infer}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_model(cfg, model, *, weights=False):
    raw = json.loads((model / "config.json").read_text())
    validate_launch_model({"model_family": "minimax_m2", "tp_size": 8}, raw)
    if cfg["context_length"] > raw["max_position_embeddings"]:
        raise ValueError("context exceeds model configuration")
    # Model code is native SGLang; only pinned HF configuration code is trusted.
    from sglang.srt.utils.hf_transformers_utils import get_config, get_tokenizer, get_rope_config
    from sglang.srt.parser.reasoning_parser import ReasoningParser
    model_cfg = get_config(str(model), trust_remote_code=True)
    rope = get_rope_config(model_cfg)
    tok = get_tokenizer(str(model), trust_remote_code=True)
    ReasoningParser(model_type="minimax-append-think", stream_reasoning=False)
    messages = [{"role": "user", "content": "Return one fenced bash command: echo hello"}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not rendered.rstrip().endswith("<think>"):
        raise ValueError("MiniMax thinking opener changed; audit reasoning parser before running")
    # Check actual unchanged harness rendering and three-turn history on CPU.
    from data.swe_bench_openenv.harness import _render_prompt
    histories = [messages, messages + [
        {"role": "assistant", "content": "```bash\necho hello\n```", "reasoning_content": "Run a command."},
        {"role": "user", "content": "hello\nExit code: 0"}]]
    lengths = []
    for history in histories:
        ids = _render_prompt(tok, history, enable_thinking=True)
        expected = tok.encode(tok.apply_chat_template(history, tokenize=False,
                              add_generation_prompt=True, enable_thinking=True, tools=None),
                              add_special_tokens=False)
        if ids != expected:
            raise ValueError("harness/tokenizer prompt mismatch")
        lengths.append(len(ids))
    if weights:
        index = json.loads((model / "model.safetensors.index.json").read_text())
        missing = [f for f in set(index["weight_map"].values()) if not (model / f).is_file()]
        if missing:
            raise ValueError("missing model shards: " + repr(missing[:5]))
        manifest = json.loads((model / "dualpd-download.json").read_text())
        if manifest["revision"] != cfg["model_revision"] or manifest["repo"] != cfg["model_repo"]:
            raise ValueError("model download revision differs from experiment")
        for f, size in manifest["files"].items():
            if not (model / f).is_file() or (model / f).stat().st_size != size:
                raise ValueError("incomplete checkpoint file: " + f)
    return {"config_class": type(model_cfg).__name__, "rope": rope, "prompt_lengths": lengths,
            "config_sha256": sha(model / "config.json"), "weights_checked": weights,
            "gpu_verified": False}


def download(cfg, model):
    from huggingface_hub import HfApi, snapshot_download
    model.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(cfg["model_repo"], revision=cfg["model_revision"], files_metadata=True)
    # No model Python implementation needed: use native SGLang, but AutoConfig
    # needs the official configuration module. Keep its revision immutable.
    files = {f.rfilename: f.size for f in info.siblings if f.rfilename.endswith(
        (".safetensors", ".json", ".jinja")) or f.rfilename == "configuration_minimax_m2.py"}
    snapshot_download(cfg["model_repo"], revision=cfg["model_revision"], local_dir=str(model),
                      allow_patterns=list(files), max_workers=4)
    for name, size in files.items():
        if (model / name).stat().st_size != size:
            raise RuntimeError("download size mismatch: " + name)
    atomic_record(model / "dualpd-download.json", {"repo": cfg["model_repo"],
                  "revision": cfg["model_revision"], "files": files})


def prepare_data(data_root):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo = "princeton-nlp/SWE-bench_Verified"
    revision = HfApi().dataset_info(repo).sha
    destination = data_root / "swe-bench-verified/test.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(load_dataset(repo, revision=revision, split="test"))
    if len(rows) != 500 or len({r["instance_id"] for r in rows}) != 500:
        raise ValueError("expected 500 distinct Verified tasks")
    with destination.open("x") as out:
        for row in rows:
            out.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    atomic_record(destination.with_suffix(".manifest.json"),
                  {"repo": repo, "revision": revision, "sha256": sha(destination), "count": 500})
    print(destination)


def preflight(cfg, model, data_root, run):
    import importlib.metadata
    if importlib.metadata.version("swebench") != "4.0.3":
        raise ValueError("install examples/pd/requirements-swe-bench.txt first (swebench==4.0.3)")
    import sglang
    if not Path(sglang.__file__).resolve().is_relative_to(ENGINE.resolve()):
        raise ValueError("sglang does not point to dualpd/sglang")
    result = check_model(cfg, model, weights=True)
    dataset = data_root / "swe-bench-verified/test.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if len(rows) != 500 or len({r["instance_id"] for r in rows}) != 500:
        raise ValueError("expected full Verified 500, not a repeated subset")
    result.update(dataset_sha256=sha(dataset), instance_ids=[r["instance_id"] for r in rows])
    listing = subprocess.check_output(["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
                                     text=True, timeout=60).splitlines()
    images = set(listing)
    needed = [r.get("image_name") or "swebench/sweb.eval.x86_64." + r["instance_id"].lower().replace("__", "_1776_") + ":latest" for r in rows]
    missing = [name for name in needed if name.removeprefix("docker.io/") not in images]
    if missing:
        raise ValueError("preload SWE Docker images before measurement: " + repr(missing[:5]))
    for gpu in cfg["gpus"]:
        owners = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                                          "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        if owners:
            raise RuntimeError(f"GPU {gpu} already has compute processes; refusing overlap: {owners}")
    result["gpus"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total",
                    "--format=csv"], text=True, timeout=15)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", cfg["port"]))
    existing = subprocess.check_output(["docker", "ps", "-aq", "--filter", f"label=pd.swe.run_id={run.name}"], text=True, timeout=30)
    if existing.strip():
        raise RuntimeError("run label already owns containers")
    return result


def cleanup(children, run, *, container_label=None):
    """Attempt every owned cleanup even if one supervisor cannot drain."""
    errors = []
    for component, child in reversed(list(children.items())):
        try:
            if child.poll() is None:
                child.terminate()
        except OSError as exc:
            errors.append(f"{component} signal: {exc}")
    deadline = time.monotonic() + 45
    for component, child in reversed(list(children.items())):
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{component} drain: {exc}")
    try:
        ids = subprocess.check_output(["docker", "ps", "-aq", "--filter",
                 f"label=pd.swe.run_id={container_label or run.name}"], text=True, timeout=30).split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"owned Docker cleanup: {exc}")
    if errors:
        raise RuntimeError("cleanup incomplete; do not start another GPU run: " + "; ".join(errors))


def run_evaluation(cfg, model, data_root, run):
    env = environment(run, data_root, cfg)
    # Preflight runs in this interpreter with the same source paths as workers.
    result = preflight(cfg, model, data_root, run)
    run.mkdir(parents=True, exist_ok=False)
    workload = run / "workload.yaml"
    workload.write_bytes((ROOT / cfg["workload_config"]).read_bytes())
    (run / "dataset.jsonl").write_bytes((data_root / "swe-bench-verified/test.jsonl").read_bytes())
    # The copied dataset is authoritative for this run; do not follow a later
    # mutation to the operator's source dataset.
    staged_data = run / "data/swe-bench-verified"
    staged_data.mkdir(parents=True)
    (staged_data / "test.jsonl").hardlink_to(run / "dataset.jsonl")
    env["PD_DATA_ROOT"] = str(run / "data")
    env["PD_SWE_RUN_ID"] = "dualpd-minimax-" + uuid.uuid4().hex
    plan = {"config": cfg, "commands": commands(cfg, model, run, workload), "env": env,
            "identity": uuid.uuid4().hex, "preflight": result}
    # Do not persist the whole inherited environment: it can contain secrets.
    subprocess_env = {key: value for key, value in env.items() if key.startswith(("PD_", "SGLANG_"))
                      or key in {"PYTHONPATH", "CUDA_VISIBLE_DEVICES", "MIN_P", "SLIME_HTTP_READ_TIMEOUT_SECONDS", "PYTHONDONTWRITEBYTECODE"}}
    plan["env"] = subprocess_env
    atomic_record(run / "plan.json", plan)
    for root, name in ((ROOT, "slime"), (ENGINE, "sglang")):
        result[name + "_commit"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        (run / (name + ".diff")).write_bytes(subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"]))
    atomic_record(run / "preflight.json", result)
    children = {}
    old_handlers = {}
    def interrupted(sig, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, interrupted)
    try:
        def start(component):
            child = subprocess.Popen([sys.executable, __file__, "_component", "--run-dir", str(run),
                                      "--component", component], env=env)
            children[component] = child
            return child
        server = start("model")
        deadline = time.monotonic() + cfg["startup_timeout_seconds"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if server.poll() is not None:
                raise RuntimeError("model exited; inspect model/service.log")
            try:
                with opener.open(f"http://127.0.0.1:{cfg['port']}/model_info", timeout=2) as response:
                    info = json.load(response)
                atomic_record(run / "model_info.json", info)
                break
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise TimeoutError("model readiness timeout")
                time.sleep(2)
        inference = start("inference")
        while inference.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("model exited during evaluation")
            time.sleep(2)
        if inference.returncode:
            raise RuntimeError("evaluation failed; inspect inference/service.log and completed records")
        subprocess.run([sys.executable, str(PD / "scripts/tools/analyze_swe_bench_run.py"), str(run)],
                       env=env, check=True)
    finally:
        # Ignore repeated interrupts while draining all owned sessions. A
        # second Ctrl-C must not leave the remaining TP ranks alive.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            cleanup(children, run, container_label=env["PD_SWE_RUN_ID"])
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "download", "prepare-data", "check-model", "preflight", "run", "_component"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=ROOT / "downloads/MiniMax-M2.7")
    parser.add_argument("--data-root", type=Path, default=ROOT / "downloads/pd-data")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--component", choices=["model", "inference"])
    args = parser.parse_args()
    cfg = read_config(args.config)
    model, data_root = args.model.resolve(), args.data_root.resolve()
    run = args.run_dir.resolve() if args.run_dir else ROOT / "runs/dualpd" / ("minimax-m27-swe-tp8-c64-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6])
    if args.action == "_component":
        plan = json.loads((run / "plan.json").read_text())
        env = environment(run, data_root, plan["config"])
        env.update(plan["env"])
        return supervise(plan["commands"][args.component], env, run / args.component,
                         plan["identity"] + ":" + args.component)
    # Configuration/tokenizer checks use the same clone as the GPU command.
    sys.path[:0] = [str(ENGINE / "python"), str(PD), str(ROOT)]
    if args.action == "plan":
        print(json.dumps({"config": cfg, "commands": commands(cfg, model, run, ROOT / cfg["workload_config"]),
                          "run_dir": str(run), "gpu_launched": False}, indent=2))
    elif args.action == "download":
        download(cfg, model)
    elif args.action == "prepare-data":
        prepare_data(data_root)
    elif args.action == "check-model":
        print(json.dumps(check_model(cfg, model), indent=2))
    elif args.action == "preflight":
        print(json.dumps(preflight(cfg, model, data_root, run), indent=2))
    elif args.action == "run":
        print("Run directory:", run, flush=True)
        run_evaluation(cfg, model, data_root, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
