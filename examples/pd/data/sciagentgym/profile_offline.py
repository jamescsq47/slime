"""Time unmodified upstream demos, NOT an LLM benchmark or fabricated workload.

The upstream task schemas identify tools; their upstream main() demos supply all
arguments. No answers are passed to a model, and no model runs in this probe.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


CASES = (1, 16, 20, 25, 26, 28, 31, 32, 34, 35, 36)
COMMIT = "e9dbbea4369d67694e38bf8be67bedbcaf9e9300"


def child(args):
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
    sys.path.insert(0, str(args.upstream))

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto",
                     "subprocess.Popen", "os.system", "os.posix_spawn", "os.fork"}:
            raise RuntimeError(f"Offline probe forbids {event}")

    # Defense in depth for reviewed local tools, not a security sandbox for
    # arbitrary model-generated code. No model-generated code is executed.
    import numpy as np
    import matplotlib.pyplot as plt
    # Trusted matplotlib initialization may query local fontconfig. Install
    # the restriction before importing any upstream benchmark module.
    sys.addaudithook(offline)
    from gym.core.tool_loader import load_tools_for_case

    cases = json.loads((args.upstream / "dataset/refine_merged_single_questions.json").read_text())
    case = next(x for x in cases if x["id"] == args.case)
    record = {"case_id": args.case, "subject": case["metadata"]["subject"],
              "question": case["question"], "events": [], "demos": []}
    result_path = args.output / f"case-{args.case}.json"
    log_path = args.output / f"case-{args.case}.log"
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            start = time.perf_counter()
            protocols, functions = load_tools_for_case(case)
            record["load_seconds"] = time.perf_counter() - start
            names = {p["function"]["name"] for p in protocols}
            record["declared_tools"] = sorted(names)
            if "main" not in functions or not names <= functions.keys():
                raise RuntimeError("Missing upstream demo or declared tool implementation")
            namespace = functions["main"].__globals__
            source = Path(namespace["__file__"])
            record["tool_source"] = str(source.relative_to(args.upstream))
            record["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            depth, repeat = 0, 0

            def timed(fn):
                @functools.wraps(fn)
                def wrapper(*a, **kw):
                    nonlocal depth
                    outer = depth == 0
                    depth += 1
                    start = time.perf_counter()
                    status = "returned"
                    try:
                        return fn(*a, **kw)
                    except BaseException:
                        status = "raised"
                        raise
                    finally:
                        elapsed = time.perf_counter() - start
                        depth -= 1
                        if outer:
                            record["events"].append({"repeat": repeat, "tool": fn.__name__,
                                                     "seconds": elapsed, "status": status})
                return wrapper

            for name in names:
                namespace[name] = timed(functions[name])
            for repeat in range(args.repeats):
                np.random.seed(2026)
                start = time.perf_counter()
                try:
                    functions["main"]()
                    record["demos"].append({"repeat": repeat, "status": "returned",
                                            "seconds": time.perf_counter() - start})
                except Exception as exc:
                    record["demos"].append({"repeat": repeat, "status": "failed",
                                            "error": repr(exc), "seconds": time.perf_counter() - start})
                    break
                finally:
                    plt.close("all")
        except Exception as exc:
            record["error"] = repr(exc)
        finally:
            result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/homes/siqic/data/SciAgentGYM"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=list(CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    args.upstream, args.output = args.upstream.resolve(), args.output.resolve()
    if args.case is not None:
        if args.case not in CASES:
            raise ValueError("Case not in reviewed offline subset")
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Expected reviewed commit {COMMIT}, got {actual}")
    dirty = subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("Upstream tracked files must remain unmodified")
    if not set(args.cases) <= set(CASES):
        raise ValueError("Case not in reviewed offline subset")
    manifest = {"commit": actual, "cases": args.cases, "repeats": args.repeats,
                "mode": "upstream_main_demo_no_llm", "timeout_seconds_per_case": 120,
                "requested_cpu_threads": 1, "children": [],
                "dataset_sha256": hashlib.sha256((args.upstream / "dataset/refine_merged_single_questions.json").read_bytes()).hexdigest()}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for case_id in args.cases:
        with tempfile.TemporaryDirectory(prefix=f"sciagentgym-{case_id}-") as scratch:
            env["MPLCONFIGDIR"] = scratch
            command = [sys.executable, str(Path(__file__).resolve()), "--upstream", str(args.upstream),
                       "--output", str(args.output), "--case", str(case_id), "--repeats", str(args.repeats)]
            try:
                proc = subprocess.run(command, cwd=scratch, env=env, capture_output=True, text=True, timeout=120)
                item = {"case_id": case_id, "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
            except subprocess.TimeoutExpired:
                item = {"case_id": case_id, "error": "120s timeout; child killed and reaped"}
            manifest["children"].append(item)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
