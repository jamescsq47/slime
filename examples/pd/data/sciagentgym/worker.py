"""One persistent local CPU worker per agent; upstream loader and tool wrapper."""
import contextlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
    upstream = Path(sys.argv[1]).resolve()
    root = Path.cwd().resolve()
    sys.path.insert(0, str(upstream))
    protocol = sys.stdout
    # Tool prints must never corrupt the line-delimited JSON RPC channel.
    sys.stdout = sys.stderr
    import numpy as np
    import matplotlib.pyplot as plt
    from jsonschema import validate
    from gym.core.tool_loader import load_tools_for_case
    from gym.tool import GenericFunctionTool, _parameters_to_arguments_schema

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen",
                     "os.system", "os.posix_spawn", "os.fork"}:
            raise RuntimeError(f"Offline tool worker forbids {event}")
        if event in {"open", "os.mkdir", "os.remove", "os.rmdir", "os.rename"}:
            paths = args[:2] if event == "os.rename" else args[:1]
            writing = event != "open" or (len(args) > 2 and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                for path in paths:
                    if isinstance(path, (str, bytes)) and not Path(os.fsdecode(path)).resolve().is_relative_to(root):
                        raise ValueError("Tool writes must remain in its private directory")

    sys.addaudithook(audit)
    case = json.loads(sys.stdin.readline())
    start = time.perf_counter()
    protocols, functions = load_tools_for_case(case)
    schemas = {p["function"]["name"]: p["function"].get("parameters", {}) for p in protocols}
    if not schemas.keys() <= functions.keys():
        raise RuntimeError("Missing declared upstream tool functions")
    tools = {p["function"]["name"]: GenericFunctionTool(
        p["function"]["name"], p["function"].get("description", ""),
        _parameters_to_arguments_schema(p["function"].get("parameters", {})),
        functions[p["function"]["name"]]) for p in protocols}
    np.random.seed(2026)
    print(json.dumps({"ready": True, "load_seconds": time.perf_counter() - start}), file=protocol, flush=True)
    for line in sys.stdin:
        call = json.loads(line)
        start = time.perf_counter()
        try:
            name, arguments = call["name"], call.get("arguments", {})
            if name not in tools:
                raise ValueError("Tool not declared for this task")
            validate(arguments, schemas[name])
            # Reject path escape before upstream tools can read model-chosen files.
            for key, value in arguments.items():
                if isinstance(value, str) and any(s in key.lower() for s in ("path", "file", "dir")):
                    if not Path(value).resolve().is_relative_to(root):
                        raise ValueError("Tool file arguments must stay in the private directory")
            observation = tools[name](None, {"arguments": arguments}).observation
            ok = not observation.startswith("错误:")
            result = {"ok": ok, "observation": observation[:32000], "truncated": len(observation) > 32000}
        except Exception as exc:
            result = {"ok": False, "observation": f"{type(exc).__name__}: {str(exc)[:2000]}"}
        finally:
            plt.close("all")
        result["execution_seconds"] = time.perf_counter() - start
        print(json.dumps(result, ensure_ascii=False), file=protocol, flush=True)


if __name__ == "__main__":
    main()
