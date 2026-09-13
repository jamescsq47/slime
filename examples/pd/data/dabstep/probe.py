"""Real DABstep dev tasks using upstream smolagents CodeAgent and isolated Python.

Not a steady-state serving benchmark. Answers are not mounted in the sandbox.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import uuid

from smolagents import CodeAgent, OpenAIServerModel
from smolagents.local_python_executor import PythonExecutor, CodeOutput

IMAGE = "sha256:0b1325d81f2f0d264c7acec73e5fcd142ff1ba0ba5e6879b91a1b2ecee058e69"


class ContainerPython(PythonExecutor):
    def __init__(self, context, timeout=60, *, image=IMAGE, memory="4g", workspace_size="512m"):
        self.name = "dabstep-probe-" + uuid.uuid4().hex
        self.events = []
        self.timeout = timeout
        self.closed = False
        self.cleanup_errors = []
        self.proc = None
        self.selector = selectors.DefaultSelector()
        self.pending = b""
        command = ["docker", "run", "--rm", "-i", "--name", self.name,
                   "--label", f"dabstep.probe_run={os.environ.get('DABSTEP_RUN_ID', self.name)}",
                   "--network", "none", "--read-only", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "128",
                   "--memory", memory, "--memory-swap", memory, "--cpus", "2",
                   "--user", "65534:65534", "--workdir", "/workspace",
                   "--tmpfs", f"/workspace:rw,nosuid,nodev,size={workspace_size},mode=1777",
                   "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m,mode=1777",
                   "--mount", f"type=bind,src={Path(context).resolve()},dst=/data/context,readonly",
                   "--mount", f"type=bind,src={Path(__file__).with_name('worker.py')},dst=/worker.py,readonly",
                   "--entrypoint", "/usr/bin/env", image, "-i", "PATH=/usr/bin:/bin", "HOME=/tmp",
                   "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                   "MPLBACKEND=Agg", "MPLCONFIGDIR=/tmp/mpl", "PYTHONDONTWRITEBYTECODE=1",
                   "/usr/bin/python", "-u", "/worker.py"]
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL)
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
            if not self.read_response(45).get("ready"):
                raise RuntimeError("Sandbox startup failed")
        except BaseException:
            self.cleanup()
            raise

    def read_response(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.pending:
            if not self.selector.select(max(0, deadline - time.monotonic())):
                raise TimeoutError("Python sandbox timed out")
            data = os.read(self.proc.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError("Python sandbox exited")
            self.pending += data
            if len(self.pending) > 256000:
                raise RuntimeError("Python sandbox response exceeded limit")
        line, self.pending = self.pending.split(b"\n", 1)
        return json.loads(line)

    def send_tools(self, tools):
        if set(tools) - {"final_answer"}:
            raise ValueError("Only the standard CodeAgent final_answer tool is supported")

    def send_variables(self, variables):
        if variables:
            raise ValueError("Task answers/host variables must not enter the container")

    def __call__(self, code_action):
        if self.closed:
            raise RuntimeError("Sandbox already closed")
        start = time.monotonic()
        event = {"code": code_action}
        try:
            self.proc.stdin.write((json.dumps({"code": code_action}) + "\n").encode())
            self.proc.stdin.flush()
            response = self.read_response(self.timeout)
            if (not isinstance(response, dict) or not {"output", "logs", "is_final_answer", "error", "execution_seconds"} <= response.keys()
                    or not isinstance(response["logs"], str) or not isinstance(response["is_final_answer"], bool)):
                raise RuntimeError("Invalid sandbox response schema")
            event.update(response)
            if response["error"]:
                raise ValueError(response["error"])
            return CodeOutput(response["output"], response["logs"], response["is_final_answer"])
        except (TimeoutError, BrokenPipeError, RuntimeError, json.JSONDecodeError) as exc:
            event["error"] = str(exc)
            self.cleanup()
            raise
        finally:
            event["wall_seconds"] = time.monotonic() - start
            self.events.append(event)

    def cleanup(self):
        if self.closed:
            return
        self.closed = True
        try:
            result = subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=30)
            if result.returncode and b"No such container" not in result.stderr:
                self.cleanup_errors.append(result.stderr.decode(errors="replace")[-1000:])
        except Exception as exc:
            self.cleanup_errors.append(repr(exc))
        finally:
            try:
                if self.proc is not None:
                    if self.proc.poll() is None:
                        self.proc.kill()
                    self.proc.wait(timeout=10)
            except Exception as exc:
                self.cleanup_errors.append(repr(exc))
            finally:
                self.selector.close()


def action_text(content, stop_sequences):
    """Apply CodeAgent's action boundary after, never inside, Qwen reasoning."""
    content = content or ""
    if "<think>" in content and "</think>" not in content:
        return ""
    content = content.rsplit("</think>", 1)[-1]
    offsets = [content.index(stop) for stop in (stop_sequences or []) if stop and stop in content]
    return content[:min(offsets)] if offsets else content


class RecordedModel(OpenAIServerModel):
    def __init__(self, endpoint):
        super().__init__(model_id="/dataset/model/qwen3/Qwen3-8B", api_base=endpoint, api_key="local",
                         client_kwargs={"timeout": 300, "max_retries": 0}, temperature=0, max_tokens=8192,
                         extra_body={"top_k": -1}, top_p=1)
        self.calls = []

    def generate(self, messages, stop_sequences=None, **kwargs):
        start = time.monotonic()
        event = {}
        try:
            # Let Qwen finish its turn; CodeAgent markers may occur inside reasoning.
            result = super().generate(messages, stop_sequences=None, **kwargs)
            event.update(raw_content=result.content, token_usage=vars(result.token_usage) if result.token_usage else None)
            result.content = action_text(result.content, stop_sequences)
            event["action_content"] = result.content
            return result
        except Exception as exc:
            event["error"] = repr(exc)
            raise
        finally:
            event["wall_seconds"] = time.monotonic() - start
            self.calls.append(event)


def upstream_task_template(path):
    # Reuse the official prompt constant without importing its obsolete API imports.
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "chat_llm_task_prompt" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Official task prompt not found")


def run_task(task, config, template):
    start = time.monotonic()
    model = RecordedModel(config["endpoint"])
    record = {k: task[k] for k in ("task_id", "question", "guidelines", "level")}
    executor = None
    try:
        executor = ContainerPython(config["context"], config["tool_timeout_seconds"])
        agent = CodeAgent(tools=[], model=model, executor=executor, max_steps=config["max_steps"],
                          additional_authorized_imports=["pandas", "numpy", "json", "os", "math", "re"],
                          verbosity_level=0, return_full_result=True)
        prompt = template.format(ctx_path="/data/context", question=task["question"], guidelines=task["guidelines"])
        result = agent.run(prompt)
        record.update(answer=str(result.output), state=result.state, steps=result.steps,
                      total_token_usage=vars(result.token_usage) if result.token_usage else None)
    except Exception as exc:
        record["error"] = repr(exc)
    finally:
        if executor:
            executor.cleanup()
        record.update(wall_seconds=time.monotonic() - start, model_calls=model.calls,
                      tool_events=executor.events if executor else [],
                      cleanup_errors=executor.cleanup_errors if executor else [])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(config["tasks"]).read_text().splitlines()][:config["max_tasks"]]
    # Only inputs cross the model boundary. Gold remains in the host dataset file.
    tasks = [{k: row[k] for k in ("task_id", "question", "guidelines", "level")} for row in rows]
    template = upstream_task_template(config["upstream_prompt"])
    manifest = {**config, "image": IMAGE, "task_ids": [t["task_id"] for t in tasks],
                "file_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(config["context"]).iterdir()) if p.is_file()}}
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2))
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        futures = [pool.submit(run_task, task, config, template) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            (args.output / f"task-{record['task_id']}.json").write_text(json.dumps(record, default=str, indent=2))
            print(json.dumps({k: record.get(k) for k in ("task_id", "state", "error", "answer", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
