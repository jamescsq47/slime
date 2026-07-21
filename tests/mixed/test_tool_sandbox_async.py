from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

import pytest


MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))

import tool_sandbox  # noqa: E402


@pytest.mark.parametrize("variant", ["mixed", "hybrid"])
def test_async_rollout_worker_uses_policy_event_loop(variant):
    rollout_path = Path(__file__).resolve().parents[2] / "examples" / variant / "fully_async_rollout.py"
    tree = ast.parse(rollout_path.read_text())
    worker_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "worker_thread_func"
    )
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(worker_method)
        if isinstance(node, ast.Call)
    }

    assert "asyncio.new_event_loop" in calls
    assert "asyncio.SelectorEventLoop" not in calls


def test_execute_code_returns_output():
    sandbox = tool_sandbox.PythonSandbox(timeout=2)

    result = asyncio.run(sandbox.execute_code("print(42)"))

    assert result == "Output:\n42"


def test_tool_registry_limits_concurrent_sandboxes(monkeypatch):
    async def run_test():
        active = 0
        peak = 0

        async def fake_execute_code(code):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return code

        monkeypatch.setattr(tool_sandbox, "SEMAPHORE", asyncio.Semaphore(2))
        registry = tool_sandbox.ToolRegistry()
        monkeypatch.setattr(registry.python_sandbox, "execute_code", fake_execute_code)
        results = await asyncio.gather(
            *(registry.execute_tool("code_interpreter", {"code": str(i)}) for i in range(8))
        )
        return peak, results

    peak, results = asyncio.run(run_test())

    assert peak == 2
    assert results == [str(i) for i in range(8)]


def test_empty_subprocess_exception_is_diagnostic(monkeypatch):
    async def create_subprocess_exec(*args, **kwargs):
        raise AssertionError()

    monkeypatch.setattr(tool_sandbox.asyncio, "create_subprocess_exec", create_subprocess_exec)

    result = asyncio.run(tool_sandbox.PythonSandbox(timeout=2).execute_code("print(42)"))

    assert result == "Error: Failed to execute code: AssertionError: AssertionError()"


def test_timed_out_code_does_not_block_event_loop():
    async def run_test():
        sandbox = tool_sandbox.PythonSandbox(timeout=0.4)
        loop = asyncio.get_running_loop()
        started = loop.time()
        execution = asyncio.create_task(sandbox.execute_code("while True:\n    pass"))

        await asyncio.sleep(0.03)
        heartbeat_delay = loop.time() - started
        fast_started = loop.time()
        fast_result = await sandbox.execute_code("print(7)")
        fast_execution_time = loop.time() - fast_started
        result = await execution
        return heartbeat_delay, fast_execution_time, fast_result, result

    heartbeat_delay, fast_execution_time, fast_result, result = asyncio.run(run_test())

    # A synchronous process.communicate(timeout=...) delays this heartbeat for
    # the full 0.4-second timeout. The async subprocess must yield meanwhile.
    assert heartbeat_delay < 0.2
    assert fast_execution_time < 0.2
    assert fast_result == "Output:\n7"
    assert result == "Error: Code execution timed out after 0.4 seconds"


def test_cancelled_rollout_kills_and_reaps_subprocess(monkeypatch):
    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.communicate_calls = 0
            self.started = asyncio.Event()
            self.killed = False

        async def communicate(self):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                self.started.set()
                await asyncio.Event().wait()
            return b"", b""

        def kill(self):
            self.killed = True
            self.returncode = -9

    async def run_test():
        process = FakeProcess()

        async def create_subprocess_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(tool_sandbox.asyncio, "create_subprocess_exec", create_subprocess_exec)
        sandbox = tool_sandbox.PythonSandbox(timeout=10)
        execution = asyncio.create_task(sandbox.execute_code("print(42)"))
        await process.started.wait()
        execution.cancel()

        with pytest.raises(asyncio.CancelledError):
            await execution
        return process

    process = asyncio.run(run_test())

    assert process.killed
    assert process.communicate_calls == 2
