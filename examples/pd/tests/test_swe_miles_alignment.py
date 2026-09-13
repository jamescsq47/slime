"""CPU-only regression tests: no model server, Docker, or GPU required."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest

from data.swe_bench.verifier import RepositoryBaseline
from data.swe_bench_openenv import harness as h
from slime.utils.types import Sample


@pytest.fixture
def episode(monkeypatch):
    async def run(responses, *, options=None, execute_error=None, execute_code=0):
        seen = {"calls": [], "commands": [], "final": [], "tool": [], "closed": 0}

        class Task:
            image = "fake"
            metrics = {}

            async def start(self):
                pass

            async def execute(self, command, **kwargs):
                seen["commands"].append(command)
                if execute_error:
                    raise execute_error
                return execute_code, "ok"

            async def close(self):
                seen["closed"] += 1

        async def baseline(*args):
            return RepositoryBaseline(
                official_base_commit="a" * 40, official_tree="b" * 40,
                image_commit="a" * 40, image_tree="b" * 40, kind="exact",
                image_commits_ahead=0, fingerprint="x",
            )

        async def capture(*args):
            seen["final_before_capture"] = list(seen["final"])
            return ""

        async def turn(**kwargs):
            seen["calls"].append(copy.deepcopy(kwargs))
            result = responses[len(seen["calls"]) - 1]
            if isinstance(result, BaseException):
                raise result
            reply, calls, finish, count = result
            return reply, [], dict(
                turn=kwargs["turn"] + 1, output_tokens=count, input_tokens=10,
                generation_seconds=0.1, tool_calls=calls, finish_type=finish,
                reasoning_content="reasoning kept in history",
            ), None

        tokenizer = SimpleNamespace(encode=lambda *a, **k: [1])
        monkeypatch.setattr(h, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer))
        monkeypatch.setattr(h, "_create_task", lambda *args: Task())
        monkeypatch.setattr(h, "prepare_repository_baseline", baseline)
        monkeypatch.setattr(h, "capture_repository_patch", capture)
        monkeypatch.setattr(h, "_model_turn", turn)
        monkeypatch.setattr(h, "confirm_agentic_generation_final", lambda m, g, **k: seen["final"].append(g))
        monkeypatch.setattr(h, "confirm_agentic_generation_tool", lambda m, g, **k: seen["tool"].append(g))
        monkeypatch.delenv("PD_SWE_PROGRESS_FILE", raising=False)
        args = SimpleNamespace(workload_dataset_options={"test": {
            "model_api": "chat_completions", "action_protocol": "fenced_shell",
            "command_contract": "miles_pr51", "verifier_mode": "capture",
            "max_turns": 4, **(options or {}),
        }}, pd_p_ready_dir="")
        sample = Sample(metadata={"dataset_id": "test", "problem_statement": "fix", "instance_id": "a__b-1"})
        try:
            result = await h.generate(args, sample, {})
        except asyncio.CancelledError:
            seen["cancelled"] = True
            result = sample
        return result, seen
    return run


def tool(name="shell", command="pwd"):
    return [{"id": "call1", "type": "function", "function": {
        "name": name, "arguments": {"command": command},
    }}]


def test_miles_fenced_history_and_submission(episode):
    result, seen = asyncio.run(episode([
        ("```bash\npwd\n```", [], "stop", 3),
        ("TASK_COMPLETE", [], "stop", 1),
    ]))
    history = seen["calls"][1]["messages"]
    assert history[0]["content"] == h._SYSTEM_PROMPT
    assert history[-2]["reasoning_content"] == "reasoning kept in history"
    assert history[-1] == {"role": "user", "content": "<shell_result exit_code=0>\nok\n</shell_result>"}
    assert seen["tool"] == [0]
    assert seen["final"] == seen["final_before_capture"] == [1]
    assert seen["closed"] == 1
    assert result.metadata["stop_reason"] == "task_complete"


@pytest.mark.parametrize("reply,calls,protocol", [
    ("```bash\nrm unfinished\n```", [], "fenced_shell"),
    ("", tool(command="python -c \"unfinished"), "openai_tools"),
    ("TASK_COMPLETE", [], "openai_tools"),
])
def test_length_never_executes_or_submits(episode, reply, calls, protocol):
    result, seen = asyncio.run(episode([(reply, calls, "length", 8192)], options={"action_protocol": protocol}))
    assert not seen["commands"] and not seen["tool"]
    assert seen["final"] == [0]
    assert result.metadata["stop_reason"] == "max_tokens_per_turn"


def test_alias_history_canonical_but_raw_trace_unchanged(episode):
    original = tool("shell=shell")
    result, seen = asyncio.run(episode([
        ("", original, "tool_calls", 3), ("TASK_COMPLETE", [], "stop", 1),
    ], options={"action_protocol": "openai_tools"}))
    assert seen["calls"][1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "shell"
    assert original[0]["function"]["name"] == "shell=shell"
    assert result.metadata["turn_metrics"][0]["tool_calls"][0]["function"]["name"] == "shell=shell"


def test_total_budget_caps_last_call(episode):
    result, seen = asyncio.run(episode([
        ("```bash\npwd\n```", [], "stop", 3), ("unfinished", [], "length", 2),
    ], options={"max_response_tokens": 5, "max_tokens_per_turn": 4}))
    assert [c["options"]["max_tokens_per_turn"] for c in seen["calls"]] == [4, 2]
    assert len(seen["commands"]) == 1
    assert result.metadata["stop_reason"] == "max_response_tokens"
    assert seen["final"] == [1]


@pytest.mark.parametrize("error", [RuntimeError("tool failed"), asyncio.CancelledError()])
def test_tool_failure_or_cancellation_finalizes(episode, error):
    result, seen = asyncio.run(episode([("```bash\npwd\n```", [], "stop", 3)], execute_error=error))
    assert seen["tool"] == [0] and seen["final"] == [0]
    assert seen["closed"] == 1
    if isinstance(error, asyncio.CancelledError):
        assert seen["cancelled"]
    else:
        assert result.status == Sample.Status.FAILED


def test_malformed_xml_and_reasoning_not_submission():
    assert h.structured_terminal_reason("<tool_call> treasure\n<parameter=command>pwd", []) == "tool_format_error"
    assert h.structured_terminal_reason("", [], "Maybe reply:\nTASK_COMPLETE") == "no_command"


def test_miles_plain_summary_is_not_submission(episode):
    result, seen = asyncio.run(episode([("Done, fixed it.", [], "stop", 3)]))
    assert result.metadata["stop_reason"] == "no_command"
    assert seen["final"] == [0] and not seen["commands"]


def test_miles_review_is_not_a_shell_command(episode):
    result, seen = asyncio.run(episode([("REQUEST_REVIEW", [], "stop", 3)]))
    assert result.metadata["stop_reason"] == "unsupported_review_request"
    assert not seen["commands"]


@pytest.mark.parametrize("command,reason", [
    ("task_complete", "task_complete"),
    ("TASK_COMPLETE done", "task_complete"),
    ("REQUEST_REVIEW tests", "unsupported_review_request"),
])
def test_miles_control_tokens_inside_fence(episode, command, reason):
    result, seen = asyncio.run(episode([(f"```bash\n{command}\n```", [], "stop", 3)]))
    assert result.metadata["stop_reason"] == reason
    assert not seen["commands"]


@pytest.mark.parametrize("error", [RuntimeError("render failed"), asyncio.CancelledError()])
def test_next_turn_failure_closes_previous_parent_too(episode, error):
    result, seen = asyncio.run(episode([("```bash\npwd\n```", [], "stop", 3), error]))
    assert seen["final"] == [0, 1]
    assert seen["closed"] == 1


def test_render_uses_runtime_schema_without_mutating_tools():
    from sglang.srt.entrypoints.openai.protocol import Tool

    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured.update(messages=messages, **kwargs)
            return "prompt"

        def encode(self, *args, **kwargs):
            return [1]

    tools = [copy.deepcopy(h._SHELL_TOOL)]
    before = copy.deepcopy(tools)
    messages = [{"role": "assistant", "content": None, "tool_calls": [{
        "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
    }]}]
    h._render_prompt(Tokenizer(), messages, enable_thinking=True, tools=tools)
    assert tools == before
    assert captured["tools"] == [Tool.model_validate(before[0]).model_dump()]
    assert captured["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"command":"pwd"}'


@pytest.mark.parametrize("options", [{}, {"docker_user": "root", "docker_cpu": 2, "docker_memory_gb": 4}])
def test_docker_parity_is_opt_in(monkeypatch, options):
    from data.swe_bench.harness import DockerTask, _create_task

    commands = []

    async def fake_run(self, *args, timeout):
        commands.append(args)
        return 0, "fake-container"

    monkeypatch.setattr(DockerTask, "_run_host", fake_run)
    task = _create_task({"instance_id": "a__b-1", "image_name": "fake-image"}, options)
    asyncio.run(task.start())
    command = next(c for c in commands if c[:2] == ("docker", "run"))
    if options:
        assert command[command.index("--user") + 1] == "root"
        assert command[command.index("--cpus") + 1] == "2"
        assert command[command.index("--memory") + 1] == "4g"
        assert "OMP_NUM_THREADS=2" in command and "OPENBLAS_NUM_THREADS=2" in command
        assert task.metrics["container_user"] == "root"
    else:
        assert "--user" not in command and "--cpus" not in command and "--memory" not in command


def test_tool_timeout_stops_before_next_model_call(episode):
    result, seen = asyncio.run(episode([
        ("```bash\nsleep 10\n```", [], "stop", 3),
    ], execute_code=124))
    assert result.metadata["stop_reason"] == "command_timeout"
    assert len(seen["calls"]) == 1 and seen["final"] == [0]
    assert seen["closed"] == 1


@pytest.mark.parametrize("command,expected", [
    ("printf ok", 0),
    ("exit 137", 137),
    ("kill -KILL $$", 137),
    ("trap '' TERM; sleep 5", 124),
])
def test_timeout_supervisor_distinguishes_kill_from_deadline(command, expected):
    import subprocess
    import sys
    from data.swe_bench.harness import _DOCKER_TIMEOUT_RUNNER

    result = subprocess.run(
        [sys.executable, "-c", _DOCKER_TIMEOUT_RUNNER, ".1", "bash", "-c", command],
        capture_output=True, timeout=3,
    )
    assert result.returncode == expected
    if expected == 0:
        assert result.stdout == b"ok"


@pytest.mark.parametrize("enabled,phase,supervised", [
    (False, "agent_tool", False), (True, "agent_tool", True),
    (True, "verifier", False),
])
def test_supervisor_only_changes_opted_in_agent_tools(monkeypatch, enabled, phase, supervised):
    from data.swe_bench.harness import DockerTask, _create_task, _DOCKER_TIMEOUT_RUNNER

    captured = []
    async def run(self, *args, timeout):
        captured.append(args)
        return 0, "ok"
    monkeypatch.setattr(DockerTask, "_run_host", run)
    task = _create_task({"instance_id": "x__y-1"}, {"docker_normalize_tool_timeout": enabled})
    task.container_id = "test-container"
    asyncio.run(task.execute("pwd", phase=phase))
    assert (_DOCKER_TIMEOUT_RUNNER in captured[0]) is supervised


@pytest.mark.parametrize("cleanup_code", [0, 1])
def test_timeout_requires_container_quiescence(monkeypatch, cleanup_code):
    from data.swe_bench.harness import DockerTask, _DOCKER_TIMEOUT_QUIESCE

    captured = []
    async def run(self, *args, timeout):
        captured.append(args)
        return (124, "partial output") if len(captured) == 1 else (cleanup_code, "")
    monkeypatch.setattr(DockerTask, "_run_host", run)
    task = DockerTask("fake", 1, 1, "none", container_id="owned",
                      normalize_tool_timeout=True)
    if cleanup_code:
        with pytest.raises(RuntimeError, match="did not quiesce"):
            asyncio.run(task.execute("sleep 2"))
        assert not task.metrics["exec_calls"]
    else:
        assert asyncio.run(task.execute("sleep 2")) == (124, "partial output")
        assert task.metrics["exec_calls"][0]["timeout_cleanup_seconds"] >= 0
    assert captured[1][:5] == ("docker", "exec", "--user", "root", "owned")
    assert _DOCKER_TIMEOUT_QUIESCE in captured[1]


def test_cleanup_failure_does_not_capture_or_grade(episode):
    result, seen = asyncio.run(episode([
        ("```bash\nsleep 10\n```", [], "stop", 3),
    ], execute_error=RuntimeError("tool timeout cleanup did not quiesce")))
    assert result.status == Sample.Status.FAILED
    assert "final_before_capture" not in seen
    assert seen["final"] == [0] and seen["closed"] == 1
