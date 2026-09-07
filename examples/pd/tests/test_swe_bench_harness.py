from __future__ import annotations

import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

from data.api import LoadContext
from data.swe_bench.harness import (
    DaytonaTask,
    DockerTask,
    _create_task,
    _image_name,
    _verifier_semaphore,
    command_from_reply,
)
from data.swe_bench.loader import load_samples
from data.swe_bench.mini_harness import (
    _capture_workspace_patch_best_effort,
    _parse_tool_calls,
)
from data.swe_bench.verifier import (
    RepositoryBaseline,
    VerifierResult,
    capture_repository_patch,
    patch_metadata,
    prepare_repository_baseline,
    run_inline_verifier,
)
from slime.utils.types import Sample


def test_swe_bench_loader_preserves_problem_text_and_row_metadata(tmp_path):
    path = tmp_path / "swe.jsonl"
    row = {
        "instance_id": "owner__repo-1",
        "problem_statement": "Unicode 雪人 ☃\nquotes: '$HOME' and `cmd`",
        "base_commit": "abc123",
        "FAIL_TO_PASS": '["test_one"]',
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    samples = load_samples(
        LoadContext(args=SimpleNamespace(), tokenizer=None, processor=None),
        SimpleNamespace(path=str(path), options={}),
    )

    assert len(samples) == 1
    assert samples[0].prompt == ""
    assert samples[0].metadata == row


def test_swe_bench_command_parser_requires_exactly_one_fenced_action():
    assert command_from_reply("THOUGHT\n```bash\nrg -n 'needle' src\n```") == "rg -n 'needle' src"
    assert command_from_reply("plain rm -rf /not-executed") is None
    assert command_from_reply("```bash\none\n```\n```bash\ntwo\n```") is None


def test_miniswe_qwen_tool_parser_preserves_reasoning_and_multiple_actions():
    reply = (
        "<think>inspect first</think>\nI will inspect both.\n"
        '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>\n'
        '<tool_call>{"name":"bash","arguments":"{\\"command\\":\\"ls -la\\"}"}</tool_call>'
    )

    content, reasoning, calls, actions = _parse_tool_calls(reply, 3)

    assert content == "I will inspect both."
    assert reasoning == "inspect first"
    assert [action["command"] for action in actions] == ["pwd", "ls -la"]
    assert [call["id"] for call in calls] == ["call_3_0", "call_3_1"]


def test_miniswe_qwen_tool_parser_rejects_non_bash_tools():
    reply = '<tool_call>{"name":"python","arguments":{"command":"pass"}}</tool_call>'
    try:
        _parse_tool_calls(reply, 0)
    except ValueError as exc:
        assert "bash" in str(exc)
    else:
        raise AssertionError("non-bash tool call must be rejected")


def test_miniswe_patch_capture_is_diagnostic_only(monkeypatch):
    async def fail_capture(task, commit):
        raise RuntimeError(f"cannot diff {commit}")

    monkeypatch.setattr(
        "data.swe_bench.mini_harness.capture_repository_patch", fail_capture
    )
    metadata = {}
    patch = asyncio.run(
        _capture_workspace_patch_best_effort(object(), "a" * 40, metadata)
    )

    assert patch == ""
    assert metadata["workspace_patch_capture_error"].startswith(
        "RuntimeError: cannot diff"
    )


def test_miniswe_non_submission_skips_patch_capture_and_verifier(monkeypatch):
    events = []

    class Tokenizer:
        pass

    class Task:
        image = "official-image"
        metrics = {"backend": "fake", "exec_calls": [], "uploads": []}

        async def start(self):
            events.append("start")

        async def close(self):
            events.append("close")

    class Agent:
        def __init__(self, model, environment, **config):
            self.model = model
            self.environment = environment
            self.config = config

        def run(self, task):
            events.append("agent")
            return {"exit_status": "TimeExceeded", "submission": ""}

        def save(self, path):
            return {
                "info": {"exit_status": "TimeExceeded"},
                "messages": [],
                "trajectory_format": "mini-swe-agent-1.1",
            }

    baseline = RepositoryBaseline(
        official_base_commit="a" * 40,
        official_tree="b" * 40,
        image_commit="c" * 40,
        image_tree="d" * 40,
        kind="compatibility_descendant",
        image_commits_ahead=1,
        fingerprint="e" * 64,
    )

    async def fake_baseline(task, expected):
        events.append("baseline")
        return baseline

    async def forbidden(*args, **kwargs):
        raise AssertionError("non-submitted trajectories must not be graded or diffed")

    monkeypatch.setattr(
        "data.swe_bench.mini_harness.GenerateState",
        lambda args: SimpleNamespace(tokenizer=Tokenizer()),
    )
    monkeypatch.setattr(
        "data.swe_bench.mini_harness._create_task", lambda metadata, options: Task()
    )
    monkeypatch.setattr(
        "data.swe_bench.mini_harness.prepare_repository_baseline", fake_baseline
    )
    monkeypatch.setattr(
        "data.swe_bench.mini_harness.capture_repository_patch", forbidden
    )
    monkeypatch.setattr(
        "data.swe_bench.mini_harness.run_inline_verifier", forbidden
    )
    monkeypatch.setattr("minisweagent.agents.default.DefaultAgent", Agent)

    args = SimpleNamespace(
        workload_dataset_options={
            "swe_bench": {
                "verifier_mode": "inline",
                "agent_workers": 1,
                "step_limit": 250,
                "wall_time_limit_seconds": 3600,
                "max_consecutive_format_errors": 3,
            }
        },
        max_seq_len=40960,
        sglang_server_concurrency=1,
        hf_checkpoint="Qwen3-32B",
        pd_p_ready_dir="",
    )
    sample = Sample(
        metadata={
            "dataset_id": "swe_bench",
            "instance_id": "owner__repo-1",
            "problem_statement": "fix it",
            "base_commit": "a" * 40,
        }
    )

    result = asyncio.run(
        __import__("data.swe_bench.mini_harness", fromlist=["generate"]).generate(
            args, sample, {"temperature": 0.0, "top_p": 1.0}
        )
    )

    assert events == ["start", "baseline", "agent", "close"]
    assert result.status == Sample.Status.TRUNCATED
    assert result.reward == 0
    assert result.metadata["stop_reason"] == "TimeExceeded"
    assert result.metadata["workspace_patch_capture_skipped"] == "not_submitted"
    assert result.metadata["swe_bench_verifier"]["status"] == "not_submitted"


def test_swe_bench_image_name_allows_per_row_override():
    metadata = {"instance_id": "astropy__astropy-12907"}
    assert _image_name(metadata, {}) == (
        "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    )
    metadata["image_name"] = "registry.example/task:fixed"
    assert _image_name(metadata, {}) == "registry.example/task:fixed"


def test_docker_task_passes_model_command_as_one_exec_argument(monkeypatch):
    observed = []

    async def fake_run(self, *args, timeout):
        observed.append((args, timeout))
        return 0, "ok"

    monkeypatch.setattr(DockerTask, "_run_host", fake_run)
    task = DockerTask("image", command_timeout=5, start_timeout=10, network="none")
    task.container_id = "container"
    command = "printf '%s' '$HOME;`uname`'"

    asyncio.run(task.execute(command))

    assert observed[0][0][-1] == command
    assert observed[0][0][:4] == ("docker", "exec", "-w", "/testbed")


def test_docker_task_upload_uses_docker_cp_and_removes_temporary_file(monkeypatch):
    observed = []

    async def fake_run(self, *args, timeout):
        source = Path(args[-2])
        observed.append((args, timeout, source.read_bytes()))
        return 0, ""

    monkeypatch.setattr(DockerTask, "_run_host", fake_run)
    task = DockerTask("image", command_timeout=5, start_timeout=10, network="none")
    task.container_id = "container"

    asyncio.run(task.upload_bytes(b"hidden verifier", "/eval.sh"))

    args, timeout, contents = observed[0]
    assert args[0:2] == ("docker", "cp")
    assert args[-1] == "container:/eval.sh"
    assert timeout == 10
    assert contents == b"hidden verifier"
    assert not Path(args[-2]).exists()


def test_sandbox_backend_selection_keeps_daytona_optional(monkeypatch):
    metadata = {"instance_id": "owner__repo-1"}
    docker = _create_task(metadata, {"sandbox_backend": "docker"})
    assert isinstance(docker, DockerTask)

    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")
    daytona = _create_task(metadata, {"sandbox_backend": "daytona"})
    assert isinstance(daytona, DaytonaTask)
    assert daytona.api_key == "test-key"


def test_verifier_semaphore_is_shared_per_inference_args():
    args = SimpleNamespace()

    async def inspect():
        first = _verifier_semaphore(args, 4)
        second = _verifier_semaphore(args, 4)
        other = _verifier_semaphore(args, 2)
        return first, second, other

    first, second, other = asyncio.run(inspect())
    assert first is second
    assert first is not other


def test_daytona_precise_process_timeout_becomes_timeout_error():
    timeout_type = type("DaytonaProcessExecutionTimeoutError", (Exception,), {})

    class Process:
        def exec(self, command, timeout):
            raise timeout_type(f"{command} exceeded {timeout}")

    task = DaytonaTask("image", 5, 10, "key", "owner__repo-1")
    task.sandbox = SimpleNamespace(process=Process())

    async def execute():
        try:
            await task.execute("pytest", timeout=1)
        except TimeoutError:
            return True
        return False

    assert asyncio.run(execute()) is True


class _FakeSandbox:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []
        self.uploads = []

    async def execute(self, command, *, timeout=None, phase="internal"):
        self.commands.append((command, timeout, phase))
        return self.responses.pop(0)

    async def upload_bytes(self, contents, path):
        self.uploads.append((contents, path))


def test_repository_baseline_accepts_exact_clean_image():
    commit = "a" * 40
    tree = "b" * 40
    sandbox = _FakeSandbox(
        [
            (0, ""),
            (0, commit + "\n"),
            (0, tree + "\n"),
            (0, tree + "\n"),
            (0, "0\n"),
        ]
    )

    baseline = asyncio.run(prepare_repository_baseline(sandbox, commit))

    assert baseline.kind == "exact"
    assert baseline.image_commits_ahead == 0
    assert len(baseline.fingerprint) == 64


def test_repository_baseline_snapshots_dirty_image_without_rewriting_worktree():
    commit = "a" * 40
    source_tree = "b" * 40
    snapshot_commit = "c" * 40
    snapshot_tree = "d" * 40
    sandbox = _FakeSandbox(
        [
            (0, " M tox.ini\n"),
            (0, commit + "\n"),
            (0, source_tree + "\n"),
            (0, source_tree + "\n"),
            (0, f"{snapshot_commit} {snapshot_tree}\n"),
            (0, "1\n"),
        ]
    )

    baseline = asyncio.run(prepare_repository_baseline(sandbox, commit))

    assert baseline.kind == "dirty_snapshot_exact"
    assert baseline.image_commit == snapshot_commit
    assert baseline.image_tree == snapshot_tree
    assert baseline.source_image_commit == commit
    assert baseline.initial_worktree_status == "M tox.ini"
    snapshot_command = sandbox.commands[4][0]
    assert "GIT_INDEX_FILE" in snapshot_command
    assert "commit-tree" in snapshot_command


def test_patch_capture_includes_tracked_and_untracked_changes():
    commit = "a" * 40
    sandbox = _FakeSandbox(
        [
            (0, "diff --git a/a.py b/a.py\ntracked\n"),
            (0, "diff --git a/new.py b/new.py\nuntracked\n"),
        ]
    )

    patch = asyncio.run(capture_repository_patch(sandbox, commit))

    assert "tracked" in patch
    assert "untracked" in patch
    assert patch.endswith("\n")
    assert patch_metadata(patch)["model_patch_chars"] == len(patch)


def test_inline_verifier_records_canonical_resolved_result(monkeypatch):
    metadata = {"instance_id": "owner__repo-1"}
    sandbox = _FakeSandbox([(1, "official test output")])
    fake_spec = object()
    monkeypatch.setattr(
        "data.swe_bench.verifier.build_eval_script",
        lambda row: (fake_spec, "#!/bin/bash\npytest\n"),
    )
    monkeypatch.setattr(
        "data.swe_bench.verifier.grade_test_output",
        lambda row, spec, patch, output: {
            row["instance_id"]: {"resolved": True, "tests_status": {}}
        },
    )

    result = asyncio.run(
        run_inline_verifier(sandbox, metadata, "model patch", timeout_seconds=60)
    )

    assert result == VerifierResult(
        status="completed",
        resolved=True,
        reward=1,
        report={"owner__repo-1": {"resolved": True, "tests_status": {}}},
        test_exit_code=1,
        timed_out=False,
        duration_seconds=result.duration_seconds,
        output_tail="official test output",
    )
    assert sandbox.uploads == [(b"#!/bin/bash\npytest\n", "/eval.sh")]


def test_inline_verifier_timeout_is_scored_failure(monkeypatch):
    sandbox = _FakeSandbox([(124, "partial output")])
    monkeypatch.setattr(
        "data.swe_bench.verifier.build_eval_script",
        lambda row: (object(), "#!/bin/bash\npytest\n"),
    )

    result = asyncio.run(
        run_inline_verifier(
            sandbox, {"instance_id": "owner__repo-1"}, "patch", timeout_seconds=1
        )
    )

    assert result.status == "timeout"
    assert result.resolved is False
    assert result.reward == 0
    assert result.timed_out is True


def test_generate_runs_hidden_verifier_after_agent_stops(monkeypatch):
    events = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def encode(self, value, add_special_tokens=False):
            return [1, 2]

        def decode(self, tokens, skip_special_tokens=False):
            if tokens == [9]:
                return "```bash\necho SWE_TASK_COMPLETE\n```"
            return "decoded"

    class Task:
        image = "official-image"
        metrics = {"backend": "fake", "exec_calls": [], "uploads": []}

        async def start(self):
            events.append("start")

        async def close(self):
            events.append("close")

    task = Task()
    baseline = RepositoryBaseline(
        official_base_commit="a" * 40,
        official_tree="b" * 40,
        image_commit="c" * 40,
        image_tree="d" * 40,
        kind="compatibility_descendant",
        image_commits_ahead=1,
        fingerprint="e" * 64,
    )

    async def fake_post(url, payload):
        events.append("generation")
        return {
            "output_ids": [9],
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    async def fake_baseline(sandbox, expected):
        events.append("baseline")
        return baseline

    async def fake_capture(sandbox, commit):
        events.append("capture")
        return "model patch\n"

    async def fake_verify(sandbox, metadata, patch, **kwargs):
        events.append("verify")
        return VerifierResult(
            status="completed",
            resolved=True,
            reward=1,
            report={metadata["instance_id"]: {"resolved": True}},
            test_exit_code=0,
            timed_out=False,
            duration_seconds=0.1,
            output_tail="passed",
        )

    class Span:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, values):
            pass

    monkeypatch.setattr("data.swe_bench.harness.GenerateState", lambda args: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr("data.swe_bench.harness._create_task", lambda metadata, options: task)
    monkeypatch.setattr("data.swe_bench.harness.prepare_repository_baseline", fake_baseline)
    monkeypatch.setattr("data.swe_bench.harness.capture_repository_patch", fake_capture)
    monkeypatch.setattr("data.swe_bench.harness.run_inline_verifier", fake_verify)
    monkeypatch.setattr("data.swe_bench.harness.post", fake_post)
    monkeypatch.setattr("data.swe_bench.harness.lifecycle_enabled", lambda: False)
    monkeypatch.setattr("data.swe_bench.harness.generation_has_visible_content", lambda *args: True)
    monkeypatch.setattr("data.swe_bench.harness.confirm_agentic_generation_final", lambda *args, **kwargs: None)
    monkeypatch.setattr("data.swe_bench.harness.dashboard_span", lambda *args, **kwargs: Span())
    monkeypatch.setattr("data.swe_bench.harness.sglang_meta_attrs", lambda value: {})

    args = SimpleNamespace(
        workload_dataset_options={
            "swe_bench": {
                "verifier_mode": "inline",
                "store_model_patch": True,
            }
        },
        max_seq_len=40960,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        pd_p_ready_dir="",
    )
    sample = Sample(
        metadata={
            "dataset_id": "swe_bench",
            "instance_id": "owner__repo-1",
            "problem_statement": "fix it",
            "base_commit": "a" * 40,
        }
    )

    result = asyncio.run(
        __import__("data.swe_bench.harness", fromlist=["generate"]).generate(
            args, sample, {}
        )
    )

    assert events == ["start", "baseline", "generation", "capture", "verify", "close"]
    assert result.reward == 1
    assert result.metadata["model_patch"] == "model patch\n"
    assert result.metadata["swe_bench_verifier"]["resolved"] is True
