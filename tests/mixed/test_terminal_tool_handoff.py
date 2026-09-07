from __future__ import annotations

import asyncio
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample


MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))
spec = importlib.util.spec_from_file_location("terminal_agent_under_test", MIXED_DIR / "terminal_agent.py")
assert spec is not None and spec.loader is not None
terminal_agent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = terminal_agent
spec.loader.exec_module(terminal_agent)


class CharTokenizer:
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class FakeSpan:
    def update(self, attrs):
        pass


@contextmanager
def fake_span(*args, **kwargs):
    yield FakeSpan()


class FakeAction:
    def __init__(self, action_type, command=None):
        self.action_type = action_type
        self.command = command


class FakeResult:
    def __init__(self, *, instruction="", output="", reward=0.0, info=None):
        self.observation = SimpleNamespace(
            instruction=instruction,
            output=output,
            info=info or {},
        )
        self.reward = reward


def _args():
    return SimpleNamespace(
        sglang_context_length=4096,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        current_policy_version=0,
        current_rollout_id=0,
        sglang_speculative_algorithm=None,
        use_slime_dashboard=True,
    )


def test_terminal_thinking_is_enabled_by_default(monkeypatch):
    class CaptureTokenizer(CharTokenizer):
        def __init__(self):
            self.enable_thinking = None

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            self.enable_thinking = kwargs.get("enable_thinking")
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )

    monkeypatch.delenv("TERMINAL_ENABLE_THINKING", raising=False)
    tokenizer = CaptureTokenizer()
    terminal_agent._encode_prompt(tokenizer, [], "task")
    assert tokenizer.enable_thinking is True

    monkeypatch.setenv("TERMINAL_ENABLE_THINKING", "0")
    terminal_agent._encode_prompt(tokenizer, [], "task")
    assert tokenizer.enable_thinking is False


def test_terminal_prompt_documents_stateless_shell_and_workdir():
    tokenizer = CharTokenizer()
    token_ids = terminal_agent._encode_prompt(
        tokenizer,
        [{"role": "system", "content": "terminal agent"}],
        "solve the task",
        "/app/personal-site",
    )
    rendered = tokenizer.decode(token_ids)
    assert "solve the task" in rendered
    assert "standalone `cd` does not persist" in rendered
    assert "Initial working directory: /app/personal-site" in rendered


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("```bash\necho hello\n```", "echo hello"),
        ("```bash\necho hello\n```<|im_end|>", "echo hello"),
        ("TASK_COMPLETE<|im_end|>", "TASK_COMPLETE"),
        ("echo bare-command<|im_end|>", None),
        ("The required package is unavailable. Please install it.", None),
        (
            "<think>inspect first</think>\n```bash\npython test.py\n```",
            "python test.py",
        ),
        (
            "<think>reasoning with a misleading ```bash\\nwrong\\n``` block"
            "</think>\n```bash\nright\n```",
            "right",
        ),
        ("reasoning without an opening tag</think>\n```bash\npwd\n```", "pwd"),
        ("<think>unfinished reasoning\n```", None),
        ("```bash\nunfinished command", None),
        ("```python\nprint('not a shell command')\n```", None),
    ],
)
def test_terminal_command_parser_separates_reasoning_from_executable_content(reply, expected):
    assert terminal_agent._command_from_reply(reply) == expected


def test_terminal_client_timeout_covers_long_official_verifiers():
    client = terminal_agent._Tbench2Client("http://127.0.0.1:8003")
    assert client.message_timeout > 3600


def test_terminal_clean_capacity_close_is_retried():
    error = terminal_agent.RecoverableTerminalInfraError(
        "environment reset failed: received 1000 (OK); then sent 1000 (OK)"
    )
    assert terminal_agent._is_capacity_error(error)


def test_terminal_live_session_limit_waits_until_a_session_closes():
    async def scenario():
        manager = terminal_agent.TerminalSessionManager()

        class FakeEnv:
            async def __aexit__(self, *args):
                pass

        async def fake_open(task_id):
            return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

        manager._open = fake_open
        first = asyncio.create_task(manager.acquire("one", "task", live_limit=2))
        second = asyncio.create_task(manager.acquire("two", "task", live_limit=2))
        third = asyncio.create_task(manager.acquire("three", "task", live_limit=2))
        await asyncio.gather(first, second)
        await asyncio.sleep(0)

        assert len(manager.sessions) == 2
        assert not third.done()
        assert manager._live_waiters == 1

        await manager.close("one")
        await asyncio.wait_for(third, timeout=1)
        assert len(manager.sessions) == 2
        assert manager._live_in_use == 2

        await manager.close_all()
        assert manager._live_in_use == 0

    asyncio.run(scenario())


def test_terminal_live_wait_notifies_scheduler_once():
    async def scenario():
        manager = terminal_agent.TerminalSessionManager()
        waits = 0

        class FakeEnv:
            async def __aexit__(self, *args):
                pass

        async def fake_open(task_id):
            return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

        def on_wait():
            nonlocal waits
            waits += 1

        manager._open = fake_open
        await manager.acquire("one", "task", live_limit=1)
        second = asyncio.create_task(
            manager.acquire("two", "task", live_limit=1, on_live_wait=on_wait)
        )
        await asyncio.sleep(0)
        assert waits == 1
        assert not second.done()

        await manager.close("one")
        await asyncio.wait_for(second, timeout=1)
        assert waits == 1
        await manager.close_all()

    asyncio.run(scenario())


def test_terminal_reset_concurrency_is_independent_of_live_sessions(monkeypatch):
    async def scenario():
        manager = terminal_agent.TerminalSessionManager()
        release_resets = asyncio.Event()
        two_resets_started = asyncio.Event()
        active_resets = 0
        max_active_resets = 0
        reset_count = 0

        class FakeEnv:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def reset(self, task_id):
                nonlocal active_resets, max_active_resets, reset_count
                active_resets += 1
                reset_count += 1
                max_active_resets = max(max_active_resets, active_resets)
                if reset_count >= 2:
                    two_resets_started.set()
                await release_resets.wait()
                active_resets -= 1
                return SimpleNamespace(
                    observation=SimpleNamespace(
                        error="", instruction=f"instruction-{task_id}", info={}
                    )
                )

        monkeypatch.setattr(terminal_agent, "_load_tbench2", lambda: (FakeEnv, FakeAction))
        tasks = [
            asyncio.create_task(
                manager.acquire(f"sample-{index}", f"task-{index}", live_limit=4, reset_limit=2)
            )
            for index in range(3)
        ]
        await asyncio.wait_for(two_resets_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert manager._resets_active == 2
        assert manager._reset_waiters == 1
        assert sum(task.done() for task in tasks) == 0

        release_resets.set()
        await asyncio.gather(*tasks)
        assert max_active_resets == 2
        assert len(manager.sessions) == 3
        await manager.close_all()

    asyncio.run(scenario())


def test_unterminated_reasoning_is_never_executed(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()
    actions = []

    class FakeEnv:
        async def step(self, action):
            actions.append(action.action_type)
            assert action.action_type == "evaluate"
            return FakeResult(reward=0.0)

        async def __aexit__(self, *args):
            pass

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        text = "<think>\nI am stuck and never closed the reasoning block.\n```"
        ids = state.tokenizer.encode(text)
        return ids, [-0.1] * len(ids), "stop", {
            "finish_reason": {"type": "stop"},
            "completion_tokens": len(ids),
        }

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "4096")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-malformed-reasoning"},
        session_id="sample-malformed-reasoning",
    )
    completed = asyncio.run(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))

    assert completed.status == Sample.Status.TRUNCATED
    assert completed.metadata["terminal_reward"] == 0.0
    assert completed.metadata["terminal_turns"] == 1
    assert completed.metadata["terminal_tool_call_count"] == 0
    assert completed.metadata["stop_reason"] == "invalid_reply"
    assert actions == ["evaluate"]


def test_explanatory_prose_is_not_executed_or_retried(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()
    generation_calls = 0
    actions = []

    class FakeEnv:
        async def step(self, action):
            actions.append(action.action_type)
            assert action.action_type == "evaluate"
            return FakeResult(reward=0.0)

        async def __aexit__(self, *args):
            pass

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        nonlocal generation_calls
        generation_calls += 1
        text = "The required package is unavailable. Please install it.<|im_end|>"
        ids = state.tokenizer.encode(text)
        return ids, [-0.1] * len(ids), "stop", {
            "finish_reason": {"type": "stop"},
            "completion_tokens": len(ids),
        }

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "4096")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-prose"},
        session_id="sample-prose",
    )
    completed = asyncio.run(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))

    assert completed.status == Sample.Status.TRUNCATED
    assert completed.metadata["stop_reason"] == "invalid_reply"
    assert completed.metadata["terminal_turns"] == 1
    assert completed.metadata["terminal_tool_call_count"] == 0
    assert generation_calls == 1
    assert actions == ["evaluate"]


def test_shell_finishes_across_update_and_same_session_resumes(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    generation_inputs = []
    commands = []

    class FakeEnv:
        exited = 0

        async def step(self, action):
            if action.action_type == "exec":
                commands.append(action.command)
                state.abort_epoch += 1
                state.aborted = True
                return FakeResult(output="command output")
            assert action.action_type == "evaluate"
            return FakeResult(reward=1.0)

        async def __aexit__(self, *args):
            self.exited += 1

    manager = terminal_agent.TerminalSessionManager()
    env = FakeEnv()

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(env, FakeAction, task_id, "solve this task")

    manager._open = fake_open

    async def fake_generate_step(url, tokens, params):
        generation_inputs.append(list(tokens))
        text = "```bash\necho hello\n```" if len(generation_inputs) == 1 else "TASK_COMPLETE"
        ids = state.tokenizer.encode(text)
        return ids, [-0.1] * len(ids), "stop", {
            "finish_reason": {"type": "stop"},
            "completion_tokens": len(ids),
        }

    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "4096")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-a"},
        session_id="sample-a",
    )

    async def scenario():
        partial = await terminal_agent.generate(_args(), sample, {"max_new_tokens": 128})
        assert partial.status == Sample.Status.ABORTED
        assert partial.metadata["stop_reason"] == "tool_completed_after_weight_update"
        assert "command output" in partial.response
        assert partial.metadata["terminal_task_instruction"] == "solve this task"
        assert "sample-a" in manager.sessions
        saved_tokens = list(partial.tokens)

        state.aborted = False
        completed = await terminal_agent.generate(_args(), partial, {"max_new_tokens": 128})
        assert completed.status == Sample.Status.COMPLETED
        assert completed.metadata["terminal_reward"] == 1.0
        assert generation_inputs[1] == saved_tokens
        assert commands == ["echo hello"]
        assert "sample-a" not in manager.sessions
        assert env.exited == 1

    asyncio.run(scenario())


def test_aborted_assistant_prefix_is_prefilled_and_command_runs_once(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    generation_inputs = []
    commands = []

    class FakeEnv:
        async def step(self, action):
            if action.action_type == "exec":
                commands.append(action.command)
                return FakeResult(output="ok")
            return FakeResult(reward=1.0)

        async def __aexit__(self, *args):
            pass

    manager = terminal_agent.TerminalSessionManager()

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    manager._open = fake_open
    prefix = "```bash\necho par"
    suffix = "tial\n```"

    async def fake_generate_step(url, tokens, params):
        generation_inputs.append(list(tokens))
        if len(generation_inputs) == 1:
            state.abort_epoch += 1
            state.aborted = True
            text, reason = prefix, "abort"
        elif len(generation_inputs) == 2:
            text, reason = suffix, "stop"
        else:
            text, reason = "TASK_COMPLETE", "stop"
        ids = state.tokenizer.encode(text)
        return ids, [-0.1] * len(ids), reason, {
            "finish_reason": {"type": reason},
            "completion_tokens": len(ids),
        }

    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "4096")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-b"},
        session_id="sample-b",
    )

    async def scenario():
        partial = await terminal_agent.generate(_args(), sample, {"max_new_tokens": 128})
        assert partial.status == Sample.Status.ABORTED
        assert partial.response.endswith(prefix)
        partial_prefix_length = partial.metadata["partial_rollout_prefix_length"]
        assert partial_prefix_length == partial.response_length
        saved_tokens = list(partial.tokens)

        state.aborted = False
        completed = await terminal_agent.generate(_args(), partial, {"max_new_tokens": 128})
        assert completed.status == Sample.Status.COMPLETED
        assert completed.metadata["partial_rollout_prefix_length"] == partial_prefix_length
        assert completed.metadata["partial_rollout_prefix_length"] < completed.response_length
        assert generation_inputs[1] == saved_tokens
        assert commands == ["echo partial"]

    asyncio.run(scenario())


def test_length_limited_generation_continues_same_assistant_turn(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    generation_inputs = []
    commands = []

    class FakeEnv:
        async def step(self, action):
            if action.action_type == "exec":
                commands.append(action.command)
                return FakeResult(output="ok")
            return FakeResult(reward=1.0)

        async def __aexit__(self, *args):
            pass

    manager = terminal_agent.TerminalSessionManager()

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    manager._open = fake_open

    async def fake_generate_step(url, tokens, params):
        generation_inputs.append(list(tokens))
        if len(generation_inputs) == 1:
            text, reason = "<think>long reasoning", "length"
        elif len(generation_inputs) == 2:
            text, reason = "</think>\n```bash\npwd\n```", "stop"
        else:
            text, reason = "TASK_COMPLETE", "stop"
        ids = state.tokenizer.encode(text)
        return ids, [-0.1] * len(ids), reason, {
            "finish_reason": {"type": reason},
            "completion_tokens": len(ids),
        }

    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "4096")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-length"},
        session_id="sample-length",
    )
    completed = asyncio.run(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))

    assert completed.status == Sample.Status.COMPLETED
    assert completed.metadata["terminal_reward"] == 1.0
    assert commands == ["pwd"]
    assert len(generation_inputs) == 3
    assert completed.metadata["terminal_generation_chunk_count"] == 3
    assert completed.metadata["terminal_generation_length_chunk_count"] == 1
    assert "<think>long reasoning</think>" in completed.response


def test_length_limited_generation_has_a_per_turn_chunk_cap(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()
    generation_inputs = []
    actions = []

    class FakeEnv:
        async def step(self, action):
            actions.append(action.action_type)
            assert action.action_type == "evaluate"
            return FakeResult(reward=0.0)

        async def __aexit__(self, *args):
            pass

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        generation_inputs.append(list(tokens))
        text = "<think>still reasoning"
        ids = state.tokenizer.encode(text)
        return ids, [-0.1] * len(ids), "length", {
            "finish_reason": {"type": "length"},
            "completion_tokens": len(ids),
        }

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "100000")
    monkeypatch.setenv("TERMINAL_MAX_GENERATION_CHUNKS_PER_TURN", "2")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-length-cap"},
        session_id="sample-length-cap",
    )
    completed = asyncio.run(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))

    assert completed.status == Sample.Status.TRUNCATED
    assert completed.metadata["stop_reason"] == "generation_length_limit"
    assert completed.metadata["terminal_turns"] == 1
    assert completed.metadata["terminal_generation_chunk_count"] == 2
    assert completed.metadata["terminal_generation_length_chunk_count"] == 2
    assert completed.metadata["terminal_current_assistant_chunk_count"] == 2
    assert len(generation_inputs) == 2
    assert actions == ["evaluate"]


def test_generation_chunk_cap_survives_partial_resume(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()
    generation_calls = 0
    actions = []

    class FakeEnv:
        async def step(self, action):
            actions.append(action.action_type)
            assert action.action_type == "evaluate"
            return FakeResult(reward=0.0)

        async def __aexit__(self, *args):
            pass

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        nonlocal generation_calls
        generation_calls += 1
        state.abort_epoch += 1
        text = "<think>partial reasoning"
        ids = state.tokenizer.encode(text)
        return ids, [-0.1] * len(ids), "abort", {
            "finish_reason": {"type": "abort"},
            "completion_tokens": len(ids),
        }

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "100000")
    monkeypatch.setenv("TERMINAL_MAX_GENERATION_CHUNKS_PER_TURN", "1")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-resume-length-cap"},
        session_id="sample-resume-length-cap",
    )

    async def scenario():
        partial = await terminal_agent.generate(_args(), sample, {"max_new_tokens": 128})
        assert partial.status == Sample.Status.ABORTED
        assert partial.metadata["terminal_continue_partial_assistant"] is True
        assert partial.metadata["terminal_current_assistant_chunk_count"] == 1

        state.aborted = False
        completed = await terminal_agent.generate(_args(), partial, {"max_new_tokens": 128})
        assert completed.status == Sample.Status.TRUNCATED
        assert completed.metadata["stop_reason"] == "generation_length_limit"
        assert completed.metadata["terminal_turns"] == 1

    asyncio.run(scenario())
    assert generation_calls == 1
    assert actions == ["evaluate"]


def test_terminal_default_turn_and_chunk_limits(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()
    commands = []
    generation_limits = []

    class FakeEnv:
        async def step(self, action):
            if action.action_type == "exec":
                commands.append(action.command)
                return FakeResult(output="ok")
            return FakeResult(reward=0.0)

        async def __aexit__(self, *args):
            pass

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        generation_limits.append(params["max_new_tokens"])
        text = "```bash\ntrue\n```"
        ids = state.tokenizer.encode(text)
        return ids, [0.0] * len(ids), "stop", {
            "finish_reason": {"type": "stop"},
            "completion_tokens": len(ids),
        }

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.delenv("TERMINAL_MAX_TURNS", raising=False)
    monkeypatch.delenv("TERMINAL_TURN_MAX_NEW_TOKENS", raising=False)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "100000")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-default-limits"},
        session_id="sample-default-limits",
    )
    completed = asyncio.run(
        terminal_agent.generate(_args(), sample, {"max_new_tokens": 36864})
    )

    assert completed.metadata["stop_reason"] == "max_turns"
    assert completed.metadata["terminal_turns"] == 64
    assert completed.metadata["terminal_generation_chunk_count"] == 64
    assert completed.metadata["terminal_generation_length_chunk_count"] == 0
    assert len(commands) == 64
    assert generation_limits == [4096] * 64


def test_final_generation_chunk_uses_remaining_context(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()
    observed_limit = None

    class FakeEnv:
        async def step(self, action):
            assert action.action_type == "evaluate"
            return FakeResult(reward=1.0)

        async def __aexit__(self, *args):
            pass

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        nonlocal observed_limit
        observed_limit = params["max_new_tokens"]
        assert observed_limit == 1200 - len(tokens) - terminal_agent._BUDGET_MARGIN
        ids = state.tokenizer.encode("TASK_COMPLETE")
        return ids, [0.0] * len(ids), "stop", {
            "finish_reason": {"type": "stop"},
            "completion_tokens": len(ids),
        }

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.delenv("TERMINAL_TURN_MAX_NEW_TOKENS", raising=False)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "1200")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-final-chunk"},
        session_id="sample-final-chunk",
    )
    completed = asyncio.run(
        terminal_agent.generate(_args(), sample, {"max_new_tokens": 36864})
    )

    assert observed_limit is not None
    assert 0 < observed_limit < 4096
    assert completed.status == Sample.Status.COMPLETED
    assert completed.metadata["terminal_reward"] == 1.0
    assert completed.metadata["terminal_generation_chunk_count"] == 1


def test_missing_live_session_restarts_instead_of_replaying_shell_history(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()

    class FakeEnv:
        async def step(self, action):
            return FakeResult(reward=0.0)

        async def __aexit__(self, *args):
            pass

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(FakeEnv(), FakeAction, task_id, "fresh instruction")

    manager._open = fake_open

    async def fake_generate_step(url, tokens, params):
        ids = state.tokenizer.encode("TASK_COMPLETE")
        return ids, [0.0] * len(ids), "stop", {
            "finish_reason": {"type": "stop"},
            "completion_tokens": len(ids),
        }

    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "4096")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        response="old command and output",
        response_length=22,
        tokens=[1, 2, 3],
        loss_mask=[1, 1],
        rollout_log_probs=[0.0, 0.0],
        status=Sample.Status.ABORTED,
        metadata={
            "task_type": "terminal",
            "task_id": "task-c",
            "terminal_initialized": True,
            "terminal_prompt_token_count": 1,
        },
        session_id="sample-c",
    )

    completed = asyncio.run(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))
    assert completed.status == Sample.Status.COMPLETED
    assert completed.metadata["terminal_session_restart_count"] == 1
    assert "old command" not in completed.response


def test_formal_cancellation_closes_live_terminal_session(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()
    command_started = asyncio.Event()

    class FakeEnv:
        exited = 0

        async def step(self, action):
            command_started.set()
            await asyncio.Event().wait()

        async def __aexit__(self, *args):
            self.exited += 1

    env = FakeEnv()

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(env, FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        text = "```bash\nsleep 100\n```"
        ids = state.tokenizer.encode(text)
        return ids, [0.0] * len(ids), "stop", {
            "finish_reason": {"type": "stop"},
            "completion_tokens": len(ids),
        }

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)
    monkeypatch.setenv("TERMINAL_MAX_SEQ_LEN", "4096")

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-d"},
        session_id="sample-d",
    )

    async def scenario():
        task = asyncio.create_task(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))
        await command_started.wait()
        task.cancel()
        result = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert env.exited == 1
        assert not manager.sessions

    asyncio.run(scenario())


def test_recoverable_infrastructure_failure_is_recycled_without_reward(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()

    class FakeEnv:
        exited = 0

        async def step(self, action):
            return FakeResult(reward=0.0)

        async def __aexit__(self, *args):
            self.exited += 1

    env = FakeEnv()

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(env, FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        raise terminal_agent.RecoverableTerminalInfraError("router unavailable")

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-infra"},
        session_id="sample-infra",
    )
    result = asyncio.run(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))

    assert result.status == Sample.Status.ABORTED
    assert result.reward is None
    assert "terminal_reward" not in result.metadata
    assert result.metadata["stop_reason"] == "infrastructure_error"
    assert env.exited == 1
    assert not manager.sessions


def test_programming_error_is_not_converted_to_negative_training_sample(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), abort_epoch=0, aborted=False)
    manager = terminal_agent.TerminalSessionManager()

    class FakeEnv:
        exited = 0

        async def __aexit__(self, *args):
            self.exited += 1

    env = FakeEnv()

    async def fake_open(task_id):
        return terminal_agent._TerminalSession(env, FakeAction, task_id, "instruction")

    async def fake_generate_step(url, tokens, params):
        raise ValueError("broken invariant")

    manager._open = fake_open
    monkeypatch.setattr(terminal_agent, "_SESSION_MANAGER", manager)
    monkeypatch.setattr(terminal_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(terminal_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(terminal_agent, "dashboard_span", fake_span)

    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_type": "terminal", "task_id": "task-bug"},
        session_id="sample-bug",
    )
    with pytest.raises(ValueError, match="broken invariant"):
        asyncio.run(terminal_agent.generate(_args(), sample, {"max_new_tokens": 128}))

    assert sample.reward is None
    assert env.exited == 1
    assert not manager.sessions


def test_missing_sglang_logprobs_is_recoverable(monkeypatch):
    async def fake_post(url, payload):
        return {
            "text": "TASK_COMPLETE",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 1,
            },
        }

    monkeypatch.setattr(terminal_agent, "post", fake_post)
    with pytest.raises(terminal_agent.RecoverableTerminalInfraError, match="output_token_logprobs"):
        asyncio.run(terminal_agent._generate_step("http://router/generate", [1], {"max_new_tokens": 8}))
