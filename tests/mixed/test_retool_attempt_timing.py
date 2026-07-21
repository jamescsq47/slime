from __future__ import annotations

import asyncio
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample


MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))

retool = importlib.import_module("generate_with_retool")
tool_sandbox = importlib.import_module("tool_sandbox")


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] if text else []}

    def decode(self, token_ids):
        return "answer" if token_ids else ""


class FakeGenerateState:
    def __init__(self, args):
        self.tokenizer = FakeTokenizer()
        self.aborted = False


class FakeSpan:
    def update(self, attrs):
        pass


@contextmanager
def fake_dashboard_span(*args, **kwargs):
    yield FakeSpan()


def test_partial_resume_separates_attempt_and_cumulative_time(monkeypatch):
    async def fake_post(url, payload):
        return {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 1,
                "output_token_logprobs": [(0.0, 2)],
            }
        }

    async def fake_execute_predictions(prediction):
        return "", True

    monkeypatch.setattr(retool, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(retool, "post", fake_post)
    monkeypatch.setattr(retool, "execute_predictions", fake_execute_predictions)
    monkeypatch.setattr(retool, "dashboard_span", fake_dashboard_span)

    args = SimpleNamespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=False,
        mask_offpolicy_math=None,
        mask_offpolicy_qa=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=128,
        context_parallel_size=1,
        max_tokens_per_gpu=128,
        current_policy_version=3,
        current_rollout_id=4,
        use_slime_dashboard=True,
    )
    sampling_params = {"max_new_tokens": 8}
    sample = Sample(
        prompt="question",
        metadata={"task_type": "math", "policy_version": 3, "dispatch_version": 2},
    )

    first = asyncio.run(retool.generate(args, sample, sampling_params))
    assert first.attempt_count == 1
    assert first.partial_resume_count == 0
    assert first.restart_count == 0
    assert first.sample_time == pytest.approx(first.attempt_time)
    assert len(first.metadata["attempt_history"]) == 1
    assert first.metadata["attempt_history"][0]["resume_kind"] == "initial"

    # Simulate a partial sample returned by a weight-update abort. The next
    # invocation must expose its own duration without losing lifetime timing.
    first.status = Sample.Status.ABORTED
    first.sample_time = 7.0
    first.lifetime_attempt_time = 7.0
    first.metadata["cumulative_sample_time"] = 7.0
    first.metadata["lifetime_attempt_time"] = 7.0

    resumed = asyncio.run(retool.generate(args, first, sampling_params))

    assert resumed.attempt_count == 2
    assert resumed.partial_resume_count == 1
    assert resumed.restart_count == 0
    assert resumed.sample_time == pytest.approx(7.0 + resumed.attempt_time)
    assert resumed.lifetime_attempt_time == pytest.approx(resumed.sample_time)
    assert resumed.metadata["attempt_time"] == resumed.attempt_time
    assert resumed.metadata["attempt_count"] == 2
    assert resumed.metadata["cumulative_sample_time"] == resumed.sample_time
    assert len(resumed.metadata["attempt_history"]) == 2
    partial_entry = resumed.metadata["attempt_history"][-1]
    assert partial_entry["resume_kind"] == "partial_resume"
    assert partial_entry["policy_version"] == 3
    assert partial_entry["dispatch_version"] == 2
    assert partial_entry["rollout_id"] == 4
    assert partial_entry["status"] == "completed"

    # A completed peer can be restarted when another member of its group was
    # aborted. Track that separately from a true partial resume.
    previous_lifetime = resumed.lifetime_attempt_time
    restarted = asyncio.run(retool.generate(args, resumed, sampling_params))

    assert restarted.attempt_count == 3
    assert restarted.partial_resume_count == 1
    assert restarted.restart_count == 1
    assert restarted.sample_time == pytest.approx(restarted.attempt_time)
    assert restarted.lifetime_attempt_time == pytest.approx(previous_lifetime + restarted.attempt_time)
    assert restarted.metadata["attempt_history"][-1]["resume_kind"] == "group_restart"
    assert restarted.trace["attempt"] == 2


def test_code_tool_uses_single_semaphore_permit(monkeypatch):
    async def fake_execute_code(code):
        return f"Output:\n{code}"

    monkeypatch.setattr(tool_sandbox, "SEMAPHORE", asyncio.Semaphore(1))
    monkeypatch.setattr(retool.tool_registry.python_sandbox, "execute_code", fake_execute_code)

    observation, done = asyncio.run(
        asyncio.wait_for(
            retool.execute_predictions('<code>print(42)</code>'),
            timeout=0.2,
        )
    )

    assert not done
    assert "Output:\nprint(42)" in observation


def test_tool_result_crossing_weight_update_is_saved_and_resumed_with_full_prefill(monkeypatch):
    class CharTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

        def decode(self, token_ids, **kwargs):
            return "".join(chr(token) for token in token_ids)

    state = SimpleNamespace(tokenizer=CharTokenizer(), aborted=False, abort_epoch=0)
    payloads = []
    generated = [
        '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(42)"}}</tool_call>',
        "#### \\boxed{42}",
    ]

    async def fake_post(url, payload):
        payloads.append(payload)
        text = generated[len(payloads) - 1]
        token_ids = state.tokenizer.encode(text)
        return {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": len(token_ids),
                "output_token_logprobs": [(0.0, token) for token in token_ids],
            }
        }

    tool_calls = 0

    async def fake_execute_predictions(prediction):
        nonlocal tool_calls
        tool_calls += 1
        if tool_calls == 1:
            # The already-running tool returns while weights are syncing.
            state.abort_epoch += 1
            state.aborted = True
            return "\n\n<interpreter>\nOutput:\n42\n</interpreter>\n\n", False
        return "", True

    monkeypatch.setattr(retool, "GenerateState", lambda args: state)
    monkeypatch.setattr(retool, "post", fake_post)
    monkeypatch.setattr(retool, "execute_predictions", fake_execute_predictions)
    monkeypatch.setattr(retool, "dashboard_span", fake_dashboard_span)

    args = SimpleNamespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=False,
        mask_offpolicy_math=None,
        mask_offpolicy_qa=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        context_parallel_size=1,
        max_tokens_per_gpu=4096,
        current_policy_version=0,
        current_rollout_id=0,
        use_slime_dashboard=True,
        enable_tool_delay=False,
    )
    sample = Sample(prompt="question", metadata={"task_type": "math", "policy_version": 0})

    partial = asyncio.run(retool.generate(args, sample, {"max_new_tokens": 512}))

    assert partial.status == Sample.Status.ABORTED
    assert partial.metadata["attempt_history"][-1]["reason"] == "tool_completed_after_weight_update"
    assert partial.metadata["current_turn"] == 1
    assert "<interpreter>\nOutput:\n42\n</interpreter>" in partial.response
    tokens_before_resume = list(partial.tokens)

    # after_weight_update resets the boolean, while the monotonic epoch stays.
    state.aborted = False
    args.current_policy_version = 1
    completed = asyncio.run(retool.generate(args, partial, {"max_new_tokens": 512}))

    assert completed.status == Sample.Status.COMPLETED
    assert tool_calls == 2
    # The resumed request submits the complete prompt + old response + tool
    # observation. SGLang therefore performs a fresh prefill after cache clear.
    assert payloads[1]["input_ids"] == tokens_before_resume
