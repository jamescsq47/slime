from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from slime.utils.types import Sample


MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
BROWSECOMP_DIR = Path(__file__).resolve().parents[2] / "examples" / "browsecomp"
for path in (BROWSECOMP_DIR, MIXED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location(
    "mixed_browsecomp_agent_under_test",
    MIXED_DIR / "browsecomp_agent.py",
)
assert spec is not None and spec.loader is not None
browsecomp_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browsecomp_agent)


class CharTokenizer:
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class FakeSpan:
    def update(self, attrs):
        pass


@contextmanager
def fake_dashboard_span(*args, **kwargs):
    yield FakeSpan()


def test_qa_tool_crossing_weight_update_commits_observation_then_refills(monkeypatch):
    state = SimpleNamespace(tokenizer=CharTokenizer(), aborted=False, abort_epoch=0)
    generation_inputs = []

    class FakeEnv:
        def __init__(self, question, label_answer, must_search):
            self.question = question
            self.label_answer = label_answer
            self.must_search = must_search
            self.donotgiveup = False
            self.visited_pages = set()
            self.is_finish = False
            self.predicted_answer = None
            self.stats = Counter()

        async def run_action(self, assistant_text):
            if self.stats["search"] == 0:
                self.stats["search"] += 1
                state.abort_epoch += 1
                state.aborted = True
                return {"action": "search", "observation": "search result"}
            self.stats["finish"] += 1
            self.is_finish = True
            self.predicted_answer = ("answer", "reason", "certain")
            return {"action": "finish", "observation": ""}

        async def close(self):
            pass

    async def fake_generate_step(url, tokens, sampling_params):
        generation_inputs.append(list(tokens))
        text = "search action" if len(generation_inputs) == 1 else "finish action"
        new_tokens = state.tokenizer.encode(text)
        return text, new_tokens, [0.0] * len(new_tokens), "stop"

    monkeypatch.setattr(browsecomp_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(browsecomp_agent, "BrowseCompEnv", FakeEnv)
    monkeypatch.setattr(browsecomp_agent, "_generate_step", fake_generate_step)
    monkeypatch.setattr(browsecomp_agent, "dashboard_span", fake_dashboard_span)

    args = SimpleNamespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=False,
        mask_offpolicy_math=None,
        mask_offpolicy_qa=None,
        max_seq_len=4096,
        sglang_context_length=4096,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        current_policy_version=0,
        current_rollout_id=0,
        use_slime_dashboard=True,
    )
    sample = Sample(
        prompt="question",
        label="answer",
        metadata={"question": "question", "answer": "answer", "task_type": "qa", "policy_version": 0},
    )

    partial = asyncio.run(browsecomp_agent.generate(args, sample, {"max_new_tokens": 128}))

    assert partial.status == Sample.Status.ABORTED
    assert partial.metadata["stop_reason"] == "tool_completed_after_weight_update"
    assert "search result" in partial.response
    assert partial.metadata["search_call_count"] == 1
    tokens_before_resume = list(partial.tokens)

    state.aborted = False
    args.current_policy_version = 1
    completed = asyncio.run(browsecomp_agent.generate(args, partial, {"max_new_tokens": 128}))

    assert completed.status == Sample.Status.COMPLETED
    assert completed.metadata["finish_call_count"] == 1
    assert generation_inputs[1] == tokens_before_resume
