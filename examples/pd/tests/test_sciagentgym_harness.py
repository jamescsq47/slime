import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from data.config import DatasetSpec
from data.sciagentgym import harness
from data.sciagentgym.loader import load_samples
from slime.utils.types import Sample

UPSTREAM = Path("/homes/siqic/data/SciAgentGYM")


def test_no_gold_in_loader(tmp_path):
    row = {"id": 1, "question": "real question", "answer": "SECRET_ANSWER",
           "metadata": {"subject": "Physics", "topic": "test", "golden_answer": "SECRET_GOLD",
                        "solution_steps": ["SECRET_STEPS"]}, "usage_tool_protocol": []}
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps([row]))
    sample = load_samples(None, DatasetSpec("science", "sciagentgym", str(source), options={"case_ids": [1]}))[0]
    assert "SECRET" not in str(sample.prompt) + str(sample.metadata)
    assert sample.label is None


def test_parse_calls_and_thinking():
    assert harness.parse_calls('<think>example <tool_call>bad</tool_call></think>answer') == []
    with pytest.raises(ValueError):
        harness.parse_calls('<tool_call>broken')
    with pytest.raises(ValueError):
        harness.parse_calls('<tool_call>{"name":"x","arguments":[]}</tool_call>')


def test_pd_rejected(monkeypatch):
    monkeypatch.setattr(harness, "lifecycle_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="colocated-only"):
        asyncio.run(harness.generate(None, None, {}))


class Tokenizer:
    eos_token_id = -1
    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}</end>" for m in messages) + ("<assistant>" if kwargs.get("add_generation_prompt") else "")
    def encode(self, text, **kwargs):
        return list(text.encode())
    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode(errors="replace")


@contextmanager
def span(*args, **kwargs):
    yield SimpleNamespace(update=lambda value: None)


@pytest.mark.parametrize("mode", ["normal", "cancel", "timeout", "nested_arguments", "path_escape", "nested_path"])
def test_real_worker_model_mock_exact_history_and_cleanup(monkeypatch, mode):
    if not UPSTREAM.exists():
        pytest.skip("Upstream not installed")
    monkeypatch.setattr(harness, "lifecycle_enabled", lambda: False)
    monkeypatch.setattr(harness, "GenerateState", lambda args: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr(harness, "dashboard_span", span)
    monkeypatch.setattr(harness, "sglang_meta_attrs", lambda meta: {})
    sample = load_samples(None, DatasetSpec("science", "sciagentgym", str(UPSTREAM / "dataset/refine_merged_single_questions.json"), options={"case_ids": [31 if mode in {"path_escape", "nested_path"} else 1]}))[0]
    sample.metadata["dataset_id"] = "science"
    args = SimpleNamespace(workload_dataset_options={}, max_seq_len=40960, sglang_router_ip="127.0.0.1", sglang_router_port=1)
    if mode == "timeout":
        args.workload_dataset_options = {"science": {"tool_timeout_seconds": 0}}
    calls, workers = [], []
    original = asyncio.create_subprocess_exec
    async def spawn(*a, **kw):
        worker = await original(*a, **kw)
        workers.append(worker)
        return worker
    monkeypatch.setattr(harness.asyncio, "create_subprocess_exec", spawn)
    text = '<tool_call>{"name":"analyze_pair_production_threshold","arguments":{"cmb_energy":0.001,"particle_type":"electron"}}</tool_call>'
    if mode == "nested_arguments":
        text = '<tool_call>{"name":"analyze_pair_production_threshold","arguments":{"cmb_energy":0.001,"particle_type":"electron","arguments":{"cmb_energy":0.001,"particle_type":"electron"}}}</tool_call>'
    if mode == "path_escape":
        text = '<tool_call>{"name":"load_file","arguments":{"filepath":"/etc/passwd"}}</tool_call>'
    if mode == "nested_path":
        text = '<tool_call>{"name":"load_file","arguments":{"filepath":"safe.txt","arguments":{"filepath":"/etc/passwd"}}}</tool_call>'
    async def post(url, payload):
        calls.append(payload)
        if mode == "cancel":
            raise asyncio.CancelledError()
        answer = text if len(calls) == 1 else "Final answer"
        return {"text": answer, "output_ids": list(answer.encode()), "meta_info": {"finish_reason": {"type": "stop"}}}
    monkeypatch.setattr(harness, "post", post)
    if mode in {"cancel", "timeout"}:
        with pytest.raises(asyncio.CancelledError if mode == "cancel" else TimeoutError):
            asyncio.run(harness.generate(args, sample, {}))
        assert sample.status == Sample.Status.ABORTED
    else:
        asyncio.run(harness.generate(args, sample, {}))
        assert sample.status == Sample.Status.COMPLETED
        event = sample.metadata["science_tool_events"][0]
        if mode in {"nested_arguments", "path_escape", "nested_path"}:
            assert not event["ok"]
            assert "root:x:" not in event["observation"]
        else:
            assert event["ok"] and "threshold" in event["observation"]
        prefix = calls[0]["input_ids"] + list(text.encode())
        assert calls[1]["input_ids"][:len(prefix)] == prefix
        assert all(p["return_logprob"] is False for p in calls)
        trace = sample.metadata["science_token_trace"]
        reconstructed = trace["initial_ids"] + [token for turn in trace["turns"] for key in ("output_ids", "tool_suffix_ids") for token in turn[key]]
        assert reconstructed == sample.tokens
        assert json.loads(json.dumps(sample.metadata))["science_token_trace"] == trace
    assert workers and all(w.returncode is not None for w in workers)
