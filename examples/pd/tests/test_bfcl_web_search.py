import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from data.bfcl_web_search import harness
from data.bfcl_web_search.loader import load_samples
from data.bfcl_web_search.tools import PublicResolver, WebBackendError, WebTools, public_url, validate_call
from data.config import DatasetSpec, load_workload
from slime.utils.types import Sample


def test_loader_never_reads_gold(tmp_path):
    (tmp_path / "questions.jsonl").write_text(json.dumps({"id": "web_search_0", "question": [[{"role": "user", "content": "café 🐱"}]]}))
    (tmp_path / "tools.jsonl").write_text(json.dumps({"name": "search_engine_query", "parameters": {"type": "dict"}}))
    (tmp_path / "answers.jsonl").write_text("invalid-json gold must never be read")
    samples = load_samples(None, DatasetSpec("bfcl", "bfcl_web_search", str(tmp_path / "questions.jsonl"), options={"variants": ["base", "no_snippet"]}))
    assert len(samples) == 2
    assert samples[0].prompt[-1]["content"] == "café 🐱"
    assert samples[0].label is None
    assert samples[0].metadata["bfcl_tools"][0]["function"]["parameters"]["type"] == "object"


def test_parse_ignores_thinking_and_accepts_native_json():
    call = {"name": "search_engine_query", "arguments": {"keywords": "test"}}
    assert harness.parse_calls('<think>example <tool_call>bad</tool_call></think><tool_call>' + json.dumps(call) + '</tool_call>') == [call]
    assert harness.parse_calls("The final answer is Paris.") == []
    with pytest.raises(ValueError):
        harness.parse_calls("<tool_call>broken")


@pytest.mark.parametrize("call", [
    {"name": "shell", "arguments": {}},
    {"name": "search_engine_query", "arguments": {"keywords": "x", "max_results": True}},
    {"name": "fetch_url_content", "arguments": {"url": "https://example.com", "mode": "python"}},
    [],
])
def test_reject_bad_tools(call):
    with pytest.raises(ValueError):
        validate_call(call)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1", "http://[::1]/", "http://169.254.169.254/", "http://user:password@example.com/"])
def test_reject_local_fetch(url):
    with pytest.raises(ValueError):
        asyncio.run(public_url(url))


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        text += "<assistant>" if add_generation_prompt else ""
        return list(text.encode()) if tokenize else text

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode(errors="replace")


@contextmanager
def span(*args, **kwargs):
    yield SimpleNamespace(update=lambda x: None)


def sample():
    return Sample(prompt=[{"role": "user", "content": "Who?"}], metadata={"dataset_id": "bfcl", "bfcl_tools": [], "bfcl_variant": "base"})


def prepare(monkeypatch):
    monkeypatch.setattr(harness, "lifecycle_enabled", lambda: False)
    monkeypatch.setattr(harness, "GenerateState", lambda args: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr(harness, "dashboard_span", span)
    monkeypatch.setattr(harness, "sglang_meta_attrs", lambda meta: {})
    return SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=1, max_seq_len=40960, workload_dataset_options={})


def test_two_turn_exact_prefix_and_stats(monkeypatch):
    args = prepare(monkeypatch)
    requests = []
    call = '<tool_call>{"name":"search_engine_query","arguments":{"keywords":"x"}}</tool_call>'
    outputs = [call, "Answer"]
    async def post(url, payload):
        requests.append(payload)
        text = outputs.pop(0)
        return {"text": text, "output_ids": list(text.encode()), "meta_info": {"finish_reason": {"type": "stop"}}}
    async def search(self, args):
        return [{"title": "test", "href": "https://example.com", "body": "text"}]
    monkeypatch.setattr(harness, "post", post)
    monkeypatch.setattr(WebTools, "_search", search)
    result = asyncio.run(harness.generate(args, sample(), {"max_new_tokens": 8192}))
    prefix = requests[0]["input_ids"] + list(call.encode())
    assert requests[1]["input_ids"][:len(prefix)] == prefix
    assert result.metadata["num_turns"] == 2
    assert result.status == Sample.Status.COMPLETED
    assert result.tool_call_count == 1
    assert result.tool_token_count > 0
    assert all(p["return_logprob"] is False for p in requests)


def test_network_error_not_reported_as_success(monkeypatch):
    args = prepare(monkeypatch)
    args.workload_dataset_options = {"bfcl": {"max_tool_errors": 1}}
    async def post(*args):
        return {"text": '<tool_call>{"name":"search_engine_query","arguments":{"keywords":"x"}}</tool_call>', "output_ids": [1], "meta_info": {"finish_reason": {"type": "stop"}}}
    async def search(*args):
        raise TimeoutError("network down")
    monkeypatch.setattr(harness, "post", post)
    monkeypatch.setattr(WebTools, "_search", search)
    result = sample()
    with pytest.raises(WebBackendError):
        asyncio.run(harness.generate(args, result, {}))
    assert result.status == Sample.Status.ABORTED
    assert result.metadata["stop_reason"] == "web_backend_error"
    assert not result.metadata["web_tool_events"][0]["ok"]


def test_recoverable_tool_error_is_given_to_model(monkeypatch):
    args = prepare(monkeypatch)
    prompts = []
    outputs = ['<tool_call>{"name":"search_engine_query","arguments":{"keywords":"x"}}</tool_call>', "Final answer"]
    async def post(url, payload):
        prompts.append(payload["input_ids"])
        text = outputs.pop(0)
        return {"text": text, "output_ids": list(text.encode()), "meta_info": {"finish_reason": {"type": "stop"}}}
    async def search(*args):
        raise TimeoutError("temporarily unavailable")
    monkeypatch.setattr(harness, "post", post)
    monkeypatch.setattr(WebTools, "_search", search)
    result = asyncio.run(harness.generate(args, sample(), {}))
    assert result.status == Sample.Status.COMPLETED
    assert result.metadata["tool_error_count"] == 1
    assert not result.metadata["web_tool_events"][0]["ok"]
    assert "temporarily unavailable" in bytes(prompts[1]).decode()


def test_pd_use_fails_before_model_call(monkeypatch):
    monkeypatch.setattr(harness, "lifecycle_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="colocated-only"):
        asyncio.run(harness.generate(None, sample(), {}))


def test_cancellation_propagates(monkeypatch):
    async def search(*args):
        raise asyncio.CancelledError()
    monkeypatch.setattr(WebTools, "_search", search)
    tools = WebTools()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(tools.call({"name": "search_engine_query", "arguments": {"keywords": "x"}}))
    assert tools.events[0]["error"] == "cancelled"


def test_harness_cancellation_records_abort(monkeypatch):
    args = prepare(monkeypatch)
    async def post(*args):
        raise asyncio.CancelledError()
    monkeypatch.setattr(harness, "post", post)
    item = sample()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(harness.generate(args, item, {}))
    assert item.status == Sample.Status.ABORTED
    assert item.metadata["stop_reason"] == "cancelled"


def test_connection_resolver_rejects_rebinding(monkeypatch):
    import aiohttp
    async def resolve(*args):
        return [{"host": "127.0.0.1"}]
    monkeypatch.setattr(aiohttp.resolver.DefaultResolver, "resolve", resolve)
    async def check():
        resolver = PublicResolver()
        try:
            with pytest.raises(ValueError, match="Private"):
                await resolver.resolve("public-looking.example", 443)
        finally:
            await resolver.close()
    asyncio.run(check())


def test_real_qwen_template_if_available():
    from pathlib import Path
    model = Path("/dataset/model/qwen3/Qwen3-8B")
    if not model.exists():
        pytest.skip("local Qwen3 tokenizer unavailable")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    item = sample()
    initial = harness.initial_tokens(tokenizer, item)
    suffix = harness.observation_tokens(tokenizer, [{"content": "tool result ☃"}])
    assert len(initial) > 5 and all(type(x) is int for x in initial)
    assert len(suffix) > 5
    text = tokenizer.decode(suffix)
    assert "<tool_response>" in text and "tool result ☃" in text


def test_unknown_provider_cannot_silently_become_auto():
    tools = WebTools(backend="nonexistent-provider")
    with pytest.raises(WebBackendError, match="Disabled/unknown"):
        asyncio.run(tools.call({"name": "search_engine_query", "arguments": {"keywords": "x"}}))
    assert not tools.events[0]["ok"]


def test_unspecified_region_maps_to_valid_ddgs_language(monkeypatch):
    import ddgs
    received = []
    class Search:
        def __init__(self, **kwargs):
            pass
        def text(self, query, **kwargs):
            received.append(kwargs["region"])
            return [{"title": "result", "href": "https://example.com", "body": "text"}]
    monkeypatch.setattr(ddgs, "DDGS", Search)
    tools = WebTools(backend="auto")
    asyncio.run(tools.call({"name": "search_engine_query", "arguments": {"keywords": "x"}}))
    assert received == ["us-en"]
