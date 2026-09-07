from __future__ import annotations

import asyncio
from types import SimpleNamespace

from data.swe_bench.verifier import RepositoryBaseline, VerifierResult
from data.swe_bench_openenv.harness import (
    _model_turn,
    bounded_observation,
    compact_messages,
    extract_command,
    extract_tool_command,
    structured_terminal_reason,
)
from slime.utils.types import Sample


def test_openenv_command_contract_is_strict_and_thinking_safe():
    assert extract_command("```bash\nrg -n needle src\n```") == "rg -n needle src"
    assert extract_command("<think>```python\npass\n```</think>\n```bash\npwd\n```") == "pwd"
    assert extract_command("Thinking Process:\n...\n</think>\n\nTASK_COMPLETE") == "TASK_COMPLETE"
    assert extract_command("thinking\n</think>\n```bash\npwd\n```<|im_end|>") == "pwd"
    assert extract_command("TASK_COMPLETE") == "TASK_COMPLETE"
    assert extract_command("plain shell text") == ""
    assert extract_command("```bash\none\n```\n```bash\ntwo\n```") == ""


def test_openenv_structured_tool_contract_is_strict():
    valid = [
        {
            "id": "functions.shell:0",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command":"pwd"}',
            },
        }
    ]
    assert extract_tool_command(valid) == ("pwd", None)
    as_dict = [
        {
            "function": {
                "name": "shell",
                "arguments": {"command": "  rg -n needle src  "},
            }
        }
    ]
    assert extract_tool_command(as_dict) == ("rg -n needle src", None)
    for compatibility_name in ("function=shell", "shell=shell"):
        aliased = [
            {
                "function": {
                    "name": compatibility_name,
                    "arguments": '{"command":"pwd"}',
                }
            }
        ]
        assert extract_tool_command(aliased) == ("pwd", None)
    assert extract_tool_command([]) == ("", "missing_tool_call")
    assert extract_tool_command(valid * 2) == ("", "multiple_tool_calls")
    assert extract_tool_command(
        [{"function": {"name": "other", "arguments": {"command": "pwd"}}}]
    ) == ("", "unexpected_tool_name")


def test_openenv_structured_terminal_semantics():
    tool_call = [{"function": {"name": "shell", "arguments": '{"command":"pwd"}'}}]
    assert structured_terminal_reason("", tool_call) is None
    assert structured_terminal_reason("TASK_COMPLETE", []) == "task_complete"
    assert (
        structured_terminal_reason(
            "The fix is complete.\n\nTASK_COMPLETE<|im_end|>", []
        )
        == "task_complete"
    )
    assert structured_terminal_reason("The requested fix is complete.", []) == "final_answer"
    assert (
        structured_terminal_reason("", [], "Tests pass.\nTASK_COMPLETE<|im_end|>")
        == "task_complete"
    )
    assert structured_terminal_reason("  <|im_end|>  ", []) == "no_command"


def test_openenv_observation_and_history_are_bounded():
    value, truncated = bounded_observation("abcdefghij", 6)
    assert truncated is True
    assert value.startswith("abc") and value.endswith("hij")

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *({"role": "user", "content": str(index)} for index in range(12)),
    ]
    compacted = compact_messages(messages, 4)
    assert compacted[:2] == messages[:2]
    assert compacted[-4:] == messages[-4:]
    assert "compacted" in compacted[2]["content"]


def test_openenv_chat_completions_uses_official_reasoning_split(monkeypatch):
    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return "rendered prompt"

        def encode(self, value, add_special_tokens=False):
            return list(range(len(value.split())))

    async def fake_post(url, payload):
        captured.update(url=url, payload=payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "private analysis",
                        "content": "```bash\npwd\n```",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 9,
                "reasoning_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 7},
            },
        }

    class Span:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, values):
            pass

    module = "data.swe_bench_openenv.harness"
    monkeypatch.setattr(f"{module}.post", fake_post)
    monkeypatch.setattr(f"{module}.lifecycle_enabled", lambda: False)
    monkeypatch.setattr(f"{module}.dashboard_span", lambda *args, **kwargs: Span())

    reply, output_ids, metric, compacted = asyncio.run(
        _model_turn(
            args=SimpleNamespace(
                max_seq_len=131072,
                sglang_router_ip="127.0.0.1",
                sglang_router_port=30000,
                hf_checkpoint="Qwen/Qwen3.5-27B",
            ),
            sample=Sample(metadata={}),
            tokenizer=Tokenizer(),
            metadata={"task_type": "swe_bench"},
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ],
            sampling_params={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            options={
                "model_api": "chat_completions",
                "enable_thinking": True,
                "max_tokens_per_turn": 8192,
            },
            turn=0,
        )
    )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["input_ids"] == [0, 1]
    assert captured["payload"]["separate_reasoning"] is True
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert reply == "```bash\npwd\n```"
    assert len(output_ids) == 3
    assert metric["output_tokens"] == 9
    assert metric["reasoning_tokens"] == 5
    assert metric["reasoning_content"] == "private analysis"
    assert metric["cached_tokens"] == 7
    assert compacted is None


def test_openenv_chat_completions_exposes_official_shell_tool_call(monkeypatch):
    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "shell"
            return "rendered prompt with tools"

        def encode(self, value, add_special_tokens=False):
            return list(range(len(value.split())))

    async def fake_post(url, payload):
        captured.update(url=url, payload=payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "inspect the repository",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "functions.shell:0",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 12,
                "reasoning_tokens": 7,
            },
        }

    class Span:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, values):
            pass

    module = "data.swe_bench_openenv.harness"
    monkeypatch.setattr(f"{module}.post", fake_post)
    monkeypatch.setattr(f"{module}.lifecycle_enabled", lambda: False)
    monkeypatch.setattr(f"{module}.dashboard_span", lambda *args, **kwargs: Span())

    reply, output_ids, metric, compacted = asyncio.run(
        _model_turn(
            args=SimpleNamespace(
                max_seq_len=131072,
                sglang_router_ip="127.0.0.1",
                sglang_router_port=30000,
                hf_checkpoint="Qwen/Qwen3.5-27B",
            ),
            sample=Sample(metadata={}),
            tokenizer=Tokenizer(),
            metadata={"task_type": "swe_bench"},
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ],
            sampling_params={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            options={
                "model_api": "chat_completions",
                "action_protocol": "openai_tools",
                "enable_thinking": True,
                "max_tokens_per_turn": 8192,
            },
            turn=0,
        )
    )

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["parallel_tool_calls"] is False
    assert captured["payload"]["tools"][0]["function"]["name"] == "shell"
    assert reply == ""
    assert output_ids == []
    assert metric["finish_type"] == "tool_calls"
    assert metric["tool_calls"][0]["function"]["name"] == "shell"
    assert extract_tool_command(metric["tool_calls"]) == ("pwd", None)
    assert compacted is None


def test_openenv_chat_completions_carries_agentic_lifecycle_metadata(monkeypatch):
    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "rendered prompt"

        def encode(self, value, add_special_tokens=False):
            return list(range(len(value.split())))

    async def fake_post(url, payload):
        captured.update(url=url, payload=payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "TASK_COMPLETE",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    class Span:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, values):
            pass

    module = "data.swe_bench_openenv.harness"
    monkeypatch.setattr(f"{module}.post", fake_post)
    monkeypatch.setattr(f"{module}.lifecycle_enabled", lambda: True)
    monkeypatch.setattr(f"{module}.dashboard_span", lambda *args, **kwargs: Span())

    metadata = {"task_type": "swe_bench"}
    asyncio.run(
        _model_turn(
            args=SimpleNamespace(
                max_seq_len=131072,
                sglang_router_ip="127.0.0.1",
                sglang_router_port=30000,
                hf_checkpoint="Qwen/Qwen3.5-27B",
            ),
            sample=Sample(metadata=metadata),
            tokenizer=Tokenizer(),
            metadata=metadata,
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ],
            sampling_params={"temperature": 0.6},
            options={"model_api": "chat_completions", "max_tokens_per_turn": 8},
            turn=1,
        )
    )

    custom = captured["payload"]["custom_params"]
    assert custom["agentic_request_id"] == metadata["agentic_request_id"]
    assert custom["agentic_generation"] == 1
    assert custom["agentic_parent_generation"] == 0
    assert custom["agentic_prompt_token_count"] == 2
    assert captured["payload"]["input_ids"] == [0, 1]
    assert captured["payload"]["extra_key"].startswith("agentic-v1e:")


def test_openenv_episode_grades_durable_patch_after_policy_stops(monkeypatch):
    events: list[str] = []
    replies = [
        ("```bash\nprintf fix > solution.txt\n```", [11]),
        ("TASK_COMPLETE", [12]),
    ]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def encode(self, value, add_special_tokens=False):
            return [1, 2, 3]

        def decode(self, tokens, skip_special_tokens=False):
            return replies[0][0] if tokens == [11] else replies[1][0]

    class Task:
        image = "official-image"
        metrics = {"backend": "fake", "exec_calls": [], "uploads": []}

        async def start(self):
            events.append("start")

        async def execute(self, command, **kwargs):
            events.append("exec")
            assert command.endswith("printf fix > solution.txt")
            return 0, ""

        async def close(self):
            events.append("close")

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
        output_ids = [11] if events.count("generation") == 1 else [12]
        return {
            "output_ids": output_ids,
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    async def fake_baseline(task, expected):
        events.append("baseline")
        return baseline

    async def fake_capture(task, commit):
        events.append("capture")
        return "diff --git a/solution.txt b/solution.txt\n"

    async def fake_verify(task, metadata, patch, **kwargs):
        events.append("verify")
        assert patch.startswith("diff --git")
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

    module = "data.swe_bench_openenv.harness"
    monkeypatch.setattr(f"{module}.GenerateState", lambda args: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr(f"{module}._create_task", lambda metadata, options: Task())
    monkeypatch.setattr(f"{module}.prepare_repository_baseline", fake_baseline)
    monkeypatch.setattr(f"{module}.capture_repository_patch", fake_capture)
    monkeypatch.setattr(f"{module}.run_inline_verifier", fake_verify)
    monkeypatch.setattr(f"{module}.post", fake_post)
    monkeypatch.setattr(f"{module}.lifecycle_enabled", lambda: False)
    monkeypatch.setattr(f"{module}.confirm_agentic_generation_tool", lambda *args, **kwargs: None)
    monkeypatch.setattr(f"{module}.confirm_agentic_generation_final", lambda *args, **kwargs: None)
    monkeypatch.setattr(f"{module}.dashboard_span", lambda *args, **kwargs: Span())
    monkeypatch.setattr(f"{module}.sglang_meta_attrs", lambda value: {})

    args = SimpleNamespace(
        workload_dataset_options={
            "swe_bench_verified_openenv": {
                "verifier_mode": "inline",
                "max_turns": 4,
                "verifier_max_concurrent": 1,
            }
        },
        max_seq_len=81920,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        pd_p_ready_dir="",
    )
    sample = Sample(
        metadata={
            "dataset_id": "swe_bench_verified_openenv",
            "instance_id": "owner__repo-1",
            "problem_statement": "fix it",
            "base_commit": "a" * 40,
        }
    )

    result = asyncio.run(
        __import__(module, fromlist=["generate"]).generate(
            args,
            sample,
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        )
    )

    assert events == [
        "start",
        "baseline",
        "generation",
        "exec",
        "generation",
        "capture",
        "verify",
        "close",
    ]
    assert result.reward == 1
    assert result.metadata["stop_reason"] == "task_complete"
    assert result.metadata["agent_harness"] == "openenv-swebench-pr51"
    assert result.metadata["swe_bench_verifier"]["resolved"] is True
    assert result.metadata["openenv_trajectory"]["patch"].startswith("diff --git")
