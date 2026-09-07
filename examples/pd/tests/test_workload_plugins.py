from __future__ import annotations

import asyncio
import json
import random
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample

from data.api import LoadContext
from data.browsecomp import agent as browsecomp_agent
from data.browsecomp import env as browsecomp_env
from data.config import DatasetSpec, SamplingSpec, WorkloadSpec, legacy_workload, load_workload
from data.dispatch import exact_counts, select_samples
from data.registry import get_harness
from data.retool import harness as retool_harness
from data.terminal_bench import harness as terminal_harness
from data.terminal_bench.loader import load_samples as load_terminal_samples
from inference import balanced_dispatch_samples


class CharTokenizer:
    bos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        if kwargs.get("tools"):
            rendered = f"<tools>{len(kwargs['tools'])}</tools>" + rendered
        return rendered + ("<assistant>" if add_generation_prompt else "")


class FakeSpan:
    def update(self, attrs):
        pass


@contextmanager
def fake_span(*args, **kwargs):
    yield FakeSpan()


def _sample(dataset_id: str, harness_id: str, source_position: int) -> Sample:
    return Sample(
        metadata={
            "task_type": dataset_id,
            "dataset_id": dataset_id,
            "harness_id": harness_id,
            "source_position": source_position,
        }
    )


def test_workload_config_expands_paths_and_keeps_explicit_shuffle(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_TEST_DATA", str(tmp_path / "sources"))
    path = tmp_path / "workload.yaml"
    path.write_text(
        """
schema_version: 1
datasets:
  - id: easy_math
    harness: retool
    path: ${PD_TEST_DATA}/math.jsonl
    weight: 2
  - id: web_qa
    harness: browsecomp
    path: ${PD_TEST_DATA}/qa.jsonl
    weight: 1
sampling:
  policy: random
  seed: 17
  preserve_source_order: false
  shuffle_algorithm: legacy_two_stage_v1
"""
    )

    workload = load_workload(path)

    assert workload.dataset_ids == ("easy_math", "web_qa")
    assert workload.sampling.seed == 17
    assert workload.sampling.shuffle_algorithm == "legacy_two_stage_v1"
    assert workload.datasets[0].path == str((tmp_path / "sources/math.jsonl").resolve())
    assert exact_counts(10, workload) == {"easy_math": 7, "web_qa": 3}


def test_pure_browsecomp_requires_canonical_source_order_schedule(tmp_path):
    path = tmp_path / "workload.yaml"
    path.write_text(
        """
schema_version: 1
datasets:
  - id: qa
    harness: browsecomp
    path: /qa.jsonl
sampling:
  policy: random
  preserve_source_order: false
  shuffle_algorithm: legacy_two_stage_v1
"""
    )

    with pytest.raises(ValueError, match="source-order n680"):
        load_workload(path)


def test_pure_browsecomp_accepts_canonical_source_order_schedule(tmp_path):
    schedule = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "workloads"
        / "fixed_browsecomp_source_order_n680.json"
    )
    path = tmp_path / "workload.yaml"
    path.write_text(
        f"""
schema_version: 1
datasets:
  - id: qa
    harness: browsecomp
    path: /qa.jsonl
sampling:
  policy: fixed
  seed: 2026
  preserve_source_order: true
  shuffle_algorithm: source_order
  count_algorithm: largest_remainder_v1
  pool_reuse_algorithm: cycle_as_needed_v1
  schedule_file: {schedule}
"""
    )

    workload = load_workload(path)

    assert workload.sampling.preserve_source_order is True
    assert workload.sampling.schedule_file == str(schedule.resolve())


def test_legacy_pure_browsecomp_forces_canonical_source_order_schedule():
    workload = legacy_workload(
        math_path="/math.jsonl",
        qa_path="/qa.jsonl",
        math_ratio=0.0,
        policy="random",
        seed=7,
        preserve_source_order=False,
        schedule_file=None,
    )

    assert workload.sampling.policy == "fixed"
    assert workload.sampling.preserve_source_order is True
    assert workload.sampling.shuffle_algorithm == "source_order"
    assert workload.sampling.pool_reuse_algorithm == "cycle_as_needed_v1"
    assert Path(workload.sampling.schedule_file).name == (
        "fixed_browsecomp_source_order_n680.json"
    )


def test_generic_dispatch_reproduces_the_legacy_two_stage_shuffle():
    seed = 2026
    math = [_sample("math", "retool", index) for index in range(24)]
    qa = [_sample("qa", "browsecomp", index) for index in range(24)]

    # The old source first sorted each task pool by seeded random keys, then
    # inference shuffled each reconstructed pool once more with the same seed.
    stage_one_rng = random.Random(seed)
    stage_one_math = sorted(math, key=lambda _sample: stage_one_rng.random())
    stage_one_qa = sorted(qa, key=lambda _sample: stage_one_rng.random())
    old_source = SimpleNamespace(origin_samples=stage_one_math + stage_one_qa)
    old, _ = balanced_dispatch_samples(
        old_source,
        measured_count=20,
        warmup_count=2,
        policy="random",
        seed=seed,
        math_ratio=0.5,
    )

    workload = legacy_workload(
        math_path="/math.jsonl",
        qa_path="/qa.jsonl",
        math_ratio=0.5,
        policy="random",
        seed=seed,
        preserve_source_order=False,
        schedule_file=None,
    )
    new, schedule = select_samples(
        {"math": stage_one_math, "qa": stage_one_qa},
        workload,
        measured_count=20,
        warmup_count=2,
    )

    key = lambda sample: (sample.metadata["task_type"], sample.metadata["source_position"])
    assert [key(sample) for sample in new] == [key(sample) for sample in old]
    assert len(schedule) == 20
    assert all("source_position" in entry for entry in schedule)


def test_legacy_odd_count_keeps_original_bankers_rounding():
    workload = legacy_workload(
        math_path="/math.jsonl",
        qa_path="/qa.jsonl",
        math_ratio=0.5,
        policy="random",
        seed=2026,
        preserve_source_order=False,
        schedule_file=None,
    )

    assert exact_counts(5, workload) == {"math": 2, "qa": 3}


def test_legacy_smaller_pool_is_recycled_before_the_second_shuffle():
    workload = legacy_workload(
        math_path="/math.jsonl",
        qa_path="/qa.jsonl",
        math_ratio=0.5,
        policy="random",
        seed=7,
        preserve_source_order=False,
        schedule_file=None,
    )
    pools = {
        "math": [_sample("math", "retool", index) for index in range(8)],
        "qa": [_sample("qa", "browsecomp", index) for index in range(2)],
    }

    selected, _ = select_samples(
        pools,
        workload,
        measured_count=8,
        warmup_count=0,
    )

    qa_positions = [
        sample.metadata["source_position"]
        for sample in selected
        if sample.metadata["dataset_id"] == "qa"
    ]
    assert len(qa_positions) == 4
    assert set(qa_positions) == {0, 1}


def test_three_dataset_mixture_can_replay_the_resolved_schedule():
    datasets = (
        DatasetSpec(id="a", harness="retool", path="/a", weight=1),
        DatasetSpec(id="b", harness="browsecomp", path="/b", weight=2),
        DatasetSpec(id="c", harness="terminal_bench", path="/c", weight=3),
    )
    workload = WorkloadSpec(
        datasets=datasets,
        sampling=SamplingSpec(
            policy="random",
            seed=11,
            shuffle_algorithm="legacy_two_stage_v1",
            count_algorithm="largest_remainder_v1",
        ),
    )
    pools = {
        dataset.id: [_sample(dataset.id, dataset.harness, index) for index in range(20)]
        for dataset in datasets
    }

    selected, resolved = select_samples(
        pools,
        workload,
        measured_count=12,
        warmup_count=0,
    )
    fixed = replace(workload, sampling=replace(workload.sampling, policy="fixed"))
    replayed, replay_log = select_samples(
        pools,
        fixed,
        measured_count=12,
        warmup_count=0,
        schedule=resolved,
    )

    identity = lambda sample: (
        sample.metadata["dataset_id"],
        sample.metadata["source_position"],
    )
    assert exact_counts(12, workload) == {"a": 2, "b": 4, "c": 6}
    assert list(map(identity, replayed)) == list(map(identity, selected))
    assert [entry["source_position"] for entry in replay_log] == [
        entry["source_position"] for entry in resolved
    ]


def test_harness_registry_declares_but_does_not_start_external_services():
    assert get_harness("retool").required_services == ()
    assert get_harness("browsecomp").required_services == ("browsecomp_search",)
    assert get_harness("terminal_bench").required_services == ("tbench2_env",)


def test_terminal_loader_accepts_label_free_task_rows(tmp_path):
    path = tmp_path / "terminal.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "system", "content": "terminal agent"}],
                "metadata": {"task_id": "headless-terminal"},
            }
        )
        + "\n"
    )
    context = LoadContext(
        args=SimpleNamespace(
            rollout_max_prompt_len=4096,
            input_key="prompt",
            metadata_key="metadata",
            rollout_seed=1,
        ),
        tokenizer=CharTokenizer(),
        processor=None,
    )
    dataset = SimpleNamespace(
        path=str(path),
        options={},
    )

    samples = load_terminal_samples(context, dataset)

    assert len(samples) == 1
    assert samples[0].label is None
    assert samples[0].metadata["task_id"] == "headless-terminal"


def test_terminal_command_parser_never_executes_reasoning_or_bare_text():
    assert terminal_harness.command_from_reply("```bash\necho ok\n```") == "echo ok"
    assert terminal_harness.command_from_reply("<think>x</think>\n```bash\npwd\n```") == "pwd"
    assert terminal_harness.command_from_reply("echo unsafe") is None
    assert terminal_harness.command_from_reply("<think>unfinished\n```bash\nrm -rf /\n```") is None


def test_terminal_harness_runs_shell_then_evaluates(monkeypatch):
    tokenizer = CharTokenizer()
    replies = ["```bash\necho hello\n```", "TASK_COMPLETE"]
    payloads = []
    clients = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.commands = []
            self.closed = False
            clients.append(self)

        async def connect(self):
            pass

        async def reset(self, task_id):
            return SimpleNamespace(
                instruction="print hello",
                info={"working_directory": "/app"},
            )

        async def execute(self, command):
            self.commands.append(command)
            return SimpleNamespace(output="hello")

        async def evaluate(self):
            return SimpleNamespace(reward=1.0)

        async def close(self):
            self.closed = True

    async def fake_post(url, payload):
        payloads.append(payload)
        text = replies.pop(0)
        return {
            "text": text,
            "output_ids": tokenizer.encode(text),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": len(payload["input_ids"]),
                "completion_tokens": len(text),
            },
        }

    monkeypatch.setattr(terminal_harness, "Tbench2Client", FakeClient)
    monkeypatch.setattr(terminal_harness, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer))
    monkeypatch.setattr(terminal_harness, "post", fake_post)
    monkeypatch.setattr(terminal_harness, "dashboard_span", fake_span)
    monkeypatch.setattr(terminal_harness, "sglang_meta_attrs", lambda meta: {})
    monkeypatch.setattr(terminal_harness, "lifecycle_enabled", lambda: False)
    sample = Sample(
        prompt=[{"role": "system", "content": "terminal agent"}],
        metadata={"task_id": "headless-terminal", "dataset_id": "terminal", "task_type": "terminal"},
    )
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30002,
        max_seq_len=40960,
        pd_p_ready_dir="",
        workload_dataset_options={
            "terminal": {
                "environment_url": "http://127.0.0.1:8003",
                "max_turns": 4,
                "max_tokens_per_turn": 512,
            }
        },
    )

    result = asyncio.run(terminal_harness.generate(args, sample, {"max_new_tokens": 2048}))

    assert result.status is Sample.Status.COMPLETED
    assert result.reward == 1.0
    assert result.metadata["stop_reason"] == "task_complete"
    assert result.metadata["shell_call_count"] == 1
    assert clients[0].commands == ["echo hello"]
    assert clients[0].closed is True
    assert all(payload["return_logprob"] is False for payload in payloads)


def test_terminal_harness_retries_openenv_capacity_without_finishing_sample(
    monkeypatch,
):
    tokenizer = CharTokenizer()
    clients = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.index = len(clients)
            self.closed = False
            clients.append(self)

        async def connect(self):
            pass

        async def reset(self, task_id):
            if self.index == 0:
                raise terminal_harness.TerminalEnvironmentError(
                    "Server at capacity", "CAPACITY_REACHED"
                )
            if self.index == 1:
                raise terminal_harness.TerminalEnvironmentError(
                    "received 1000 (OK)", "RESET_CONNECTION_CLOSED_OK"
                )
            return SimpleNamespace(
                instruction="finish",
                info={"working_directory": "/app"},
            )

        async def evaluate(self):
            return SimpleNamespace(reward=1.0)

        async def close(self):
            self.closed = True

    async def fake_post(url, payload):
        text = "TASK_COMPLETE"
        return {
            "text": text,
            "output_ids": tokenizer.encode(text),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": len(payload["input_ids"]),
                "completion_tokens": len(text),
            },
        }

    async def no_wait(_seconds):
        pass

    monkeypatch.setattr(terminal_harness, "Tbench2Client", FakeClient)
    monkeypatch.setattr(
        terminal_harness, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer)
    )
    monkeypatch.setattr(terminal_harness, "post", fake_post)
    monkeypatch.setattr(terminal_harness, "dashboard_span", fake_span)
    monkeypatch.setattr(terminal_harness, "sglang_meta_attrs", lambda meta: {})
    monkeypatch.setattr(terminal_harness, "lifecycle_enabled", lambda: False)
    monkeypatch.setattr(terminal_harness.asyncio, "sleep", no_wait)
    sample = Sample(
        prompt=[],
        metadata={"task_id": "headless-terminal", "dataset_id": "terminal"},
    )
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30002,
        max_seq_len=40960,
        pd_p_ready_dir="",
        workload_dataset_options={
            "terminal": {
                "environment_url": "http://127.0.0.1:8003",
                "max_turns": 2,
                "max_tokens_per_turn": 512,
            }
        },
    )

    result = asyncio.run(
        terminal_harness.generate(args, sample, {"max_new_tokens": 512})
    )

    assert result.status is Sample.Status.COMPLETED
    assert len(clients) == 3
    assert all(client.closed for client in clients)


def test_retool_harness_finishes_without_training_payload(monkeypatch):
    tokenizer = CharTokenizer()
    payloads = []
    reply = r"Reasoning.\n#### \boxed{2}"

    async def fake_post(url, payload):
        payloads.append(payload)
        return {
            "text": reply,
            "output_ids": tokenizer.encode(reply),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": len(payload["input_ids"]),
                "completion_tokens": len(reply),
            },
        }

    monkeypatch.setattr(
        retool_harness,
        "GenerateState",
        lambda args: SimpleNamespace(tokenizer=tokenizer),
    )
    monkeypatch.setattr(retool_harness, "post", fake_post)
    monkeypatch.setattr(retool_harness, "dashboard_span", fake_span)
    monkeypatch.setattr(retool_harness, "sglang_meta_attrs", lambda meta: {})
    monkeypatch.setattr(retool_harness, "lifecycle_enabled", lambda: False)
    sample = Sample(prompt="What is 1+1?", metadata={"dataset_id": "math", "task_type": "math"})
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30002,
        retool_local_router_port=None,
        rollout_max_context_len=40960,
        context_parallel_size=1,
        max_tokens_per_gpu=40960,
        pd_p_ready_dir="",
        enable_tool_delay=False,
    )

    result = asyncio.run(retool_harness.generate(args, sample, {"max_new_tokens": 2048}))

    assert result.status is Sample.Status.COMPLETED
    assert result.response_length == len(reply)
    assert result.metadata["tool_call_count"] == 0
    assert payloads[0]["return_logprob"] is False
    assert "loss_mask" not in result.metadata
    assert "token_logprobs" not in result.metadata


def test_browsecomp_harness_finishes_without_starting_search_service(monkeypatch):
    tokenizer = CharTokenizer()
    payloads = []
    environments = []
    reply = (
        "<function=finish>\n"
        "<parameter=answer>answer</parameter>\n"
        "<parameter=explanation>evidence</parameter>\n"
        "<parameter=confidence>high</parameter>\n"
        "</function>"
    )

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            self.stats = {"search": 1, "open_page": 0, "finish": 1}
            self.predicted_answer = ("answer", "evidence", "high")
            self.closed = False
            environments.append(self)

        async def run_action(self, text):
            assert text == reply
            return {"action": "finish"}

        async def close(self):
            self.closed = True

    async def fake_post(url, payload):
        payloads.append(payload)
        return {
            "text": reply,
            "output_ids": tokenizer.encode(reply),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": len(payload["input_ids"]),
                "completion_tokens": len(reply),
            },
        }

    monkeypatch.setattr(browsecomp_agent, "BrowseCompEnv", FakeEnv)
    monkeypatch.setattr(
        browsecomp_agent,
        "GenerateState",
        lambda args: SimpleNamespace(tokenizer=tokenizer),
    )
    monkeypatch.setattr(browsecomp_agent, "post", fake_post)
    monkeypatch.setattr(browsecomp_agent, "dashboard_span", fake_span)
    monkeypatch.setattr(browsecomp_agent, "sglang_meta_attrs", lambda meta: {})
    monkeypatch.setattr(browsecomp_agent, "lifecycle_enabled", lambda: False)
    sample = Sample(
        prompt=[{"role": "user", "content": "Question?"}],
        label="answer",
        metadata={
            "question": "Question?",
            "answer": "answer",
            "dataset_id": "qa",
            "task_type": "qa",
        },
    )
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30002,
        max_seq_len=40960,
        pd_p_ready_dir="",
        workload_dataset_options={
            "qa": {
                "search_url": "http://127.0.0.1:8000",
                "max_turns": 4,
                "max_tokens_per_turn": 512,
            }
        },
    )

    result = asyncio.run(browsecomp_agent.generate(args, sample, {"max_new_tokens": 2048}))

    assert result.status is Sample.Status.COMPLETED
    assert result.metadata["stop_reason"] == "finish"
    assert environments[0].closed is True
    assert payloads[0]["return_logprob"] is False
    assert "loss_mask" not in result.metadata
    assert "token_logprobs" not in result.metadata


def test_browsecomp_optional_shared_search_budget_does_not_change_default(monkeypatch):
    class FakeSearchClient:
        def __init__(self, *args, **kwargs):
            self.calls = []

        async def search(self, query, k):
            self.calls.append((query, k))
            return [
                {"docid": f"{query}-{index}", "url": f"https:///{index}", "text": "x"}
                for index in range(50)
            ]

        async def close(self):
            pass

    monkeypatch.setattr(browsecomp_env, "AsyncSearchClient", FakeSearchClient)
    response = "\n".join(
        f"<function=search>\n<parameter=query>q{index}</parameter>\n"
        "<parameter=topk>10</parameter>\n</function>"
        for index in range(5)
    )

    default_env = browsecomp_env.BrowseCompEnv("q", "a", base_url="http://unused")
    default_result = asyncio.run(default_env.run_action(response))
    bounded_env = browsecomp_env.BrowseCompEnv(
        "q",
        "a",
        base_url="http://unused",
        search_total_topk_per_turn=10,
    )
    bounded_result = asyncio.run(bounded_env.run_action(response))

    assert default_result["observation"].count("--- #") == 50
    assert bounded_result["observation"].count("--- #") == 10
    assert bounded_env.stats["search"] == 5


def test_browsecomp_optional_observation_word_limits_do_not_change_defaults(monkeypatch):
    class FakeSearchClient:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, query, k):
            return [{"docid": "d", "url": "https:///d", "text": "zzword " * 600}]

        async def open(self, url, docid):
            return [{"docid": "d", "url": "https:///d", "text": "zzword " * 5000}]

        async def close(self):
            pass

    monkeypatch.setattr(browsecomp_env, "AsyncSearchClient", FakeSearchClient)
    default_env = browsecomp_env.BrowseCompEnv("q", "a", base_url="http://unused")
    bounded_env = browsecomp_env.BrowseCompEnv(
        "q",
        "a",
        base_url="http://unused",
        search_snippet_words=256,
        open_page_words=2048,
    )
    search = "<function=search><parameter=query>q</parameter></function>"
    opened = "<function=open_page><parameter=docid>d</parameter></function>"
    default_search = asyncio.run(default_env.run_action(search))["observation"]
    bounded_search = asyncio.run(bounded_env.run_action(search))["observation"]
    default_open = asyncio.run(default_env.run_action(opened))["observation"]
    bounded_open = asyncio.run(bounded_env.run_action(opened))["observation"]

    assert default_search.count("zzword") == 512
    assert bounded_search.count("zzword") == 256
    assert default_open.count("zzword") == 4096
    assert bounded_open.count("zzword") == 2048


def test_browsecomp_optional_strict_result_count_bounds_revisited_pages(monkeypatch):
    class FakeSearchClient:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, query, k):
            return [
                {"docid": f"d{index}", "url": f"https:///{index}", "text": "body"}
                for index in range(20)
            ]

        async def close(self):
            pass

    monkeypatch.setattr(browsecomp_env, "AsyncSearchClient", FakeSearchClient)
    default_env = browsecomp_env.BrowseCompEnv("q", "a", base_url="http://unused")
    strict_env = browsecomp_env.BrowseCompEnv(
        "q",
        "a",
        base_url="http://unused",
        search_count_revisits_as_full=True,
    )
    action = "<function=search><parameter=query>q</parameter><parameter=topk>5</parameter></function>"
    asyncio.run(default_env.run_action(action))
    asyncio.run(strict_env.run_action(action))
    default_revisit = asyncio.run(default_env.run_action(action))["observation"]
    strict_revisit = asyncio.run(strict_env.run_action(action))["observation"]

    assert default_revisit.count("--- #") > 5
    assert strict_revisit.count("--- #") == 5


def test_browsecomp_qwen35_profile_is_isolated_and_replaces_system_prompt(tmp_path):
    profile = tmp_path / "qwen35.yaml"
    profile.write_text(
        "schema_version: 1\nsystem_prompt: qwen35-only\nsearch_call_budget: 3\n"
    )

    loaded = browsecomp_agent._load_agent_profile(str(profile))
    original = [
        {"role": "system", "content": "generic-qwen"},
        {"role": "user", "content": "question"},
    ]
    replaced = browsecomp_agent._replace_system_prompt(
        original, loaded["system_prompt"]
    )

    assert original[0]["content"] == "generic-qwen"
    assert replaced[0]["content"] == "qwen35-only"
    assert loaded["search_call_budget"] == 3
    assert browsecomp_agent._load_agent_profile(None) == {}


def test_browsecomp_qwen35_uses_native_tools_and_tool_response_role():
    tokenizer = CharTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
    ]
    tools = [{"type": "function", "function": {"name": "search"}}]

    initial = browsecomp_agent._initial_tokens(
        tokenizer, messages, tools=tools
    )
    observation = browsecomp_agent._observation_tokens(
        tokenizer, "result", native_tool_response=True
    )

    assert "<tools>1</tools>" in tokenizer.decode(initial)
    assert "<tool>result" in tokenizer.decode(observation)


def test_browsecomp_qwen35_profile_limits_one_action_and_exposes_budget(monkeypatch):
    class FakeSearchClient:
        def __init__(self, *args, **kwargs):
            self.queries = []

        async def search(self, query, k):
            self.queries.append(query)
            return [{"docid": query, "url": f"https:///{query}", "text": "evidence"}]

        async def close(self):
            pass

    monkeypatch.setattr(browsecomp_env, "AsyncSearchClient", FakeSearchClient)
    env = browsecomp_env.BrowseCompEnv(
        "q",
        "a",
        base_url="http://unused",
        max_tool_calls_per_turn=1,
        search_call_budget=1,
        open_page_call_budget=0,
        followup_prompt=(
            "remaining search={search_remaining}, open={open_remaining}"
        ),
        budget_exhausted_prompt="finish now",
    )
    two_calls = (
        '<tool_call>{"name":"search","arguments":{"query":["q1"]}}</tool_call>\n'
        '<tool_call>{"name":"search","arguments":{"query":["q2"]}}</tool_call>'
    )

    result = asyncio.run(env.run_action(two_calls))

    assert env.client.queries == ["q1"]
    assert env.stats["search"] == 1
    assert "finish now" in result["observation"]
