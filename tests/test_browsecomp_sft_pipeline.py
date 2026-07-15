import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from slime.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[1]
BROWSECOMP_DIR = REPO_ROOT / "examples" / "browsecomp"
SFT_DIR = BROWSECOMP_DIR / "sft"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SFT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_sft():
    return _load_module("browsecomp_export_sft_test", "export_sft.py")


@pytest.fixture
def browsecomp_rm(monkeypatch):
    monkeypatch.syspath_prepend(str(BROWSECOMP_DIR))
    monkeypatch.syspath_prepend(str(SFT_DIR))
    return _load_module("browsecomp_sft_rm_test", "sft_rm.py")


def _valid_sample(instance_id: str = "q1", token_count: int = 200) -> dict:
    return {
        "status": "completed",
        "reward": 1.0,
        "tokens": list(range(token_count)),
        "loss_mask": [1] * 100,
        "metadata": {
            "data_source": "bc_train_hard",
            "instance_id": instance_id,
            "question": "question",
            "answer": "answer",
            "predicted_answer": "answer",
            "stop_reason": "finish",
            "num_turns": 3,
            "tool_stats": {"search": 1, "open_page": 1, "finish": 1, "change_answer": 0},
            "trajectory_messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "search and answer"},
            ],
            "grading_source": "exact_match",
        },
    }


@pytest.mark.unit
def test_export_rejects_non_train_incorrect_unopened_and_incomplete(export_sft):
    args = type(
        "Args",
        (),
        dict(min_reward=1.0, min_searches=1, min_open_pages=1, min_assistant_tokens=64, max_tokens=36864),
    )()

    valid = _valid_sample()
    assert export_sft._reject_reason(valid, args) is None

    test_sample = copy.deepcopy(valid)
    test_sample["metadata"]["data_source"] = "bc_test"
    assert export_sft._reject_reason(test_sample, args) == "non_train_split"

    incorrect = copy.deepcopy(valid)
    incorrect["reward"] = 0
    assert export_sft._reject_reason(incorrect, args) == "incorrect"

    unopened = copy.deepcopy(valid)
    unopened["metadata"]["tool_stats"]["open_page"] = 0
    assert export_sft._reject_reason(unopened, args) == "no_open_page"

    truncated = copy.deepcopy(valid)
    truncated["status"] = "truncated"
    assert export_sft._reject_reason(truncated, args) == "incomplete"


@pytest.mark.unit
def test_export_keeps_shortest_two_unique_trajectories_per_question(export_sft, monkeypatch, tmp_path):
    samples = []
    for index, token_count in enumerate((300, 100, 200)):
        sample = _valid_sample(token_count=token_count)
        sample["metadata"]["trajectory_messages"][-1]["content"] += str(index)
        samples.append(sample)

    input_path = tmp_path / "rollout.pt"
    output_path = tmp_path / "sft.jsonl"
    torch.save({"rollout_id": 0, "samples": samples}, input_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_sft.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--max-per-question",
            "2",
        ],
    )

    export_sft.main()

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert [row["metadata"]["token_count"] for row in rows] == [100, 200]
    assert all(row["metadata"]["instance_id"] == "q1" for row in rows)


@pytest.mark.unit
def test_reward_records_exact_match_without_calling_judge(browsecomp_rm, monkeypatch):
    async def unexpected_judge(*args):
        raise AssertionError("exact match must not call the LLM judge")

    monkeypatch.setattr(browsecomp_rm, "judge", unexpected_judge)
    sample = Sample(
        label="The Answer",
        metadata={"question": "question", "answer": "The Answer", "predicted_answer": "the answer"},
    )

    reward = asyncio.run(browsecomp_rm.reward_func(None, sample))

    assert reward == 1
    assert sample.metadata["grading_source"] == "exact_match"
    assert sample.metadata["grading_score"] == 1


@pytest.mark.unit
def test_non_exact_sft_reward_requires_two_positive_judgements(browsecomp_rm, monkeypatch):
    verdicts = iter((1, 0))
    calls = 0

    async def fake_judge(*args):
        nonlocal calls
        calls += 1
        return next(verdicts)

    monkeypatch.setattr(browsecomp_rm, "judge", fake_judge)
    monkeypatch.setenv("BROWSECOMP_JUDGE_CONSENSUS", "1")
    sample = Sample(
        label="answer",
        metadata={"question": "question", "answer": "answer", "predicted_answer": "different wording"},
    )

    reward = asyncio.run(browsecomp_rm.reward_func(None, sample))

    assert calls == 2
    assert reward == 0
    assert sample.metadata["grading_source"] == "llm_judge_consensus"
    assert sample.metadata["grading_score"] == 0
