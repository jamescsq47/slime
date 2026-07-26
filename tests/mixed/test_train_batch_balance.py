from types import SimpleNamespace

import pytest

from slime.rollout import sglang_rollout
from slime.utils.types import Sample


def _make_group(task_type: str, group_size: int = 8) -> list[Sample]:
    return [
        Sample(
            prompt=f"{task_type}-{index}",
            response="done",
            reward={"score": 1.0},
            status=Sample.Status.COMPLETED,
            metadata={"task_type": task_type},
        )
        for index in range(group_size)
    ]


def test_strict_train_batch_balance_defers_only_excess_completed_groups():
    quotas = {"math": 16, "qa": 16}
    selected_counts = {"math": 0, "qa": 0}
    accepted = []
    deferred = []

    completion_order = [_make_group("math") for _ in range(20)] + [_make_group("qa") for _ in range(16)]
    for group in completion_order:
        destination = (
            accepted
            if sglang_rollout._admit_group_to_train_batch(group, quotas, selected_counts)
            else deferred
        )
        destination.append(group)

    assert selected_counts == {"math": 16, "qa": 16}
    assert len(accepted) == 32
    assert [sglang_rollout._get_group_task_type(group) for group in deferred] == ["math"] * 4
    assert all(sample.status == Sample.Status.COMPLETED for group in deferred for sample in group)


def test_train_batch_balance_requires_one_known_task_per_group():
    selected_counts = {"math": 0, "qa": 0}

    with pytest.raises(ValueError, match="one task_type per group"):
        sglang_rollout._admit_group_to_train_batch(
            [Sample(prompt="missing metadata")],
            {"math": 16, "qa": 16},
            selected_counts,
        )

    with pytest.raises(ValueError, match="only supports task_type=math or qa"):
        sglang_rollout._admit_group_to_train_batch(
            _make_group("other"),
            {"math": 16, "qa": 16},
            selected_counts,
        )


def test_train_batch_task_quotas_are_optional_or_complete():
    assert (
        sglang_rollout._get_train_batch_task_quotas(
            SimpleNamespace(train_batch_math_groups=None, train_batch_qa_groups=None)
        )
        is None
    )
    assert sglang_rollout._get_train_batch_task_quotas(
        SimpleNamespace(train_batch_math_groups=16, train_batch_qa_groups=16)
    ) == {"math": 16, "qa": 16}

    with pytest.raises(ValueError, match="Both train batch task quotas"):
        sglang_rollout._get_train_batch_task_quotas(
            SimpleNamespace(train_batch_math_groups=16, train_batch_qa_groups=None)
        )
