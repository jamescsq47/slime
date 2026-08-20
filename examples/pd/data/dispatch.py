"""Deterministic, reusable dispatch schedules for arbitrary dataset mixtures."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from slime.utils.types import Sample

from .config import WorkloadSpec


def exact_counts(total: int, workload: WorkloadSpec) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    weight_sum = sum(dataset.weight for dataset in workload.datasets)
    if workload.sampling.count_algorithm == "legacy_two_dataset_round_v1":
        first, second = workload.datasets
        first_count = round(total * first.weight / weight_sum)
        return {first.id: first_count, second.id: total - first_count}
    raw = {
        dataset.id: total * dataset.weight / weight_sum for dataset in workload.datasets
    }
    counts = {dataset.id: int(raw[dataset.id]) for dataset in workload.datasets}
    remainder = total - sum(counts.values())
    order = sorted(
        range(len(workload.datasets)),
        key=lambda index: (-(raw[workload.datasets[index].id] - counts[workload.datasets[index].id]), index),
    )
    for index in order[:remainder]:
        counts[workload.datasets[index].id] += 1
    return counts


def warmup_labels(total: int, workload: WorkloadSpec) -> list[str]:
    """Preserve the old two-pool warmup sequence where requested."""

    if workload.sampling.count_algorithm != "legacy_two_dataset_round_v1":
        return weighted_alternating_labels(total, workload)
    first, second = workload.datasets
    ratio = first.weight / (first.weight + second.weight)
    if ratio == 1.0:
        return [first.id] * total
    if ratio == 0.0:
        return [second.id] * total
    return [
        first.id
        if ((index + 1) * ratio).__ceil__() > (index * ratio).__ceil__()
        else second.id
        for index in range(total)
    ]


def weighted_alternating_labels(total: int, workload: WorkloadSpec) -> list[str]:
    """Smooth weighted round-robin with deterministic config-order tie breaking."""

    counts = exact_counts(total, workload)
    emitted = {dataset.id: 0 for dataset in workload.datasets}
    labels: list[str] = []
    for position in range(total):
        candidates = [dataset for dataset in workload.datasets if emitted[dataset.id] < counts[dataset.id]]
        choice = max(
            candidates,
            key=lambda dataset: (
                (position + 1) * counts[dataset.id] / max(total, 1) - emitted[dataset.id],
                -workload.dataset_ids.index(dataset.id),
            ),
        )
        labels.append(choice.id)
        emitted[choice.id] += 1
    return labels


def generated_labels(total: int, workload: WorkloadSpec) -> list[str]:
    counts = exact_counts(total, workload)
    labels = [dataset.id for dataset in workload.datasets for _ in range(counts[dataset.id])]
    if workload.sampling.policy == "random":
        random.Random(workload.sampling.seed + 1).shuffle(labels)
    elif workload.sampling.policy == "alternating":
        labels = weighted_alternating_labels(total, workload)
    elif workload.sampling.policy not in {"fixed", "profile_balanced", "dynamic"}:
        raise ValueError(f"unsupported sampling policy: {workload.sampling.policy}")
    return labels


def _cycle_to_length(values: list[Sample], target: int) -> list[Sample]:
    if not values:
        return []
    return [values[index % len(values)] for index in range(target)]


def _cover_all_counts(
    pools: dict[str, list[Sample]], workload: WorkloadSpec
) -> dict[str, int]:
    """Size the mixed epoch so every source is represented at its configured ratio."""

    weight_sum = sum(dataset.weight for dataset in workload.datasets)
    total = max(
        math.ceil(len(pools[dataset.id]) * weight_sum / dataset.weight)
        for dataset in workload.datasets
    )
    while True:
        counts = exact_counts(total, workload)
        if all(counts[dataset.id] >= len(pools[dataset.id]) for dataset in workload.datasets):
            return counts
        total += 1


def select_samples(
    pools: dict[str, list[Sample]],
    workload: WorkloadSpec,
    *,
    measured_count: int,
    warmup_count: int,
    schedule: list[dict[str, Any]] | None = None,
) -> tuple[list[Sample], list[dict[str, Any]]]:
    """Select exact-composition samples and return the reusable resolved schedule."""

    selected_pools = {dataset_id: list(values) for dataset_id, values in pools.items()}
    if workload.sampling.pool_reuse_algorithm == "cover_all_cycle_v1":
        cover_counts = _cover_all_counts(selected_pools, workload)
        selected_pools = {
            dataset_id: _cycle_to_length(values, cover_counts[dataset_id])
            for dataset_id, values in selected_pools.items()
        }
    if not workload.sampling.preserve_source_order:
        rng = random.Random(workload.sampling.seed)
        for values in selected_pools.values():
            rng.shuffle(values)

    selected_warmup_labels = warmup_labels(warmup_count, workload)
    labels = generated_labels(measured_count, workload)
    if workload.sampling.policy in {"fixed", "profile_balanced"}:
        if schedule is None or len(schedule) != measured_count:
            raise ValueError("fixed schedule must contain exactly measured_count entries")
        labels = [str(entry.get("dataset_id") or entry.get("task_type")) for entry in schedule]
        if {dataset.id: labels.count(dataset.id) for dataset in workload.datasets} != exact_counts(
            measured_count, workload
        ):
            raise ValueError("fixed schedule composition does not match workload weights")

    required = {
        dataset.id: labels.count(dataset.id) + selected_warmup_labels.count(dataset.id)
        for dataset in workload.datasets
    }
    for dataset_id, count in required.items():
        if not selected_pools[dataset_id]:
            raise RuntimeError(f"dataset {dataset_id!r} has no samples")
        if len(selected_pools[dataset_id]) < count:
            selected_pools[dataset_id] = _cycle_to_length(
                selected_pools[dataset_id], count
            )

    offsets = {dataset.id: 0 for dataset in workload.datasets}
    selected: list[Sample] = []
    for dataset_id in selected_warmup_labels:
        selected.append(copy.deepcopy(selected_pools[dataset_id][offsets[dataset_id]]))
        offsets[dataset_id] += 1

    resolved: list[dict[str, Any]] = []
    for position, dataset_id in enumerate(labels):
        if dataset_id not in selected_pools:
            raise ValueError(f"schedule references unknown dataset {dataset_id!r}")
        if workload.sampling.policy in {"fixed", "profile_balanced"}:
            entry = schedule[position]
            sample_id = str(entry["experiment_sample_id"])
            prefix = f"{dataset_id}-"
            if not sample_id.startswith(prefix):
                raise ValueError(f"schedule entry {sample_id!r} does not match {dataset_id!r}")
            pool_index = int(sample_id.removeprefix(prefix))
            sample = copy.deepcopy(selected_pools[dataset_id][pool_index])
        else:
            pool_index = offsets[dataset_id]
            sample = copy.deepcopy(selected_pools[dataset_id][pool_index])
            offsets[dataset_id] += 1
            sample_id = f"{dataset_id}-{pool_index}"
        metadata = dict(sample.metadata or {})
        metadata.update(
            {
                "dispatch_policy": workload.sampling.policy,
                "dispatch_position": position,
                "experiment_sample_id": sample_id,
            }
        )
        sample.metadata = metadata
        selected.append(sample)
        resolved.append(
            {
                "position": position,
                "dataset_id": dataset_id,
                "harness_id": metadata["harness_id"],
                "task_type": metadata["task_type"],
                "experiment_sample_id": sample_id,
                "source_position": metadata["source_position"],
            }
        )

    for index, sample in enumerate(selected):
        sample.index = index
        sample.group_index = index
    return selected, resolved
