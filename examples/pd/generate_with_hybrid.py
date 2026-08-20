"""Dispatch a serving sample to its config-selected inference harness."""

from typing import Any

from data.registry import get_harness
from slime.utils.types import Sample


_LEGACY_HARNESS = {"math": "retool", "qa": "browsecomp"}


async def generate_unified(
    args: Any, sample: Sample, sampling_params: dict[str, Any]
) -> Sample:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_type = str(metadata.get("task_type", "math"))
    harness_id = str(metadata.get("harness_id") or _LEGACY_HARNESS.get(task_type, task_type))
    dataset_id = str(metadata.get("dataset_id") or task_type)
    options_by_dataset = getattr(args, "workload_dataset_options", {})
    options = options_by_dataset.get(dataset_id, {})
    return await get_harness(harness_id).run(
        args,
        sample,
        sampling_params,
        options=options,
    )
