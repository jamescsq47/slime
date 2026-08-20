"""Load every configured source through its harness-specific adapter."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

from .api import LoadContext
from .config import WorkloadSpec
from .registry import get_harness


@dataclass
class LoadedWorkload:
    origin_samples: list[Sample]
    pools: dict[str, list[Sample]]
    spec: WorkloadSpec


def load_samples(args: Any, workload: WorkloadSpec) -> LoadedWorkload:
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
    context = LoadContext(args=args, tokenizer=tokenizer, processor=processor)
    pools: dict[str, list[Sample]] = {}
    origin_samples: list[Sample] = []
    for dataset in workload.datasets:
        harness = get_harness(dataset.harness)
        samples = harness.load_samples(context, dataset)
        if not samples:
            raise ValueError(f"dataset {dataset.id!r} loaded no samples from {dataset.path}")
        for source_position, sample in enumerate(samples):
            metadata = dict(sample.metadata or {})
            metadata.update(
                {
                    "dataset_id": dataset.id,
                    "harness_id": dataset.harness,
                    "source_position": source_position,
                    "tools_available": list(harness.tools),
                }
            )
            # Keep the old analysis/PD metadata key during migration.
            metadata.setdefault("task_type", dataset.id)
            sample.metadata = metadata
        pools[dataset.id] = samples
        origin_samples.extend(samples)
    # Preserve the exact two-stage shuffle used by the original
    # CustomDataSource + inference dispatcher.  Naming the algorithm in the
    # resolved config makes the selected order reproducible after refactors.
    if workload.sampling.shuffle_algorithm == "legacy_two_stage_v1":
        rng = random.Random(workload.sampling.seed)
        for dataset in workload.datasets:
            pools[dataset.id] = sorted(pools[dataset.id], key=lambda _sample: rng.random())
        origin_samples = [sample for dataset in workload.datasets for sample in pools[dataset.id]]
    return LoadedWorkload(origin_samples=origin_samples, pools=pools, spec=workload)
