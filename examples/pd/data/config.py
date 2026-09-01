"""Workload configuration for repeatable multi-harness inference experiments."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_BROWSECOMP_SOURCE_ORDER_SCHEDULE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "workloads"
    / "fixed_browsecomp_source_order_n680.json"
).resolve()


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    harness: str
    path: str
    weight: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingSpec:
    policy: str = "random"
    seed: int = 2026
    preserve_source_order: bool = False
    shuffle_algorithm: str = "legacy_two_stage_v1"
    count_algorithm: str = "largest_remainder_v1"
    pool_reuse_algorithm: str = "cycle_as_needed_v1"
    schedule_file: str | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    datasets: tuple[DatasetSpec, ...]
    sampling: SamplingSpec = SamplingSpec()
    schema_version: int = 1
    source: str | None = None

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(dataset.id for dataset in self.datasets)

    def dataset(self, dataset_id: str) -> DatasetSpec:
        return next(dataset for dataset in self.datasets if dataset.id == dataset_id)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["datasets"] = list(value["datasets"])
        return value


def _expand_path(value: str, *, base_dir: Path) -> str:
    expanded = Path(os.path.expanduser(os.path.expandvars(value)))
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return str(expanded.resolve())


def _validate(payload: dict[str, Any], *, source: Path | None) -> WorkloadSpec:
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("only workload schema_version=1 is supported")
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("workload.datasets must be a non-empty list")

    base_dir = source.parent if source is not None else Path.cwd()
    datasets: list[DatasetSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_datasets):
        if not isinstance(raw, dict):
            raise TypeError(f"datasets[{index}] must be an object")
        dataset_id = str(raw.get("id", ""))
        harness = str(raw.get("harness", ""))
        path = str(raw.get("path", ""))
        if not _ID_RE.fullmatch(dataset_id):
            raise ValueError(f"invalid dataset id: {dataset_id!r}")
        if dataset_id in seen:
            raise ValueError(f"duplicate dataset id: {dataset_id}")
        if not _ID_RE.fullmatch(harness):
            raise ValueError(f"invalid harness name: {harness!r}")
        if not path:
            raise ValueError(f"datasets[{index}].path is required")
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"dataset {dataset_id} weight must be positive")
        options = raw.get("options") or {}
        if not isinstance(options, dict):
            raise TypeError(f"dataset {dataset_id} options must be an object")
        datasets.append(
            DatasetSpec(
                id=dataset_id,
                harness=harness,
                path=_expand_path(path, base_dir=base_dir),
                weight=weight,
                options=dict(options),
            )
        )
        seen.add(dataset_id)

    raw_sampling = payload.get("sampling") or {}
    if not isinstance(raw_sampling, dict):
        raise TypeError("workload.sampling must be an object")
    policy = str(raw_sampling.get("policy", "random"))
    if policy not in {"random", "fixed", "profile_balanced", "alternating", "dynamic"}:
        raise ValueError(f"unsupported sampling policy: {policy}")
    schedule_file = raw_sampling.get("schedule_file")
    if schedule_file is not None:
        schedule_file = _expand_path(str(schedule_file), base_dir=base_dir)
    if policy in {"fixed", "profile_balanced"} and schedule_file is None:
        raise ValueError(f"{policy} sampling requires sampling.schedule_file")
    preserve_source_order = bool(raw_sampling.get("preserve_source_order", False))
    shuffle_algorithm = str(
        raw_sampling.get(
            "shuffle_algorithm",
            "source_order" if preserve_source_order else "legacy_two_stage_v1",
        )
    )
    if shuffle_algorithm not in {"legacy_two_stage_v1", "source_order"}:
        raise ValueError(f"unsupported shuffle_algorithm: {shuffle_algorithm}")
    if preserve_source_order != (shuffle_algorithm == "source_order"):
        raise ValueError(
            "preserve_source_order and shuffle_algorithm disagree; use "
            "preserve_source_order=true with shuffle_algorithm=source_order"
        )
    count_algorithm = str(
        raw_sampling.get("count_algorithm", "largest_remainder_v1")
    )
    if count_algorithm not in {
        "largest_remainder_v1",
        "legacy_two_dataset_round_v1",
    }:
        raise ValueError(f"unsupported count_algorithm: {count_algorithm}")
    if count_algorithm == "legacy_two_dataset_round_v1" and len(datasets) != 2:
        raise ValueError("legacy_two_dataset_round_v1 requires exactly two datasets")
    pool_reuse_algorithm = str(
        raw_sampling.get("pool_reuse_algorithm", "cycle_as_needed_v1")
    )
    if pool_reuse_algorithm not in {
        "cycle_as_needed_v1",
        "cover_all_cycle_v1",
    }:
        raise ValueError(f"unsupported pool_reuse_algorithm: {pool_reuse_algorithm}")

    # Pure BrowseComp measurements use one canonical replay order.  Enforce it
    # here, rather than relying on shell defaults, so an environment override
    # cannot silently turn a comparable source-order run into a shuffled run.
    if len(datasets) == 1 and datasets[0].harness == "browsecomp":
        canonical = (
            policy == "fixed"
            and preserve_source_order
            and shuffle_algorithm == "source_order"
            and count_algorithm == "largest_remainder_v1"
            and pool_reuse_algorithm == "cycle_as_needed_v1"
            and schedule_file is not None
            and Path(schedule_file).resolve() == _BROWSECOMP_SOURCE_ORDER_SCHEDULE
        )
        if not canonical:
            raise ValueError(
                "pure BrowseComp workloads must replay source-order n680: "
                "policy=fixed, preserve_source_order=true, "
                "shuffle_algorithm=source_order, "
                "pool_reuse_algorithm=cycle_as_needed_v1, schedule_file="
                f"{_BROWSECOMP_SOURCE_ORDER_SCHEDULE.name}"
            )

    return WorkloadSpec(
        datasets=tuple(datasets),
        sampling=SamplingSpec(
            policy=policy,
            seed=int(raw_sampling.get("seed", 2026)),
            preserve_source_order=preserve_source_order,
            shuffle_algorithm=shuffle_algorithm,
            count_algorithm=count_algorithm,
            pool_reuse_algorithm=pool_reuse_algorithm,
            schedule_file=schedule_file,
        ),
        schema_version=1,
        source=str(source) if source is not None else None,
    )


def load_workload(path: str | Path) -> WorkloadSpec:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise TypeError("workload config must contain one object")
    return _validate(payload, source=path)


def legacy_workload(
    *,
    math_path: str,
    qa_path: str,
    math_ratio: float,
    policy: str,
    seed: int,
    preserve_source_order: bool,
    schedule_file: str | None,
) -> WorkloadSpec:
    """Translate the old two-domain CLI without changing its public behavior."""

    if not 0.0 <= math_ratio <= 1.0:
        raise ValueError(f"math_ratio must be in [0, 1], got {math_ratio}")
    datasets = []
    if math_ratio > 0:
        datasets.append(
            {
                "id": "math",
                "harness": "retool",
                "path": math_path,
                "weight": math_ratio,
                "options": {
                    "max_response_tokens": int(
                        os.getenv("MIXED_RETOOL_MAX_RESPONSE_LEN", "8192")
                    )
                },
            }
        )
    if math_ratio < 1:
        datasets.append(
            {
                "id": "qa",
                "harness": "browsecomp",
                "path": qa_path,
                "weight": 1 - math_ratio,
                "options": {
                    "max_response_tokens": int(
                        os.getenv("MIXED_BROWSECOMP_MAX_RESPONSE_LEN", "36864")
                    )
                },
            }
        )
    pure_browsecomp = len(datasets) == 1 and datasets[0]["harness"] == "browsecomp"
    if pure_browsecomp:
        policy = "fixed"
        preserve_source_order = True
        schedule_file = str(_BROWSECOMP_SOURCE_ORDER_SCHEDULE)

    return _validate(
        {
            "schema_version": 1,
            "datasets": datasets,
            "sampling": {
                "policy": policy,
                "seed": seed,
                "preserve_source_order": preserve_source_order,
                "shuffle_algorithm": (
                    "source_order" if preserve_source_order else "legacy_two_stage_v1"
                ),
                "count_algorithm": (
                    "legacy_two_dataset_round_v1"
                    if len(datasets) == 2
                    else "largest_remainder_v1"
                ),
                "pool_reuse_algorithm": (
                    "cycle_as_needed_v1"
                    if pure_browsecomp
                    else "cover_all_cycle_v1"
                ),
                "schedule_file": schedule_file,
            },
        },
        source=None,
    )
