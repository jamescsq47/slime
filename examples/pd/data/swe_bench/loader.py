"""Load official SWE-bench JSONL/JSON/Parquet rows without prompt rewriting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from slime.utils.types import Sample

from data.api import LoadContext
from data.config import DatasetSpec


LOG = logging.getLogger(__name__)
_REQUIRED = ("instance_id", "problem_statement")


def _json_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        if path.suffix.lower() == ".jsonl":
            for line_number, raw in enumerate(source, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise TypeError(f"SWE-bench row at {path}:{line_number} is not an object")
                yield value
            return
        value = json.load(source)
        rows = value if isinstance(value, list) else value.get("data") if isinstance(value, dict) else None
        if not isinstance(rows, list):
            raise TypeError(f"{path} must contain a JSON array or a data array")
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"SWE-bench row in {path} is not an object")
            yield row


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        yield from _json_rows(path)
        return
    if suffix == ".parquet":
        import pyarrow.parquet as parquet

        yield from parquet.read_table(path).to_pylist()
        return
    raise ValueError(f"unsupported SWE-bench source {path}; use .jsonl, .json, or .parquet")


def load_samples(context: LoadContext, dataset: DatasetSpec) -> list[Sample]:
    del context  # SWE-bench prompts are rendered by the interactive harness.
    path = Path(dataset.path)
    samples: list[Sample] = []
    seen: set[str] = set()
    for position, raw_row in enumerate(_rows(path)):
        row = {str(key): value for key, value in raw_row.items()}
        missing = [key for key in _REQUIRED if not str(row.get(key) or "").strip()]
        if missing:
            raise ValueError(f"SWE-bench row {position} in {path} is missing {missing}")
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-bench instance_id {instance_id!r} in {path}")
        seen.add(instance_id)
        # Keep the official row intact. In particular, patches and problem
        # statements may contain Unicode, CRLFs, NUL-like escape sequences,
        # and arbitrary shell punctuation; none is interpolated into a host
        # shell command by the harness.
        row["instance_id"] = instance_id
        row["problem_statement"] = str(row["problem_statement"])
        samples.append(Sample(prompt="", metadata=row))
    LOG.info("Loaded %d SWE-bench samples from %s", len(samples), path)
    return samples
