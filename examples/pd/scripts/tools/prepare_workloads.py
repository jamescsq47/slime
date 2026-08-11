#!/usr/bin/env python3
"""Prepare the JSONL workload files consumed by examples/pd/inference.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_jsonl(rows, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(output)
    print(f"Wrote {count} rows to {output}")


def prepare_browsecomp(args: argparse.Namespace) -> None:
    import pandas as pd

    dataframe = pd.read_parquet(args.input)

    def records():
        for index, raw in dataframe.iterrows():
            row = _plain(raw.to_dict())
            extra_info = row.get("extra_info") or {}
            answer = str(row.get("answer") or extra_info.get("answer") or "").strip()
            prompt = row.get("prompt")
            question = extra_info.get("query")
            if not prompt or not answer or not question:
                raise ValueError(
                    f"BrowseComp row {index} requires prompt, answer, and extra_info.query"
                )
            yield {
                "prompt": prompt,
                "label": answer,
                "metadata": {
                    "question": question,
                    "answer": answer,
                    "data_source": row.get("data_source"),
                    "instance_id": extra_info.get("instance_id"),
                },
            }

    _write_jsonl(records(), args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    browsecomp = subparsers.add_parser(
        "browsecomp", help="convert a FoldAgent bc_train/bc_test parquet"
    )
    browsecomp.add_argument("--input", type=Path, required=True)
    browsecomp.add_argument("--output", type=Path, required=True)
    browsecomp.set_defaults(func=prepare_browsecomp)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
