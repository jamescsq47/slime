#!/usr/bin/env python3
"""Export an official SWE-bench split to the harness JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument(
        "--image-template",
        default="",
        help="optional per-row image_name template with {instance} and {instance_id}",
    )
    args = parser.parse_args()
    selected = set(args.instance_id)
    rows = load_dataset(args.dataset, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as destination:
        for raw in rows:
            row = dict(raw)
            instance_id = str(row["instance_id"])
            if selected and instance_id not in selected:
                continue
            if args.image_template:
                row["image_name"] = args.image_template.format(
                    instance=instance_id.lower().replace("__", "_1776_"),
                    instance_id=instance_id,
                )
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    missing = selected - {
        str(row["instance_id"]) for row in rows if str(row["instance_id"]) in selected
    }
    if missing:
        raise SystemExit(f"instance ids not present in {args.dataset}:{args.split}: {sorted(missing)}")
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
