#!/usr/bin/env python3
"""Compare two deterministic PD correctness captures token by token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_row(left: dict, right: dict) -> dict:
    left_ids = left["output_ids"]
    right_ids = right["output_ids"]
    shared = min(len(left_ids), len(right_ids))
    first_difference = next(
        (index for index in range(shared) if left_ids[index] != right_ids[index]),
        None,
    )
    if first_difference is None and len(left_ids) != len(right_ids):
        first_difference = shared
    left_lp = left["output_logprobs"]
    right_lp = right["output_logprobs"]
    max_logprob_delta = max(
        (abs(a - b) for a, b in zip(left_lp, right_lp)),
        default=0.0,
    )
    return {
        "exact_output_id_match": left_ids == right_ids,
        "first_difference_token_index": first_difference,
        "left_output_tokens": len(left_ids),
        "right_output_tokens": len(right_ids),
        "max_abs_selected_token_logprob_delta": max_logprob_delta,
        "left_cached_tokens": left["cached_tokens"],
        "right_cached_tokens": right["cached_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    sections = {
        "sequential": [compare_row(left["sequential"], right["sequential"])],
        "turn1": [compare_row(a, b) for a, b in zip(left["turn1"], right["turn1"])],
        "turn2": [compare_row(a, b) for a, b in zip(left["turn2"], right["turn2"])],
    }
    rows = [row for section in sections.values() for row in section]
    result = {
        "all_output_ids_exact": all(row["exact_output_id_match"] for row in rows),
        "compared_requests": len(rows),
        "first_failed_section": next(
            (
                name
                for name, section in sections.items()
                if not all(row["exact_output_id_match"] for row in section)
            ),
            None,
        ),
        "sections": sections,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_output_ids_exact"] else 1)


if __name__ == "__main__":
    main()
