#!/usr/bin/env python3
"""Compute first-k empirical pass rates from saved BrowseComp eval trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

KS = (1, 2, 4, 8, 16, 32, 64)


def reward_score(sample: dict) -> float:
    reward = sample.get("reward", 0)
    if isinstance(reward, dict):
        reward = reward.get("score", 0)
    if hasattr(reward, "item"):
        reward = reward.item()
    return float(reward or 0)


def analyze(path: Path, samples_per_prompt: int = 64) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = sorted(payload["samples"], key=lambda sample: int(sample["index"]))
    if len(samples) % samples_per_prompt:
        raise ValueError(f"{path}: {len(samples)} samples is not divisible by {samples_per_prompt}")
    groups = [samples[i : i + samples_per_prompt] for i in range(0, len(samples), samples_per_prompt)]
    for group_id, group in enumerate(groups):
        indices = [int(sample["index"]) for sample in group]
        expected = list(range(group_id * samples_per_prompt, (group_id + 1) * samples_per_prompt))
        if indices != expected:
            raise ValueError(f"{path}: non-contiguous indices in group {group_id}")
    result = {
        "model": path.name.removesuffix("-browsecomp.pt"),
        "path": str(path),
        "questions": len(groups),
        "samples": len(samples),
    }
    for k in KS:
        passed = sum(any(reward_score(sample) > 0 for sample in group[:k]) for group in groups)
        result[f"pass@{k}"] = passed / len(groups)
        result[f"passed@{k}"] = passed
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=64)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    rows = [analyze(path, args.samples_per_prompt) for path in args.paths]
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, indent=2) + "\n")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    headers = ["model", *[f"pass@{k}" for k in KS]]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] + ["---:"] * len(KS)) + " |")
    for row in rows:
        print("| " + " | ".join([row["model"], *[f"{100 * row[f'pass@{k}']:.2f}%" for k in KS]]) + " |")


if __name__ == "__main__":
    main()
