"""Filter saved BrowseComp train rollouts into structured SFT JSONL.

The input is one or more files created by --save-debug-rollout-data.  Only
successful, searched, non-truncated train trajectories are eligible.  For each
training question we retain a small number of the shortest evidence-backed
trajectories to avoid over-representing easy questions.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def _reward_value(sample: dict[str, Any]) -> float:
    reward = sample.get("reward", 0)
    if isinstance(reward, dict):
        reward = reward.get("score", 0)
    return float(reward or 0)


def _status_value(sample: dict[str, Any]) -> str:
    status = sample.get("status", "")
    return getattr(status, "value", status)


def _assistant_tokens(sample: dict[str, Any]) -> int:
    mask = sample.get("loss_mask")
    return int(sum(mask)) if mask is not None else int(sample.get("response_length", 0))


def _trajectory_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quality_key(sample: dict[str, Any]) -> tuple[int, int, int, int]:
    metadata = sample["metadata"]
    stats = metadata.get("tool_stats") or {}
    # Prefer opened evidence, then fewer turns/tokens/repeated answer changes.
    return (
        -int(stats.get("open_page", 0) > 0),
        int(metadata.get("num_turns", 10**9)),
        len(sample.get("tokens") or []),
        int(stats.get("change_answer", 0)),
    )


def _reject_reason(sample: dict[str, Any], args: argparse.Namespace) -> str | None:
    metadata = sample.get("metadata") or {}
    stats = metadata.get("tool_stats") or {}
    messages = metadata.get("trajectory_messages")
    data_source = str(metadata.get("data_source", ""))

    if not data_source.startswith("bc_train"):
        return "non_train_split"
    if _reward_value(sample) < args.min_reward:
        return "incorrect"
    if _status_value(sample) != "completed" or metadata.get("stop_reason") != "finish":
        return "incomplete"
    if not (metadata.get("predicted_answer") or "").strip():
        return "no_answer"
    if int(stats.get("search", 0)) < args.min_searches:
        return "no_search"
    if int(stats.get("open_page", 0)) < args.min_open_pages:
        return "no_open_page"
    if not isinstance(messages, list) or not messages:
        return "missing_messages"
    if _assistant_tokens(sample) < args.min_assistant_tokens:
        return "too_short"
    if len(sample.get("tokens") or []) > args.max_tokens:
        return "too_long"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="PT files or glob patterns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-question", type=int, default=4)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--min-searches", type=int, default=1)
    parser.add_argument("--min-open-pages", type=int, default=1)
    parser.add_argument("--min-assistant-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=36864)
    args = parser.parse_args()

    paths = sorted({path for pattern in args.input for path in glob.glob(pattern)})
    if not paths:
        parser.error("no input files matched")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    total = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            total += 1
            reason = _reject_reason(sample, args)
            if reason:
                rejected[reason] += 1
                continue
            messages = sample["metadata"]["trajectory_messages"]
            digest = _trajectory_hash(messages)
            if digest in seen_hashes:
                rejected["duplicate_trajectory"] += 1
                continue
            seen_hashes.add(digest)
            metadata = sample["metadata"]
            question_id = str(metadata.get("instance_id") or metadata.get("question"))
            grouped[question_id].append(sample)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for question_id in sorted(grouped):
            for sample in sorted(grouped[question_id], key=_quality_key)[: args.max_per_question]:
                metadata = sample["metadata"]
                record = {
                    "messages": metadata["trajectory_messages"],
                    "metadata": {
                        "instance_id": metadata.get("instance_id"),
                        "question": metadata.get("question"),
                        "answer": metadata.get("answer"),
                        "predicted_answer": metadata.get("predicted_answer"),
                        "grading_source": metadata.get("grading_source", "legacy_reward"),
                        "reward": _reward_value(sample),
                        "num_turns": metadata.get("num_turns"),
                        "tool_stats": metadata.get("tool_stats"),
                        "token_count": len(sample.get("tokens") or []),
                        "assistant_token_count": _assistant_tokens(sample),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "input_files": len(paths),
        "candidate_samples": total,
        "eligible_questions": len(grouped),
        "written": written,
        "rejected": dict(sorted(rejected.items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
