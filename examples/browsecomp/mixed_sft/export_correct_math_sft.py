#!/usr/bin/env python3
"""Export correct DAPO math rollouts into exact-mask SFT records."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

USER_PATTERN = re.compile(r"<\|im_start\|>user\s*\n?(.*?)<\|im_end\|>", re.DOTALL)


def _score(reward: Any) -> float:
    return float((reward.get("score", 0.0) if isinstance(reward, dict) else reward) or 0.0)


def _status(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _user(prompt: Any) -> str:
    if isinstance(prompt, list):
        users = [m.get("content", "") for m in prompt if m.get("role") == "user"]
        if len(users) == 1:
            return str(users[0])
    if isinstance(prompt, str):
        users = USER_PATTERN.findall(prompt)
        if len(users) == 1:
            return users[0].strip()
        return prompt
    raise TypeError(f"unsupported prompt type: {type(prompt)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=558)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-total-tokens", type=int, default=10240)
    args = parser.parse_args()

    paths = sorted({p for pattern in args.input for p in glob.glob(pattern)})
    if not paths:
        parser.error("no rollout files matched")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for sample in payload.get("samples", []):
            meta = sample.get("metadata") or {}
            tokens, mask, response = sample.get("tokens") or [], sample.get("loss_mask") or [], sample.get("response") or ""
            if (meta.get("task_type", "math") != "math" or _score(sample.get("reward")) < 1 or
                    _status(sample.get("status")) != "completed" or not response or not mask or
                    len(tokens) < len(mask) or len(tokens) > args.max_total_tokens):
                rejected += 1
                continue
            prompt = _user(sample.get("prompt"))
            prompt_id = hashlib.sha256(prompt.encode()).hexdigest()
            grouped[prompt_id].append({
                "prompt": prompt, "response": response, "label": sample.get("label"),
                "tokens": tokens, "mask": mask, "reward": _score(sample.get("reward")),
                "file": path, "prompt_id": prompt_id,
                "response_id": hashlib.sha256(response.encode()).hexdigest(),
            })

    candidates, seen = [], set()
    for prompt_id in sorted(grouped):
        for item in sorted(grouped[prompt_id], key=lambda x: (len(x["tokens"]), x["response_id"])):
            if item["response_id"] not in seen:
                candidates.append(item)
                seen.add(item["response_id"])
    if len(candidates) < args.target:
        raise SystemExit(f"need {args.target} correct trajectories, found {len(candidates)}")
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[:args.target]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in selected:
            record = {
                "messages": [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response"]}],
                "metadata": {
                    "sft_source": "dapo_math_correct_rollout", "prompt_id": item["prompt_id"],
                    "response_id": item["response_id"], "answer": item["label"], "reward": item["reward"],
                    "source_file": item["file"], "pretokenized_tokens": item["tokens"],
                    "pretokenized_loss_mask": item["mask"], "total_tokens": len(item["tokens"]),
                    "supervised_tokens": int(sum(item["mask"])),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"input_files": len(paths), "eligible": len(candidates), "rejected": rejected, "written": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
