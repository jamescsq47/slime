#!/usr/bin/env python3
"""Integration check that a second turn reuses first-turn decode KV cache."""

from __future__ import annotations

import argparse
import json
import time
import uuid

import requests
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="wait for asynchronous decode KV backup before sending turn two",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    marker = uuid.uuid4().hex
    prompt_ids = tokenizer.encode(
        f"Prefix-cache integration test {marker}. Continue briefly:",
        add_special_tokens=False,
    )
    first = requests.post(
        f"{args.url.rstrip('/')}/generate",
        json={
            "input_ids": prompt_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=120,
    ).json()
    output_ids = [item[1] for item in first["meta_info"]["output_token_logprobs"]]
    observation_ids = tokenizer.encode("\nTool result: continue.", add_special_tokens=False)
    second_ids = prompt_ids + output_ids + observation_ids

    # Give the scheduler a moment to insert the finished request into radix cache.
    time.sleep(args.settle_seconds)
    second = requests.post(
        f"{args.url.rstrip('/')}/generate",
        json={
            "input_ids": second_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        },
        timeout=120,
    ).json()
    cached = int(second["meta_info"].get("cached_tokens") or 0)
    # SGLang intentionally recomputes the token immediately before the new
    # suffix to produce logits, so `cached_tokens` is not expected to equal
    # the complete historical sequence.  Any cached token beyond the original
    # P-side prompt proves that D-generated KV was written back and reused.
    decode_cached_tokens = max(0, cached - len(prompt_ids))
    result = {
        "settle_seconds": args.settle_seconds,
        "first_prompt_tokens": len(prompt_ids),
        "first_decode_tokens": len(output_ids),
        "second_prompt_tokens": len(second_ids),
        "second_cached_tokens": cached,
        "decode_cached_tokens": decode_cached_tokens,
        "decode_kv_reused": decode_cached_tokens > 0,
    }
    print(json.dumps(result, indent=2))
    if not result["decode_kv_reused"]:
        raise RuntimeError("second turn did not reuse the first turn's decode KV cache")


if __name__ == "__main__":
    main()
