#!/usr/bin/env python3
"""Force allocator reuse while deterministic Decode requests are still active."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def build_prompt(tokenizer, label: str, target: int) -> list[int]:
    fact = tokenizer.encode(
        f"{label}: preserve alpha=1729 beta=4099 gamma=8191. ",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\nAnalyze these values repeatedly and continue until the output limit.",
        add_special_tokens=False,
    )
    repeats = max(1, (target - len(suffix) + len(fact) - 1) // len(fact))
    return (fact * repeats)[: target - len(suffix)] + suffix


def generate(url: str, rid: str, prompt: list[int], output_tokens: int) -> dict:
    response = requests.post(
        f"{url.rstrip('/')}/generate",
        json={
            "rid": rid,
            "input_ids": prompt,
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "max_new_tokens": output_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        },
        timeout=900,
    )
    response.raise_for_status()
    payload = response.json()
    logprobs = payload["meta_info"]["output_token_logprobs"]
    return {
        "input_tokens": len(prompt),
        "output_ids": [int(item[1]) for item in logprobs],
        "output_logprobs": [float(item[0]) for item in logprobs],
        "cached_tokens": int(payload["meta_info"].get("cached_tokens") or 0),
    }


def metric_value(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def wait_for_running(decode_url: str, minimum: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{decode_url.rstrip('/')}/metrics", timeout=5)
        response.raise_for_status()
        running = metric_value(response.text, "sglang:num_running_reqs")
        if running >= minimum:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Decode did not reach {minimum} running requests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=12)
    parser.add_argument("--churn-count", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    anchors = [
        build_prompt(tokenizer, f"anchor-{index}", 2944 + 128 * (index % 4))
        for index in range(args.anchor_count)
    ]
    churn = [
        build_prompt(tokenizer, f"churn-{index}", 1408 + 128 * (index % 2))
        for index in range(args.churn_count)
    ]
    barrier = threading.Barrier(args.anchor_count)

    def anchor(item: tuple[int, list[int]]) -> tuple[int, dict]:
        index, prompt = item
        barrier.wait(timeout=60)
        return index, generate(
            args.url, f"{args.label}-anchor-{index}", prompt, 1024
        )

    def churn_request(item: tuple[int, list[int]]) -> tuple[int, dict]:
        index, prompt = item
        return index, generate(
            args.url, f"{args.label}-churn-{index}", prompt, 128
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.anchor_count + args.churn_count
    ) as pool:
        anchor_futures = [pool.submit(anchor, item) for item in enumerate(anchors)]
        wait_for_running(args.decode_url, max(4, args.anchor_count // 2), 120)
        # At roughly 100+ decoded tokens/request/s this allows the first 64-token
        # chunk to be offloaded before new allocations begin.
        time.sleep(1.0)
        churn_futures = []
        for start in range(0, args.churn_count, 8):
            churn_futures.extend(
                pool.submit(churn_request, item)
                for item in list(enumerate(churn))[start : start + 8]
            )
            time.sleep(0.25)
        anchor_results = [future.result() for future in anchor_futures]
        churn_results = [future.result() for future in churn_futures]

    result = {
        "label": args.label,
        "anchors": [value for _, value in sorted(anchor_results)],
        "churn": [value for _, value in sorted(churn_results)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "anchor_outputs": [len(row["output_ids"]) for row in result["anchors"]],
                "churn_outputs": [len(row["output_ids"]) for row in result["churn"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
