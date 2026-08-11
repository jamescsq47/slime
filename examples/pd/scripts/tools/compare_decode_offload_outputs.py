#!/usr/bin/env python3
"""Capture deterministic PD outputs and compare Decode-offload A/B runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def build_prompts(tokenizer):
    prefix = tokenizer.encode(
        "Archive record: amber=17, cobalt=29, ivory=43, and juniper=61. "
        "These values are immutable and must be remembered. ",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\nWithout using tools, reason carefully about the archive and finish "
        "your response with exactly: FINAL cobalt=29.",
        add_special_tokens=False,
    )
    prompts = []
    for target in (1024, 4096):
        repeat_count = max(1, (target - len(suffix) + len(prefix) - 1) // len(prefix))
        body = (prefix * repeat_count)[: max(1, target - len(suffix))]
        prompts.append(body + suffix)
    return prompts


def generate(url: str, prompt_ids: list[int]) -> dict:
    response = requests.post(
        f"{url.rstrip('/')}/generate",
        json={
            "input_ids": prompt_ids,
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "max_new_tokens": 256,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    output_ids = [item[1] for item in payload["meta_info"]["output_token_logprobs"]]
    return {
        "prompt_tokens": len(prompt_ids),
        "output_ids": output_ids,
        "output_text": payload.get("text", ""),
        "finish_reason": payload["meta_info"].get("finish_reason"),
        "cached_tokens": payload["meta_info"].get("cached_tokens"),
    }


def compare(left: dict, right: dict) -> dict:
    rows = []
    for index, (a, b) in enumerate(zip(left["requests"], right["requests"])):
        first_difference = next(
            (i for i, pair in enumerate(zip(a["output_ids"], b["output_ids"])) if pair[0] != pair[1]),
            None,
        )
        exact = a["output_ids"] == b["output_ids"]
        rows.append(
            {
                "request_index": index,
                "prompt_tokens": a["prompt_tokens"],
                "exact_output_id_match": exact,
                "first_difference_token_index": first_difference,
                "left_output_tokens": len(a["output_ids"]),
                "right_output_tokens": len(b["output_ids"]),
            }
        )
    return {"all_exact": all(row["exact_output_id_match"] for row in rows), "requests": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-left", type=Path)
    parser.add_argument("--compare-right", type=Path)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Issue prompts one at a time to avoid batch-shape numerical differences.",
    )
    args = parser.parse_args()

    if args.compare_left and args.compare_right:
        result = compare(
            json.loads(args.compare_left.read_text()),
            json.loads(args.compare_right.read_text()),
        )
    else:
        if not all((args.url, args.model, args.label, args.output)):
            parser.error("capture mode requires --url, --model, --label, and --output")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = build_prompts(tokenizer)
        if args.sequential:
            requests_data = [generate(args.url, ids) for ids in prompts]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                requests_data = list(pool.map(lambda ids: generate(args.url, ids), prompts))
        result = {"label": args.label, "requests": requests_data}
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
