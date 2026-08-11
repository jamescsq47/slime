#!/usr/bin/env python3
"""Capture deterministic long-output PD results for Decode-offload A/B tests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def build_prompts(tokenizer, count: int) -> list[list[int]]:
    records = []
    for index in range(count):
        facts = (
            f"Record {index}: amber={1000 + index}, cobalt={2000 + 3 * index}, "
            f"ivory={3000 + 7 * index}. Preserve these exact values. "
        )
        target = 1024 + (index % 4) * 512
        fact_ids = tokenizer.encode(facts, add_special_tokens=False)
        suffix = tokenizer.encode(
            "\nWrite a careful analysis. Keep referring to the record values and "
            "continue until the generation limit. Do not use tools.",
            add_special_tokens=False,
        )
        repeats = max(1, (target - len(suffix) + len(fact_ids) - 1) // len(fact_ids))
        records.append((fact_ids * repeats)[: target - len(suffix)] + suffix)
    return records


def generate(
    url: str,
    request_id: str,
    input_ids: list[int],
    max_new_tokens: int,
    session_params: dict | None = None,
    logical_input_tokens: int | None = None,
) -> dict:
    request = {
        "rid": request_id,
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    if session_params is not None:
        request["session_params"] = session_params
    response = requests.post(
        f"{url.rstrip('/')}/generate",
        json=request,
        timeout=900,
    )
    response.raise_for_status()
    payload = response.json()
    output_logprobs = payload["meta_info"]["output_token_logprobs"]
    return {
        "input_tokens": (
            len(input_ids) if logical_input_tokens is None else logical_input_tokens
        ),
        "output_ids": [int(item[1]) for item in output_logprobs],
        "output_logprobs": [float(item[0]) for item in output_logprobs],
        "cached_tokens": int(payload["meta_info"].get("cached_tokens") or 0),
        "finish_reason": payload["meta_info"].get("finish_reason"),
    }


def concurrent_generate(
    url: str,
    label: str,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> list[dict]:
    barrier = threading.Barrier(len(prompts))

    def one(item: tuple[int, list[int]]) -> tuple[int, dict]:
        index, prompt = item
        barrier.wait(timeout=60)
        return index, generate(url, f"{label}-{index}", prompt, max_new_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(pool.map(one, enumerate(prompts)))
    return [result for _, result in sorted(results)]


def open_session(url: str, session_id: str) -> None:
    response = requests.post(
        f"{url.rstrip('/')}/open_session",
        json={"capacity_of_str_len": 200000, "session_id": session_id},
        timeout=60,
    )
    response.raise_for_status()
    if response.json() != session_id:
        raise RuntimeError(f"failed to open session {session_id}: {response.text}")


def close_session(url: str, session_id: str) -> None:
    response = requests.post(
        f"{url.rstrip('/')}/close_session",
        json={"session_id": session_id},
        timeout=60,
    )
    response.raise_for_status()


def concurrent_session_generate(
    url: str,
    label: str,
    inputs: list[list[int]],
    max_new_tokens: int,
    session_ids: list[str],
    parent_rids: list[str | None],
    logical_input_tokens: list[int],
) -> list[dict]:
    barrier = threading.Barrier(len(inputs))

    def one(item: tuple[int, list[int]]) -> tuple[int, dict]:
        index, input_ids = item
        request_id = f"{label}-{index}"
        params = {"id": session_ids[index]}
        if parent_rids[index] is not None:
            params["rid"] = parent_rids[index]
        barrier.wait(timeout=60)
        return index, generate(
            url,
            request_id,
            input_ids,
            max_new_tokens,
            session_params=params,
            logical_input_tokens=logical_input_tokens[index],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        results = list(pool.map(one, enumerate(inputs)))
    return [result for _, result in sorted(results)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--first-turn-tokens", type=int, default=256)
    parser.add_argument("--inter-turn-delay", type=float, default=0.0)
    parser.add_argument("--session-reference", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tokenizer, args.concurrency)

    sequential = generate(
        args.url, f"{args.label}-sequential", prompts[0], args.first_turn_tokens
    )
    session_ids = [f"{args.label}-session-{index}" for index in range(len(prompts))]
    if args.session_reference:
        for session_id in session_ids:
            open_session(args.url, session_id)
        first_turn = concurrent_session_generate(
            args.url,
            f"{args.label}-turn1",
            prompts,
            args.first_turn_tokens,
            session_ids,
            [None] * len(prompts),
            [len(prompt) for prompt in prompts],
        )
    else:
        first_turn = concurrent_generate(
            args.url, f"{args.label}-turn1", prompts, args.first_turn_tokens
        )

    tool_suffix = tokenizer.encode(
        "\n<tool_result>The archive service confirms all values above.</tool_result>\n"
        "Now continue the analysis and restate the exact record values.",
        add_special_tokens=False,
    )
    if args.inter_turn_delay > 0:
        time.sleep(args.inter_turn_delay)
    second_prompts = [
        prompt + first["output_ids"] + tool_suffix
        for prompt, first in zip(prompts, first_turn)
    ]
    if args.session_reference:
        second_turn = concurrent_session_generate(
            args.url,
            f"{args.label}-turn2",
            [tool_suffix] * len(prompts),
            128,
            session_ids,
            [f"{args.label}-turn1-{index}" for index in range(len(prompts))],
            [len(prompt) for prompt in second_prompts],
        )
        for session_id in session_ids:
            close_session(args.url, session_id)
    else:
        second_turn = concurrent_generate(
            args.url, f"{args.label}-turn2", second_prompts, 128
        )

    result = {
        "label": args.label,
        "concurrency": args.concurrency,
        "page_boundary_exercised": all(
            len(row["output_ids"]) >= args.first_turn_tokens for row in first_turn
        ),
        "sequential": sequential,
        "turn1": first_turn,
        "turn2": second_turn,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "label": args.label,
        "page_boundary_exercised": result["page_boundary_exercised"],
        "turn1_outputs": [len(row["output_ids"]) for row in first_turn],
        "turn2_cached_tokens": [row["cached_tokens"] for row in second_turn],
    }, indent=2))


if __name__ == "__main__":
    main()
