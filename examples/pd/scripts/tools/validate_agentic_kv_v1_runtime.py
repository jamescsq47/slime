#!/usr/bin/env python3
"""Exercise the Agentic KV V1 direct, Mooncake, and terminal paths.

This is a deliberately small serving-level test.  It sends real tokenized
requests through the PD router and uses a fixed regex output only to make the
tool-call boundary deterministic; KV movement is performed by the real P/D
workers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

PD_DIR = Path(__file__).resolve().parents[2]
if str(PD_DIR) not in sys.path:
    sys.path.insert(0, str(PD_DIR))

from agentic_kv_request import build_agentic_extra_key


TOOL_MARKER = " VALIDATION_TOOL_END"
TOOL_OUTPUT = (
    '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(2)"}}'
    f"</tool_call>{TOOL_MARKER}"
)
FINAL_OUTPUT = "<answer>done</answer>"


def output_ids(payload: dict) -> list[int]:
    return [int(item[1]) for item in payload["meta_info"]["output_token_logprobs"]]


def generate(
    url: str,
    input_ids: list[int],
    *,
    request_id: str,
    generation: int,
    parent_generation: int | None,
    exact_output: str | None,
    is_tool_output: bool,
    tokenizer,
    max_new_tokens: int = 128,
) -> tuple[dict, float]:
    custom_params = {
        "agentic_request_id": request_id,
        "agentic_generation": generation,
        "agentic_tool_type": "validation_tool",
        # Constrained decoding can represent the same text with different BPE
        # pieces on different workers.  EOS is deterministic, so this runtime
        # harness declares EOS as the tool boundary only on first-turn calls.
        # Production agents continue to use their real tool suffixes.
        "agentic_tool_suffix_token_ids": (
            [[int(tokenizer.eos_token_id)]] if is_tool_output else []
        ),
        "agentic_tool_suffix_strings": [TOOL_MARKER] if is_tool_output else [],
        "agentic_terminal_marker_token_ids": [
            tokenizer.encode("<answer>", add_special_tokens=False)
        ],
        "agentic_terminal_marker_strings": ["<answer>"],
    }
    if parent_generation is not None:
        custom_params["agentic_parent_generation"] = parent_generation
    sampling_params = {
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "max_new_tokens": max_new_tokens,
        "custom_params": custom_params,
    }
    if exact_output is not None:
        sampling_params["regex"] = re.escape(exact_output)
    body = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        # Request output-token logprobs without forcing prompt logprobs.  A
        # value of 0 intentionally caps radix matching at zero in SGLang and
        # would make this cache-reuse validator invalidate its own experiment.
        "logprob_start_len": -1,
        "extra_key": build_agentic_extra_key(request_id, sampling_params),
    }
    started = time.monotonic()
    response = requests.post(f"{url.rstrip('/')}/generate", json=body, timeout=300)
    response.raise_for_status()
    elapsed = time.monotonic() - started
    result = response.json()
    if exact_output is not None and result.get("text") != exact_output:
        raise AssertionError(
            f"fixed output mismatch for {request_id}: {result.get('text')!r}"
        )
    return result, elapsed


def run_reuse_equivalence(url: str, tokenizer, label: str, delay: float) -> dict:
    """Compare restored D->P KV with a full-Prefill control.

    The first turn is constrained only to make the parent token sequence
    identical and deterministic.  The two second turns are unconstrained:
    one consumes the reverse-KV snapshot and the other recomputes the exact
    same input from scratch.  At temperature zero their token IDs should be
    identical.  This catches wrong-page and prefix-boundary bugs that a fully
    constrained path test cannot observe.
    """

    request_id = f"agentic-v1-validation-{label}-reuse"
    first_ids = initial_prompt(tokenizer, label)
    first, _ = generate(
        url,
        first_ids,
        request_id=request_id,
        generation=0,
        parent_generation=None,
        exact_output=TOOL_OUTPUT,
        is_tool_output=True,
        tokenizer=tokenizer,
    )
    first_output_ids = output_ids(first)
    time.sleep(delay)
    observation_ids = tokenizer.encode(
        "\n<tool_response>2</tool_response>\n"
        "Restate amber, cobalt, ivory, and juniper with their exact values.",
        add_special_tokens=False,
    )
    second_ids = first_ids + first_output_ids + observation_ids

    # Claim the Direct offer before its two-second fast-path deadline.  The
    # later full-Prefill control uses a different generation-scoped extra key,
    # so it cannot match the reused request's local radix entry.
    reused, reused_elapsed = generate(
        url,
        second_ids,
        request_id=request_id,
        generation=1,
        parent_generation=0,
        exact_output=None,
        is_tool_output=False,
        tokenizer=tokenizer,
        max_new_tokens=64,
    )
    control, control_elapsed = generate(
        url,
        second_ids,
        request_id=f"{request_id}-control",
        generation=0,
        parent_generation=None,
        exact_output=None,
        is_tool_output=False,
        tokenizer=tokenizer,
        max_new_tokens=64,
    )
    control_ids = output_ids(control)
    reused_ids = output_ids(reused)
    common = 0
    for left, right in zip(control_ids, reused_ids):
        if left != right:
            break
        common += 1
    return {
        "request_id": request_id,
        "delay_seconds": delay,
        "second_prompt_tokens": len(second_ids),
        "control_elapsed_seconds": control_elapsed,
        "reused_elapsed_seconds": reused_elapsed,
        "control_cached_tokens": control["meta_info"].get("cached_tokens"),
        "reused_cached_tokens": reused["meta_info"].get("cached_tokens"),
        "control_output_ids": control_ids,
        "reused_output_ids": reused_ids,
        "common_prefix_tokens": common,
        "token_exact_match": control_ids == reused_ids,
        "control_text": control.get("text", ""),
        "reused_text": reused.get("text", ""),
    }


def initial_prompt(tokenizer, label: str) -> list[int]:
    # Keep the snapshot comfortably above one 64-token page and make each
    # trajectory distinct so a local radix hit cannot masquerade as D->P reuse.
    record = (
        f"Agentic KV runtime validation trajectory {label}. "
        "The following archive is stable context and must remain available: "
        "amber 17, cobalt 29, ivory 43, juniper 61. "
    )
    text = record * 12 + "Return the requested structured result now."
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 128:
        raise AssertionError(f"validation prompt unexpectedly short: {len(ids)}")
    return ids


def run_two_turn(url: str, tokenizer, label: str, delay: float) -> dict:
    request_id = f"agentic-v1-validation-{label}"
    first_ids = initial_prompt(tokenizer, label)
    first, first_elapsed = generate(
        url,
        first_ids,
        request_id=request_id,
        generation=0,
        parent_generation=None,
        exact_output=TOOL_OUTPUT,
        is_tool_output=True,
        tokenizer=tokenizer,
    )
    first_output_ids = output_ids(first)
    time.sleep(delay)
    observation_ids = tokenizer.encode(
        "\n<tool_response>2</tool_response>\nNow provide the final answer.",
        add_special_tokens=False,
    )
    second_ids = first_ids + first_output_ids + observation_ids
    second, second_elapsed = generate(
        url,
        second_ids,
        request_id=request_id,
        generation=1,
        parent_generation=0,
        exact_output=FINAL_OUTPUT,
        is_tool_output=False,
        tokenizer=tokenizer,
    )
    return {
        "request_id": request_id,
        "delay_seconds": delay,
        "first_prompt_tokens": len(first_ids),
        "tool_output_tokens": len(first_output_ids),
        "second_prompt_tokens": len(second_ids),
        "first_elapsed_seconds": first_elapsed,
        "second_elapsed_seconds": second_elapsed,
        "first_cached_tokens": first["meta_info"].get("cached_tokens"),
        "second_cached_tokens": second["meta_info"].get("cached_tokens"),
        "second_finish_reason": second["meta_info"].get("finish_reason"),
    }


def run_terminal(url: str, tokenizer, label: str) -> dict:
    request_id = f"agentic-v1-validation-{label}"
    ids = initial_prompt(tokenizer, label)
    result, elapsed = generate(
        url,
        ids,
        request_id=request_id,
        generation=0,
        parent_generation=None,
        exact_output=FINAL_OUTPUT,
        is_tool_output=False,
        tokenizer=tokenizer,
    )
    return {
        "request_id": request_id,
        "prompt_tokens": len(ids),
        "output_tokens": len(output_ids(result)),
        "elapsed_seconds": elapsed,
        "cached_tokens": result["meta_info"].get("cached_tokens"),
        "finish_reason": result["meta_info"].get("finish_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:36503")
    parser.add_argument("--model", default="/dataset/model/qwen3/Qwen3-8B")
    parser.add_argument("--direct-delay", type=float, default=0.02)
    parser.add_argument("--fallback-delay", type=float, default=1.0)
    parser.add_argument("--run-id", default=str(time.time_ns()))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--equivalence-only",
        action="store_true",
        help="run only the unconstrained restored-KV/full-prefill A/B check",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    direct_label = f"direct-{args.run_id}"
    fallback_label = f"fallback-{args.run_id}"
    terminal_label = f"terminal-{args.run_id}"
    if args.equivalence_only:
        result = {
            "reuse_equivalence": run_reuse_equivalence(
                args.url,
                tokenizer,
                f"equivalence-{args.run_id}",
                args.direct_delay,
            )
        }
    else:
        result = {
            "direct": run_two_turn(
                args.url, tokenizer, direct_label, args.direct_delay
            ),
            "fallback": run_two_turn(
                args.url, tokenizer, fallback_label, args.fallback_delay
            ),
            "terminal": run_terminal(args.url, tokenizer, terminal_label),
            "reuse_equivalence": run_reuse_equivalence(
            args.url, tokenizer, f"equivalence-{args.run_id}", args.direct_delay
            ),
        }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
