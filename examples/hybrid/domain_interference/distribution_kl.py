#!/usr/bin/env python3
"""Measure token-distribution drift between two checkpoints on fixed traces."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


IM_START = 151644
IM_END = 151645


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def assistant_mask(input_ids: list[int], tokenizer) -> list[bool]:
    """Mask Qwen chat-template assistant spans, excluding role/control tokens."""
    assistant_role = tokenizer.encode("assistant", add_special_tokens=False)
    mask = [False] * len(input_ids)
    index = 0
    while index < len(input_ids):
        if input_ids[index] != IM_START:
            index += 1
            continue
        role_start = index + 1
        role_end = role_start + len(assistant_role)
        is_assistant = input_ids[role_start:role_end] == assistant_role
        content_start = role_end
        while content_start < len(input_ids) and input_ids[content_start] in (198, 220):
            content_start += 1
        end = content_start
        while end < len(input_ids) and input_ids[end] != IM_END:
            end += 1
        if is_assistant:
            for pos in range(content_start, end):
                mask[pos] = True
        index = end + 1
    return mask


def load_records(path: Path, count: int, seed: int) -> list[dict]:
    reservoir: list[dict] = []
    rng = random.Random(seed)
    with path.open() as handle:
        for seen, line in enumerate(handle):
            record = json.loads(line)
            if "messages" not in record:
                continue
            if len(reservoir) < count:
                reservoir.append(record)
            else:
                replacement = rng.randint(0, seen)
                if replacement < count:
                    reservoir[replacement] = record
    return reservoir


def prepare(record: dict, tokenizer, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer.apply_chat_template(
        record["messages"], tokenize=True, add_generation_prompt=False
    )
    mask = assistant_mask(ids, tokenizer)
    # Causal-LM KL requires a contiguous prefix. Never splice a head and tail:
    # tokens after such a join would be conditioned on a context the model
    # never saw in the original trajectory.
    if len(ids) > max_length:
        ids = ids[:max_length]
        mask = mask[:max_length]
    input_ids = torch.tensor(ids, dtype=torch.long)
    target_mask = torch.tensor(mask[1:], dtype=torch.bool)
    if not target_mask.any():
        raise ValueError("No assistant target tokens remain after truncation")
    return input_ids, target_mask


def load_model(path: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    return model.eval().to(device)


@torch.inference_mode()
def compare_trace(ref_model, cand_model, ids, target_mask, device: str) -> dict:
    ids = ids.unsqueeze(0).to(device)
    ref_logits = ref_model(ids, use_cache=False).logits[0, :-1]
    cand_logits = cand_model(ids, use_cache=False).logits[0, :-1]
    positions = target_mask.to(device).nonzero(as_tuple=False).flatten()
    targets = ids[0, 1:].index_select(0, positions)

    sums = {"forward_kl": 0.0, "reverse_kl": 0.0, "ref_nll": 0.0, "cand_nll": 0.0}
    token_forward: list[float] = []
    chunk_size = 32
    for start in range(0, positions.numel(), chunk_size):
        pos = positions[start : start + chunk_size]
        target = targets[start : start + chunk_size]
        rlogp = torch.log_softmax(ref_logits.index_select(0, pos).float(), dim=-1)
        clogp = torch.log_softmax(cand_logits.index_select(0, pos).float(), dim=-1)
        rp = rlogp.exp()
        cp = clogp.exp()
        forward = torch.sum(rp * (rlogp - clogp), dim=-1)
        reverse = torch.sum(cp * (clogp - rlogp), dim=-1)
        sums["forward_kl"] += forward.sum().item()
        sums["reverse_kl"] += reverse.sum().item()
        sums["ref_nll"] -= rlogp.gather(1, target[:, None]).sum().item()
        sums["cand_nll"] -= clogp.gather(1, target[:, None]).sum().item()
        token_forward.extend(forward.cpu().tolist())

    count = positions.numel()
    del ref_logits, cand_logits
    return {
        "tokens": count,
        **{key: value / count for key, value in sums.items()},
        "forward_kl_p50": float(np.quantile(token_forward, 0.5)),
        "forward_kl_p90": float(np.quantile(token_forward, 0.9)),
        "forward_kl_p99": float(np.quantile(token_forward, 0.99)),
    }


def weighted_summary(rows: list[dict]) -> dict:
    total = sum(row["tokens"] for row in rows)
    keys = ("forward_kl", "reverse_kl", "ref_nll", "cand_nll")
    return {
        "samples": len(rows),
        "tokens": total,
        **{
            key: sum(row[key] * row["tokens"] for row in rows) / total
            for key in keys
        },
        "sample_forward_kl_p50": float(np.quantile([row["forward_kl"] for row in rows], 0.5)),
        "sample_forward_kl_p90": float(np.quantile([row["forward_kl"] for row in rows], 0.9)),
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.reference)
    records = load_records(args.data, args.samples, args.seed)
    prepared = [prepare(record, tokenizer, args.max_length) for record in records]
    print(f"loaded {len(prepared)} {args.domain} traces")
    ref_model = load_model(args.reference, args.device)
    cand_model = load_model(args.candidate, args.device)

    rows = []
    for index, (ids, mask) in enumerate(prepared):
        row = compare_trace(ref_model, cand_model, ids, mask, args.device)
        row["index"] = index
        row["sequence_tokens"] = ids.numel()
        rows.append(row)
        print(index, json.dumps(row))

    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "data": str(args.data),
        "domain": args.domain,
        "seed": args.seed,
        "max_length": args.max_length,
        "summary": weighted_summary(rows),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
