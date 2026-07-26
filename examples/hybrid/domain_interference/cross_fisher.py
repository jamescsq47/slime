#!/usr/bin/env python3
"""Matrix-free cross-domain empirical Fisher measurements.

For a checkpoint theta this computes

  g_QA^T F_math g_QA = mean_i (s_math_i^T g_QA)^2
  g_math^T F_QA g_math = mean_i (s_QA_i^T g_math)^2

where all gradients are gradients of mean assistant-token log likelihood.
The normalization is intentionally fixed across checkpoints and recorded in
the output.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distribution_kl import load_records, prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--math-data", type=Path, required=True)
    parser.add_argument("--qa-data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(path: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).to(device)
    model.train()
    model.config.use_cache = False
    return model


def trace_logp(model, ids: torch.Tensor, target_mask: torch.Tensor, device: str):
    ids = ids.unsqueeze(0).to(device)
    mask = target_mask.to(device)
    logits = model(ids, use_cache=False).logits[0, :-1]
    positions = mask.nonzero(as_tuple=False).flatten()
    targets = ids[0, 1:].index_select(0, positions)
    selected = logits.index_select(0, positions).float()
    logp = torch.log_softmax(selected, dim=-1)
    return logp.gather(1, targets[:, None]).mean(), positions.numel()


def trainable_parameters(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def mean_gradient(model, traces, device: str):
    model.zero_grad(set_to_none=True)
    token_counts = []
    for ids, mask in traces:
        logp, count = trace_logp(model, ids, mask, device)
        (-logp / len(traces)).backward()
        token_counts.append(count)
    direction = {
        name: param.grad.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        for name, param in trainable_parameters(model)
        if param.grad is not None
    }
    norm2 = sum(t.float().square().sum(dtype=torch.float64).item() for t in direction.values())
    model.zero_grad(set_to_none=True)
    return direction, math.sqrt(norm2), token_counts


def cpu_dot(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    if a.keys() != b.keys():
        raise ValueError("Gradient parameter sets differ")
    return sum(
        torch.sum(a[name].float() * b[name].float(), dtype=torch.float64).item()
        for name in a
    )


def direction_to_device(direction: dict[str, torch.Tensor], device: str):
    return {name: tensor.to(device) for name, tensor in direction.items()}


def empirical_quadratic(model, traces, direction_cpu, device: str):
    direction = direction_to_device(direction_cpu, device)
    projections = []
    token_counts = []
    params = trainable_parameters(model)
    for ids, mask in traces:
        model.zero_grad(set_to_none=True)
        logp, count = trace_logp(model, ids, mask, device)
        logp.backward()
        projection = torch.zeros((), dtype=torch.float64, device=device)
        for name, param in params:
            if param.grad is not None:
                projection += torch.sum(
                    param.grad.float() * direction[name].float(), dtype=torch.float64
                )
        projections.append(projection.item())
        token_counts.append(count)
    model.zero_grad(set_to_none=True)
    del direction
    gc.collect()
    torch.cuda.empty_cache()
    values = torch.tensor(projections, dtype=torch.float64)
    return {
        "value": values.square().mean().item(),
        "projection_mean": values.mean().item(),
        "projection_std": values.std(unbiased=False).item(),
        "projections": projections,
        "target_token_counts": token_counts,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    math_records = load_records(args.math_data, args.samples, args.seed)
    qa_records = load_records(args.qa_data, args.samples, args.seed)
    math_traces = [prepare(row, tokenizer, args.max_length) for row in math_records]
    qa_traces = [prepare(row, tokenizer, args.max_length) for row in qa_records]
    model = load_model(args.checkpoint, args.device)

    print("computing mean math gradient", flush=True)
    g_math, math_norm, math_tokens = mean_gradient(model, math_traces, args.device)
    print("computing mean QA gradient", flush=True)
    g_qa, qa_norm, qa_tokens = mean_gradient(model, qa_traces, args.device)
    dot = cpu_dot(g_math, g_qa)
    cosine = dot / (math_norm * qa_norm)

    print("computing g_QA^T F_math g_QA", flush=True)
    qa_to_math = empirical_quadratic(model, math_traces, g_qa, args.device)
    print("computing g_math^T F_QA g_math", flush=True)
    math_to_qa = empirical_quadratic(model, qa_traces, g_math, args.device)

    result = {
        "checkpoint": str(args.checkpoint),
        "normalization": "each score and domain gradient is mean assistant-token log likelihood; domain gradient is mean over traces",
        "samples_per_domain": args.samples,
        "seed": args.seed,
        "max_length": args.max_length,
        "math_gradient_norm": math_norm,
        "qa_gradient_norm": qa_norm,
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "math_source_token_counts": math_tokens,
        "qa_source_token_counts": qa_tokens,
        "qa_to_math": {
            **qa_to_math,
            "normalized_by_direction_norm2": qa_to_math["value"] / (qa_norm * qa_norm),
        },
        "math_to_qa": {
            **math_to_qa,
            "normalized_by_direction_norm2": math_to_qa["value"] / (math_norm * math_norm),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
