#!/usr/bin/env python3
"""Stream Qwen3-8B checkpoint deltas and summarize update-direction similarity."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


MODELS = {
    "base": Path("/workspace/Qwen3-8B"),
    "math099": Path("/workspace/Qwen3-8B_hybrid_math_400/iter099"),
    "math199": Path("/workspace/Qwen3-8B_hybrid_math_400/iter199"),
    "math299": Path("/workspace/Qwen3-8B_hybrid_math_400/iter299"),
    "math399": Path("/workspace/Qwen3-8B_hybrid_math_400/iter399"),
    "qa099": Path("/workspace/Qwen3-8B_hybrid_batch400_400/iter099"),
    "qa199": Path("/workspace/Qwen3-8B_hybrid_batch400_400/iter199"),
    "qa299": Path("/workspace/Qwen3-8B_hybrid_batch400_400/iter299"),
    "qa399": Path("/workspace/Qwen3-8B_hybrid_batch400_400/iter399"),
}

DELTAS = {
    "math_000_400": ("math399", "base"),
    "qa_400_800": ("qa399", "math399"),
    "math_000_100": ("math099", "base"),
    "math_100_200": ("math199", "math099"),
    "math_200_300": ("math299", "math199"),
    "math_300_400": ("math399", "math299"),
    "qa_400_500": ("qa099", "math399"),
    "qa_500_600": ("qa199", "qa099"),
    "qa_600_700": ("qa299", "qa199"),
    "qa_700_800": ("qa399", "qa299"),
    "math_cum_000_200": ("math199", "base"),
    "math_cum_000_300": ("math299", "base"),
    "qa_cum_400_600": ("qa199", "math399"),
    "qa_cum_400_700": ("qa299", "math399"),
}

COMPARISONS = {
    "domain_math000_400_vs_qa400_800": ("math_000_400", "qa_400_800"),
    "math_adjacent_000_100_vs_100_200": ("math_000_100", "math_100_200"),
    "math_adjacent_100_200_vs_200_300": ("math_100_200", "math_200_300"),
    "math_adjacent_200_300_vs_300_400": ("math_200_300", "math_300_400"),
    "qa_adjacent_400_500_vs_500_600": ("qa_400_500", "qa_500_600"),
    "qa_adjacent_500_600_vs_600_700": ("qa_500_600", "qa_600_700"),
    "qa_adjacent_600_700_vs_700_800": ("qa_600_700", "qa_700_800"),
    "math_cum_000_200_vs_000_400": ("math_cum_000_200", "math_000_400"),
    "math_cum_000_300_vs_000_400": ("math_cum_000_300", "math_000_400"),
    "qa_cum_400_600_vs_400_800": ("qa_cum_400_600", "qa_400_800"),
    "qa_cum_400_700_vs_400_800": ("qa_cum_400_700", "qa_400_800"),
}


def index_for(model: Path) -> dict[str, str]:
    with (model / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def tensor(model: str, key: str, indexes: dict[str, dict[str, str]]) -> torch.Tensor:
    shard = MODELS[model] / indexes[model][key]
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


def layer_type(key: str) -> str:
    if "embed_tokens" in key:
        return "embed"
    if "lm_head" in key:
        return "lm_head"
    if "norm" in key:
        return "norm"
    if "self_attn" in key:
        return "attn"
    if ".mlp." in key:
        return "mlp"
    return "other"


def empty_stats() -> dict[str, float]:
    return {
        "dot": 0.0,
        "norm_a2": 0.0,
        "norm_b2": 0.0,
        "sqdiff": 0.0,
        "sign_same": 0,
        "count": 0,
        "tensors": 0,
    }


def add_stats(stats: dict[str, float], a: torch.Tensor, b: torch.Tensor) -> None:
    af = a.reshape(-1)
    bf = b.reshape(-1)
    stats["dot"] += torch.dot(af, bf).double().item()
    stats["norm_a2"] += torch.dot(af, af).double().item()
    stats["norm_b2"] += torch.dot(bf, bf).double().item()
    diff = af - bf
    stats["sqdiff"] += torch.dot(diff, diff).double().item()
    stats["sign_same"] += (torch.sign(af) == torch.sign(bf)).sum().item()
    stats["count"] += af.numel()
    stats["tensors"] += 1


def finalize(stats: dict[str, float]) -> dict[str, float]:
    norm_a = math.sqrt(stats["norm_a2"])
    norm_b = math.sqrt(stats["norm_b2"])
    cos = stats["dot"] / (norm_a * norm_b) if norm_a and norm_b else float("nan")
    cos = max(-1.0, min(1.0, cos))
    return {
        "cosine": cos,
        "angle_deg": math.degrees(math.acos(cos)),
        "norm_a": norm_a,
        "norm_b": norm_b,
        "norm_ratio": min(norm_a, norm_b) / max(norm_a, norm_b) if max(norm_a, norm_b) else 1.0,
        "relative_l2_distance": math.sqrt(stats["sqdiff"]) / max(norm_a, norm_b) if max(norm_a, norm_b) else 0.0,
        "sign_agreement": stats["sign_same"] / stats["count"] if stats["count"] else float("nan"),
        "param_count": stats["count"],
        "tensor_count": stats["tensors"],
    }


def main() -> None:
    indexes = {name: index_for(path) for name, path in MODELS.items()}
    keys = sorted(set.intersection(*(set(index.keys()) for index in indexes.values())))
    results = {}

    for comp_name, (delta_a, delta_b) in COMPARISONS.items():
        a_hi, a_lo = DELTAS[delta_a]
        b_hi, b_lo = DELTAS[delta_b]
        global_stats = empty_stats()
        by_type = defaultdict(empty_stats)

        for i, key in enumerate(keys, 1):
            a = tensor(a_hi, key, indexes) - tensor(a_lo, key, indexes)
            b = tensor(b_hi, key, indexes) - tensor(b_lo, key, indexes)
            add_stats(global_stats, a, b)
            add_stats(by_type[layer_type(key)], a, b)
            if i % 50 == 0:
                print(f"{comp_name}: {i}/{len(keys)} tensors", flush=True)

        results[comp_name] = {
            "delta_a": delta_a,
            "delta_b": delta_b,
            "global": finalize(global_stats),
            "by_type": {name: finalize(stats) for name, stats in sorted(by_type.items())},
        }
        print(json.dumps({comp_name: results[comp_name]["global"]}, indent=2), flush=True)

    output = Path("/workspace/slime/param_analysis_results/qwen3_8b_update_orthogonality.json")
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
