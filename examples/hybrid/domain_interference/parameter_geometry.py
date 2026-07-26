#!/usr/bin/env python3
"""Stream exact checkpoint-vector geometry without loading whole models.

The script compares two parameter-space vectors

    u = checkpoint_a - checkpoint_b
    v = checkpoint_c - checkpoint_d

and reports their norms, dot product, cosine, and per-layer/module
contributions.  All checkpoints must be Hugging Face safetensors exports with
the same parameter names and shapes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--d", type=Path, required=True)
    parser.add_argument("--name-u", default="a_minus_b")
    parser.add_argument("--name-v", default="c_minus_d")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_weight_map(path: Path) -> dict[str, str]:
    with (path / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def category(name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    layer = f"layer_{int(match.group(1)):02d}" if match else "non_layer"
    if "self_attn" in name:
        module = "attention"
    elif ".mlp." in name:
        module = "mlp"
    elif "norm" in name:
        module = "norm"
    elif "embed_tokens" in name or "lm_head" in name:
        module = "embedding"
    else:
        module = "other"
    return f"{layer}/{module}"


def empty_stats() -> dict[str, float]:
    return {"u2": 0.0, "v2": 0.0, "dot": 0.0, "numel": 0}


def finish(stats: dict[str, float]) -> dict[str, float]:
    u_norm = math.sqrt(stats["u2"])
    v_norm = math.sqrt(stats["v2"])
    denom = u_norm * v_norm
    return {
        **stats,
        "u_norm": u_norm,
        "v_norm": v_norm,
        "cosine": stats["dot"] / denom if denom else float("nan"),
    }


def main() -> None:
    args = parse_args()
    paths = [args.a, args.b, args.c, args.d]
    maps = [load_weight_map(path) for path in paths]
    names = set(maps[0])
    for path, weight_map in zip(paths[1:], maps[1:], strict=True):
        if set(weight_map) != names:
            raise ValueError(f"Parameter names differ for {path}")

    # Group by the 4-tuples of shard filenames so each file is opened once.
    shard_groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for name in sorted(names):
        shard_groups[tuple(weight_map[name] for weight_map in maps)].append(name)

    total = empty_stats()
    groups: dict[str, dict[str, float]] = defaultdict(empty_stats)
    tensors: dict[str, dict[str, float]] = {}

    for shard_names, param_names in shard_groups.items():
        handles = [
            safe_open(str(path / shard), framework="pt", device="cpu")
            for path, shard in zip(paths, shard_names, strict=True)
        ]
        with handles[0] as fa, handles[1] as fb, handles[2] as fc, handles[3] as fd:
            for name in param_names:
                u = fa.get_tensor(name).float().sub_(fb.get_tensor(name).float())
                v = fc.get_tensor(name).float().sub_(fd.get_tensor(name).float())
                if u.shape != v.shape:
                    raise ValueError(f"Shape mismatch for {name}: {u.shape} vs {v.shape}")
                values = {
                    "u2": torch.sum(u * u, dtype=torch.float64).item(),
                    "v2": torch.sum(v * v, dtype=torch.float64).item(),
                    "dot": torch.sum(u * v, dtype=torch.float64).item(),
                    "numel": u.numel(),
                }
                tensors[name] = finish(values)
                group = groups[category(name)]
                for key in ("u2", "v2", "dot", "numel"):
                    group[key] += values[key]
                    total[key] += values[key]

    result = {
        "vector_u": {"name": args.name_u, "positive": str(args.a), "negative": str(args.b)},
        "vector_v": {"name": args.name_v, "positive": str(args.c), "negative": str(args.d)},
        "total": finish(total),
        "groups": {name: finish(stats) for name, stats in sorted(groups.items())},
        "tensors": tensors,
    }
    encoded = json.dumps(result, indent=2, allow_nan=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(json.dumps({k: result[k] for k in ("vector_u", "vector_v", "total")}, indent=2))


if __name__ == "__main__":
    main()
