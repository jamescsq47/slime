#!/usr/bin/env python3
"""
Analyze parameter update direction similarity across different RL-trained models.

Compares the weight deltas (RL_model - base_model) between:
  1. math_iter399 vs qa_iter399   (different RL tasks, same step)
  2. math_iter199 vs math_iter399 (same RL task, different steps)

Metrics:
  - Per-layer cosine similarity of flattened delta vectors
  - Per-layer CKA (Centered Kernel Alignment) on delta matrices
  - Per-layer angular distance (in degrees)
  - Global summary statistics
"""

import json
import os
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from safetensors import safe_open


# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL = "/workspace/Qwen3-4B"

RL_MODELS = OrderedDict(
    math_iter399="/workspace/Qwen3-4B_sync_math/Qwen3-4B_sync_math_iter399",
    qa_iter399="/workspace/Qwen3-4B_sync_qa/Qwen3-4B_sync_qa_iter399",
    math_iter199="/workspace/Qwen3-4B_sync_math/Qwen3-4B_sync_math_iter199",
)

# Which pairs to compare
COMPARISON_PAIRS = [
    ("math_iter399", "qa_iter399", "math399 vs qa399  (diff task, same step)"),
    ("math_iter199", "math_iter399", "math199 vs math399 (same task, diff step)"),
]

OUTPUT_DIR = "/workspace/slime/param_analysis_results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """Load all safetensors shards into a single dict."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    weights = {}
    for shard in shards:
        shard_path = os.path.join(model_dir, shard)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    return weights


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two flattened vectors (float64, law of cosines for stability)."""
    a_flat = a.double().flatten()
    b_flat = b.double().flatten()
    a_norm = a_flat.norm()
    b_norm = b_flat.norm()
    if a_norm < 1e-12 or b_norm < 1e-12:
        return float("nan")
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b  =>  cos = (||a||^2 + ||b||^2 - ||a-b||^2) / (2||a||||b||)
    d_norm = (a_flat - b_flat).norm()
    cos_val = (a_norm ** 2 + b_norm ** 2 - d_norm ** 2) / (2 * a_norm * b_norm)
    return cos_val.clamp(-1, 1).item()


def angular_distance_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    """Angular distance in degrees between two flattened vectors."""
    cos_sim = cosine_similarity_flat(a, b)
    return float(np.degrees(np.arccos(cos_sim)))


def cka_linear(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Linear CKA (Centered Kernel Alignment) between two 2-D matrices.
    Reshapes tensors to 2-D (rows × cols) if needed.
    """
    def _to_2d(t):
        t = t.float()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        return t

    a = _to_2d(a)
    b = _to_2d(b)

    # Ensure same number of rows (features); if not, use min
    min_rows = min(a.shape[0], b.shape[0])
    a = a[:min_rows]
    b = b[:min_rows]

    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)

    a_ta = a @ a.T
    b_tb = b @ b.T

    hsic_ab = (a_ta * b_tb).sum()
    hsic_aa = (a_ta * a_ta).sum()
    hsic_bb = (b_tb * b_tb).sum()

    denom = torch.sqrt(hsic_aa * hsic_bb)
    if denom < 1e-12:
        return 0.0
    return (hsic_ab / denom).item()


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """Ratio of L2 norms: min(||a||, ||b||) / max(||a||, ||b||)."""
    na = a.float().norm().item()
    nb = b.float().norm().item()
    if max(na, nb) < 1e-12:
        return 1.0
    return min(na, nb) / max(na, nb)


def sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of parameters where the sign of the delta agrees."""
    sa = torch.sign(a.float())
    sb = torch.sign(b.float())
    return (sa == sb).float().mean().item()


# ── Layer grouping ────────────────────────────────────────────────────────────

def parse_layer_info(key: str) -> str:
    """
    Return a human-readable grouping key for a weight name.
    Groups by layer index and sub-module type.
    """
    if "model.layers." in key:
        parts = key.split(".")
        layer_idx = parts[2]
        # Extract sub-module: self_attn.q_proj, mlp.gate_proj, etc.
        sub_parts = parts[3:]
        # Remove .weight / .bias suffix
        if sub_parts[-1] in ("weight", "bias"):
            sub_parts = sub_parts[:-1]
        sub_module = ".".join(sub_parts)
        return f"layer.{layer_idx}.{sub_module}"
    elif "embed_tokens" in key:
        return "embed_tokens"
    elif "norm" in key and "layers" not in key:
        return "final_norm"
    elif "lm_head" in key:
        return "lm_head"
    else:
        return key


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load base model
    print(f"Loading base model: {BASE_MODEL}")
    base_weights = load_all_weights(BASE_MODEL)
    print(f"  → {len(base_weights)} tensors loaded")

    # 2. Load RL models and compute deltas
    deltas = {}
    for name, path in RL_MODELS.items():
        print(f"Loading RL model: {name} ({path})")
        rl_weights = load_all_weights(path)
        print(f"  → {len(rl_weights)} tensors loaded")

        # Compute delta = RL - base (only for keys present in both)
        common_keys = sorted(set(base_weights.keys()) & set(rl_weights.keys()))
        delta = {}
        for k in common_keys:
            delta[k] = rl_weights[k].float() - base_weights[k].float()
        deltas[name] = delta
        print(f"  → {len(delta)} delta tensors computed")

        # Free memory
        del rl_weights

    del base_weights  # free base model memory

    # 3. Compare pairs
    all_results = {}
    for name_a, name_b, desc in COMPARISON_PAIRS:
        print(f"\n{'='*80}")
        print(f"Comparing: {desc}")
        print(f"{'='*80}")

        delta_a = deltas[name_a]
        delta_b = deltas[name_b]

        common_keys = sorted(set(delta_a.keys()) & set(delta_b.keys()))

        results = []
        for key in common_keys:
            da = delta_a[key]
            db = delta_b[key]

            cos_sim = cosine_similarity_flat(da, db)
            ang_dist = angular_distance_deg(da, db)
            cka_val = cka_linear(da, db)
            nr = norm_ratio(da, db)
            sign_agr = sign_agreement(da, db)

            group = parse_layer_info(key)
            results.append({
                "key": key,
                "group": group,
                "cosine_similarity": cos_sim,
                "angular_distance_deg": ang_dist,
                "cka": cka_val,
                "norm_ratio": nr,
                "sign_agreement": sign_agr,
                "norm_a": da.float().norm().item(),
                "norm_b": db.float().norm().item(),
            })

        all_results[(name_a, name_b)] = results

        # ── Print summary table ───────────────────────────────────────────
        print(f"\n{'Layer Group':<45} {'CosSim':>8} {'Angle°':>8} {'CKA':>8} {'NormR':>8} {'SignAgr':>8}")
        print("-" * 95)

        # Group by layer for a more compact view
        grouped = {}
        for r in results:
            g = r["group"]
            if g not in grouped:
                grouped[g] = []
            grouped[g].append(r)

        layer_order = sorted(
            grouped.keys(),
            key=lambda x: (
                int(x.split(".")[1]) if x.startswith("layer.") else -1,
                x,
            ),
        )

        for g in layer_order:
            items = grouped[g]
            avg_cos = np.mean([r["cosine_similarity"] for r in items])
            avg_ang = np.mean([r["angular_distance_deg"] for r in items])
            avg_cka = np.mean([r["cka"] for r in items])
            avg_nr = np.mean([r["norm_ratio"] for r in items])
            avg_sign = np.mean([r["sign_agreement"] for r in items])

            print(f"{g:<45} {avg_cos:>8.4f} {avg_ang:>8.2f} {avg_cka:>8.4f} {avg_nr:>8.4f} {avg_sign:>8.4f}")

        # ── Global stats ──────────────────────────────────────────────────
        all_cos = [r["cosine_similarity"] for r in results]
        all_ang = [r["angular_distance_deg"] for r in results]
        all_cka = [r["cka"] for r in results]
        all_sign = [r["sign_agreement"] for r in results]

        print(f"\n{'─── Global Summary ───':─^95}")
        print(f"  Cosine Similarity  — mean: {np.mean(all_cos):.4f}  std: {np.std(all_cos):.4f}  "
              f"min: {np.min(all_cos):.4f}  max: {np.max(all_cos):.4f}")
        print(f"  Angular Distance   — mean: {np.mean(all_ang):.2f}°  std: {np.std(all_ang):.2f}°  "
              f"min: {np.min(all_ang):.2f}°  max: {np.max(all_ang):.2f}°")
        print(f"  CKA                — mean: {np.mean(all_cka):.4f}  std: {np.std(all_cka):.4f}  "
              f"min: {np.min(all_cka):.4f}  max: {np.max(all_cka):.4f}")
        print(f"  Sign Agreement     — mean: {np.mean(all_sign):.4f}  std: {np.std(all_sign):.4f}  "
              f"min: {np.min(all_sign):.4f}  max: {np.max(all_sign):.4f}")

        # ── Per-layer-type aggregation ────────────────────────────────────
        print(f"\n{'─── By Layer Type ───':─^95}")
        type_groups = {"embed": [], "attn": [], "mlp": [], "norm": [], "other": []}
        for r in results:
            g = r["group"].lower()
            if "embed" in g:
                type_groups["embed"].append(r)
            elif "attn" in g or "self_attn" in g or "q_proj" in g or "k_proj" in g or "v_proj" in g or "o_proj" in g:
                type_groups["attn"].append(r)
            elif "mlp" in g or "gate_proj" in g or "up_proj" in g or "down_proj" in g:
                type_groups["mlp"].append(r)
            elif "norm" in g:
                type_groups["norm"].append(r)
            else:
                type_groups["other"].append(r)

        for tname, items in type_groups.items():
            if not items:
                continue
            avg_cos = np.mean([r["cosine_similarity"] for r in items])
            avg_ang = np.mean([r["angular_distance_deg"] for r in items])
            avg_cka = np.mean([r["cka"] for r in items])
            avg_sign = np.mean([r["sign_agreement"] for r in items])
            print(f"  {tname:<12}  cos={avg_cos:.4f}  angle={avg_ang:.2f}°  cka={avg_cka:.4f}  sign={avg_sign:.4f}  (n={len(items)})")

    # 4. Save detailed results to JSON
    output_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    serializable = {}
    for (na, nb), results in all_results.items():
        key = f"{na}_vs_{nb}"
        serializable[key] = results
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")



if __name__ == "__main__":
    main()
