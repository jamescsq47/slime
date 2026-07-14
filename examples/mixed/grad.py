"""Compare model parameter changes between QA and Math GRPO training.

Computes deltas (trained - base) at per-parameter and per-element granularity,
then analyzes whether different training domains modify the same parameters.

Usage:
    python3 slime/examples/hybrid/grad.py
"""

import heapq
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import spearmanr

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE_MODEL = Path("/workspace/qwen3-4b-sft")

MODEL_PATHS = {
    "qa_iter199": Path("/workspace/Qwen3-4B_sync_qa/Qwen3-4B_sync_qa_iter199"),
    "qa_iter399": Path("/workspace/Qwen3-4B_sync_qa/Qwen3-4B_sync_qa_iter399"),
    "math_iter199": Path("/workspace/Qwen3-4B_sync_math/Qwen3-4B_sync_math_iter199"),
    "math_iter399": Path("/workspace/Qwen3-4B_sync_math/Qwen3-4B_sync_math_iter399"),
}

TOP_K_VALUES = [100, 500, 1000, 5000, 10000, 50000]

# Progressive overlap at parameter level (not element)
PARAM_TOP_N_VALUES = [10, 20, 50, 100, 200, 500]


# ─── Helpers ───────────────────────────────────────────────────────────────

def _param_type(name: str) -> str:
    """Categorize a parameter name into a high-level group."""
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding"
    if "self_attn" in name:
        return "attention"
    if "mlp" in name:
        return "mlp"
    if "layernorm" in name or "input_layernorm" in name or "post_attention_layernorm" in name:
        return "norm"
    return "other"


def _layer_id(name: str) -> int:
    """Extract layer index from param name; -1 for non-layer params."""
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def _short_name(name: str) -> str:
    return name.replace("model.", "", 1)


# ─── Model Loading ─────────────────────────────────────────────────────────

def load_base_model(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all base model parameters into CPU memory."""
    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    file_params = defaultdict(list)
    for pname, fname in weight_map.items():
        file_params[fname].append(pname)

    model = {}
    for fname, pnames in file_params.items():
        fpath = model_dir / fname
        with safe_open(str(fpath), framework="pt") as f:
            for pname in pnames:
                model[pname] = f.get_tensor(pname)
    return model


def iter_trained_params(model_dir: Path):
    """Generator: yield (param_name, tensor) from a safetensors checkpoint."""
    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    file_params = defaultdict(list)
    for pname, fname in weight_map.items():
        file_params[fname].append(pname)

    for fname, pnames in file_params.items():
        fpath = model_dir / fname
        with safe_open(str(fpath), framework="pt") as f:
            for pname in pnames:
                yield pname, f.get_tensor(pname)


# ─── Delta Computation ─────────────────────────────────────────────────────

def compute_per_param_delta_stats(
    base: dict[str, torch.Tensor],
    model_dir: Path,
    exclude_emb: bool = True,
) -> dict[str, dict]:
    """Per-parameter delta statistics between trained and base model.

    Returns {param_name: {l2_norm, mean_abs, max_abs, numel, shape, ...}}.
    Streams trained model params one at a time to limit peak memory.
    """
    stats = {}
    for pname, trained_t in iter_trained_params(model_dir):
        if exclude_emb and ("embed_tokens" in pname or "lm_head" in pname):
            continue
        if pname not in base:
            continue

        base_t = base[pname]
        delta = trained_t.float() - base_t.float()
        abs_delta = delta.abs()

        stats[pname] = {
            "l2_norm": delta.norm().item(),
            "mean_abs": abs_delta.mean().item(),
            "max_abs": abs_delta.max().item(),
            "numel": delta.numel(),
            "shape": list(delta.shape),
            "layer": _layer_id(pname),
            "param_type": _param_type(pname),
        }
    return stats


def find_global_topk_elements(
    base: dict[str, torch.Tensor],
    model_dir: Path,
    k: int = 5000,
    exclude_emb: bool = True,
) -> list[dict]:
    """Find top-k elements (by |delta|) across all parameters.

    Returns list sorted by abs_delta descending:
        [{param_name, flat_index, multi_index, shape, abs_delta, delta}, ...]
    """
    heap: list[tuple[float, str, int, float]] = []

    for pname, trained_t in iter_trained_params(model_dir):
        if exclude_emb and ("embed_tokens" in pname or "lm_head" in pname):
            continue
        if pname not in base:
            continue

        base_t = base[pname]
        delta = trained_t.float() - base_t.float()
        abs_delta = delta.abs()

        local_k = min(k, delta.numel())
        topk_vals, topk_idxs = torch.topk(abs_delta.flatten(), k=local_k)

        for val, flat_i in zip(topk_vals.tolist(), topk_idxs.tolist()):
            if len(heap) < k:
                heapq.heappush(heap, (val, pname, flat_i, delta.flatten()[flat_i].item()))
            elif val > heap[0][0]:
                heapq.heapreplace(heap, (val, pname, flat_i, delta.flatten()[flat_i].item()))

    entries = []
    while heap:
        val, pname, flat_i, d_val = heapq.heappop(heap)
        shape = base[pname].shape
        multi_idx = []
        remaining = flat_i
        for dim in reversed(shape):
            multi_idx.insert(0, remaining % dim)
            remaining //= dim
        entries.append({
            "param_name": pname,
            "flat_index": flat_i,
            "multi_index": multi_idx,
            "shape": list(shape),
            "abs_delta": val,
            "delta": d_val,
        })

    entries.reverse()  # largest first
    return entries


# ─── Comparison Functions ──────────────────────────────────────────────────

def compare_param_rankings(
    stats_a: dict[str, dict],
    stats_b: dict[str, dict],
    top_n_values: list[int] | None = None,
) -> dict:
    """Compare per-parameter delta L2 norm rankings between two domains.

    Returns Spearman correlation + overlap stats at multiple top-N thresholds.
    """
    common = set(stats_a) & set(stats_b)
    if not common:
        return {"error": "no common params"}

    a_vec = [stats_a[n]["l2_norm"] for n in common]
    b_vec = [stats_b[n]["l2_norm"] for n in common]

    if len(common) > 3:
        corr, pval = spearmanr(a_vec, b_vec)
    else:
        corr, pval = float("nan"), 1.0

    a_ranked = sorted(stats_a.items(), key=lambda x: x[1]["l2_norm"], reverse=True)
    b_ranked = sorted(stats_b.items(), key=lambda x: x[1]["l2_norm"], reverse=True)

    if top_n_values is None:
        top_n_values = [20]

    result = {
        "spearman_r": corr,
        "spearman_p": pval,
        "common_params": len(common),
        "a_ranked": [n for n, _ in a_ranked],
        "b_ranked": [n for n, _ in b_ranked],
    }

    for tn in sorted(set(top_n_values)):
        a_top = {n for n, _ in a_ranked[:tn]}
        b_top = {n for n, _ in b_ranked[:tn]}
        overlap = len(a_top & b_top)
        union = len(a_top | b_top)
        result[f"top_{tn}_overlap"] = overlap
        result[f"top_{tn}_union"] = union
        result[f"top_{tn}_jaccard"] = overlap / union if union > 0 else 0
        result[f"top_{tn}_overlap_pct"] = overlap / tn * 100

    return result


def compare_element_overlap(
    topk_a: list[dict],
    topk_b: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Element-level top-k position overlap between two domains."""
    if k_values is None:
        k_values = [100, 500, 1000, 5000, 10000, 50000]

    k_values = [min(k, len(topk_a), len(topk_b)) for k in k_values]

    results = {}
    for k in set(k_values):
        set_a = {(e["param_name"], e["flat_index"]) for e in topk_a[:k]}
        set_b = {(e["param_name"], e["flat_index"]) for e in topk_b[:k]}
        overlap = len(set_a & set_b)
        union = len(set_a | set_b)
        results[f"top_{k}"] = {
            "overlap": overlap,
            "union": union,
            "jaccard": overlap / union if union > 0 else 0,
            "overlap_pct": overlap / k * 100,
        }
    return results


def compute_layer_aggregated(stats: dict[str, dict]) -> dict:
    """Aggregate delta L2 norms by layer number."""
    by_layer = defaultdict(list)
    for pname, s in stats.items():
        lid = s["layer"]
        by_layer[lid].append(s["l2_norm"])

    return {
        f"layer_{lid}" if lid >= 0 else "non_layer": {
            "mean_l2": float(np.mean(vals)),
            "sum_l2": float(np.sum(vals)),
            "count": len(vals),
        }
        for lid, vals in sorted(by_layer.items())
    }


def compute_type_aggregated(stats: dict[str, dict]) -> dict:
    """Aggregate delta L2 norms by parameter type."""
    by_type = defaultdict(list)
    for pname, s in stats.items():
        by_type[s["param_type"]].append(s["l2_norm"])

    return {
        t: {
            "mean_l2": float(np.mean(vals)),
            "sum_l2": float(np.sum(vals)),
            "count": len(vals),
        }
        for t, vals in sorted(by_type.items())
    }


def compute_per_layer_cosine(
    base: dict[str, torch.Tensor],
    model_a_dir: Path,
    model_b_dir: Path,
    exclude_emb: bool = True,
) -> dict:
    """Per-layer cosine similarity of delta vectors between two trained models.

    Streams one file at a time from each model, computes delta = trained - base,
    then cosine similarity between the two flattened delta vectors per param.

    Returns {layer_id: {avg_cos, min_cos, max_cos, count}} plus "all" and "by_type".
    """
    with open(model_a_dir / "model.safetensors.index.json") as f:
        wm_a = json.load(f)["weight_map"]
    with open(model_b_dir / "model.safetensors.index.json") as f:
        wm_b = json.load(f)["weight_map"]

    file_a = defaultdict(list)
    for pn, fn in wm_a.items():
        file_a[fn].append(pn)
    file_b = defaultdict(list)
    for pn, fn in wm_b.items():
        file_b[fn].append(pn)

    per_layer: dict[int, list[float]] = defaultdict(list)
    type_cos: dict[str, list[float]] = defaultdict(list)

    for fname in sorted(set(file_a) & set(file_b)):
        fpath_a = model_a_dir / fname
        fpath_b = model_b_dir / fname

        # Load model B's shard first
        tensors_b = {}
        with safe_open(str(fpath_b), framework="pt") as f:
            for pn in file_b[fname]:
                tensors_b[pn] = f.get_tensor(pn)

        # Load model A's shard and compare
        with safe_open(str(fpath_a), framework="pt") as f:
            for pname in file_a[fname]:
                if pname not in tensors_b or pname not in base:
                    continue
                if exclude_emb and ("embed_tokens" in pname or "lm_head" in pname):
                    continue

                ta = f.get_tensor(pname)
                tb = tensors_b[pname]
                bt = base[pname]

                da = (ta.float() - bt.float()).flatten()
                db = (tb.float() - bt.float()).flatten()
                na, nb = da.norm(), db.norm()
                if na < 1e-8 or nb < 1e-8:
                    continue

                cs = torch.dot(da, db) / (na * nb)

                m = re.search(r"layers\.(\d+)", pname)
                lid = int(m.group(1)) if m else -1
                per_layer[lid].append(cs.item())

                typ = "attn" if "self_attn" in pname else "mlp" if "mlp" in pname else "norm"
                type_cos[typ].append(cs.item())

        del tensors_b

    result: dict = {}
    for lid in sorted(per_layer):
        vals = per_layer[lid]
        result[lid] = {
            "avg_cos": float(np.mean(vals)),
            "min_cos": float(np.min(vals)),
            "max_cos": float(np.max(vals)),
            "count": len(vals),
        }
    all_vals = [v for vals in per_layer.values() for v in vals]
    result["all"] = {
        "avg_cos": float(np.mean(all_vals)),
        "min_cos": float(np.min(all_vals)),
        "max_cos": float(np.max(all_vals)),
        "count": len(all_vals),
    }
    result["by_type"] = {
        t: {
            "avg_cos": float(np.mean(vals)),
            "min_cos": float(np.min(vals)),
            "max_cos": float(np.max(vals)),
            "count": len(vals),
        }
        for t, vals in sorted(type_cos.items())
    }
    return result


# ─── Formatting / Reporting ────────────────────────────────────────────────

def _fmt(v, width=12) -> str:
    if isinstance(v, float):
        if abs(v) < 0.01 or abs(v) >= 10000:
            return f"{v:>{width}.2e}"
        return f"{v:>{width}.4f}"
    return f"{v:>{width}}"


def print_header(title: str, width: int = 90):
    print(f"\n{'=' * width}")
    print(f"   {title}")
    print(f"{'=' * width}")


def print_per_param_table(stats: dict[str, dict], title: str, top_n: int = 25):
    print(f"\n{title}")
    print(f"{'param':<55} {'layer':<6} {'type':<10} {'l2_norm':>12} {'mean_abs':>12} {'max_abs':>12} {'numel':>10}")
    print("-" * 117)

    ranked = sorted(stats.items(), key=lambda x: x[1]["l2_norm"], reverse=True)
    for pname, s in ranked[:top_n]:
        layer_str = str(s["layer"]) if s["layer"] >= 0 else "N/A"
        print(
            f"{_short_name(pname):<55} {layer_str:<6} {s['param_type']:<10} "
            f"{_fmt(s['l2_norm'])} {_fmt(s['mean_abs'])} {_fmt(s['max_abs'])} "
            f"{s['numel']:>10,}"
        )


def print_overlapped_elements(
    topk_a: list[dict], topk_b: list[dict],
    topk_a_key: str, topk_b_key: str,
    max_k: int = 5000, show: int = 20,
):
    set_a = {(e["param_name"], e["flat_index"]) for e in topk_a[:max_k]}
    set_b = {(e["param_name"], e["flat_index"]) for e in topk_b[:max_k]}
    overlap = set_a & set_b
    if not overlap:
        return
    print(f"\n    Overlapping element positions (top-{max_k}, showing ≤{show}):")
    for count, (pname, fidx) in enumerate(sorted(overlap)):
        if count >= show:
            print(f"      ... and {len(overlap) - show} more")
            break
        shape = topk_a[0]["shape"] if any(e["param_name"] == pname for e in topk_a) else []
        multi = []
        remaining = fidx
        for dim in reversed(shape):
            multi.insert(0, remaining % dim)
            remaining //= dim
        idx_str = f"[{', '.join(map(str, multi))}]"
        print(f"      {_short_name(pname)}  {idx_str}")


def print_per_layer_cosine_table(
    result: dict,
    label_a: str,
    label_b: str,
    title: str = "",
):
    """Print per-layer cosine similarity table from compute_per_layer_cosine()."""
    if title:
        print(f"\n  {title}")
    print(f"  {label_a} vs {label_b} — per-layer cosine similarity")
    print(f"  {'Layer':>6}  {'avg_cos':>8}  {'min_cos':>8}  {'max_cos':>8}  {'n':>4}")
    print(f"  {'-'*40}")
    for lid in sorted(k for k in result if isinstance(k, int)):
        v = result[lid]
        print(f"  {lid:>6}  {v['avg_cos']:>8.4f}  {v['min_cos']:>8.4f}  {v['max_cos']:>8.4f}  {v['count']:>4}")
    print(f"  {'-'*40}")
    v = result["all"]
    print(f"  {'ALL':>6}  {v['avg_cos']:>8.4f}  {v['min_cos']:>8.4f}  {v['max_cos']:>8.4f}  {v['count']:>4}")

    # Per-type breakdown
    if "by_type" in result:
        bt = result["by_type"]
        print(f"\n    By type:")
        for t in ["mlp", "attn", "norm"]:
            if t in bt:
                w = bt[t]
                print(f"      {t:<6} avg={w['avg_cos']:>8.4f}  min={w['min_cos']:>8.4f}  "
                      f"max={w['max_cos']:>8.4f}  n={w['count']}")


def print_cross_domain_comparison(
    name_a: str, name_b: str,
    stats_a: dict, stats_b: dict,
    topk_a: list, topk_b: list,
    label: str = "",
):
    suffix = f" ({label})" if label else ""
    print_header(f"COMPARISON: {name_a} vs {name_b}{suffix}")

    # Per-param ranking
    pr = compare_param_rankings(stats_a, stats_b, top_n_values=PARAM_TOP_N_VALUES)
    print(f"\n  Per-parameter ranking:")
    print(f"    Spearman ρ = {pr['spearman_r']:.4f}  "
          f"(p={pr['spearman_p']:.2e}, {pr['common_params']} common params)")

    # Progressive overlap table
    print(f"\n  Progressive param-level overlap:")
    print(f"    {'top-N':>8} {'overlap':>8} {'union':>8} {'Jaccard':>8} {'overlap%':>9}")
    print(f"    {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
    for tn in PARAM_TOP_N_VALUES:
        ov = pr[f"top_{tn}_overlap"]
        un = pr[f"top_{tn}_union"]
        jac = pr[f"top_{tn}_jaccard"]
        pct = pr[f"top_{tn}_overlap_pct"]
        print(f"    {tn:>8} {ov:>8} {un:>8} {jac:>8.3f} {pct:>8.1f}%")

    # Top-20 unique/shared details
    a_top20 = set(pr["a_ranked"][:20])
    b_top20 = set(pr["b_ranked"][:20])
    only_a = a_top20 - b_top20
    only_b = b_top20 - a_top20
    common_p = a_top20 & b_top20

    if common_p:
        print(f"\n    Shared in both top-20 ({len(common_p)}):")
        for n in sorted(common_p):
            print(f"      {_short_name(n):<55}  {name_a}={stats_a[n]['l2_norm']:.2e}  "
                  f"{name_b}={stats_b[n]['l2_norm']:.2e}")

    if only_a:
        print(f"\n    Unique to {name_a} top-20:")
        for n in sorted(only_a, key=lambda x: stats_a[x]["l2_norm"], reverse=True):
            print(f"      {_short_name(n):<55}  {name_a}={stats_a[n]['l2_norm']:.2e}  "
                  f"{name_b}.get={stats_b.get(n, {}).get('l2_norm', 0):.2e}")

    if only_b:
        print(f"\n    Unique to {name_b} top-20:")
        for n in sorted(only_b, key=lambda x: stats_b[x]["l2_norm"], reverse=True):
            print(f"      {_short_name(n):<55}  {name_b}={stats_b[n]['l2_norm']:.2e}  "
                  f"{name_a}.get={stats_a.get(n, {}).get('l2_norm', 0):.2e}")

    # Element-level overlap
    print(f"\n  Element-level top-k overlap:")
    el = compare_element_overlap(topk_a, topk_b)
    for k_str, v in el.items():
        print(f"    {k_str}: overlap={v['overlap']}, Jaccard={v['jaccard']:.4g}, "
              f"overlap%={v['overlap_pct']:.4f}%")

    print_overlapped_elements(topk_a, topk_b, "a", "b", max_k=5000)


def print_layer_distribution(stats: dict[str, dict], title: str, top_n: int = 10):
    by_layer = compute_layer_aggregated(stats)
    print(f"\n{title}")
    print(f"{'layer':<12} {'mean_l2':>12} {'sum_l2':>12} {'count':>6}")
    print("-" * 42)
    for layer_name, vals in sorted(by_layer.items(), key=lambda x: x[1]["sum_l2"], reverse=True)[:top_n]:
        print(f"{layer_name:<12} {_fmt(vals['mean_l2'])} {_fmt(vals['sum_l2'])} {vals['count']:>6}")


def print_type_distribution(stats: dict[str, dict], title: str):
    by_type = compute_type_aggregated(stats)
    print(f"\n{title}")
    print(f"{'type':<12} {'mean_l2':>12} {'sum_l2':>12} {'count':>6}")
    print("-" * 42)
    for t in ["attention", "mlp", "norm", "embedding"]:
        if t in by_type:
            v = by_type[t]
            print(f"{t:<12} {_fmt(v['mean_l2'])} {_fmt(v['sum_l2'])} {v['count']:>6}")


def plot_analysis(results: dict, output_dir: Path = None):
    """Generate visualization plots (optional, needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [matplotlib not available, skipping plots]")
        return

    if output_dir is None:
        output_dir = Path("/workspace/slime/examples/hybrid/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-iteration detailed plots ──
    for iter_label in ["iter199", "iter399"]:
        qa_key = f"qa_{iter_label}_stats"
        math_key = f"math_{iter_label}_stats"
        if qa_key not in results or math_key not in results:
            continue

        qs, ms = results[qa_key], results[math_key]

        # Aggregate per-layer by type
        def _layer_type_agg(stats):
            by_layer_attn = defaultdict(float)
            by_layer_mlp = defaultdict(float)
            by_layer_norm = defaultdict(float)
            for s in stats.values():
                lid = s["layer"]
                if s["param_type"] == "attention":
                    by_layer_attn[lid] += s["l2_norm"]
                elif s["param_type"] == "mlp":
                    by_layer_mlp[lid] += s["l2_norm"]
                elif s["param_type"] == "norm":
                    by_layer_norm[lid] += s["l2_norm"]
            return by_layer_attn, by_layer_mlp, by_layer_norm

        qa_attn, qa_mlp, qa_norm = _layer_type_agg(qs)
        math_attn, math_mlp, math_norm = _layer_type_agg(ms)

        all_lids = sorted(
            set(k for k in list(qa_attn) + list(qa_mlp) + list(qa_norm) if k >= 0)
            | set(k for k in list(math_attn) + list(math_mlp) + list(math_norm) if k >= 0)
        )

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"Parameter Change Distribution: QA vs Math ({iter_label})", fontsize=16)

        # QA stacked
        ax = axes[0, 0]
        qa_a = [qa_attn.get(lid, 0) for lid in all_lids]
        qa_m = [qa_mlp.get(lid, 0) for lid in all_lids]
        qa_n = [qa_norm.get(lid, 0) for lid in all_lids]
        ax.bar(all_lids, qa_a, label="attention", color="steelblue", alpha=0.8)
        ax.bar(all_lids, qa_m, bottom=qa_a, label="mlp", color="coral", alpha=0.8)
        bot = [a + m for a, m in zip(qa_a, qa_m)]
        ax.bar(all_lids, qa_n, bottom=bot, label="norm", color="lightgreen", alpha=0.5)
        ax.set_title(f"QA per-layer stacked L2")
        ax.set_xlabel("Layer"); ax.set_ylabel("Sum L2 Norm")
        ax.legend(fontsize=8)
        total = sum(qa_a) + sum(qa_m) + sum(qa_n)
        ax.text(0.98, 0.95, f"total L2={total:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", alpha=0.8))

        # Math stacked
        ax = axes[0, 1]
        math_a = [math_attn.get(lid, 0) for lid in all_lids]
        math_m = [math_mlp.get(lid, 0) for lid in all_lids]
        math_n = [math_norm.get(lid, 0) for lid in all_lids]
        ax.bar(all_lids, math_a, label="attention", color="steelblue", alpha=0.8)
        ax.bar(all_lids, math_m, bottom=math_a, label="mlp", color="coral", alpha=0.8)
        bot = [a + m for a, m in zip(math_a, math_m)]
        ax.bar(all_lids, math_n, bottom=bot, label="norm", color="lightgreen", alpha=0.5)
        ax.set_title(f"Math per-layer stacked L2")
        ax.set_xlabel("Layer"); ax.set_ylabel("Sum L2 Norm")
        ax.legend(fontsize=8)
        total = sum(math_a) + sum(math_m) + sum(math_n)
        ax.text(0.98, 0.95, f"total L2={total:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", alpha=0.8))

        # QA - Math diff stacked
        ax = axes[1, 0]
        diff_a = [qa_attn.get(lid, 0) - math_attn.get(lid, 0) for lid in all_lids]
        diff_m = [qa_mlp.get(lid, 0) - math_mlp.get(lid, 0) for lid in all_lids]
        diff_n = [qa_norm.get(lid, 0) - math_norm.get(lid, 0) for lid in all_lids]
        ax.bar(all_lids, diff_a, label="attention", color="steelblue", alpha=0.7)
        ax.bar(all_lids, diff_m, bottom=diff_a, label="mlp", color="coral", alpha=0.7)
        bot = [a + m for a, m in zip(diff_a, diff_m)]
        ax.bar(all_lids, diff_n, bottom=bot, label="norm", color="lightgreen", alpha=0.5)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title("QA - Math per-layer ΔL2")
        ax.set_xlabel("Layer"); ax.set_ylabel("Δ L2 Norm")
        ax.legend(fontsize=8)

        # Scatter
        ax = axes[1, 1]
        common = set(qs) & set(ms)
        x = [qs[n]["l2_norm"] for n in common]
        y = [ms[n]["l2_norm"] for n in common]
        ax.scatter(x, y, alpha=0.4, s=8)
        ax.set_xlabel("QA delta L2 norm"); ax.set_ylabel("Math delta L2 norm")
        ax.set_xscale("log"); ax.set_yscale("log")
        lo, hi = min(x + y), max(x + y)
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5)
        corr, _ = spearmanr(x, y)
        ax.text(0.05, 0.95, f"Spearman ρ={corr:.4f}", transform=ax.transAxes,
                fontsize=12, va="top", bbox=dict(boxstyle="round", alpha=0.8))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fpath = output_dir / f"param_analysis_{iter_label}.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  [Saved plot: {fpath}]")

    # ── Summary: progressive overlap curve ──
    fig, ax = plt.subplots(figsize=(8, 6))
    comparisons_plotted = 0
    for label, key_a, key_b in [
        ("QA vs Math @iter199", "qa_iter199_stats", "math_iter199_stats"),
        ("QA vs Math @iter399", "qa_iter399_stats", "math_iter399_stats"),
        ("QA iter199→399", "qa_iter199_stats", "qa_iter399_stats"),
        ("Math iter199→399", "math_iter199_stats", "math_iter399_stats"),
    ]:
        if key_a not in results or key_b not in results:
            continue
        pr = compare_param_rankings(results[key_a], results[key_b], top_n_values=PARAM_TOP_N_VALUES)
        ks = sorted(PARAM_TOP_N_VALUES)
        overlaps = [pr[f"top_{k}_overlap_pct"] for k in ks]
        ax.plot(ks, overlaps, marker="o", ms=5, label=label)
        comparisons_plotted += 1

    if comparisons_plotted > 0:
        ax.set_title("Progressive Parameter Overlap (by L2 norm ranking)")
        ax.set_xlabel("Top-N parameters"); ax.set_ylabel("Overlap %")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks(PARAM_TOP_N_VALUES)
        ax.set_xticklabels([str(k) for k in PARAM_TOP_N_VALUES])
        ax.set_ylim(-5, 105)

        fpath = output_dir / "progressive_overlap.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Saved plot: {fpath}]")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("   GRPO Parameter Delta Analysis: QA vs Math")
    print("=" * 90)

    exclude_emb = True

    # Check which models are available
    available = {name: path for name, path in MODEL_PATHS.items() if path.exists()}
    print(f"\n  Base model:  {BASE_MODEL}")
    print(f"  Excluding embedding/lm_head: {exclude_emb}")
    print(f"  Top-K values: {TOP_K_VALUES}")
    print(f"  Available models ({len(available)}):")
    for name, path in available.items():
        print(f"    {name:<15} {path}")

    if len(available) < 2:
        print(f"\n  Need at least 2 models. Found {len(available)}.")
        return

    # Step 1: Load base model
    print(f"\n{'─' * 90}")
    print("  [1/4] Loading base model...")
    base = load_base_model(BASE_MODEL)
    n_params = sum(t.numel() for t in base.values())
    n_bytes = sum(t.numel() * t.element_size() for t in base.values())
    print(f"  {len(base)} params, {n_params:,} elements ({n_bytes/1e9:.1f} GB)")

    if exclude_emb:
        for excl in ["lm_head.weight", "model.embed_tokens.weight"]:
            if excl in base:
                print(f"  Will exclude {excl} ({base[excl].numel():,} elements)")

    # Step 2: Compute deltas
    print(f"\n{'─' * 90}")
    print("  [2/4] Computing deltas...")

    results = {}
    for name, path in available.items():
        print(f"\n    {name}: computing per-param stats...", end=" ")
        stats = compute_per_param_delta_stats(base, path, exclude_emb=exclude_emb)
        results[f"{name}_stats"] = stats
        total_l2 = sum(s["l2_norm"] for s in stats.values())
        print(f"done ({len(stats)} params, total L2={total_l2:.4f})")

        print(f"    {name}: finding top-{max(TOP_K_VALUES)} elements...", end=" ")
        topk = find_global_topk_elements(base, path, k=max(TOP_K_VALUES), exclude_emb=exclude_emb)
        results[f"{name}_topk"] = topk
        print(f"done (top element |Δ|={topk[0]['abs_delta']:.2e})")

    # Step 3: Cross comparisons
    print(f"\n{'─' * 90}")
    print("  [3/4] Cross-domain comparisons...")

    for iter_label in ["iter199", "iter399"]:
        qa_s = results.get(f"qa_{iter_label}_stats")
        ma_s = results.get(f"math_{iter_label}_stats")
        qa_t = results.get(f"qa_{iter_label}_topk")
        ma_t = results.get(f"math_{iter_label}_topk")
        if qa_s and ma_s and qa_t and ma_t:
            print_cross_domain_comparison("QA", "Math", qa_s, ma_s, qa_t, ma_t, label=iter_label)

    # Cross-iteration within each domain
    for domain, dlabel in [("qa", "QA"), ("math", "Math")]:
        s199 = results.get(f"{domain}_iter199_stats")
        s399 = results.get(f"{domain}_iter399_stats")
        t199 = results.get(f"{domain}_iter199_topk")
        t399 = results.get(f"{domain}_iter399_topk")
        if s199 and s399 and t199 and t399:
            print_cross_domain_comparison(
                f"{dlabel}@iter199", f"{dlabel}@iter399",
                s199, s399, t199, t399,
                label=f"{dlabel} iteration stability",
            )

    # Step 4: Summaries
    print(f"\n{'─' * 90}")
    print("  [4/4] Per-model summaries...")

    for name_key in results:
        if not name_key.endswith("_stats"):
            continue
        label = name_key.replace("_stats", "")
        stats = results[name_key]

        total_l2 = sum(s["l2_norm"] for s in stats.values())
        print_header(f"MODEL: {label}")
        print(f"  Total delta L2 norm: {total_l2:.4f}")
        print(f"  Params with non-zero delta: {sum(1 for s in stats.values() if s['l2_norm'] > 0)}/{len(stats)}")
        by_type = compute_type_aggregated(stats)
        for t, vals in sorted(by_type.items(), key=lambda x: x[1]["sum_l2"], reverse=True):
            pct = vals["sum_l2"] / total_l2 * 100 if total_l2 > 0 else 0
            print(f"    {t:<12}: sum_l2={vals['sum_l2']:.2e} ({pct:.1f}%), count={vals['count']}")

        print_per_param_table(stats, f"  Top-20 params by delta L2 norm:")
        print_layer_distribution(stats, f"  Layer distribution (top-10 by sum L2):")
        print_type_distribution(stats, f"  Type distribution:")

    # Step 5: Per-layer cosine similarity analysis
    print(f"\n{'─' * 90}")
    print("  [5/5] Per-layer cosine similarity...")

    cos_pairs = [
        ("math_iter199", "math_iter399", "Math@iter199", "Math@iter399", "Within-domain: Math 199 -> 399"),
        ("qa_iter199", "qa_iter399", "QA@iter199", "QA@iter399", "Within-domain: QA 199 -> 399"),
        ("math_iter199", "qa_iter199", "Math@iter199", "QA@iter199", "Cross-domain: Math vs QA @ iter199"),
        ("math_iter399", "qa_iter399", "Math@iter399", "QA@iter399", "Cross-domain: Math vs QA @ iter399"),
    ]
    for key_a, key_b, la, lb, desc in cos_pairs:
        if key_a in MODEL_PATHS and key_b in MODEL_PATHS:
            print()
            cos_result = compute_per_layer_cosine(base, MODEL_PATHS[key_a], MODEL_PATHS[key_b],
                                                  exclude_emb=exclude_emb)
            print_per_layer_cosine_table(cos_result, la, lb, title=desc)

    # Plots
    print(f"\n{'─' * 90}")
    print("  Generating plots...")
    plot_analysis(results)

    # Final summary
    print(f"\n{'=' * 90}")
    print("   SUMMARY")
    print(f"{'=' * 90}")

    for iter_label in ["iter199", "iter399"]:
        qa_s = results.get(f"qa_{iter_label}_stats")
        ma_s = results.get(f"math_{iter_label}_stats")
        qa_t = results.get(f"qa_{iter_label}_topk")
        ma_t = results.get(f"math_{iter_label}_topk")
        if not (qa_s and ma_s):
            continue

        pr = compare_param_rankings(qa_s, ma_s)
        el = compare_element_overlap(qa_t, ma_t, k_values=[100, 1000, 10000, 50000])

        print(f"\n  QA vs Math @ {iter_label}:")
        print(f"    Per-param Spearman ρ = {pr['spearman_r']:.4f}")
        print(f"    Top-20 param overlap: {pr['top_20_overlap']}/20 ({pr['top_20_overlap_pct']:.0f}%)")
        for k_str, v in el.items():
            print(f"    Element overlap {k_str}: {v['overlap']} positions "
                  f"(Jaccard={v['jaccard']:.4g})")

    print(f"\n  Iteration stability (consistency check):")
    for domain, dlabel in [("qa", "QA"), ("math", "Math")]:
        s199 = results.get(f"{domain}_iter199_stats")
        s399 = results.get(f"{domain}_iter399_stats")
        t199 = results.get(f"{domain}_iter199_topk")
        t399 = results.get(f"{domain}_iter399_topk")
        if s199 and s399:
            pr = compare_param_rankings(s199, s399)
            el = compare_element_overlap(t199, t399, k_values=[1000, 10000])
            print(f"    {dlabel} iter199→iter399: Spearman ρ = {pr['spearman_r']:.4f}, "
                  f"Top-20 Jaccard={pr['top_20_jaccard']:.3f}")
            for k_str, v in el.items():
                print(f"      Element overlap {k_str}: {v['overlap']} (Jaccard={v['jaccard']:.4g})")

    # Bottom line
    print(f"\n{'─' * 90}")
    for iter_label in ["iter199", "iter399"]:
        qa_s = results.get(f"qa_{iter_label}_stats")
        ma_s = results.get(f"math_{iter_label}_stats")
        if qa_s and ma_s:
            pr = compare_param_rankings(qa_s, ma_s, top_n_values=PARAM_TOP_N_VALUES)
            r = pr["spearman_r"]
            print(f"\n  {iter_label}: QA vs Math (Spearman ρ={r:.3f})")
            print(f"    {'top-N':>6} {'overlap':>7} {'Jaccard':>7} {'overlap%':>8}")
            print(f"    {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
            for tn in PARAM_TOP_N_VALUES:
                ov = pr[f"top_{tn}_overlap"]
                jac = pr[f"top_{tn}_jaccard"]
                pct = pr[f"top_{tn}_overlap_pct"]
                print(f"    {tn:>6} {ov:>7} {jac:>7.3f} {pct:>7.1f}%")
            if r > 0.8:
                print(f"  → Spearman ρ={r:.3f}，两个领域修改的参数高度一致，但在具体排名上有差异")
            elif r > 0.5:
                print(f"  → Spearman ρ={r:.3f}，部分重叠但各有侧重")
            else:
                print(f"  → Spearman ρ={r:.3f}，两个领域关注不同的参数")
    print()


def plot_top20_overlap(output_dir: Path = None, target_size: int = 200):
    """Visualize overlapping params as 2D heatmaps with pooling.

    For each param, computes delta = trained - base (NOT absolute value),
    pools to ~target_size×target_size, and shows Math vs QA side-by-side
    with a diverging colormap (red=positive, blue=negative).

    Also computes element-wise cosine similarity and sign agreement.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("matplotlib not available")
        return

    if output_dir is None:
        output_dir = Path("/workspace/slime/examples/hybrid/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    base = load_base_model(BASE_MODEL)
    qa = {n: t for n, t in iter_trained_params(MODEL_PATHS["qa_iter399"])}
    math = {n: t for n, t in iter_trained_params(MODEL_PATHS["math_iter399"])}

    overlap_params = [
        "model.layers.14.mlp.down_proj.weight",
        "model.layers.15.mlp.down_proj.weight",
        "model.layers.16.mlp.down_proj.weight",
        "model.layers.16.mlp.up_proj.weight",
        "model.layers.17.mlp.down_proj.weight",
        "model.layers.17.mlp.up_proj.weight",
        "model.layers.18.mlp.down_proj.weight",
        "model.layers.18.mlp.gate_proj.weight",
        "model.layers.18.mlp.up_proj.weight",
        "model.layers.19.mlp.down_proj.weight",
        "model.layers.19.mlp.gate_proj.weight",
        "model.layers.19.mlp.up_proj.weight",
        "model.layers.20.mlp.down_proj.weight",
        "model.layers.20.mlp.gate_proj.weight",
        "model.layers.21.mlp.down_proj.weight",
        "model.layers.21.mlp.gate_proj.weight",
        "model.layers.21.mlp.up_proj.weight",
    ]

    def _pool2d(t: torch.Tensor, target: int) -> np.ndarray:
        """Average-pool 2D tensor to roughly target×target."""
        h, w = t.shape
        if h <= target and w <= target:
            return t.cpu().numpy()
        bh = max(1, h // target)
        bw = max(1, w // target)
        h_trim = h - (h % bh)
        w_trim = w - (w % bw)
        pooled = t[:h_trim, :w_trim].reshape(h_trim // bh, bh, w_trim // bw, bw).mean(dim=(1, 3))
        return pooled.cpu().numpy()

    for pname in overlap_params:
        if pname not in base or pname not in qa or pname not in math:
            print(f"  Skipping {pname}")
            continue

        delta_m = (math[pname].float() - base[pname].float()).detach()
        delta_q = (qa[pname].float() - base[pname].float()).detach()
        shape = delta_m.shape
        short = _short_name(pname)

        # Pooled 2D heatmaps
        pooled_m = _pool2d(delta_m, target_size)
        pooled_q = _pool2d(delta_q, target_size)

        # Symmetric color scale
        vmax = max(abs(pooled_m).max(), abs(pooled_q).max())
        vmin = -vmax

        # Element-wise metrics (computed on FULL tensors, not pooled)
        flat_m = delta_m.flatten()
        flat_q = delta_q.flatten()
        cos_sim = torch.dot(flat_m, flat_q) / (flat_m.norm() * flat_q.norm() + 1e-12)
        cos_sim = cos_sim.item()

        # Sign agreement (among elements with meaningful change)
        eps = 1e-8
        mask = (flat_m.abs() > eps) | (flat_q.abs() > eps)
        if mask.sum() > 0:
            same_sign = ((flat_m[mask] * flat_q[mask]) > 0).float().mean().item()
        else:
            same_sign = 1.0

        norm_m = flat_m.norm().item()
        norm_q = flat_q.norm().item()

        # Coefficient of variation of the delta ratio
        nonzero_mask = (flat_m.abs() > 1e-8)
        if nonzero_mask.sum() > 100:
            ratio = flat_q[nonzero_mask] / flat_m[nonzero_mask]
            ratio_mean = ratio.mean().item()
            ratio_std = ratio.std().item()
            ratio_cv = ratio_std / abs(ratio_mean) if abs(ratio_mean) > 0 else 0
        else:
            ratio_mean, ratio_cv = 0, 0

        # ─── Plot: 2 pooled heatmaps + metrics panel ───
        fig, axes = plt.subplots(1, 3, figsize=(16, 6.5),
                                 gridspec_kw={"width_ratios": [1, 1, 0.8]})
        fig.suptitle(f"{short}  ({shape[0]}×{shape[1]}", fontsize=12, y=0.97)

        # Panel 1: Math
        ax = axes[0]
        im = ax.imshow(pooled_m, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax),
                        aspect="auto", interpolation="none")
        ax.set_title(f"Math Δ  (norm={norm_m:.2e})", fontsize=10)
        ax.set_xlabel("col (pooled)"); ax.set_ylabel("row (pooled)")

        # Panel 2: QA
        ax = axes[1]
        im = ax.imshow(pooled_q, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax),
                        aspect="auto", interpolation="none")
        ax.set_title(f"QA Δ  (norm={norm_q:.2e})", fontsize=10)
        ax.set_xlabel("col (pooled)"); ax.set_ylabel("row (pooled)")

        # Shared colorbar
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.04)
        cbar.set_label("Δ (positive=red, negative=blue)")

        # Panel 3: Metrics
        ax = axes[2]
        ax.axis("off")
        metrics_text = (
            f"ELEMENT-WISE METRICS\n"
            f"{'─'*24}\n\n"
            f"Cosine similarity:\n  {cos_sim:.4f}\n"
            f"  (1 = same direction,\n"
            f"   0 = random,\n"
            f"  -1 = opposite)\n\n"
            f"Sign agreement:\n  {same_sign*100:.1f}%\n"
            f"  (same sign among\n"
            f"   non-zero deltas)\n\n"
            f"Magnitude:\n"
            f"  L2(math) = {norm_m:.2e}\n"
            f"  L2(qa)   = {norm_q:.2e}\n"
            f"  ratio    = {norm_q/norm_m:.3f}\n\n"
            f"Pool size:\n  pooled to\n"
            f"  {pooled_m.shape[0]}×{pooled_m.shape[1]}"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=9, fontfamily="monospace", va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

        plt.tight_layout()
        fname = short.replace(".", "_")
        fpath = output_dir / f"heatmap_{fname}.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")

    # ── Additional: pooled heatmap summary for ALL params + metrics table ──
    print(f"\n{'='*80}")
    print(f"  Cosine similarity & sign agreement for all 24 params (layers 14-21)")
    print(f"{'='*80}")
    print(f"{'param':<50} {'cos_sim':>8} {'sign_agree':>10} {'L2_ratio':>8}")
    print(f"{'-'*80}")
    for l in range(14, 22):
        for t in ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"]:
            pname = f"model.layers.{l}.{t}.weight"
            if pname not in base or pname not in qa or pname not in math:
                continue
            dm = (math[pname].float() - base[pname].float()).flatten()
            dq = (qa[pname].float() - base[pname].float()).flatten()
            cs = torch.dot(dm, dq) / (dm.norm() * dq.norm() + 1e-12)
            eps = 1e-8
            mask = (dm.abs() > eps) | (dq.abs() > eps)
            sa = ((dm[mask] * dq[mask]) > 0).float().mean().item() if mask.sum() > 0 else 1.0
            ratio = dq.norm().item() / (dm.norm().item() + 1e-12)
            short = pname.replace("model.", "")
            print(f"{short:<50} {cs.item():>8.4f} {sa*100:>9.1f}% {ratio:>7.3f}")

    print(f"\nDone — {len(overlap_params)} heatmaps saved to {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--overlap-viz":
        plot_top20_overlap()
    else:
        main()
