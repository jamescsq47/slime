#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from compare_four_gpu_baselines import queue_summary


CASES = (
    ("pd-no-reverse-2p2d", "2P:2D, no reverse KV"),
    ("pd-native-mooncake-2p2d", "2P:2D, native Mooncake"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, label in CASES:
        run_dir = args.run_root / name
        summary = json.loads((run_dir / "offload_analysis_summary.json").read_text())
        boundary = json.loads((run_dir / "closed_loop_boundaries.json").read_text())
        reuse = summary.get("completion_weighted_reuse", {})
        mooncake = summary.get("mooncake", {})
        row = {
            "case": name,
            "label": label,
            **summary,
            "prompt_prefix_hit_fraction": reuse.get("prefix_hit_fraction", 0.0),
            "reverse_reuse_fraction": reuse.get(
                "page_aligned_reverse_reuse_fraction", 0.0
            ),
            "prefill_gpu_busy_fraction_per_gpu": summary.get(
                "prefill_gpu_busy_fraction", 0.0
            )
            / 2,
            "engine_cache_fraction": summary.get("prefill_cache_hit_tps", 0.0)
            / max(
                summary.get("prefill_compute_tps", 0.0)
                + summary.get("prefill_cache_hit_tps", 0.0),
                1.0,
            ),
            "mooncake_resident_mean_fraction": mooncake.get(
                "measurement_resident_mean_fraction", 0.0
            ),
            "mooncake_resident_max_fraction": mooncake.get(
                "measurement_resident_max_fraction", 0.0
            ),
            "mooncake_eviction_successes": mooncake.get(
                "eviction_successes", 0
            ),
            "mooncake_evicted_gb": mooncake.get("evicted_gb", 0.0),
        }
        row.update(
            queue_summary(
                run_dir,
                boundary["measurement_start_wall"],
                boundary["measurement_end_wall"],
            )
        )
        row.pop("completion_weighted_reuse", None)
        row.pop("mooncake", None)
        rows.append(row)

    (args.run_root / "architecture_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    columns = [
        "case",
        "completed_agents",
        "completion_rps",
        "prefill_compute_tps",
        "prefill_cache_hit_tps",
        "decode_tps",
        "decode_tps_per_gpu",
        "prefill_gpu_busy_fraction",
        "prefill_gpu_busy_fraction_per_gpu",
        "decode_gpu_busy_fraction_per_gpu",
        "prefill_gpu_active_tps",
        "decode_gpu_active_tps",
        "decode_running_per_engine",
        "decode_prealloc_per_engine",
        "decode_transfer_per_engine",
        "decode_kv_usage_per_engine",
        "prefill_kv_usage_per_engine",
        "prefill_queue_per_engine",
        "prefill_inflight_per_engine",
        "prompt_prefix_hit_fraction",
        "engine_cache_fraction",
        "reverse_reuse_fraction",
        "mooncake_resident_mean_fraction",
        "mooncake_resident_max_fraction",
        "mooncake_eviction_successes",
        "mooncake_evicted_gb",
    ]
    with (args.run_root / "architecture_comparison.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Four-model-GPU 2P:2D mixed-agent baseline comparison",
        "",
        "Both cases use the same fixed Mixed 1:1 schedule, seed, c256, 300-second warmup, and 1200-second measurement window.",
        "",
        "| Case | Agent/s | P compute token/s | P cache token/s | P busy/GPU | P KV | D token/s | D token/s/GPU | D active token/s/GPU | D busy/GPU | D running/GPU | D KV | D prealloc/GPU | D transfer/GPU | Engine cache fraction | Mooncake mean/max | Evictions / GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['label']} | {row['completion_rps']:.3f} | "
            f"{row['prefill_compute_tps']:.0f} | "
            f"{row['prefill_cache_hit_tps']:.0f} | "
            f"{100 * row['prefill_gpu_busy_fraction_per_gpu']:.1f}% | "
            f"{100 * row['prefill_kv_usage_per_engine']:.1f}% | "
            f"{row['decode_tps']:.0f} | "
            f"{row['decode_tps_per_gpu']:.0f} | "
            f"{row['decode_gpu_active_tps']:.0f} | "
            f"{100 * row['decode_gpu_busy_fraction_per_gpu']:.1f}% | "
            f"{row['decode_running_per_engine']:.1f} | "
            f"{100 * row['decode_kv_usage_per_engine']:.1f}% | "
            f"{row['decode_prealloc_per_engine']:.1f} | "
            f"{row['decode_transfer_per_engine']:.1f} | "
            f"{100 * row['engine_cache_fraction']:.1f}% | "
            f"{100 * row['mooncake_resident_mean_fraction']:.1f}%/"
            f"{100 * row['mooncake_resident_max_fraction']:.1f}% | "
            f"{row['mooncake_eviction_successes']} / "
            f"{row['mooncake_evicted_gb']:.1f} |"
        )
    (args.run_root / "architecture_comparison.md").write_text("\n".join(md) + "\n")

    labels = [label for _, label in CASES]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    axes[0].bar(labels, [row["completion_rps"] for row in rows], color="#0072b2")
    axes[0].set_ylabel("Completed agents/s")
    axes[1].bar(
        labels, [row["decode_tps_per_gpu"] for row in rows], color="#009e73"
    )
    axes[1].set_ylabel("Decode tokens/s/GPU")
    axes[2].bar(
        labels,
        [100 * row["decode_kv_usage_per_engine"] for row in rows],
        color="#e69f00",
    )
    axes[2].set_ylabel("Decode KV usage (%)")
    axes[3].bar(
        labels,
        [100 * row["prefill_gpu_busy_fraction_per_gpu"] for row in rows],
        color="#cc79a7",
    )
    axes[3].set_ylabel("Prefill GPU busy (%)")
    axes[4].bar(
        labels,
        [row["decode_running_per_engine"] for row in rows],
        color="#56b4e9",
    )
    axes[4].set_ylabel("Decode running requests/GPU")
    axes[5].bar(
        labels,
        [100 * row["engine_cache_fraction"] for row in rows],
        color="#d55e00",
    )
    axes[5].set_ylabel("Engine prefill cache fraction (%)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(args.run_root / "architecture_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
