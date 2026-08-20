#!/usr/bin/env python3
"""Combine pure-workload colocated and stock-PD baseline results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from compare_pure_workload_pd_baselines import completion_means, engine_means


COLOCATED_CASES = (
    ("browsecomp-only", "BrowseComp"),
    ("retool-only", "Retool"),
)


def load_colocated(run_dir: Path, workload: str) -> dict[str, object]:
    summary = json.loads((run_dir / "offload_analysis_summary.json").read_text())
    boundary = json.loads((run_dir / "closed_loop_boundaries.json").read_text())
    reuse = summary.get("completion_weighted_reuse", {})
    row: dict[str, object] = {
        "case": run_dir.name,
        "workload": workload,
        "layout": "4-GPU colocated",
        "mode": "Local RadixCache",
        "physical_gpus": 4,
        "prefill_gpus": 4,
        "decode_gpus": 4,
        **summary,
        "prefill_busy_per_gpu": summary["prefill_gpu_busy_fraction"] / 4,
        "prefix_hit_fraction": reuse.get("prefix_hit_fraction", 0.0),
        "reverse_reuse_fraction": reuse.get(
            "page_aligned_reverse_reuse_fraction", 0.0
        ),
        "mooncake_mean_fraction": 0.0,
        "mooncake_max_fraction": 0.0,
        "mooncake_evictions": 0,
        "mooncake_evicted_gb": 0.0,
        "host_cache_full_messages": 0,
    }
    row.update(
        engine_means(
            run_dir,
            boundary["measurement_start_wall"],
            boundary["measurement_end_wall"],
        )
    )
    row.update(
        completion_means(
            run_dir,
            boundary["measurement_start_wall"],
            boundary["measurement_end_wall"],
        )
    )
    row.pop("completion_weighted_reuse", None)
    row.pop("mooncake", None)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd-root", type=Path, required=True)
    parser.add_argument("--colocated-root", type=Path, required=True)
    args = parser.parse_args()

    pd_rows = json.loads(
        (args.pd_root / "pure_workload_pd_comparison.json").read_text()
    )
    for row in pd_rows:
        row["layout"] = f"{row['prefill_gpus']}P:{row['decode_gpus']}D"
        row["physical_gpus"] = row["prefill_gpus"] + row["decode_gpus"]
    colocated_rows = [
        load_colocated(args.colocated_root / case, workload)
        for case, workload in COLOCATED_CASES
    ]

    rows: list[dict[str, object]] = []
    for workload in ("BrowseComp", "Retool"):
        rows.extend(row for row in colocated_rows if row["workload"] == workload)
        rows.extend(row for row in pd_rows if row["workload"] == workload)

    columns = [
        "case", "workload", "layout", "mode", "physical_gpus",
        "completed_agents", "completion_rps", "prefill_compute_tps",
        "prefill_busy_per_gpu", "prefill_compute_tokens_per_completed_agent",
        "decode_tps", "decode_tps_per_gpu", "decode_gpu_active_tps",
        "decode_gpu_busy_fraction_per_gpu", "decode_running", "prefill_kv",
        "decode_kv", "decode_prealloc", "decode_transfer",
        "decode_tokens_per_completed_agent", "generation_turns_per_agent",
        "response_tokens_per_agent", "prefix_hit_fraction",
        "reverse_reuse_fraction", "truncated_fraction",
        "mooncake_mean_fraction", "mooncake_max_fraction",
        "mooncake_evictions", "mooncake_evicted_gb", "host_cache_full_messages",
    ]
    csv_path = args.pd_root / "pure_workload_grand_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.pd_root / "pure_workload_grand_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    md = [
        "# Pure-workload four-GPU baseline comparison",
        "",
        "All cases use c256, seed 2026, a 300-second warmup, and a 1200-second measurement window.",
        "",
        "| Workload | Layout | KV mode | Agent/s | P compute t/s | P busy/GPU | P t/agent | D t/s total | D t/s/GPU | D active/GPU | D busy/GPU | D running/GPU | D KV | Prealloc / transfer | Prefix hit | Cross-turn reuse | Truncated |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['workload']} | {row['layout']} | {row['mode']} | "
            f"{row['completion_rps']:.3f} | {row['prefill_compute_tps']:.0f} | "
            f"{100 * row['prefill_busy_per_gpu']:.1f}% | "
            f"{row['prefill_compute_tokens_per_completed_agent']:.0f} | "
            f"{row['decode_tps']:.0f} | {row['decode_tps_per_gpu']:.0f} | "
            f"{row['decode_gpu_active_tps']:.0f} | "
            f"{100 * row['decode_gpu_busy_fraction_per_gpu']:.1f}% | "
            f"{row['decode_running']:.1f} | {100 * row['decode_kv']:.1f}% | "
            f"{row['decode_prealloc']:.1f} / {row['decode_transfer']:.1f} | "
            f"{100 * row['prefix_hit_fraction']:.1f}% | "
            f"{100 * row['reverse_reuse_fraction']:.1f}% | "
            f"{100 * row['truncated_fraction']:.1f}% |"
        )
    md.extend(
        [
            "",
            "## Completion-set and cache health",
            "",
            "| Workload | Layout | KV mode | Completed | Turns/agent | Response tokens/agent | Mooncake mean / max | Mooncake evictions | Evicted data | Host-cache-full messages |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        md.append(
            f"| {row['workload']} | {row['layout']} | {row['mode']} | "
            f"{row['completed_agents']} | {row['generation_turns_per_agent']:.2f} | "
            f"{row['response_tokens_per_agent']:.0f} | "
            f"{100 * row['mooncake_mean_fraction']:.1f}% / "
            f"{100 * row['mooncake_max_fraction']:.1f}% | "
            f"{row['mooncake_evictions']} | {row['mooncake_evicted_gb']:.1f} GiB | "
            f"{row['host_cache_full_messages']} |"
        )
    (args.pd_root / "pure_workload_grand_comparison.md").write_text(
        "\n".join(md) + "\n"
    )

    labels = []
    for row in rows:
        workload = "BC" if row["workload"] == "BrowseComp" else "Retool"
        layout = "Colocated" if row["layout"] == "4-GPU colocated" else row["layout"]
        mode = {
            "Local RadixCache": "Local KV",
            "No reverse KV": "No reverse",
            "Native Mooncake": "Mooncake",
        }[row["mode"]]
        labels.append(f"{workload} {layout}\n{mode}")
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    plots = (
        ("completion_rps", "Completed agents/s", 1.0),
        ("prefill_compute_tps", "P compute tokens/s", 1.0),
        ("decode_tps_per_gpu", "D tokens/s/GPU", 1.0),
        ("decode_running", "D running/GPU", 1.0),
        ("decode_kv", "D KV usage", 100.0),
        ("reverse_reuse_fraction", "Cross-turn KV reuse", 100.0),
    )
    colors = ["#444444", "#0072B2", "#56B4E9", "#009E73", "#66C2A5",
              "#777777", "#CC79A7", "#E78AC3", "#D55E00", "#E69F00"]
    for axis, (key, title, scale) in zip(axes.ravel(), plots):
        values = [float(row[key]) * scale for row in rows]
        bars = axis.bar(labels, values, color=colors[: len(rows)])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=16, labelsize=7)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if key == "completion_rps" else f"{value:.1f}",
                ha="center", va="bottom", fontsize=7,
            )
    fig.suptitle("Pure-workload four-GPU baselines (c256, 300s + 1200s)")
    fig.tight_layout()
    fig.savefig(args.pd_root / "pure_workload_grand_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
