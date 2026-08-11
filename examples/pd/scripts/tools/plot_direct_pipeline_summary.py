#!/usr/bin/env python3
"""Summarize the long-running direct KV pipeline experiments."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = (
    ("Retool\n1P:3D c256", "direct-retool-1p3d-concurrency-sweep/c256"),
    ("Retool\n1P:3D c320", "direct-retool-1p3d-concurrency-sweep/c320"),
    ("Retool\n1P:7D c512", "direct-retool-1p7d-concurrency-sweep/c512"),
    ("Mixed 1:1\n1P:3D c256", "direct-mixed-1p3d-forced-wait/c256"),
)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def endpoint_state_means(run_dir: Path, start: float, end: float):
    keys = (
        "sglang_num_running_reqs",
        "sglang_num_decode_prealloc_queue_reqs",
        "sglang_num_decode_transfer_queue_reqs",
    )
    totals = {key: [] for key in keys}
    with (run_dir / "engine_metrics.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("role") != "decode" or not start <= row["ts"] <= end:
                continue
            for key in keys:
                totals[key].append(
                    sum(item["metrics"].get(key, 0.0) for item in row["endpoint_metrics"])
                )
    return {key: float(np.mean(values)) for key, values in totals.items()}


def collect(runs_root: Path):
    output = []
    for label, relative in CASES:
        run_dir = runs_root / relative
        summary = load_json(run_dir / "offload_analysis_summary.json")
        bounds = load_json(run_dir / "closed_loop_boundaries.json")
        direct = load_json(run_dir / "agentic_kv_analysis.json")["direct"]
        states = endpoint_state_means(
            run_dir,
            bounds["measurement_start_wall"],
            bounds["measurement_end_wall"],
        )
        output.append(
            {
                "label": label,
                "relative_run_dir": relative,
                "workload": "mixed" if "Mixed" in label else "retool",
                **summary,
                **states,
                "direct_offers": direct["offers"],
                "direct_hits": direct["hits"],
                "direct_fallbacks": direct["fallbacks"],
                "direct_hit_rate": direct["hit_rate_per_offer"],
                "direct_send_seconds": direct["mean_send_complete_seconds"],
            }
        )
    return output


def render(cases, output: Path):
    labels = [case["label"] for case in cases]
    x = np.arange(len(cases))
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#b279a2"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax = axes[0, 0]
    aggregate = [case["decode_tps"] for case in cases]
    bars = ax.bar(x, aggregate, color=colors)
    for bar, case in zip(bars, cases):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 150,
            f'{case["decode_tps"]:.0f}\n({case["decode_tps_per_gpu"]:.0f}/D)',
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Steady decode throughput")
    ax.set_ylabel("generated token/s")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(aggregate) * 1.22)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    completions = [case["completion_rps"] for case in cases]
    bars = ax.bar(x, completions, color=colors)
    for bar, value in zip(bars, completions):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    ax.set_title("Completed agent trajectories")
    ax.set_ylabel("agent/s")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(completions) * 1.2)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    running = np.array([case["sglang_num_running_reqs"] for case in cases])
    prealloc = np.array([case["sglang_num_decode_prealloc_queue_reqs"] for case in cases])
    transfer = np.array([case["sglang_num_decode_transfer_queue_reqs"] for case in cases])
    ax.bar(x, running, label="D running", color="#54a24b")
    ax.bar(x, prealloc, bottom=running, label="D prealloc", color="#eeca3b")
    ax.bar(x, transfer, bottom=running + prealloc, label="D transfer", color="#e45756")
    ax.set_title("Mean request state across all D GPUs")
    ax.set_ylabel("requests")
    ax.set_xticks(x, labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    width = 0.25
    p_busy = [100 * case["prefill_gpu_busy_fraction"] for case in cases]
    d_busy = [100 * case["decode_gpu_busy_fraction_per_gpu"] for case in cases]
    direct_hit = [100 * case["direct_hit_rate"] for case in cases]
    ax.bar(x - width, p_busy, width, label="P GPU busy", color="#4c78a8")
    ax.bar(x, d_busy, width, label="D GPU busy / card", color="#f58518")
    ax.bar(x + width, direct_hit, width, label="direct hits / offers", color="#72b7b2")
    ax.set_title("Utilization and direct-path success")
    ax.set_ylabel("percent")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Agentic PD direct-KV pipeline — 300 s warmup + 1200 s steady measurement\n"
        "Unstable overload points excluded: 1P:3D c384 (P crash), 1P:7D c640 (KV/retraction collapse)",
        fontsize=14,
    )
    fig.savefig(output, dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = collect(args.runs_root)
    (args.output_dir / "direct_pipeline_summary.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )
    render(cases, args.output_dir / "direct_pipeline_summary.png")


if __name__ == "__main__":
    main()
