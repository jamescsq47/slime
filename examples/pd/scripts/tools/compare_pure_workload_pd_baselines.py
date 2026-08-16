#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


CASES = (
    ("browsecomp-no-reverse-3p1d", "BrowseComp", "No reverse KV", 3, 1),
    ("browsecomp-native-mooncake-3p1d", "BrowseComp", "Native Mooncake", 3, 1),
    ("browsecomp-no-reverse-2p2d", "BrowseComp", "No reverse KV", 2, 2),
    ("browsecomp-native-mooncake-2p2d", "BrowseComp", "Native Mooncake", 2, 2),
    ("retool-no-reverse-1p3d", "Retool", "No reverse KV", 1, 3),
    ("retool-native-mooncake-1p3d", "Retool", "Native Mooncake", 1, 3),
    ("retool-no-reverse-2p2d", "Retool", "No reverse KV", 2, 2),
    ("retool-native-mooncake-2p2d", "Retool", "Native Mooncake", 2, 2),
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def engine_means(run_dir: Path, start: float, end: float) -> dict[str, float]:
    series = {
        "prefill_kv": [],
        "decode_running": [],
        "decode_prealloc": [],
        "decode_transfer": [],
        "decode_kv": [],
    }
    with (run_dir / "engine_metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if not start <= float(row["ts"]) <= end:
                continue
            for endpoint in row.get("endpoint_metrics", []):
                metrics = endpoint["metrics"]
                if row.get("role") == "prefill":
                    series["prefill_kv"].append(
                        metrics.get("sglang_token_usage", 0.0)
                    )
                elif row.get("role") == "decode":
                    series["decode_running"].append(
                        metrics.get("sglang_num_running_reqs", 0.0)
                    )
                    series["decode_prealloc"].append(
                        metrics.get("sglang_num_decode_prealloc_queue_reqs", 0.0)
                    )
                    series["decode_transfer"].append(
                        metrics.get("sglang_num_decode_transfer_queue_reqs", 0.0)
                    )
                    series["decode_kv"].append(
                        metrics.get("sglang_token_usage", 0.0)
                    )
    return {key: mean(values) for key, values in series.items()}


def completion_means(run_dir: Path, start: float, end: float) -> dict[str, float]:
    status_counts: dict[str, int] = {}
    turns: list[float] = []
    response_tokens: list[float] = []
    with (run_dir / "requests.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            finished = row.get("finished_ts")
            if finished is None or not start <= float(finished) <= end:
                continue
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            turns.append(float(row.get("generation_turns", 0.0)))
            response_tokens.append(float(row.get("response_tokens", 0.0)))
    total = sum(status_counts.values())
    return {
        "completed_status_count": status_counts.get("completed", 0),
        "truncated_status_count": status_counts.get("truncated", 0),
        "truncated_fraction": status_counts.get("truncated", 0) / max(total, 1),
        "generation_turns_per_agent": mean(turns),
        "response_tokens_per_agent": mean(response_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case, workload, mode, prefill_gpus, decode_gpus in CASES:
        run_dir = args.run_root / case
        summary = json.loads((run_dir / "offload_analysis_summary.json").read_text())
        boundary = json.loads((run_dir / "closed_loop_boundaries.json").read_text())
        reuse = summary.get("completion_weighted_reuse", {})
        mooncake = summary.get("mooncake", {})
        row = {
            "case": case,
            "workload": workload,
            "mode": mode,
            "prefill_gpus": prefill_gpus,
            "decode_gpus": decode_gpus,
            **summary,
            "prefill_busy_per_gpu": summary["prefill_gpu_busy_fraction"]
            / prefill_gpus,
            "prefix_hit_fraction": reuse.get("prefix_hit_fraction", 0.0),
            "reverse_reuse_fraction": reuse.get(
                "page_aligned_reverse_reuse_fraction", 0.0
            ),
            "mooncake_mean_fraction": mooncake.get(
                "measurement_resident_mean_fraction", 0.0
            ),
            "mooncake_max_fraction": mooncake.get(
                "measurement_resident_max_fraction", 0.0
            ),
            "mooncake_evictions": mooncake.get("eviction_successes", 0),
            "mooncake_evicted_gb": mooncake.get("evicted_gb", 0.0),
            "host_cache_full_messages": sum(
                text.count("Not enough host memory for request")
                for text in (
                    path.read_text(errors="replace")
                    for path in sorted((run_dir / "logs").glob("decode-*.log"))
                )
            ),
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
        rows.append(row)

    columns = [
        "case", "workload", "mode", "prefill_gpus", "decode_gpus",
        "completed_agents", "completion_rps", "prefill_compute_tps",
        "prefill_cache_hit_tps", "prefill_busy_per_gpu", "prefill_kv",
        "decode_tps", "decode_tps_per_gpu", "decode_gpu_active_tps",
        "decode_gpu_busy_fraction_per_gpu", "decode_running",
        "decode_prealloc", "decode_transfer", "decode_kv",
        "prefill_compute_tokens_per_completed_agent",
        "decode_tokens_per_completed_agent", "generation_turns_per_agent",
        "response_tokens_per_agent", "completed_status_count",
        "truncated_status_count", "truncated_fraction",
        "prefix_hit_fraction", "reverse_reuse_fraction",
        "mooncake_mean_fraction", "mooncake_max_fraction",
        "mooncake_evictions", "mooncake_evicted_gb",
        "host_cache_full_messages",
    ]
    with (args.run_root / "pure_workload_pd_comparison.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.run_root / "pure_workload_pd_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    md = [
        "# Pure-workload stock PD baseline comparison",
        "",
        "All cases use c256, seed 2026, a 300-second warmup, and a 1200-second measurement window.",
        "",
        "| Workload / layout | KV mode | Agent/s | P compute token/s | P busy/GPU | P token/agent | D token/s/GPU | D active token/s/GPU | D busy/GPU | D running/GPU | D KV | Prealloc / transfer | Reverse reuse | Truncated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        layout = f"{row['workload']} {row['prefill_gpus']}P:{row['decode_gpus']}D"
        md.append(
            f"| {layout} | {row['mode']} | {row['completion_rps']:.3f} | "
            f"{row['prefill_compute_tps']:.0f} | "
            f"{100 * row['prefill_busy_per_gpu']:.1f}% | "
            f"{row['prefill_compute_tokens_per_completed_agent']:.0f} | "
            f"{row['decode_tps_per_gpu']:.0f} | "
            f"{row['decode_gpu_active_tps']:.0f} | "
            f"{100 * row['decode_gpu_busy_fraction_per_gpu']:.1f}% | "
            f"{row['decode_running']:.1f} | {100 * row['decode_kv']:.1f}% | "
            f"{row['decode_prealloc']:.1f} / {row['decode_transfer']:.1f} | "
            f"{100 * row['reverse_reuse_fraction']:.1f}% | "
            f"{100 * row['truncated_fraction']:.1f}% |"
        )
    md.extend(
        [
            "",
            "## Native Mooncake delta versus no reverse KV",
            "",
            "| Workload | Agent/s | P compute tokens/agent | D tokens/s/GPU | D running/GPU |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for workload, prefill_gpus, decode_gpus in (
        ("BrowseComp", 3, 1),
        ("BrowseComp", 2, 2),
        ("Retool", 1, 3),
        ("Retool", 2, 2),
    ):
        base, native = [
            row
            for row in rows
            if row["workload"] == workload
            and row["prefill_gpus"] == prefill_gpus
            and row["decode_gpus"] == decode_gpus
        ]
        delta = lambda key: 100 * (native[key] / base[key] - 1)
        md.append(
            f"| {workload} {prefill_gpus}P:{decode_gpus}D | "
            f"{delta('completion_rps'):+.1f}% | "
            f"{delta('prefill_compute_tokens_per_completed_agent'):+.1f}% | "
            f"{delta('decode_tps_per_gpu'):+.1f}% | "
            f"{delta('decode_running'):+.1f}% |"
        )
    (args.run_root / "pure_workload_pd_comparison.md").write_text(
        "\n".join(md) + "\n"
    )

    labels = [f"{row['workload']}\n{row['mode']}" for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plots = (
        ("completion_rps", "Completed agents/s", 1.0),
        ("prefill_compute_tps", "P compute tokens/s", 1.0),
        ("decode_tps_per_gpu", "D tokens/s/GPU", 1.0),
        ("decode_running", "D running/GPU", 1.0),
        ("decode_kv", "D KV usage", 100.0),
        ("reverse_reuse_fraction", "Reverse KV reuse", 100.0),
    )
    colors = [
        "#0072B2", "#56B4E9", "#009E73", "#66C2A5",
        "#CC79A7", "#E78AC3", "#D55E00", "#E69F00",
    ]
    for axis, (key, title, scale) in zip(axes.ravel(), plots):
        values = [row[key] * scale for row in rows]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=12)
        if scale == 100.0:
            axis.set_ylabel("Percent")
        annotation_format = ".3f" if key == "completion_rps" else ".1f"
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format(value, annotation_format),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Stock SGLang pure-workload PD baselines (c256, 300s + 1200s)")
    fig.tight_layout()
    fig.savefig(args.run_root / "pure_workload_pd_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
