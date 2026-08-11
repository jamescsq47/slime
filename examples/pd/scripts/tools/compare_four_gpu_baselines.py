#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


CASES = (
    ("colocated-4gpu", "4× Colocated", 4),
    ("pd-no-reverse-1p3d", "1P:3D, no reverse KV", 3),
    ("pd-native-mooncake-1p3d", "1P:3D, native Mooncake", 3),
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def queue_summary(run_dir: Path, start: float, end: float) -> dict[str, float]:
    values = {
        "prefill_queue": [],
        "prefill_inflight": [],
        "prefill_kv_usage": [],
        "decode_running": [],
        "decode_prealloc": [],
        "decode_transfer": [],
        "decode_kv_usage": [],
    }
    with (run_dir / "engine_metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if not start <= float(row["ts"]) <= end:
                continue
            role = row.get("role")
            for endpoint in row.get("endpoint_metrics", []):
                metrics = endpoint["metrics"]
                if role == "prefill":
                    values["prefill_queue"].append(metrics.get("sglang_num_queue_reqs", 0.0))
                    values["prefill_inflight"].append(
                        metrics.get("sglang_num_prefill_inflight_queue_reqs", 0.0)
                    )
                    values["prefill_kv_usage"].append(
                        metrics.get("sglang_token_usage", 0.0)
                    )
                elif role == "decode":
                    values["decode_running"].append(metrics.get("sglang_num_running_reqs", 0.0))
                    values["decode_prealloc"].append(
                        metrics.get("sglang_num_decode_prealloc_queue_reqs", 0.0)
                    )
                    values["decode_transfer"].append(
                        metrics.get("sglang_num_decode_transfer_queue_reqs", 0.0)
                    )
                    values["decode_kv_usage"].append(metrics.get("sglang_token_usage", 0.0))
    return {f"{key}_per_engine": mean(series) for key, series in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, label, expected_decode_gpus in CASES:
        run_dir = args.run_root / name
        summary = json.loads((run_dir / "offload_analysis_summary.json").read_text())
        boundary = json.loads((run_dir / "closed_loop_boundaries.json").read_text())
        reuse = summary.get("completion_weighted_reuse", {})
        row = {
            "case": name,
            "label": label,
            **summary,
            "expected_decode_gpus": expected_decode_gpus,
            "reverse_reuse_fraction": reuse.get("page_aligned_reverse_reuse_fraction", 0.0),
            "prompt_prefix_hit_fraction": reuse.get("prefix_hit_fraction", 0.0),
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
        "case", "completed_agents", "completion_rps", "prefill_compute_tps",
        "prefill_cache_hit_tps", "decode_tps", "decode_tps_per_gpu",
        "prefill_gpu_busy_fraction", "decode_gpu_busy_fraction_per_gpu",
        "decode_running_per_engine", "decode_prealloc_per_engine",
        "decode_transfer_per_engine", "decode_kv_usage_per_engine",
        "prefill_queue_per_engine", "prefill_inflight_per_engine",
        "prompt_prefix_hit_fraction", "reverse_reuse_fraction",
    ]
    with (args.run_root / "architecture_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Four-model-GPU mixed-agent baseline comparison",
        "",
        "All cases use the same fixed Mixed 1:1 schedule, seed, concurrency, warmup, and measurement window.",
        "",
        "| Case | Agent/s | Prefill token/s | Decode token/s | Decode token/s/GPU | D busy/GPU | D running/GPU | Prefix hit | Reverse KV reuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['label']} | {row['completion_rps']:.3f} | "
            f"{row['prefill_compute_tps']:.0f} | {row['decode_tps']:.0f} | "
            f"{row['decode_tps_per_gpu']:.0f} | "
            f"{100 * row['decode_gpu_busy_fraction_per_gpu']:.1f}% | "
            f"{row['decode_running_per_engine']:.1f} | "
            f"{100 * row['prompt_prefix_hit_fraction']:.1f}% | "
            f"{100 * row['reverse_reuse_fraction']:.1f}% |"
        )
    (args.run_root / "architecture_comparison.md").write_text("\n".join(md) + "\n")

    labels = [label for _, label, _ in CASES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(labels, [row["completion_rps"] for row in rows], color="#0072b2")
    axes[0].set_ylabel("Completed agents/s")
    axes[1].bar(labels, [row["decode_tps_per_gpu"] for row in rows], color="#009e73")
    axes[1].set_ylabel("Decode tokens/s/GPU")
    axes[2].bar(
        labels,
        [100 * row["decode_gpu_busy_fraction_per_gpu"] for row in rows],
        color="#e69f00",
    )
    axes[2].set_ylabel("Decode GPU busy (%)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(args.run_root / "architecture_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
