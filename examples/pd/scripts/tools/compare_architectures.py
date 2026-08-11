#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


CASES = (
    ("colocated-6gpu", "6×Colocated"),
    ("pd-no-reverse-1p5d", "PD 1P:5D\nno reverse KV"),
    ("pd-no-reverse-2p4d", "PD 2P:4D\nno reverse KV"),
    ("pd-native-mooncake-1p5d", "PD 1P:5D\nnative Mooncake"),
)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def queue_summary(run_dir: Path, start: float, end: float) -> dict[str, float]:
    p_queue, p_inflight, d_running, d_prealloc, d_transfer, d_usage = ([] for _ in range(6))
    with (run_dir / "engine_metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if not start <= float(row["ts"]) <= end:
                continue
            if row.get("role") == "prefill":
                for endpoint in row.get("endpoint_metrics", []):
                    metrics = endpoint["metrics"]
                    p_queue.append(metrics.get("sglang_num_queue_reqs", 0.0))
                    p_inflight.append(metrics.get("sglang_num_prefill_inflight_queue_reqs", 0.0))
            elif row.get("role") == "decode":
                for endpoint in row.get("endpoint_metrics", []):
                    metrics = endpoint["metrics"]
                    d_running.append(metrics.get("sglang_num_running_reqs", 0.0))
                    d_prealloc.append(metrics.get("sglang_num_decode_prealloc_queue_reqs", 0.0))
                    d_transfer.append(metrics.get("sglang_num_decode_transfer_queue_reqs", 0.0))
                    d_usage.append(metrics.get("sglang_token_usage", 0.0))
    return {
        "prefill_queue_per_p": mean(p_queue),
        "prefill_inflight_per_p": mean(p_inflight),
        "decode_running_per_d": mean(d_running),
        "decode_prealloc_per_d": mean(d_prealloc),
        "decode_transfer_per_d": mean(d_transfer),
        "decode_kv_usage_per_d": mean(d_usage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, label in CASES:
        run_dir = args.run_root / name
        summary = json.loads((run_dir / "offload_analysis_summary.json").read_text())
        boundary = json.loads((run_dir / "closed_loop_boundaries.json").read_text())
        row = {"case": name, "label": label.replace("\n", " "), **summary}
        row.update(queue_summary(run_dir, boundary["measurement_start_wall"], boundary["measurement_end_wall"]))
        row["reverse_reuse_fraction"] = summary["completion_weighted_reuse"].get(
            "page_aligned_reverse_reuse_fraction", 0.0
        )
        row.pop("completion_weighted_reuse", None)
        row.pop("mooncake", None)
        rows.append(row)

    (args.run_root / "architecture_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    columns = [
        "case", "completed_agents", "completion_rps", "prefill_compute_tps",
        "decode_tps", "decode_tps_per_gpu", "prefill_gpu_busy_fraction",
        "decode_gpu_busy_fraction_per_gpu", "decode_running_per_d",
        "decode_prealloc_per_d", "decode_transfer_per_d", "decode_kv_usage_per_d",
        "prefill_queue_per_p", "prefill_inflight_per_p", "reverse_reuse_fraction",
    ]
    with (args.run_root / "architecture_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    md = [
        "# Six-GPU mixed-agent architecture comparison", "",
        "All cases use the same fixed Mixed 1:1 schedule, seed, concurrency, warmup and measurement window.", "",
        "| Case | Agent/s | Prefill token/s | Decode token/s | Decode token/s/GPU | D busy/GPU | D running/GPU | D prealloc/GPU | Reverse KV reuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['label']} | {row['completion_rps']:.3f} | {row['prefill_compute_tps']:.0f} | "
            f"{row['decode_tps']:.0f} | {row['decode_tps_per_gpu']:.0f} | "
            f"{100*row['decode_gpu_busy_fraction_per_gpu']:.1f}% | {row['decode_running_per_d']:.1f} | "
            f"{row['decode_prealloc_per_d']:.1f} | {100*row['reverse_reuse_fraction']:.1f}% |"
        )
    (args.run_root / "architecture_comparison.md").write_text("\n".join(md) + "\n")

    labels = [label for _, label in CASES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(labels, [row["completion_rps"] for row in rows], color="#0072b2")
    axes[0].set_ylabel("completed agents/s")
    axes[1].bar(labels, [row["decode_tps_per_gpu"] for row in rows], color="#009e73")
    axes[1].set_ylabel("Decode tokens/s/GPU")
    axes[2].bar(labels, [100 * row["decode_gpu_busy_fraction_per_gpu"] for row in rows], color="#e69f00")
    axes[2].set_ylabel("Decode GPU busy (%)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(args.run_root / "architecture_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
