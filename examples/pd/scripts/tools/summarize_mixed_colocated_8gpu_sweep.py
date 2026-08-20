#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def load_valid_run(run_dir: Path) -> dict[str, object]:
    summary = json.loads((run_dir / "offload_analysis_summary.json").read_text())
    boundary = json.loads((run_dir / "closed_loop_boundaries.json").read_text())
    start = float(boundary["measurement_start_wall"])
    end = float(boundary["measurement_end_wall"])

    endpoint_series: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with (run_dir / "engine_metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("role") != "decode" or not start <= float(row["ts"]) <= end:
                continue
            for item in row.get("endpoint_metrics", []):
                endpoint = str(item["endpoint"])
                metrics = item["metrics"]
                for output_name, metric_name in (
                    ("running", "sglang_num_running_reqs"),
                    ("queue", "sglang_num_queue_reqs"),
                    ("kv", "sglang_token_usage"),
                ):
                    endpoint_series[endpoint][output_name].append(
                        float(metrics.get(metric_name, 0.0))
                    )

    endpoint_rows = []
    for endpoint in sorted(endpoint_series):
        port = int(endpoint.rsplit(":", 1)[1])
        gpu = port - 32100
        series = endpoint_series[endpoint]
        endpoint_rows.append(
            {
                "gpu": gpu,
                "endpoint": endpoint,
                "memory_fraction": 0.60 if gpu == 7 else 0.80,
                "running_mean": mean(series["running"]),
                "running_p95": percentile(series["running"], 0.95),
                "queue_mean": mean(series["queue"]),
                "queue_p95": percentile(series["queue"], 0.95),
                "kv_mean": mean(series["kv"]),
                "kv_p95": percentile(series["kv"], 0.95),
            }
        )

    status_counts: dict[str, int] = defaultdict(int)
    turns: list[float] = []
    response_tokens: list[float] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    latencies: list[float] = []
    with (run_dir / "requests.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            finished = row.get("finished_ts")
            if finished is None or not start <= float(finished) <= end:
                continue
            status_counts[str(row.get("status", "unknown"))] += 1
            turns.append(float(row.get("generation_turns", 0.0)))
            response_tokens.append(float(row.get("response_tokens", 0.0)))
            prompt_tokens.append(float(row.get("model_prompt_tokens", 0.0)))
            completion_tokens.append(float(row.get("model_completion_tokens", 0.0)))
            latencies.append(float(row.get("agent_latency_seconds", 0.0)))

    aggregate = {
        "concurrency": int(summary["concurrency"]),
        "measurement_seconds": float(summary["measurement_seconds"]),
        "completed_agents": int(summary["completed_agents"]),
        "completion_rps": float(summary["completion_rps"]),
        "math_completed": int(summary["math_completed"]),
        "qa_completed": int(summary["qa_completed"]),
        "prefill_compute_tps": float(summary["prefill_compute_tps"]),
        "prefill_cache_hit_tps": float(summary["prefill_cache_hit_tps"]),
        "decode_tps": float(summary["decode_tps"]),
        "decode_tps_per_gpu": float(summary["decode_tps_per_gpu"]),
        "prefill_busy_per_gpu": float(summary["prefill_gpu_busy_fraction"]) / 8,
        "decode_busy_per_gpu": float(summary["decode_gpu_busy_fraction_per_gpu"]),
        "decode_active_tps_per_gpu": float(summary["decode_gpu_active_tps"]),
        "prefix_hit_fraction": float(
            summary["completion_weighted_reuse"]["prefix_hit_fraction"]
        ),
        "running_mean_per_gpu": mean(
            [float(row["running_mean"]) for row in endpoint_rows]
        ),
        "queue_mean_per_gpu": mean(
            [float(row["queue_mean"]) for row in endpoint_rows]
        ),
        "kv_mean_per_gpu": mean([float(row["kv_mean"]) for row in endpoint_rows]),
        "completed_status_count": status_counts.get("completed", 0),
        "truncated_status_count": status_counts.get("truncated", 0),
        "truncated_fraction": status_counts.get("truncated", 0)
        / max(sum(status_counts.values()), 1),
        "turns_per_agent": mean(turns),
        "response_tokens_per_agent": mean(response_tokens),
        "model_prompt_tokens_per_agent": mean(prompt_tokens),
        "model_completion_tokens_per_agent": mean(completion_tokens),
        "agent_latency_mean_seconds": mean(latencies),
        "agent_latency_p95_seconds": percentile(latencies, 0.95),
    }
    return {"aggregate": aggregate, "endpoints": endpoint_rows}


def load_failed_run(run_dir: Path, concurrency: int) -> dict[str, object]:
    log_paths = [run_dir / "inference.log", *sorted((run_dir / "logs").glob("model-*.log"))]
    log = "\n".join(path.read_text(errors="replace") for path in log_paths)
    queues = [int(value) for value in re.findall(r"#queue-req:\s*(\d+)", log)]
    attempts = re.findall(
        r"Tried to allocate ([0-9.]+) GiB.*?([0-9.]+) GiB is free", log, re.S
    )
    allocation, free = attempts[-1] if attempts else ("", "")
    return {
        "concurrency": concurrency,
        "status": "OOM before valid measurement",
        "max_observed_local_queue": max(queues, default=0),
        "failed_allocation_gib": float(allocation) if allocation else None,
        "free_at_failure_gib": float(free) if free else None,
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    valid = load_valid_run(args.run_root / "c512")
    failures = [
        load_failed_run(args.run_root / "c640-overload-oom", 640),
        load_failed_run(args.run_root / "c768-overload-oom", 768),
    ]
    payload = {"stable": valid, "overloads": failures}
    (args.run_root / "saturation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    with (args.run_root / "c512_endpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(valid["endpoints"][0]))
        writer.writeheader()
        writer.writerows(valid["endpoints"])

    aggregate = valid["aggregate"]
    md = [
        "# 8-GPU mixed colocated saturation sweep",
        "",
        "Stock SGLang baseline, fixed Mixed 1:1 schedule (seed 2026), 300 s warmup + 1200 s measurement.",
        "GPU0-6 use mem-fraction-static=0.80; GPU7 shares the BrowseComp search model and uses 0.60.",
        "",
        "| Concurrency | Outcome | Agent/s | P compute token/s | D token/s | D token/s/GPU | P busy/GPU | D busy/GPU | Running/GPU | Queue/GPU | KV/GPU |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 512 | stable full load | {aggregate['completion_rps']:.3f} | "
        f"{aggregate['prefill_compute_tps']:.0f} | {aggregate['decode_tps']:.0f} | "
        f"{aggregate['decode_tps_per_gpu']:.0f} | "
        f"{100 * aggregate['prefill_busy_per_gpu']:.1f}% | "
        f"{100 * aggregate['decode_busy_per_gpu']:.1f}% | "
        f"{aggregate['running_mean_per_gpu']:.1f} | "
        f"{aggregate['queue_mean_per_gpu']:.1f} | "
        f"{100 * aggregate['kv_mean_per_gpu']:.1f}% |",
    ]
    for row in failures:
        md.append(
            f"| {row['concurrency']} | OOM before valid measurement "
            f"(max local queue {row['max_observed_local_queue']}) | — | — | — | — | — | — | — | — | — |"
        )
    md.extend(
        [
            "",
            "## Stable c512 completion characteristics",
            "",
            f"- Completed agents: {aggregate['completed_agents']} "
            f"(Retool {aggregate['math_completed']}, BrowseComp {aggregate['qa_completed']})",
            f"- Completed/truncated: {aggregate['completed_status_count']} / "
            f"{aggregate['truncated_status_count']} "
            f"({100 * aggregate['truncated_fraction']:.1f}% truncated)",
            f"- Mean turns: {aggregate['turns_per_agent']:.2f}",
            f"- Mean response tokens/agent: {aggregate['response_tokens_per_agent']:.0f}",
            f"- Mean model prompt/completion tokens per agent: "
            f"{aggregate['model_prompt_tokens_per_agent']:.0f} / "
            f"{aggregate['model_completion_tokens_per_agent']:.0f}",
            f"- Mean/P95 agent latency: {aggregate['agent_latency_mean_seconds']:.1f} / "
            f"{aggregate['agent_latency_p95_seconds']:.1f} s",
            f"- Prefix hit: {100 * aggregate['prefix_hit_fraction']:.1f}%",
            f"- Decode active throughput: {aggregate['decode_active_tps_per_gpu']:.0f} token/s/GPU",
        ]
    )
    (args.run_root / "saturation_summary.md").write_text("\n".join(md) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(
        ["Prefill compute", "Decode"],
        [aggregate["prefill_compute_tps"], aggregate["decode_tps"]],
        color=["#0072B2", "#009E73"],
    )
    axes[0].set_ylabel("tokens/s (8 GPUs total)")
    axes[0].set_title("Stable c512 throughput")

    p_busy = 100 * aggregate["prefill_busy_per_gpu"]
    d_busy = 100 * aggregate["decode_busy_per_gpu"]
    idle = max(0.0, 100 - p_busy - d_busy)
    axes[1].bar(["mean GPU"], [p_busy], label="Prefill", color="#0072B2")
    axes[1].bar(["mean GPU"], [d_busy], bottom=[p_busy], label="Decode", color="#009E73")
    axes[1].bar(["mean GPU"], [idle], bottom=[p_busy + d_busy], label="Other/idle", color="#BBBBBB")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("GPU forward time (%)")
    axes[1].set_title("c512 is compute-saturated")
    axes[1].legend(fontsize=8)

    concurrencies = [512, 640, 768]
    colors = ["#009E73", "#D55E00", "#D55E00"]
    axes[2].bar([str(value) for value in concurrencies], [1, 1, 1], color=colors)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Closed-loop concurrency")
    axes[2].set_title("Capacity outcome")
    axes[2].text(0, 0.5, "stable\n1200 s", ha="center", va="center", color="white", weight="bold")
    axes[2].text(1, 0.5, "OOM", ha="center", va="center", color="white", weight="bold")
    axes[2].text(2, 0.5, "OOM", ha="center", va="center", color="white", weight="bold")
    for axis in axes[:2]:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("8-GPU baseline-colocated Mixed 1:1 saturation sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.run_root / "saturation_summary.png", dpi=180)


if __name__ == "__main__":
    main()
