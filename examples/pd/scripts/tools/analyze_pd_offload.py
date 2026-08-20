#!/usr/bin/env python3
"""Summarize and visualize a closed-loop PD run with Decode KV offload."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


P_COMPUTE = "sglang_realtime_tokens_total|mode=prefill_compute"
P_CACHE = "sglang_realtime_tokens_total|mode=prefill_cache"
D_DECODE = "sglang_realtime_tokens_total|mode=decode"
P_GPU = "sglang_gpu_execution_seconds_total|category=forward_extend"
D_GPU = "sglang_gpu_execution_seconds_total|category=forward_decode"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def counter_delta(
    rows: list[dict],
    role: str,
    key: str,
    start: float,
    end: float,
    *,
    optional: bool = False,
) -> tuple[float, float]:
    selected = [
        row for row in rows
        if row.get("role") == role
        and start <= row["ts"] <= end
        and key in row.get("metrics", {})
    ]
    if not selected:
        if optional:
            return 0.0, 0.0
        raise ValueError(f"missing required counter {key!r} for role {role!r}")
    if len(selected) < 2:
        raise ValueError(f"not enough samples for counter {key!r} and role {role!r}")
    elapsed = selected[-1]["ts"] - selected[0]["ts"]
    delta = selected[-1]["metrics"][key] - selected[0]["metrics"][key]
    return delta, elapsed


def counter_rates(rows: list[dict], role: str, key: str, origin: float, start: float, end: float) -> tuple[list[float], list[float]]:
    selected = [
        row for row in rows
        if row.get("role") == role
        and start <= row["ts"] <= end
        and key in row.get("metrics", {})
    ]
    xs, rates = [], []
    for old, new in zip(selected, selected[1:]):
        elapsed = new["ts"] - old["ts"]
        if elapsed <= 0:
            continue
        xs.append(new["ts"] - origin)
        rates.append(max(0.0, new["metrics"][key] - old["metrics"][key]) / elapsed)
    return xs, rates


def moving_average(values: list[float], width: int = 30) -> list[float]:
    return [statistics.fmean(values[max(0, index - width + 1):index + 1]) for index in range(len(values))]


def request_reuse(requests: list[dict], page_size: int = 64) -> dict[str, float | int]:
    turns = [turn for row in requests for turn in row.get("turn_metrics", [])]
    prompt = sum(turn["prompt_tokens"] for turn in turns)
    cached = sum(turn["cached_tokens"] for turn in turns)
    generated = sum(turn["completion_tokens"] for turn in turns)
    reverse_reuse = 0
    eligible_decode = 0
    reverse_hit_turns = 0
    later_turns = 0
    page_eligible_decode = 0
    page_reverse_reuse = 0
    page_eligible_turns = 0
    page_reverse_hit_turns = 0
    page_full_hit_turns = 0
    page_parent_expected = 0
    page_parent_reused = 0
    page_parent_full_hit_turns = 0
    page_parent_partial_hit_turns = 0
    page_parent_zero_hit_turns = 0
    for row in requests:
        row_turns = row.get("turn_metrics", [])
        for index, turn in enumerate(row_turns[1:], 1):
            previous = row_turns[index - 1]
            reused = min(
                previous["completion_tokens"],
                max(0, turn["cached_tokens"] - previous["prompt_tokens"]),
            )
            reverse_reuse += reused
            eligible_decode += previous["completion_tokens"]
            reverse_hit_turns += reused > 0
            later_turns += 1

            # HiCache/RadixCache can only retain complete pages.  The final
            # partial page of the previous turn is therefore not reusable by
            # the next turn even when Decode offload works perfectly.
            previous_prompt = previous["prompt_tokens"]
            previous_completion = previous["completion_tokens"]
            # Decode stores output_ids[:-1]: the last sampled token has no KV
            # until it is consumed by a subsequent forward.  Subtract one
            # before page alignment so an exact page boundary is not counted
            # as a complete, reusable page prematurely.
            aligned_end = (
                (previous_prompt + previous_completion - 1) // page_size
            ) * page_size
            parent_reused = min(aligned_end, turn["cached_tokens"])
            page_parent_expected += aligned_end
            page_parent_reused += parent_reused
            if parent_reused == aligned_end:
                page_parent_full_hit_turns += 1
            elif parent_reused > 0:
                page_parent_partial_hit_turns += 1
            else:
                page_parent_zero_hit_turns += 1
            page_eligible = max(
                0,
                min(previous_completion, aligned_end - previous_prompt),
            )
            if page_eligible > 0:
                page_reused = max(
                    0,
                    min(page_eligible, turn["cached_tokens"] - previous_prompt),
                )
                page_eligible_decode += page_eligible
                page_reverse_reuse += page_reused
                page_eligible_turns += 1
                page_reverse_hit_turns += page_reused > 0
                page_full_hit_turns += page_reused == page_eligible
    return {
        "agents": len(requests),
        "model_calls": len(turns),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "actual_prefill_tokens": prompt - cached,
        "decode_tokens_completion_weighted": generated,
        "prefix_hit_fraction": cached / prompt,
        "reverse_reused_decode_tokens_lower_bound": reverse_reuse,
        "eligible_prior_decode_tokens": eligible_decode,
        "reverse_reuse_fraction_lower_bound": (
            reverse_reuse / eligible_decode if eligible_decode else 0.0
        ),
        "reverse_hit_later_turns": reverse_hit_turns,
        "later_turns": later_turns,
        "cache_page_size": page_size,
        "page_aligned_eligible_prior_decode_tokens": page_eligible_decode,
        "page_aligned_reverse_reused_decode_tokens": page_reverse_reuse,
        "page_aligned_reverse_reuse_fraction": (
            page_reverse_reuse / page_eligible_decode if page_eligible_decode else 0.0
        ),
        "page_aligned_eligible_later_turns": page_eligible_turns,
        "page_aligned_reverse_hit_later_turns": page_reverse_hit_turns,
        "page_aligned_full_hit_later_turns": page_full_hit_turns,
        # Full parent-prefix accounting is the colocated KV-thrashing metric.
        # Unlike the reverse-KV fields above, it includes prompt/tool history
        # that must also be recomputed when an earlier Radix branch is lost.
        "page_aligned_expected_parent_prefix_tokens": page_parent_expected,
        "page_aligned_reused_parent_prefix_tokens": page_parent_reused,
        "page_aligned_extra_prefill_from_parent_kv_loss": (
            page_parent_expected - page_parent_reused
        ),
        "page_aligned_parent_prefix_reuse_fraction": (
            page_parent_reused / page_parent_expected if page_parent_expected else 0.0
        ),
        "page_aligned_parent_full_hit_later_turns": page_parent_full_hit_turns,
        "page_aligned_parent_partial_hit_later_turns": page_parent_partial_hit_turns,
        "page_aligned_parent_zero_hit_later_turns": page_parent_zero_hit_turns,
    }


def terminal_repair_summary(requests: list[dict], duration: float) -> dict:
    """Summarize parser repairs whose parent KV was deliberately finalized."""

    events = [
        (row["task_type"], event)
        for row in requests
        for event in (row.get("metadata") or {}).get("terminal_repair_events", [])
    ]
    attempted = [(task, event) for task, event in events if event.get("repair_attempted")]
    affected = [
        row
        for row in requests
        if any(
            event.get("repair_attempted")
            for event in (row.get("metadata") or {}).get("terminal_repair_events", [])
        )
    ]

    def total(key: str) -> int:
        return sum(int(event.get(key) or 0) for _, event in attempted)

    extra = total("extra_prefill_tokens")
    return {
        "affected_agents": len(affected),
        "affected_agent_fraction": len(affected) / len(requests) if requests else 0.0,
        "repair_events": len(attempted),
        "unattempted_events": len(events) - len(attempted),
        "qa_affected_agents": sum(row["task_type"] == "qa" for row in affected),
        "math_affected_agents": sum(row["task_type"] == "math" for row in affected),
        "qa_repair_events": sum(task == "qa" for task, _ in attempted),
        "math_repair_events": sum(task == "math" for task, _ in attempted),
        "logical_repair_prompt_tokens": total("actual_prompt_tokens"),
        "actual_repair_prefill_tokens": total("actual_prefill_tokens"),
        "counterfactual_repair_prefill_tokens_with_parent_kv": total(
            "counterfactual_prefill_tokens"
        ),
        "extra_prefill_tokens_from_finalized_parent": extra,
        "extra_prefill_tokens_per_second": extra / duration if duration else 0.0,
        "extra_prefill_tokens_per_completed_agent": (
            extra / len(requests) if requests else 0.0
        ),
    }


def mooncake_series(path: Path, origin: float, start: float, end: float) -> tuple[list[float], list[float], dict]:
    timestamp_pattern = re.compile(r"^I(\d{4}) (\d\d:\d\d:\d\d\.\d+)")
    storage_pattern = re.compile(
        r"Mem Storage: ([0-9.]+) GB / ([0-9.]+) GB \(([0-9.]+)%\).*"
        r"Eviction: Success/Attempts=(\d+)/(\d+), AllocFail=(\d+), keys=(\d+), size=([0-9.]+) GB"
    )
    watermark_pattern = re.compile(r"eviction_high_watermark_ratio=([0-9.]+)")
    import datetime as dt

    if not path.exists():
        return [], [], {}
    year = dt.datetime.fromtimestamp(origin, tz=dt.timezone.utc).year
    xs, usage, final = [], [], {}
    for line in path.read_text(errors="replace").splitlines():
        watermark_match = watermark_pattern.search(line)
        if watermark_match:
            final["eviction_high_watermark_fraction"] = float(
                watermark_match.group(1)
            )
        stamp_match = timestamp_pattern.search(line)
        storage_match = storage_pattern.search(line)
        if not stamp_match or not storage_match:
            continue
        stamp = dt.datetime.strptime(
            f"{year}{stamp_match.group(1)} {stamp_match.group(2)}", "%Y%m%d %H:%M:%S.%f"
        ).replace(tzinfo=dt.timezone.utc).timestamp()
        groups = storage_match.groups()
        final.update({
            "resident_gb": float(groups[0]),
            "capacity_gb": float(groups[1]),
            "resident_fraction": float(groups[2]) / 100.0,
            "eviction_successes": int(groups[3]),
            "eviction_attempts": int(groups[4]),
            "allocation_failures": int(groups[5]),
            "evicted_keys": int(groups[6]),
            "evicted_gb": float(groups[7]),
        })
        if start <= stamp <= end:
            xs.append(stamp - origin)
            usage.append(float(groups[2]))
    if usage:
        final["measurement_resident_mean_fraction"] = statistics.fmean(usage) / 100.0
        final["measurement_resident_max_fraction"] = max(usage) / 100.0
    return xs, usage, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    boundary = json.loads((args.run_dir / "closed_loop_boundaries.json").read_text())
    start = boundary["measurement_start_wall"]
    end = boundary["measurement_end_wall"]
    duration = end - start
    origin = boundary["origin_wall"]
    raw_metrics = read_jsonl(args.run_dir / "engine_metrics.jsonl")
    throughput = read_jsonl(args.run_dir / "engine_throughput_2s.jsonl")
    requests = read_jsonl(args.run_dir / "requests.jsonl")
    completed = [row for row in requests if start <= row["finished_ts"] <= end and not row.get("error")]
    decode_engine_count = max(
        (
            int(row.get("engine_count", 1))
            for row in raw_metrics
            if row.get("role") == "decode"
        ),
        default=1,
    )

    p_compute, counter_seconds = counter_delta(raw_metrics, "prefill", P_COMPUTE, start, end)
    # SGLang omits this series entirely when RadixCache is disabled.  That is
    # a valid zero-cache-hit result for the no-reverse baseline, not a failed
    # experiment.
    p_cache, _ = counter_delta(
        raw_metrics, "prefill", P_CACHE, start, end, optional=True
    )
    d_decode, _ = counter_delta(raw_metrics, "decode", D_DECODE, start, end)
    p_gpu, _ = counter_delta(raw_metrics, "prefill", P_GPU, start, end)
    d_gpu, _ = counter_delta(raw_metrics, "decode", D_GPU, start, end)
    completed_count = (
        boundary["state_at_measurement_end"]["successes"]
        - boundary["state_at_measurement_start"]["successes"]
    )
    reuse = request_reuse(completed)
    parent_loss = reuse["page_aligned_extra_prefill_from_parent_kv_loss"]
    reuse["page_aligned_extra_prefill_from_parent_kv_loss_per_second"] = (
        parent_loss / duration if duration else 0.0
    )
    reuse["page_aligned_extra_prefill_from_parent_kv_loss_per_agent"] = (
        parent_loss / len(completed) if completed else 0.0
    )
    reuse["page_aligned_extra_prefill_fraction"] = (
        parent_loss / reuse["actual_prefill_tokens"]
        if reuse["actual_prefill_tokens"] else 0.0
    )
    reuse["counterfactual_prefill_without_parent_kv_loss"] = (
        reuse["actual_prefill_tokens"] - parent_loss
    )
    terminal_repairs = terminal_repair_summary(completed, duration)
    moon_x, moon_usage, mooncake = mooncake_series(
        args.run_dir / "logs" / "mooncake-master.log", origin, start, end
    )

    summary = {
        "status": boundary["status"],
        "warmup_seconds": boundary["warmup_seconds"],
        "measurement_seconds": duration,
        "concurrency": boundary["state_at_measurement_start"]["active"],
        "decode_gpus": decode_engine_count,
        "completed_agents": completed_count,
        "completion_rps": completed_count / duration,
        "math_completed": sum(row["task_type"] == "math" for row in completed),
        "qa_completed": sum(row["task_type"] == "qa" for row in completed),
        "prefill_compute_tps": p_compute / counter_seconds,
        "prefill_cache_hit_tps": p_cache / counter_seconds,
        "decode_tps": d_decode / counter_seconds,
        "decode_tps_per_gpu": d_decode / counter_seconds / decode_engine_count,
        "prefill_gpu_busy_fraction": p_gpu / counter_seconds,
        "decode_gpu_busy_fraction": d_gpu / counter_seconds,
        "decode_gpu_busy_fraction_per_gpu": d_gpu / counter_seconds / decode_engine_count,
        "prefill_gpu_active_tps": p_compute / p_gpu if p_gpu else 0.0,
        "decode_gpu_active_tps": d_decode / d_gpu if d_gpu else 0.0,
        "prefill_compute_tokens_per_completed_agent": p_compute / completed_count,
        "decode_tokens_per_completed_agent": d_decode / completed_count,
        "ideal_decode_gpus_per_prefill_gpu_from_gpu_time": d_gpu / p_gpu if p_gpu else 0.0,
        "completion_weighted_reuse": reuse,
        "terminal_repairs": terminal_repairs,
        "mooncake": mooncake,
    }
    (args.run_dir / "offload_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    p_rows = [row for row in throughput if row["role"] == "prefill" and start <= row["ts"] <= end]
    d_rows = [row for row in throughput if row["role"] == "decode" and start <= row["ts"] <= end]
    px = [row["ts"] - start for row in p_rows]
    dx = [row["ts"] - start for row in d_rows]
    p_tps = moving_average([row["prompt_tokens_per_second"] for row in p_rows])
    d_tps = moving_average([row["generation_tokens_per_second"] for row in d_rows])
    p_gpu_x, p_gpu_rate = counter_rates(raw_metrics, "prefill", P_GPU, start, start, end)
    d_gpu_x, d_gpu_rate = counter_rates(raw_metrics, "decode", D_GPU, start, start, end)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(px, p_tps, color="#e69f00", label="Prefill compute TPS (60s mean)")
    axes[0].plot(dx, d_tps, color="#0072b2", label="Decode TPS (60s mean)")
    axes[0].set_ylabel("tokens/s")
    axes[0].legend(loc="upper right")

    axes[1].plot(dx, [row["running_requests"] for row in d_rows], color="#0072b2", label="D running")
    axes[1].plot(dx, [row["decode_transfer_requests"] for row in d_rows], color="#d55e00", label="D transfer")
    axes[1].plot(px, [row["prefill_inflight_requests"] for row in p_rows], color="#e69f00", label="P inflight")
    axes[1].set_ylabel("requests")
    axes[1].legend(loc="center right", ncol=3)

    axes[2].plot(p_gpu_x, moving_average(p_gpu_rate), color="#e69f00", label="P GPU busy")
    axes[2].plot(d_gpu_x, moving_average(d_gpu_rate), color="#0072b2", label="D GPU busy")
    axes[2].set_ylabel("GPU s / wall s")
    axes[2].set_ylim(-0.03, max(1.08, decode_engine_count * 1.08))
    axes[2].legend(loc="center right")

    axes[3].plot([x - (start - origin) for x in moon_x], moon_usage, color="#009e73")
    eviction_watermark = 100.0 * mooncake.get(
        "eviction_high_watermark_fraction", 0.85
    )
    axes[3].axhline(
        eviction_watermark,
        color="#cc79a7",
        linestyle="--",
        label=f"eviction watermark ({eviction_watermark:.0f}%)",
    )
    axes[3].set_ylabel("Mooncake used (%)")
    axes[3].set_xlabel("Measurement elapsed seconds")
    axes[3].legend(loc="lower right")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"1P:{decode_engine_count}D c{summary['concurrency']}, page=64, Decode KV offload"
        f" — {completed_count} agents / {duration:.0f}s"
    )
    fig.tight_layout()
    fig.savefig(args.run_dir / "steady_offload_analysis.png", dpi=180)


if __name__ == "__main__":
    main()
