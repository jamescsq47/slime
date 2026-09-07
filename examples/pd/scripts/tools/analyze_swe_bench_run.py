#!/usr/bin/env python3
"""Summarize full SWE-bench trajectories and Docker/verifier timing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Iterable[Any]) -> dict[str, float | int | None]:
    clean = [item for value in values if (item := number(value)) is not None]
    return {
        "count": len(clean),
        "sum": sum(clean) if clean else None,
        "mean": statistics.fmean(clean) if clean else None,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean) if clean else None,
    }


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("metadata") or {}


def harness_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    turns = metadata(row).get("turn_metrics") or []
    return turns if isinstance(turns, list) else []


def sandbox(row: dict[str, Any]) -> dict[str, Any]:
    value = metadata(row).get("sandbox_metrics") or {}
    return value if isinstance(value, dict) else {}


def exec_phase_seconds(row: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for call in sandbox(row).get("exec_calls") or []:
        result[str(call.get("phase") or "unknown")] += float(
            call.get("duration_seconds") or 0
        )
    return dict(result)


def per_request(row: dict[str, Any]) -> dict[str, Any]:
    meta = metadata(row)
    turns = harness_turns(row)
    box = sandbox(row)
    phases = exec_phase_seconds(row)
    verifier = meta.get("swe_bench_verifier") or {}
    trajectory = meta.get("mini_swe_agent_trajectory") or {}
    trajectory_info = trajectory.get("info") or {}
    agent_exit_status = str(trajectory_info.get("exit_status") or "")
    raw_status = row.get("status")
    effective_status = raw_status
    effective_stop_reason = meta.get("stop_reason")
    environment_error = str(meta.get("environment_error") or "")
    if raw_status == "failed" and environment_error.startswith(
        "mini-SWE-agent history reached the "
    ):
        effective_status = "truncated"
        effective_stop_reason = "ContextWindowExceeded"
    elif (
        raw_status == "failed"
        and agent_exit_status in {"TimeExceeded", "LimitsExceeded", "RepeatedFormatError"}
        and not str(meta.get("model_patch") or "")
        and "patch capture failed" in environment_error
    ):
        # Older runs captured a diagnostic workspace diff after a valid
        # no-submission agent exit.  A capture failure must not replace the
        # official zero-score terminal outcome.
        effective_status = "truncated"
        effective_stop_reason = agent_exit_status
    resolved = verifier.get("resolved")
    verifier_status = verifier.get("status")
    if (
        trajectory
        and effective_status == "truncated"
        and agent_exit_status != "Submitted"
    ):
        resolved = False
        verifier_status = verifier_status or "not_submitted"
    model_patch = str(meta.get("model_patch") or "")
    uploads = box.get("uploads") or []
    output_tokens = sum(int(turn.get("output_tokens") or 0) for turn in turns)
    observation_tokens = sum(int(turn.get("observation_tokens") or 0) for turn in turns)
    input_tokens = sum(int(turn.get("input_tokens") or 0) for turn in turns)
    cached_tokens = sum(int(turn.get("cached_tokens") or 0) for turn in turns)
    docker_exec_seconds = sum(phases.values())
    docker_upload_seconds = sum(float(item.get("duration_seconds") or 0) for item in uploads)
    docker_lifecycle_seconds = sum(
        float(box.get(key) or 0)
        for key in (
            "image_inspect_seconds",
            "container_start_seconds",
            "container_close_seconds",
        )
    )
    return {
        "sample_index": row.get("sample_index"),
        "instance_id": meta.get("instance_id") or meta.get("task_id"),
        "status": effective_status,
        "raw_status": raw_status,
        "stop_reason": effective_stop_reason,
        "agent_exit_status": agent_exit_status,
        "resolved": resolved,
        "reward": verifier.get("reward"),
        "verifier_status": verifier_status,
        "agent_latency_seconds": row.get("agent_latency_seconds"),
        "sample_time_seconds": meta.get("sample_time"),
        "model_seconds": meta.get("model_time"),
        "tool_seconds": meta.get("tool_time"),
        "verifier_queue_seconds": meta.get("verifier_queue_seconds"),
        "verifier_seconds": verifier.get("duration_seconds"),
        "docker_image_inspect_seconds": box.get("image_inspect_seconds"),
        "docker_container_start_seconds": box.get("container_start_seconds"),
        "docker_container_close_seconds": box.get("container_close_seconds"),
        "docker_exec_seconds": docker_exec_seconds,
        "docker_upload_seconds": docker_upload_seconds,
        "docker_accounted_seconds": (
            docker_lifecycle_seconds + docker_exec_seconds + docker_upload_seconds
        ),
        "docker_agent_tool_seconds": phases.get("agent_tool", 0.0),
        "docker_baseline_seconds": phases.get("baseline", 0.0),
        "docker_patch_capture_seconds": phases.get("patch_capture", 0.0),
        "docker_verifier_exec_seconds": phases.get("verifier", 0.0),
        "docker_setup_seconds": phases.get("sandbox_setup", 0.0),
        "docker_exec_calls": len(box.get("exec_calls") or []),
        "docker_upload_calls": len(uploads),
        "image_warm": box.get("image_present_before_start"),
        "turns": len(turns),
        "shell_calls": meta.get("shell_call_count"),
        "cumulative_model_input_tokens": input_tokens,
        "model_output_tokens": output_tokens,
        "tool_observation_tokens": observation_tokens,
        "cached_input_tokens": cached_tokens,
        "trajectory_tokens": row.get("response_tokens"),
        "patch_chars": meta.get("model_patch_chars"),
        "patch_bytes": len(model_patch.encode("utf-8")),
        "mini_swe_trajectory_recorded": bool(
            meta.get("mini_swe_agent_trajectory")
            and meta.get("mini_swe_agent_trajectory_format")
        ),
        "mini_swe_agent_version": meta.get("mini_swe_agent_version"),
        "agent_harness": meta.get("agent_harness"),
        "openenv_trajectory_recorded": bool(meta.get("openenv_trajectory")),
        "error": row.get("error") or meta.get("environment_error"),
    }


METRICS = (
    "agent_latency_seconds",
    "sample_time_seconds",
    "model_seconds",
    "tool_seconds",
    "verifier_queue_seconds",
    "verifier_seconds",
    "docker_image_inspect_seconds",
    "docker_container_start_seconds",
    "docker_container_close_seconds",
    "docker_exec_seconds",
    "docker_upload_seconds",
    "docker_accounted_seconds",
    "docker_agent_tool_seconds",
    "docker_baseline_seconds",
    "docker_patch_capture_seconds",
    "docker_verifier_exec_seconds",
    "docker_setup_seconds",
    "docker_exec_calls",
    "docker_upload_calls",
    "turns",
    "shell_calls",
    "cumulative_model_input_tokens",
    "model_output_tokens",
    "tool_observation_tokens",
    "cached_input_tokens",
    "trajectory_tokens",
    "patch_chars",
    "patch_bytes",
)


def render_markdown(summary: dict[str, Any]) -> str:
    overview = summary["overview"]
    lines = [
        "# SWE-bench full-run profile",
        "",
        "## Outcome",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "requests",
        "completed",
        "failed",
        "raw_failed",
        "truncated",
        "resolved",
        "unresolved",
        "verifier_infrastructure_errors",
        "warm_image_requests",
        "mini_swe_trajectories",
        "openenv_trajectories",
        "run_wall_seconds",
        "request_per_second",
        "agent_per_second",
    ):
        value = overview.get(key)
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Timing and length distributions",
            "",
            "| Metric | Mean | P50 | P90 | P95 | P99 | Max | Sum |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, values in summary["distributions"].items():
        def fmt(name: str) -> str:
            value = values.get(name)
            return "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {key} | {fmt('mean')} | {fmt('p50')} | {fmt('p90')} | "
            f"{fmt('p95')} | {fmt('p99')} | {fmt('max')} | {fmt('sum')} |"
        )
    lines.extend(
        [
            "",
            "## Categories",
            "",
            f"- status: `{json.dumps(summary['status_counts'], ensure_ascii=False)}`",
            f"- raw status: `{json.dumps(summary['raw_status_counts'], ensure_ascii=False)}`",
            f"- stop reason: `{json.dumps(summary['stop_reason_counts'], ensure_ascii=False)}`",
            f"- verifier status: `{json.dumps(summary['verifier_status_counts'], ensure_ascii=False)}`",
            f"- mini-SWE-agent version: `{json.dumps(summary['mini_swe_agent_version_counts'], ensure_ascii=False)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    request_path = args.run_dir / "requests.jsonl"
    rows = read_jsonl(request_path)
    profiles = [per_request(row) for row in rows]
    starts = [number(row.get("started_ts")) for row in rows]
    finishes = [number(row.get("finished_ts")) for row in rows]
    clean_starts = [value for value in starts if value is not None]
    clean_finishes = [value for value in finishes if value is not None]
    run_wall = (
        max(clean_finishes) - min(clean_starts)
        if clean_starts and clean_finishes
        else None
    )
    completed = sum(row.get("status") == "completed" for row in profiles)
    resolved = sum(row.get("resolved") is True for row in profiles)
    unresolved = sum(row.get("resolved") is False for row in profiles)
    infra = sum(row.get("verifier_status") == "infrastructure_error" for row in profiles)
    summary = {
        "overview": {
            "requests": len(profiles),
            "completed": completed,
            "failed": sum(row.get("status") == "failed" for row in profiles),
            "raw_failed": sum(row.get("raw_status") == "failed" for row in profiles),
            "truncated": sum(row.get("status") == "truncated" for row in profiles),
            "resolved": resolved,
            "unresolved": unresolved,
            "verifier_infrastructure_errors": infra,
            "warm_image_requests": sum(row.get("image_warm") is True for row in profiles),
            "mini_swe_trajectories": sum(
                row.get("mini_swe_trajectory_recorded") is True for row in profiles
            ),
            "openenv_trajectories": sum(
                row.get("openenv_trajectory_recorded") is True for row in profiles
            ),
            "run_wall_seconds": run_wall,
            "request_per_second": len(profiles) / run_wall if run_wall and run_wall > 0 else None,
            "agent_per_second": completed / run_wall if run_wall and run_wall > 0 else None,
        },
        "status_counts": dict(Counter(str(row.get("status")) for row in profiles)),
        "raw_status_counts": dict(
            Counter(str(row.get("raw_status")) for row in profiles)
        ),
        "stop_reason_counts": dict(
            Counter(str(row.get("stop_reason")) for row in profiles)
        ),
        "verifier_status_counts": dict(
            Counter(str(row.get("verifier_status")) for row in profiles)
        ),
        "mini_swe_agent_version_counts": dict(
            Counter(str(row.get("mini_swe_agent_version")) for row in profiles)
        ),
        "distributions": {
            key: distribution(row.get(key) for row in profiles) for key in METRICS
        },
    }
    json_path = args.run_dir / "swe_bench_profile_summary.json"
    md_path = args.run_dir / "swe_bench_profile_summary.md"
    csv_path = args.run_dir / "swe_bench_request_profile.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(render_markdown(summary))
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(profiles[0]) if profiles else [])
        writer.writeheader()
        writer.writerows(profiles)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
