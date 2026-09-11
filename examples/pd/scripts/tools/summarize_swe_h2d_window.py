#!/usr/bin/env python3
"""Read-only H2D diagnostics for a completed SWE run; emit JSON to stdout.

Select H2D completions in [first agent start + 300, +1500). Arrival/ready
timestamps are exact, but worker log timestamps have one-second resolution.
These are event-cohort statistics, not steady-state capacity measurements.
"""

import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def quantile(q):
        x = (len(values) - 1) * q
        i = int(x)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)

    return dict(count=len(values), minimum=values[0], mean=statistics.mean(values), p50=quantile(.5),
                p90=quantile(.9), maximum=values[-1])


def analyze(path):
    first = None
    command_seconds = []
    commands_by_snapshot = {}
    with (path / "requests.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            value = row.get("started_ts")
            if value is not None:
                first = value if first is None else min(first, value)
            for turn in row.get("metadata", {}).get("turn_metrics", []):
                seconds = turn.get("command_seconds")
                if seconds is not None and turn.get("command"):
                    command_seconds.append(float(seconds))
                    request_id = row.get("metadata", {}).get("agentic_request_id")
                    if request_id:
                        # Harness metrics use turn+1; lifecycle generation is zero-based.
                        commands_by_snapshot[f"{request_id}:{int(turn['turn']) - 1}"] = float(seconds)
    assert first is not None
    begin, end = first + 300, first + 1500
    arrivals = {}
    for file in (path / "control-final/ready/early-claims/arrivals").glob("*.json"):
        row = json.loads(file.read_text())
        arrivals[row["snapshot_id"]] = row["arrived_at"]
    mid_commands = [commands_by_snapshot[snapshot] for snapshot, timestamp in arrivals.items()
                    if begin <= timestamp < end and snapshot in commands_by_snapshot]
    events = {}
    controls = {}
    direct = set()
    fast_seen = set()
    for file in sorted((path / "logs").glob("*.log*")):
        if not file.name.startswith(("prefill-", "decode-")):
            continue
        with file.open(errors="replace") as f:
            for line in f:
                if "AgenticKV " not in line and "Agentic P async control stats" not in line:
                    continue
                if not re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", line):
                    continue
                timestamp = dt.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
                fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
                if "Agentic P async control stats" in line:
                    if begin <= timestamp < end:
                        controls.setdefault(file.name, []).append(fields)
                    continue
                snapshot = fields.get("snapshot")
                if not snapshot:
                    continue
                if "AgenticKV direct_rank_send_complete " in line and begin <= timestamp < end:
                    direct.add(snapshot)
                if "AgenticKV fast_arrival_seen " in line and begin <= timestamp < end:
                    fast_seen.add(snapshot)
                stage = None
                for name in ("shared_host_d2h_complete", "shared_host_h2d_start", "shared_host_h2d_complete"):
                    if f"AgenticKV {name} " in line:
                        stage = name
                        break
                if stage:
                    events.setdefault(snapshot, {})[stage] = dict(fields, timestamp=timestamp)
    selected = {k: v for k, v in events.items()
                if "shared_host_h2d_complete" in v
                and begin <= v["shared_host_h2d_complete"]["timestamp"] < end}
    waits, gpu_ms, wall_ms, cpu_ms, sizes, gpu_rates = [], [], [], [], [], []
    negative_waits = 0
    missing = []
    for snapshot, event in selected.items():
        done = event["shared_host_h2d_complete"]
        start = event.get("shared_host_h2d_start")
        durable = event.get("shared_host_d2h_complete")
        gpu_ms.append(float(done["elapsed_ms"]))
        wall_ms.append(float(done["wall_ms"]))
        cpu_ms.append(float(done["host_copy_ms"]))
        gpu_rates.append(float(done["gib_per_s"]))
        if start:
            sizes.append(int(start["bytes"]))
        if start and durable and snapshot in arrivals:
            eligible = max(arrivals[snapshot], float(durable["completed_at"]))
            wait = start["timestamp"] - eligible
            negative_waits += wait < 0
            waits.append(wait)  # Do not hide one-second log quantization by clipping.
        else:
            missing.append(snapshot)
    control_result = {}
    for worker, rows in controls.items():
        lanes = [int(r["h2d_lanes"].split("/")[0]) for r in rows]
        capacity = [int(r["h2d_lanes"].split("/")[1]) for r in rows]
        control_result[worker] = {
            "samples": len(rows), "reserved_lanes_sample_mean": statistics.mean(lanes),
            "lane_full_sample_fraction": statistics.mean(a == b for a, b in zip(lanes, capacity)),
            "host_ready_sample_mean": statistics.mean(int(r["host_ready"]) for r in rows),
        }
    return dict(run=str(path), window=[begin, end],
                h2d_completions=len(selected), direct_completions=len(direct),
                fast_arrival_seen=len(fast_seen),
                whole_run_command_seconds=stats(command_seconds),
                whole_run_command_under_1s_fraction=(statistics.mean(v <= 1 for v in command_seconds) if command_seconds else None),
                window_arrival_command_seconds=stats(mid_commands),
                window_arrival_command_under_1s_fraction=(statistics.mean(v <= 1 for v in mid_commands) if mid_commands else None),
                eligible_to_first_h2d_log_seconds=stats(waits),
                unmatched_wait_count=len(missing), negative_quantized_wait_count=negative_waits,
                wait_less_than_minus_one_second=sum(v < -1 for v in waits),
                gpu_event_ms=stats(gpu_ms), io_wall_ms=stats(wall_ms), host_copy_ms=stats(cpu_ms),
                snapshot_gib=stats([n / 2**30 for n in sizes]),
                completed_bytes_per_window_gib_s=sum(sizes) / 2**30 / 1200,
                gpu_event_gib_s=stats(gpu_rates), controls=control_result,
                note="TP=1 only. Wait has <1s downward timestamp quantization; control stats are ~30s snapshots, not continuous DMA occupancy. Throughput counts whole snapshots finishing in the window. window_arrival_command statistics cover retained arrival markers only, not all window tools; whole_run_command covers all recorded shell calls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([analyze(path) for path in args.run], indent=2))
