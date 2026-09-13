"""Summarize physical NUMA pool counters, never per-P virtual capacities."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(rows, start, end):
    weights = []
    for index, row in enumerate(rows):
        next_time = rows[index + 1]["time"] if index + 1 < len(rows) else end
        seconds = max(0, min(end, next_time) - max(start, row["time"]))
        if seconds:
            weights.append((row, seconds))
    duration = sum(weight for _, weight in weights)
    if not duration:
        return {}
    result = {"covered_seconds": duration}
    for direction in ("d2p", "p2d", "total"):
        values = [(sum(row["used"].values()) if direction == "total"
                   else row["used"][direction]) / 1024**3 for row, _ in weights]
        result[direction] = dict(
            mean_gib=sum(v * w for v, (_, w) in zip(values, weights)) / duration,
            peak_gib=max(values), min_gib=min(values), first_gib=values[0], last_gib=values[-1])
    result["leases_peak"] = max(row["leases"] for row, _ in weights)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    boundaries = json.loads((args.run_dir / "closed_loop_boundaries.json").read_text())
    start, end = (boundaries[key] for key in
                  ("measurement_start_wall", "measurement_end_wall"))
    pools = {}
    for path in sorted((args.run_dir / "logs").glob("host-pool-*.log")):
        rows = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "numa_host_pool_usage":
                rows.append(row)
        if rows:
            pools[rows[0]["numa"]] = rows
    if not pools:
        raise RuntimeError("no physical NUMA pool samples")
    fig, axes = plt.subplots(len(pools), 1, figsize=(12, 4 * len(pools)), squeeze=False)
    result = {}
    for axis, (node, rows) in zip(axes[:, 0], pools.items()):
        result[node] = {"measurement": summarize(rows, start, end), "segments": []}
        segment_start = start
        while end - segment_start > 0.01:
            segment_end = min(end, segment_start + 300)
            result[node]["segments"].append(dict(
                start_s=segment_start - start, end_s=segment_end - start,
                **summarize(rows, segment_start, segment_end)))
            segment_start = segment_end
        selected = [row for row in rows if start <= row["time"] <= end]
        for direction, label in (("d2p", "D to P"), ("p2d", "P to D"), ("total", "Total")):
            axis.plot([row["time"] - start for row in selected],
                      [(sum(row["used"].values()) if direction == "total" else
                        row["used"][direction]) / 1024**3 for row in selected], label=label)
        capacity = rows[0]["capacity"] / 1024**3
        axis.axhline(capacity, color="black", linestyle="--", label="Physical capacity")
        axis.set(title=f"NUMA {node}: physical Host pool", ylabel="Used GiB",
                 xlabel="Measurement seconds", ylim=(0, capacity * 1.05))
        axis.legend(loc="upper left", ncol=4)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.run_dir / "numa_host_usage.png", dpi=150)
    (args.run_dir / "numa_host_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
