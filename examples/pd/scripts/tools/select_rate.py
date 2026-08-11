#!/usr/bin/env python3
"""Select the highest tested rate whose p95 queue delay stays small."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(path.read_text()) | {"summary_path": str(path)} for path in args.summaries]
    rows.sort(key=lambda row: row["configured_arrival_rate_rps"])
    healthy = []
    for row in rows:
        queue_p95 = row["queue_delay_seconds"]["p95"] or 0.0
        latency_p50 = row["agent_latency_seconds"]["p50"] or 0.0
        engine_queue = row.get("max_engine_queue_requests") or 0.0
        threshold = max(1.0, 0.1 * latency_p50)
        row["queue_delay_threshold_seconds"] = threshold
        row["rate_is_healthy"] = row["failed"] == 0 and queue_p95 <= threshold and engine_queue <= 1
        if row["rate_is_healthy"]:
            healthy.append(row)
    selected = healthy[-1] if healthy else rows[0]
    result = {
        "recommended_arrival_rate_rps": selected["configured_arrival_rate_rps"],
        "selection_rule": (
            "highest rate with failed=0, max_engine_queue_requests<=1, and "
            "queue_delay_p95<=max(1s, 10% of agent_latency_p50)"
        ),
        "runs": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
