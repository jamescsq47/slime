"""Summarize request-generation KV lifecycle events from P/D logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EVENT_RE = re.compile(r"AgenticKV\s+(?P<event>[a-z_]+)\s+(?P<fields>.*)$")
FIELD_RE = re.compile(r"([a-z_]+)=([^\s]+)")


def parse_logs(log_dir: Path) -> dict:
    events: list[tuple[str, dict[str, str]]] = []
    for path in sorted(log_dir.glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            match = EVENT_RE.search(line)
            if match is None:
                continue
            fields = dict(FIELD_RE.findall(match.group("fields")))
            fields["log"] = path.name
            events.append((match.group("event"), fields))

    counts = Counter(event for event, _ in events)
    offers = counts["direct_offer"]
    # Current D-side success is recorded when the reverse NIXL send completes.
    # Older logs used ``direct_load_complete``, so retain that as a fallback.
    direct_hits = counts["direct_send_complete"] or counts["direct_load_complete"]
    fallbacks = counts["direct_fallback"]
    direct_pending = max(0, offers - direct_hits - fallbacks)

    def values(event_name: str, field: str) -> list[float]:
        result = []
        for event, fields in events:
            if event == event_name and field in fields:
                try:
                    result.append(float(fields[field]))
                except ValueError:
                    pass
        return result

    def mean(items: list[float]) -> float | None:
        return sum(items) / len(items) if items else None

    ready_bytes = values("mooncake_ready", "bytes")
    return {
        "events": dict(sorted(counts.items())),
        "direct": {
            "offers": offers,
            "hits": direct_hits,
            "fallbacks": fallbacks,
            "pending_at_log_end": direct_pending,
            "hit_rate_per_offer": direct_hits / offers if offers else None,
            "mean_send_complete_seconds": mean(
                values("direct_send_complete", "elapsed_s")
            ),
            "mean_fallback_seconds": mean(values("direct_fallback", "elapsed_s")),
        },
        "mooncake": {
            "ready_snapshots": counts["mooncake_ready"],
            "consumed_snapshots": counts["mooncake_consumed"],
            "mean_snapshot_gib": (
                mean(ready_bytes) / (1024**3) if ready_bytes else None
            ),
            "mean_ready_seconds": mean(values("mooncake_ready", "elapsed_s")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse_logs(args.log_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
