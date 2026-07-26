from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path


class DashboardReader:
    DISPLAY_ENGINE_METRICS = {
        "sglang_num_running_reqs",
        "sglang_num_queue_reqs",
        "sglang_gen_throughput",
        "sglang_token_usage",
        "sglang_cache_hit_rate",
    }

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = threading.RLock()
        self._records: dict[str, deque[dict]] = {}
        self._positions: dict[str, dict[Path, tuple[int, int, int]]] = {}

    def meta(self) -> dict:
        path = self.directory / "meta.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def records(self, stream: str, since: float, limit: int = 100_000) -> list[dict]:
        stream_dir = self.directory / stream
        if not stream_dir.is_dir():
            return []
        with self._lock:
            output = self._records.setdefault(stream, deque(maxlen=max(100_000, limit)))
            positions = self._positions.setdefault(stream, {})
            for path in sorted(stream_dir.glob("*.jsonl")):
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                previous = positions.get(path)
                offset = previous[2] if previous and previous[:2] == identity and stat.st_size >= previous[2] else 0
                if stat.st_size == offset:
                    continue

                with path.open("rb") as handle:
                    handle.seek(offset)
                    while True:
                        line_start = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            handle.seek(line_start)
                            break
                        try:
                            output.append(json.loads(line))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                    positions[path] = (*identity, handle.tell())

            matching = [record for record in output if float(record.get("ts", 0)) >= since]
            return matching[-limit:]

    @staticmethod
    def aggregate_engine(records: list[dict]) -> list[dict]:
        aggregated: dict[tuple[str, float], float] = {}
        for record in records:
            key = (str(record.get("metric", "")), float(record.get("ts", 0)))
            aggregated[key] = aggregated.get(key, 0.0) + float(record.get("value", 0))
        return [
            {
                "ts": timestamp,
                "metric": metric,
                "worker_addr": "aggregate",
                "labels": {"scope": "all_workers"},
                "value": value,
            }
            for (metric, timestamp), value in aggregated.items()
        ]

    @staticmethod
    def downsample(records: list[dict], key_fields: tuple[str, ...], max_points: int = 600) -> list[dict]:
        grouped: dict[tuple, list[dict]] = {}
        for record in records:
            key = tuple(record.get(field) for field in key_fields)
            grouped.setdefault(key, []).append(record)
        output = []
        for rows in grouped.values():
            if len(rows) <= max_points:
                output.extend(rows)
                continue
            stride = math.ceil(len(rows) / max_points)
            sampled = rows[::stride]
            if sampled[-1] is not rows[-1]:
                sampled.append(rows[-1])
            output.extend(sampled)
        return sorted(output, key=lambda record: float(record.get("ts", 0)))

    def trace_summary(self, since: float, now: float, recent_limit: int = 500) -> dict:
        events = self.records("trace", since, limit=200_000)
        wanted = {"generation_turn", "tool_call", "reward_model"}
        starts = {}
        spans = []
        for event in events:
            name = event.get("name")
            if name not in wanted:
                continue
            span_id = event.get("span_id")
            if event.get("type") == "span_start" and span_id:
                starts[span_id] = event
            elif event.get("type") == "span_end" and span_id in starts:
                start = starts.pop(span_id)
                attrs = dict(start.get("attrs") or {})
                attrs.update(event.get("attrs") or {})
                spans.append(
                    {
                        "ts": float(start.get("ts", 0)),
                        "end": float(event.get("ts", 0)),
                        "sample": event.get("sample_id"),
                        "group": event.get("group_id"),
                        "name": name,
                        "duration": max(0.0, float(event.get("ts", 0)) - float(start.get("ts", 0))),
                        "attrs": attrs,
                        "ongoing": False,
                    }
                )
        for start in starts.values():
            spans.append(
                {
                    "ts": float(start.get("ts", 0)),
                    "end": now,
                    "sample": start.get("sample_id"),
                    "group": start.get("group_id"),
                    "name": start.get("name"),
                    "duration": max(0.0, now - float(start.get("ts", 0))),
                    "attrs": dict(start.get("attrs") or {}),
                    "ongoing": True,
                }
            )

        generations = [span for span in spans if span["name"] == "generation_turn"]
        tools = [
            span
            for span in spans
            if span["name"] == "tool_call"
            and (span["ongoing"] or span["attrs"].get("is_tool_call") is True)
        ]

        def average(rows):
            return sum(row["duration"] for row in rows) / len(rows) if rows else 0.0

        tool_series = {}
        for domain in ("math", "qa"):
            changes = []
            for span in tools:
                if span["attrs"].get("task_type") != domain:
                    continue
                changes.append((span["ts"], 1))
                changes.append((span["end"], -1))
            changes.sort(key=lambda item: (item[0], item[1]))
            running = 0
            points = []
            for timestamp, delta in changes:
                points.append([timestamp, running])
                running += delta
                points.append([timestamp, running])
            if len(points) > 1200:
                stride = math.ceil(len(points) / 1200)
                points = points[::stride] + ([points[-1]] if points[-1] != points[::stride][-1] else [])
            tool_series[domain] = points

        return {
            "spans": sorted(spans, key=lambda span: span["ts"], reverse=True)[:recent_limit],
            "tool_series": tool_series,
            "totals": {
                "generation_turns": len(generations),
                "mean_generation_seconds": average(generations),
                "tool_calls": sum(int(span["attrs"].get("tool_calls", 1) or 0) for span in tools),
                "mean_tool_seconds": average(tools),
                "max_tool_seconds": max((span["duration"] for span in tools), default=0.0),
            },
        }

    def snapshot(self, minutes: float = 30.0, aggregate_engine: bool = False, include_raw_trace: bool = False) -> dict:
        with self._lock:
            now = time.time()
            since = now - max(1.0, min(minutes, 240.0)) * 60.0
            engine = [
                record
                for record in self.records("engine", since)
                if record.get("metric") in self.DISPLAY_ENGINE_METRICS
            ]
            if aggregate_engine:
                engine = self.aggregate_engine(engine)
            engine = self.downsample(engine, ("worker_addr", "metric"), max_points=240)
            gpu = self.downsample(self.records("gpu", since), ("node", "gpu"), max_points=360)
            return {
                "meta": self.meta(),
                "now": now,
                "since": since,
                "gpu": gpu,
                "engine": engine,
                "metrics": self.records("metrics", since, limit=20_000),
                "trace": self.records("trace", since) if include_raw_trace else [],
                "trace_summary": self.trace_summary(since, now),
            }
