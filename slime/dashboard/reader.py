from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path


class DashboardReader:
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

    def snapshot(self, minutes: float = 30.0, aggregate_engine: bool = False) -> dict:
        with self._lock:
            now = time.time()
            since = now - max(1.0, min(minutes, 240.0)) * 60.0
            engine = self.records("engine", since)
            if aggregate_engine:
                engine = self.aggregate_engine(engine)
            return {
                "meta": self.meta(),
                "now": now,
                "since": since,
                "gpu": self.records("gpu", since),
                "engine": engine,
                "metrics": self.records("metrics", since, limit=20_000),
                "trace": self.records("trace", since),
            }
