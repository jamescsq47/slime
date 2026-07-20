from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


PARTITIONED_STREAMS = frozenset({"gpu", "engine", "trace", "metrics"})


def _hour_key(timestamp: float) -> str:
    return time.strftime("%Y%m%d_%H", time.gmtime(timestamp))


class JsonlStore:
    """Append-only, hourly-partitioned telemetry store.

    One collector owns this object, so append/flush synchronization stays in
    the collector. Failed flushes leave records buffered for a later retry.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "meta.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)

    def append(self, stream: str, record: dict[str, Any]) -> None:
        if stream not in PARTITIONED_STREAMS:
            raise ValueError(f"unsupported dashboard stream: {stream}")
        self._buffers[stream].append(record)

    def buffered_count(self, stream: str) -> int:
        return len(self._buffers[stream])

    def drop_oldest(self, stream: str, keep_ratio: float = 0.9) -> int:
        buffer = self._buffers[stream]
        if not buffer:
            return 0
        dropped = max(1, int(len(buffer) * (1.0 - keep_ratio)))
        del buffer[:dropped]
        return dropped

    def flush(self) -> int:
        self.directory.mkdir(parents=True, exist_ok=True)
        written = 0
        for stream, records in self._buffers.items():
            if not records:
                continue
            partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                partitions[_hour_key(float(record.get("ts", time.time())))].append(record)

            for hour, partition_records in partitions.items():
                stream_dir = self.directory / stream
                stream_dir.mkdir(parents=True, exist_ok=True)
                path = stream_dir / f"{hour}.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    for record in partition_records:
                        handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
                written += len(partition_records)
            records.clear()
        return written
