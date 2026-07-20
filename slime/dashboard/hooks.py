from __future__ import annotations

import logging
import threading
import time

from slime.dashboard.logging_utils import RateLimitedWarner

logger = logging.getLogger(__name__)


class TraceEventSink:
    """Batch trace events before sending them to the collector actor."""

    def __init__(self, collector_handle, batch_size: int = 64, batch_seconds: float = 2.0):
        self.collector_handle = collector_handle
        self.batch_size = batch_size
        self.batch_seconds = batch_seconds
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._warner = RateLimitedWarner(logger)

    def __call__(self, event: dict) -> None:
        try:
            with self._lock:
                self._buffer.append(dict(event))
                due = len(self._buffer) >= self.batch_size or time.monotonic() - self._last_flush >= self.batch_seconds
                batch = self._take_locked() if due else None
            if batch:
                self.collector_handle.push.remote("trace", batch)
        except Exception:
            self._warner.warn("Slime dashboard trace push failed; dropping trace events")

    def flush(self) -> None:
        try:
            with self._lock:
                batch = self._take_locked()
            if batch:
                self.collector_handle.push.remote("trace", batch)
        except Exception:
            self._warner.warn("Slime dashboard trace flush failed; dropping trace events")

    def _take_locked(self) -> list[dict]:
        batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        return batch
