from __future__ import annotations

import logging
import threading
import time

from slime.dashboard.logging_utils import RateLimitedWarner

logger = logging.getLogger(__name__)


class GpuSampler:
    """NVML sampler intended to run as one small Ray actor per GPU node."""

    def __init__(self, collector_handle, node: str, interval: float = 1.0, nvml=None):
        if interval <= 0:
            raise ValueError(f"GPU sample interval must be positive, got {interval}")
        self.collector_handle = collector_handle
        self.node = node
        self.interval = interval
        self._nvml = nvml
        self._handles = []
        self._uuids: list[str] = []
        self._buffer: list[dict] = []
        self._buffer_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)

    def _initialize_nvml(self) -> bool:
        try:
            if self._nvml is None:
                import pynvml

                self._nvml = pynvml
            self._nvml.nvmlInit()
            count = self._nvml.nvmlDeviceGetCount()
            self._handles = [self._nvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)]
            self._uuids = [str(self._nvml.nvmlDeviceGetUUID(handle)) for handle in self._handles]
            return True
        except Exception as exc:
            logger.warning("NVML unavailable on %s; GPU telemetry disabled on this node: %s", self.node, exc)
            return False

    def start(self) -> bool:
        if self._thread is not None:
            return True
        if not self._initialize_nvml():
            return False
        self._thread = threading.Thread(target=self._run, name="slime-dashboard-gpu", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 6.0)
        self.flush()

    def sample_once(self, timestamp: float | None = None) -> int:
        timestamp = time.time() if timestamp is None else timestamp
        sampled = 0
        for gpu, handle in enumerate(self._handles):
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                record = {
                    "ts": timestamp,
                    "node": self.node,
                    "gpu": gpu,
                    "uuid": self._uuids[gpu],
                    "util": int(utilization.gpu),
                    "memory_util": int(utilization.memory),
                    "mem_used_mb": int(memory.used) >> 20,
                    "mem_total_mb": int(memory.total) >> 20,
                    "power_w": int(self._nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
                }
            except Exception:
                self._warner.warn("NVML read failed for GPU %s on %s", gpu, self.node)
                continue
            with self._buffer_lock:
                self._buffer.append(record)
            sampled += 1
        return sampled

    def flush(self) -> None:
        with self._buffer_lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return
        try:
            self.collector_handle.push.remote("gpu", batch)
        except Exception:
            self._warner.warn("GPU telemetry push failed on %s; dropping this batch", self.node)

    def _run(self) -> None:
        next_flush = time.monotonic() + 5.0
        while not self._stop.is_set():
            self.sample_once()
            if time.monotonic() >= next_flush:
                self.flush()
                next_flush = time.monotonic() + 5.0
            self._stop.wait(self.interval)
