from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable

from slime.dashboard.logging_utils import RateLimitedWarner

logger = logging.getLogger(__name__)

DEFAULT_METRIC_WHITELIST = (
    "sglang_num_running_reqs",
    "sglang_num_queue_reqs",
    "sglang_gen_throughput",
    "sglang_token_usage",
    "sglang_cache_hit_rate",
    "sglang_num_prefill_prealloc_queue_reqs",
    "sglang_num_prefill_inflight_queue_reqs",
    "sglang_num_decode_prealloc_queue_reqs",
    "sglang_num_decode_transfer_queue_reqs",
    "sglang_kv_transfer_speed_gb_s",
    "sglang_kv_transfer_latency_ms",
)

KEPT_LABELS = ("engine_type", "model_name", "worker_addr")


def _http_get(url: str, timeout: float) -> str:
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


class SglangScraper:
    def __init__(
        self,
        sink: Callable[[list[dict]], None],
        router_addr: str,
        interval: float = 2.0,
        timeout: float = 5.0,
        whitelist: Iterable[str] = DEFAULT_METRIC_WHITELIST,
        http_get: Callable[[str, float], str] = _http_get,
    ):
        if not router_addr:
            raise ValueError("router_addr is required")
        if interval <= 0:
            raise ValueError(f"scrape interval must be positive, got {interval}")
        self.sink = sink
        self.router_addr = router_addr.rstrip("/")
        self.interval = interval
        self.timeout = timeout
        self.whitelist = frozenset(whitelist)
        self.http_get = http_get
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="slime-dashboard-sglang", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1.0)

    def scrape_once(self, timestamp: float | None = None) -> list[dict]:
        timestamp = time.time() if timestamp is None else timestamp
        text = self.http_get(f"{self.router_addr}/engine_metrics", self.timeout)
        records = self.parse_metrics(text, timestamp)
        if records:
            self.sink(records)
        return records

    def parse_metrics(self, text: str, timestamp: float) -> list[dict]:
        from prometheus_client.parser import text_string_to_metric_families

        records = []
        for family in text_string_to_metric_families(text):
            name = family.name.replace(":", "_")
            if name not in self.whitelist:
                continue
            for sample in family.samples:
                labels = {key: value for key, value in sample.labels.items() if key in KEPT_LABELS}
                records.append(
                    {
                        "ts": timestamp,
                        "metric": name,
                        "worker_addr": sample.labels.get("worker_addr", "router"),
                        "labels": labels,
                        "value": float(sample.value),
                    }
                )
        return records

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.scrape_once()
            except Exception:
                self._warner.warn("SGLang metrics scrape failed; leaving a telemetry gap")
            self._stop.wait(max(0.5, self.interval - (time.monotonic() - started)))
