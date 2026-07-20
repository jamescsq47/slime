from __future__ import annotations

import logging
import threading
import time
from typing import Any

from slime.dashboard.logging_utils import RateLimitedWarner
from slime.dashboard.sglang_scraper import SglangScraper
from slime.dashboard.store import JsonlStore, PARTITIONED_STREAMS

logger = logging.getLogger(__name__)


class DashboardCollector:
    """Single-writer ingest actor for all dashboard telemetry."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.store = JsonlStore(config["directory"])
        self.store.write_meta(
            {
                "schema_version": 1,
                "run_name": config.get("run_name", "slime-run"),
                "start_ts": config.get("start_ts", time.time()),
                "args": config.get("args", {}),
                "source": "slime dashboard adapted from radixark/miles#1654@d9189010",
            }
        )
        self.flush_interval = float(config.get("flush_interval", 5.0))
        self.gpu_sample_interval = float(config.get("gpu_sample_interval", 1.0))
        self.sglang_scrape_interval = float(config.get("sglang_scrape_interval", 2.0))
        self.max_buffered = int(config.get("max_buffered_records", 500_000))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._scraper: SglangScraper | None = None
        self._router_addr: str | None = None
        self._samplers: dict[str, Any] = {}
        self._self_handle = None
        self._dropped = 0
        self._warner = RateLimitedWarner(logger)

    def ping(self) -> bool:
        return True

    def start(self, self_handle=None) -> None:
        if self._flush_thread is not None:
            return
        self._self_handle = self_handle
        self._reconcile_gpu_samplers()
        self._flush_thread = threading.Thread(target=self._flush_loop, name="slime-dashboard-flush", daemon=True)
        self._flush_thread.start()

    def push(self, stream: str, records: list[dict] | dict) -> None:
        if stream not in PARTITIONED_STREAMS:
            return
        batch = records if isinstance(records, list) else [records]
        with self._lock:
            for record in batch:
                if self.store.buffered_count(stream) >= self.max_buffered:
                    self._dropped += self.store.drop_oldest(stream)
                self.store.append(stream, record)

    def set_router(self, router_addr: str | None) -> None:
        if not router_addr:
            return
        router_addr = router_addr.rstrip("/")
        with self._lock:
            if self._scraper is not None and self._router_addr == router_addr:
                return
            previous = self._scraper
            scraper = SglangScraper(
                self._push_engine_records,
                router_addr=router_addr,
                interval=self.sglang_scrape_interval,
            )
            self._scraper = scraper
            self._router_addr = router_addr
        if previous is not None:
            previous.stop()
        scraper.start()
        logger.info("Slime dashboard scraping SGLang metrics from %s", router_addr)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "directory": self.config["directory"],
                "router_addr": self._router_addr,
                "gpu_sampler_nodes": len(self._samplers),
                "buffered": {stream: self.store.buffered_count(stream) for stream in PARTITIONED_STREAMS},
                "dropped": self._dropped,
            }

    def flush(self) -> int:
        with self._lock:
            if self._dropped:
                logger.error("Slime dashboard dropped %d telemetry records because its buffer was full", self._dropped)
                self._dropped = 0
            try:
                return self.store.flush()
            except OSError:
                self._warner.warn("Slime dashboard flush failed; records remain buffered")
                return 0

    def shutdown(self) -> None:
        self._stop.set()
        scraper = self._scraper
        if scraper is not None:
            scraper.stop()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=self.flush_interval + 1.0)
        samplers = list(self._samplers.values())
        try:
            import ray

            stop_refs = [sampler.stop.remote() for sampler in samplers]
            if stop_refs:
                ray.get(stop_refs, timeout=15)
        except Exception:
            self._warner.warn("Slime dashboard GPU sampler shutdown was incomplete")
        for sampler in samplers:
            try:
                import ray

                ray.kill(sampler, no_restart=True)
            except Exception:
                self._warner.warn("Slime dashboard could not terminate a GPU sampler")
        self._samplers.clear()
        self.flush()

    def _push_engine_records(self, records: list[dict]) -> None:
        self.push("engine", records)

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.flush_interval):
            try:
                self._reconcile_gpu_samplers()
            except Exception:
                self._warner.warn("Slime dashboard could not reconcile GPU samplers")
            self.flush()

    def _reconcile_gpu_samplers(self) -> None:
        if self._self_handle is None:
            return
        try:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            from slime.dashboard.gpu_sampler import GpuSampler

            nodes = [
                node
                for node in ray.nodes()
                if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0)) > 0
            ]
            alive_ids = {node["NodeID"] for node in nodes}
            for node_id in list(self._samplers):
                if node_id not in alive_ids:
                    self._samplers.pop(node_id, None)
            sampler_actor = ray.remote(GpuSampler)
            for node in nodes:
                node_id = node["NodeID"]
                if node_id in self._samplers:
                    continue
                handle = sampler_actor.options(
                    num_cpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(
                    self._self_handle,
                    node=node["NodeManagerAddress"],
                    interval=self.gpu_sample_interval,
                )
                handle.start.remote()
                self._samplers[node_id] = handle
        except Exception:
            self._warner.warn("Slime dashboard GPU sampling is unavailable")
