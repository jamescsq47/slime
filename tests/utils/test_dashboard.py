from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from slime.dashboard.api import span as dashboard_span
from slime.dashboard.gpu_sampler import GpuSampler
from slime.dashboard.hooks import TraceEventSink
from slime.dashboard.reader import DashboardReader
from slime.dashboard.sglang_scraper import SglangScraper
from slime.dashboard.store import JsonlStore
from slime.utils.trace_utils import add_trace_event_sink, remove_trace_event_sink, trace_span
from slime.utils.types import Sample


class _RemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)


class _CollectorHandle:
    def __init__(self):
        self.push = _RemoteMethod()


@pytest.mark.unit
def test_store_reader_and_server_smoke(tmp_path):
    store = JsonlStore(tmp_path)
    store.write_meta({"schema_version": 1, "run_name": "cpu-smoke"})
    store.append("gpu", {"ts": 100.0, "node": "node-a", "gpu": 0, "util": 75})
    store.append("metrics", {"ts": 101.0, "step": 3, "metrics": {"loss": 0.5}})

    assert store.flush() == 2
    reader = DashboardReader(tmp_path)
    assert reader.meta()["run_name"] == "cpu-smoke"
    assert reader.records("gpu", since=99.0)[0]["util"] == 75

    from fastapi.testclient import TestClient
    from slime.dashboard.server import make_app

    client = TestClient(make_app(tmp_path))
    assert client.get("/api/health").json()["ok"] is True
    index = client.get("/")
    assert index.status_code == 200
    assert index.headers["cache-control"] == "no-cache"


@pytest.mark.unit
def test_reader_incrementally_loads_appended_records(tmp_path):
    store = JsonlStore(tmp_path)
    store.append("gpu", {"ts": 100.0, "node": "node-a", "gpu": 0, "util": 75})
    store.flush()
    reader = DashboardReader(tmp_path)

    assert [record["util"] for record in reader.records("gpu", since=99.0)] == [75]

    store.append("gpu", {"ts": 101.0, "node": "node-a", "gpu": 0, "util": 80})
    store.flush()
    assert [record["util"] for record in reader.records("gpu", since=99.0)] == [75, 80]


@pytest.mark.unit
def test_server_aggregates_engine_workers_by_default(tmp_path):
    store = JsonlStore(tmp_path)
    timestamp = time.time()
    for worker, value in (("worker-a", 2.0), ("worker-b", 3.0)):
        store.append(
            "engine",
            {
                "ts": timestamp,
                "metric": "sglang_num_running_reqs",
                "worker_addr": worker,
                "value": value,
            },
        )
    store.flush()

    from fastapi.testclient import TestClient
    from slime.dashboard.server import make_app

    client = TestClient(make_app(tmp_path))
    response = client.get("/api/snapshot", params={"minutes": 240})

    assert response.status_code == 200
    assert response.json()["engine"] == [
        {
            "ts": timestamp,
            "metric": "sglang_num_running_reqs",
            "worker_addr": "aggregate",
            "labels": {"scope": "all_workers"},
            "value": 5.0,
        }
    ]


@pytest.mark.unit
def test_snapshot_compacts_trace_into_tool_summary(tmp_path):
    store = JsonlStore(tmp_path)
    timestamp = time.time() - 2
    common = {
        "name": "tool_call",
        "span_id": "tool-1",
        "sample_id": 7,
        "group_id": "group-1",
    }
    store.append(
        "trace",
        {
            **common,
            "type": "span_start",
            "ts": timestamp,
            "attrs": {"task_type": "math"},
        },
    )
    store.append(
        "trace",
        {
            **common,
            "type": "span_end",
            "ts": timestamp + 1,
            "attrs": {"is_tool_call": True, "tool_calls": 1},
        },
    )
    store.flush()

    snapshot = DashboardReader(tmp_path).snapshot(minutes=1)

    assert snapshot["trace"] == []
    assert snapshot["trace_summary"]["totals"]["tool_calls"] == 1
    assert snapshot["trace_summary"]["spans"][0]["duration"] == pytest.approx(1.0)
    assert snapshot["trace_summary"]["tool_series"]["math"] == [
        [timestamp, 0],
        [timestamp, 1],
        [timestamp + 1, 1],
        [timestamp + 1, 0],
    ]


@pytest.mark.unit
def test_dashboard_downsample_preserves_latest_point_per_series():
    records = [
        {"ts": float(index), "node": "node-a", "gpu": gpu, "util": index}
        for gpu in (0, 1)
        for index in range(10)
    ]

    sampled = DashboardReader.downsample(records, ("node", "gpu"), max_points=3)

    for gpu in (0, 1):
        gpu_rows = [record for record in sampled if record["gpu"] == gpu]
        assert len(gpu_rows) <= 4
        assert gpu_rows[-1]["ts"] == 9.0


@pytest.mark.unit
def test_sglang_scraper_parses_current_colon_metric_names():
    payload = """
# HELP sglang:num_running_reqs Number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="qwen",worker_addr="10.0.0.1:30000"} 3
# HELP sglang:num_queue_reqs Number of queued requests
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="qwen",worker_addr="10.0.0.1:30000"} 7
# HELP unrelated_metric Ignored
# TYPE unrelated_metric gauge
unrelated_metric 9
"""
    batches = []
    scraper = SglangScraper(
        batches.append,
        router_addr="http://router:30000/",
        http_get=lambda url, timeout: payload,
    )

    records = scraper.scrape_once(timestamp=123.0)

    assert [(record["metric"], record["value"]) for record in records] == [
        ("sglang_num_running_reqs", 3.0),
        ("sglang_num_queue_reqs", 7.0),
    ]
    assert records[0]["worker_addr"] == "10.0.0.1:30000"
    assert batches == [records]


@pytest.mark.unit
def test_sglang_scraper_attaches_explicit_worker_gpu():
    metrics = "sglang:num_running_reqs{worker_addr=\"http://10.0.0.1:30000\"} 3\n"

    def http_get(url, timeout):
        if url.endswith("/get_server_info"):
            return '{"base_gpu_id": 5}'
        return metrics

    scraper = SglangScraper(lambda records: None, router_addr="http://router:30000", http_get=http_get)

    records = scraper.scrape_once(timestamp=123.0)

    assert records[0]["labels"]["gpu"] == "5"


class _FakeNvml:
    class _Utilization:
        gpu = 82
        memory = 41

    class _Memory:
        used = 4 * 1024 * 1024
        total = 8 * 1024 * 1024

    def nvmlDeviceGetUtilizationRates(self, handle):
        return self._Utilization()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return self._Memory()

    def nvmlDeviceGetPowerUsage(self, handle):
        return 125_500


@pytest.mark.unit
def test_gpu_sampler_uses_nvml_without_cuda_or_gpu(tmp_path):
    collector = _CollectorHandle()
    sampler = GpuSampler(collector, node="node-a", interval=1.0, nvml=_FakeNvml())
    sampler._handles = [object()]
    sampler._uuids = ["GPU-test"]

    assert sampler.sample_once(timestamp=456.0) == 1
    sampler.flush()

    stream, records = collector.push.calls[0]
    assert stream == "gpu"
    assert records == [
        {
            "ts": 456.0,
            "node": "node-a",
            "gpu": 0,
            "uuid": "GPU-test",
            "util": 82,
            "memory_util": 41,
            "mem_used_mb": 4,
            "mem_total_mb": 8,
            "power_w": 125.5,
        }
    ]


@pytest.mark.unit
def test_trace_sink_batches_spans_without_dump_details():
    collector = _CollectorHandle()
    sink = TraceEventSink(collector, batch_size=2, batch_seconds=60)
    sample = Sample(index=7, prompt="test")
    add_trace_event_sink(sink)
    try:
        with trace_span(sample, "generation_turn", attrs={"turn": 1}) as context:
            context.update({"completion_tokens": 4})
    finally:
        remove_trace_event_sink(sink)
        sink.flush()

    assert len(collector.push.calls) == 1
    stream, records = collector.push.calls[0]
    assert stream == "trace"
    assert [record["type"] for record in records] == ["span_start", "span_end"]
    assert records[1]["attrs"]["completion_tokens"] == 4


@pytest.mark.unit
def test_dashboard_disabled_does_not_create_trace_carrier():
    args = SimpleNamespace(use_slime_dashboard=False)
    sample = Sample(index=8, prompt="test")

    with dashboard_span(args, sample, "generation_turn") as context:
        context.update({"completion_tokens": 4})

    assert getattr(sample, "trace", None) is None


@pytest.mark.unit
def test_failing_trace_sink_cannot_break_rollout_trace():
    def failing_sink(event):
        raise RuntimeError("telemetry unavailable")

    sample = Sample(index=9, prompt="test")
    add_trace_event_sink(failing_sink)
    try:
        with trace_span(sample, "generation_turn"):
            pass
    finally:
        remove_trace_event_sink(failing_sink)

    assert [event["type"] for event in sample.trace["events"]] == ["span_start", "span_end"]
