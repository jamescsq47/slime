"""Small metric helpers used only by the standalone PD experiment."""

from __future__ import annotations

import math
from typing import Any


META_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "request_received_ts",
    "prefill_finished_ts",
    "decode_finished_ts",
    "e2e_latency",
    "inference_time",
    "queue_time",
    "decode_throughput",
    "pd_prefill_bootstrap_queue_duration",
    "pd_prefill_forward_duration",
    "pd_prefill_transfer_queue_duration",
    "pd_prefill_retry_count",
    "pd_decode_prealloc_duration",
    "pd_decode_transfer_duration",
    "pd_decode_forward_duration",
    "pd_bootstrap_duration",
    "pd_alloc_waiting_duration",
    "pd_transfer_speed_gb_s",
    "pd_transfer_total_mb",
)


def sglang_meta_attrs(meta: dict[str, Any]) -> dict[str, Any]:
    attrs = {key: meta[key] for key in META_KEYS if meta.get(key) is not None}
    reason = meta.get("finish_reason")
    attrs["finish_reason"] = reason.get("type", "unknown") if isinstance(reason, dict) else str(reason or "unknown")

    request_ts = meta.get("request_received_ts")
    prefill_ts = meta.get("prefill_finished_ts")
    decode_ts = meta.get("decode_finished_ts")
    if request_ts is not None and prefill_ts is not None:
        attrs["ttft_seconds"] = max(0.0, float(prefill_ts) - float(request_ts))
    completion_tokens = int(meta.get("completion_tokens") or 0)
    if prefill_ts is not None and decode_ts is not None and completion_tokens > 1:
        attrs["tpot_seconds"] = max(0.0, float(decode_ts) - float(prefill_ts)) / (completion_tokens - 1)
    elif meta.get("e2e_latency") is not None and meta.get("inference_time") is not None:
        decode_time = max(0.0, float(meta["inference_time"]))
        attrs["ttft_seconds"] = max(0.0, float(meta["e2e_latency"]) - decode_time)
        attrs["ttft_source"] = "e2e_minus_inference_time"
        if completion_tokens > 1:
            attrs["tpot_seconds"] = decode_time / (completion_tokens - 1)
            attrs["tpot_source"] = "inference_time_per_decode_interval"
    return attrs


def generation_turns(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace:
        return []
    turns = []
    for event in trace.get("events", []):
        if event.get("type", event.get("kind")) == "span_end" and event.get("name") == "generation_turn":
            turns.append(dict(event.get("attrs") or {}))
    return turns


def percentile(values: list[float], quantile: float) -> float | None:
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean) if clean else None,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean) if clean else None,
    }
