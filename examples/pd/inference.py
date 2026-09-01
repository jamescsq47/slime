#!/usr/bin/env python3
"""Standalone Math + BrowseComp agentic inference load generator for SGLang PD."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from pd_metrics import distribution, generation_turns

if TYPE_CHECKING:
    from slime.utils.types import Sample

LOG = logging.getLogger("pd-inference")
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
COUNTERS = {
    "sglang_prompt_tokens_total",
    "sglang_generation_tokens_total",
    "sglang_cached_tokens_total",
    "sglang_num_requests_total",
}
GAUGES = {
    "sglang_num_running_reqs",
    "sglang_num_queue_reqs",
    "sglang_num_used_tokens",
    "sglang_token_usage",
    "sglang_max_total_num_tokens",
    "sglang_cache_hit_rate",
    "sglang_gen_throughput",
    "sglang_num_prefill_prealloc_queue_reqs",
    "sglang_num_prefill_inflight_queue_reqs",
    "sglang_num_decode_prealloc_queue_reqs",
    "sglang_num_decode_transfer_queue_reqs",
    "sglang_kv_transfer_speed_gb_s",
}
HISTOGRAMS = {
    "sglang_time_to_first_token_seconds",
    "sglang_inter_token_latency_seconds",
}


def make_runtime_args(cli: argparse.Namespace, workload: Any | None = None) -> Namespace:
    return Namespace(
        # Existing slime-compatible data/generation interface.
        hf_checkpoint=cli.model,
        rollout_global_dataset=True,
        rollout_shuffle=not cli.preserve_source_order,
        rollout_seed=cli.seed,
        rollout_max_prompt_len=cli.max_context_length,
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        multimodal_keys=None,
        apply_chat_template=True,
        apply_chat_template_kwargs={},
        n_samples_per_prompt=1,
        # GenerateState and copied Math/QA agents.
        sglang_router_ip=cli.router_host,
        sglang_router_port=cli.router_port,
        retool_local_router_port=cli.retool_local_router_port,
        sglang_server_concurrency=cli.max_inflight,
        # One model call can legitimately remain queued/decoding for more
        # than ten minutes in the c256 mixed workload.  A 600-second client
        # timeout caused GenerateState to retry the same request-generation;
        # the router then sent the duplicate to another D, so both producers
        # raced on one lifecycle manifest and duplicated Decode work.
        sglang_router_request_timeout_secs=cli.router_request_timeout_seconds,
        use_distributed_post=False,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=cli.temperature,
        rollout_top_p=cli.top_p,
        rollout_top_k=cli.top_k,
        rollout_max_response_len=cli.max_response_length,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_max_context_len=cli.max_context_length,
        max_seq_len=cli.max_context_length,
        sglang_context_length=cli.max_context_length,
        context_parallel_size=1,
        max_tokens_per_gpu=cli.max_context_length,
        use_slime_dashboard=True,
        enable_tool_delay=False,
        workload_dataset_options=(
            {dataset.id: dict(dataset.options) for dataset in workload.datasets}
            if workload is not None
            else {}
        ),
    )


def arrival_offsets(count: int, rate: float, distribution_name: str, seed: int) -> list[float]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, count):
        interval = 1.0 / rate if distribution_name == "fixed" else rng.expovariate(rate)
        offsets.append(offsets[-1] + interval)
    return offsets


def balanced_dispatch_samples(
    source: Any,
    *,
    measured_count: int,
    warmup_count: int,
    policy: str,
    seed: int,
    math_ratio: float = 0.5,
    profile_schedule: list[dict[str, Any]] | None = None,
    preserve_source_order: bool = False,
) -> tuple[list["Sample"], list[dict[str, Any]]]:
    """Select a fixed-composition sample set, changing only its dispatch order."""
    if not 0.0 <= math_ratio <= 1.0:
        raise ValueError(f"math_ratio must be in [0, 1], got {math_ratio}")
    pools: dict[str, list["Sample"]] = {"math": [], "qa": []}
    for sample in source.origin_samples:
        task_type = (sample.metadata or {}).get("task_type")
        if task_type in pools:
            pools[task_type].append(sample)
    if not preserve_source_order:
        rng = random.Random(seed)
        for pool in pools.values():
            rng.shuffle(pool)

    math_count = round(measured_count * math_ratio)
    qa_count = measured_count - math_count
    if math_ratio == 1.0:
        warmup_types = ["math"] * warmup_count
    elif math_ratio == 0.0:
        warmup_types = ["qa"] * warmup_count
    else:
        warmup_types = [
            "math" if ((index + 1) * math_ratio).__ceil__() > (index * math_ratio).__ceil__() else "qa"
            for index in range(warmup_count)
        ]
    required = {
        "math": math_count + warmup_types.count("math"),
        "qa": qa_count + warmup_types.count("qa"),
    }
    for task_type, count in required.items():
        if len(pools[task_type]) < count:
            raise RuntimeError(f"need {count} {task_type} samples, only found {len(pools[task_type])}")

    warmups = []
    offsets = {"math": 0, "qa": 0}
    for task_type in warmup_types:
        warmups.append(copy.deepcopy(pools[task_type][offsets[task_type]]))
        offsets[task_type] += 1

    labels = ["math"] * math_count + ["qa"] * qa_count
    if policy in {"profile_balanced", "fixed"}:
        if profile_schedule is None or len(profile_schedule) != measured_count:
            raise ValueError(f"{policy} requires a schedule with exactly --requests entries")
        schedule_ids = [entry["experiment_sample_id"] for entry in profile_schedule]
        if len(set(schedule_ids)) != measured_count:
            raise ValueError("profile_balanced schedule contains duplicate sample IDs")
        labels = [entry["task_type"] for entry in profile_schedule]
        if labels.count("math") != math_count or labels.count("qa") != qa_count:
            raise ValueError("profile_balanced schedule does not match --math-ratio")
    elif policy == "random":
        random.Random(seed + 1).shuffle(labels)
    elif policy in {"alternating", "dynamic"}:
        if math_count != qa_count:
            raise ValueError(f"{policy} currently requires an exact 1:1 Math/QA mix")
        labels = ["math" if index % 2 == 0 else "qa" for index in range(measured_count)]
    else:
        raise ValueError(f"unsupported dispatch policy: {policy}")

    measured = []
    dispatch_log = []
    for position, task_type in enumerate(labels):
        if policy in {"profile_balanced", "fixed"}:
            sample_id = profile_schedule[position]["experiment_sample_id"]
            expected_prefix = f"{task_type}-"
            if not sample_id.startswith(expected_prefix):
                raise ValueError(f"schedule entry {sample_id} does not match {task_type}")
            pool_index = int(sample_id.removeprefix(expected_prefix))
            sample = copy.deepcopy(pools[task_type][pool_index])
        else:
            sample = copy.deepcopy(pools[task_type][offsets[task_type]])
            offsets[task_type] += 1
            sample_id = f"{task_type}-{offsets[task_type] - 1}"
        metadata = dict(sample.metadata or {})
        metadata.update(
            {
                "dispatch_policy": policy,
                "dispatch_position": position,
                "experiment_sample_id": sample_id,
            }
        )
        sample.metadata = metadata
        measured.append(sample)
        dispatch_log.append(
            {
                "position": position,
                "task_type": task_type,
                "experiment_sample_id": metadata["experiment_sample_id"],
            }
        )

    samples = warmups + measured
    for index, sample in enumerate(samples):
        sample.index = index
        sample.group_index = index
    return samples, dispatch_log


async def run_dynamic_slot(
    *,
    args: Namespace,
    position: int,
    scheduled_offset: float,
    scheduled_monotonic: float,
    scheduled_wall: float,
    pools: dict[str, list["Sample"]],
    pool_lock: asyncio.Lock,
    scheduler: DynamicScheduler,
    sampler: EngineSampler,
    semaphore: asyncio.Semaphore,
    cli: argparse.Namespace,
    dispatch_log: list[dict[str, Any]],
    selected_samples: dict[int, "Sample"],
) -> dict[str, Any]:
    await asyncio.sleep(max(0.0, scheduled_monotonic - time.monotonic()))
    pressure = sampler.pressure_snapshot(
        recent_seconds=cli.dynamic_recent_seconds,
        history_start_seconds=cli.dynamic_history_start_seconds,
        history_end_seconds=cli.dynamic_history_end_seconds,
    )
    async with pool_lock:
        available = {task_type for task_type, values in pools.items() if values}
        task_type, reason = scheduler.choose(available, pressure)
        sample = pools[task_type].pop(0)
        sample.index = position
        sample.group_index = position
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["dispatch_position"] = position
        selected_samples[position] = sample
        dispatch_log.append(
            {
                "position": position,
                "scheduled_offset_seconds": scheduled_offset,
                "decision_ts": time.time(),
                "task_type": task_type,
                "experiment_sample_id": sample.metadata.get("experiment_sample_id"),
                "reason": reason,
                "remaining_math": len(pools["math"]),
                "remaining_qa": len(pools["qa"]),
                **pressure,
            }
        )
    return await run_one(
        args,
        sample,
        scheduled_monotonic,
        scheduled_wall,
        semaphore,
    )


def _parse_engine_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name.replace(":", "_")
            if name == "sglang_realtime_tokens_total":
                mode = sample.labels.get("mode")
                if mode:
                    key = f"{name}|mode={mode}"
                    values[key] = values.get(key, 0.0) + float(sample.value)
            if name == "sglang_gpu_execution_seconds_total":
                category = sample.labels.get("category")
                if category:
                    key = f"{name}|category={category}"
                    values[key] = values.get(key, 0.0) + float(sample.value)
            if name in COUNTERS or name in GAUGES:
                values[name] = values.get(name, 0.0) + float(sample.value)
            for histogram in HISTOGRAMS:
                if name in {f"{histogram}_sum", f"{histogram}_count"}:
                    values[name] = values.get(name, 0.0) + float(sample.value)
                elif name == f"{histogram}_bucket":
                    key = f"{name}|le={sample.labels['le']}"
                    values[key] = values.get(key, 0.0) + float(sample.value)
    return values


class EngineSampler:
    def __init__(self, endpoints: dict[str, str | list[str]], interval: float):
        self.endpoints = {
            role: [endpoint] if isinstance(endpoint, str) else endpoint
            for role, endpoint in endpoints.items()
        }
        self.interval = interval
        self.records: list[dict[str, Any]] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            while not self._stop.is_set():
                started = time.time()
                for role, endpoints in self.endpoints.items():
                    aggregate: dict[str, float] = {}
                    endpoint_metrics: list[dict[str, Any]] = []
                    errors = []
                    for endpoint in endpoints:
                        try:
                            response = await client.get(f"{endpoint.rstrip('/')}/metrics")
                            response.raise_for_status()
                            parsed = _parse_engine_metrics(response.text)
                            endpoint_metrics.append({"endpoint": endpoint, "metrics": parsed})
                            for key, value in parsed.items():
                                aggregate[key] = aggregate.get(key, 0.0) + value
                        except Exception as exc:
                            errors.append(f"{endpoint}: {exc}")
                    record: dict[str, Any] = {
                        "ts": started,
                        "role": role,
                        "engine_count": len(endpoints),
                    }
                    if aggregate:
                        record["metrics"] = aggregate
                    if endpoint_metrics:
                        record["endpoint_metrics"] = endpoint_metrics
                    if errors:
                        record["errors"] = errors
                    self.records.append(record)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()

    def pressure_snapshot(
        self,
        *,
        recent_seconds: float,
        history_start_seconds: float,
        history_end_seconds: float,
    ) -> dict[str, float]:
        """Compare P throughput and D occupancy with each node's own history."""
        now = self.records[-1]["ts"] if self.records else time.time()
        valid = [row for row in self.records if "metrics" in row]

        def window(role: str, age_min: float, age_max: float) -> list[dict[str, Any]]:
            return [
                row
                for row in valid
                if row.get("role") == role
                and now - age_max <= row["ts"] <= now - age_min
            ]

        def counter_rate(rows: list[dict[str, Any]], counter: str) -> float | None:
            if len(rows) < 2 or rows[-1]["ts"] <= rows[0]["ts"]:
                return None
            first, last = rows[0], rows[-1]
            if counter not in first["metrics"] or counter not in last["metrics"]:
                return None
            return max(0.0, last["metrics"][counter] - first["metrics"][counter]) / (
                last["ts"] - first["ts"]
            )

        def gauge_mean(rows: list[dict[str, Any]], gauge: str) -> float | None:
            values = [row["metrics"][gauge] for row in rows if gauge in row["metrics"]]
            return sum(values) / len(values) if values else None

        p_recent_rows = window("prefill", 0.0, recent_seconds)
        p_history_rows = window("prefill", history_start_seconds, history_end_seconds)
        d_recent_rows = window("decode", 0.0, recent_seconds)
        d_history_rows = window("decode", history_start_seconds, history_end_seconds)
        p_counter = "sglang_realtime_tokens_total|mode=prefill_compute"
        p_recent = counter_rate(p_recent_rows, p_counter)
        p_history = counter_rate(p_history_rows, p_counter)
        d_recent = gauge_mean(d_recent_rows, "sglang_num_running_reqs")
        d_history = gauge_mean(d_history_rows, "sglang_num_running_reqs")

        # A zero historical baseline is not a usable relative comparison.  The
        # scheduler stays in deterministic cold-start balancing until both nodes
        # have non-zero historical activity.
        ready = p_recent is not None and p_history not in (None, 0.0) and d_recent is not None and d_history not in (None, 0.0)
        return {
            "prefill_recent_throughput": p_recent or 0.0,
            "prefill_history_throughput": p_history or 0.0,
            "decode_recent_running": d_recent or 0.0,
            "decode_history_running": d_history or 0.0,
            "prefill_relative_activity": p_recent / p_history if ready else 0.0,
            "decode_relative_activity": d_recent / d_history if ready else 0.0,
            "relative_activity_ready": float(ready),
        }


class DynamicScheduler:
    """Deterministically dispatch away from the node with higher relative activity."""

    def __init__(
        self, *, hysteresis: float, max_imbalance: int, max_consecutive: int, seed: int
    ):
        self.hysteresis = hysteresis
        self.max_imbalance = max_imbalance
        self.max_consecutive = max_consecutive
        self.rng = random.Random(seed)
        self.dispatched = {"math": 0, "qa": 0}
        self.last_choice: str | None = None
        self.consecutive = 0

    def choose(self, available: set[str], pressure: dict[str, float]) -> tuple[str, str]:
        if len(available) == 1:
            choice = next(iter(available))
            reason = "other_queue_empty"
        else:
            p_relative = pressure["prefill_relative_activity"]
            d_relative = pressure["decode_relative_activity"]
            if not pressure["relative_activity_ready"]:
                choice, reason = self._deterministic_tie(), "cold_start_balance"
            elif p_relative > d_relative:
                choice, reason = "math", "prefill_relatively_heavier"
            elif d_relative > p_relative:
                choice, reason = "qa", "decode_relatively_heavier"
            else:
                choice, reason = self._deterministic_tie(), "relative_activity_tie"
        self.dispatched[choice] += 1
        self.consecutive = self.consecutive + 1 if choice == self.last_choice else 1
        self.last_choice = choice
        return choice, reason

    def _deterministic_tie(self) -> str:
        if self.dispatched["math"] < self.dispatched["qa"]:
            return "math"
        if self.dispatched["qa"] < self.dispatched["math"]:
            return "qa"
        return "qa" if self.last_choice == "math" else "math"


def sample_record(
    sample: "Sample",
    *,
    scheduled_ts: float,
    arrival_ts: float,
    started_ts: float,
    finished_ts: float,
    error: str | None = None,
) -> dict[str, Any]:
    turns = generation_turns(getattr(sample, "trace", None))
    metadata = sample.metadata or {}
    task_type = metadata.get("task_type", "unknown")
    return {
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "task_type": task_type,
        "dataset_id": metadata.get("dataset_id", task_type),
        "harness_id": metadata.get("harness_id"),
        "scheduled_ts": scheduled_ts,
        "arrival_ts": arrival_ts,
        "started_ts": started_ts,
        "finished_ts": finished_ts,
        "schedule_lag_seconds": max(0.0, arrival_ts - scheduled_ts),
        "queue_delay_seconds": max(0.0, started_ts - arrival_ts),
        "agent_latency_seconds": max(0.0, finished_ts - started_ts),
        "status": sample.status.value,
        "error": error,
        "response": sample.response,
        "response_tokens": sample.response_length,
        "generation_turns": len(turns),
        "model_prompt_tokens": sum(int(turn.get("prompt_tokens") or 0) for turn in turns),
        "model_completion_tokens": sum(int(turn.get("completion_tokens") or 0) for turn in turns),
        "first_turn_ttft_seconds": turns[0].get("ttft_seconds") if turns else None,
        "turn_metrics": turns,
        "code_calls": int(getattr(sample, "code_call_count", 0) or 0),
        "search_calls": int(getattr(sample, "search_call_count", 0) or 0),
        "tool_time_seconds": float(getattr(sample, "tool_time", 0.0) or 0.0),
        "metadata": sample.metadata,
    }


async def run_one(
    args: Namespace,
    sample: "Sample",
    scheduled_monotonic: float,
    scheduled_wall: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    from generate_with_hybrid import generate_unified
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.types import Sample

    await asyncio.sleep(max(0.0, scheduled_monotonic - time.monotonic()))
    arrival_ts = time.time()
    error = None
    async with semaphore:
        started_ts = time.time()
        try:
            state = GenerateState(args)
            sample = await generate_unified(args, sample, state.sampling_params.copy())
        except Exception as exc:
            LOG.exception("sample %s failed", sample.index)
            error = f"{type(exc).__name__}: {exc}"
            sample.status = Sample.Status.FAILED
        finished_ts = time.time()
    return sample_record(
        sample,
        scheduled_ts=scheduled_wall,
        arrival_ts=arrival_ts,
        started_ts=started_ts,
        finished_ts=finished_ts,
        error=error,
    )


def _role_throughput(records: list[dict[str, Any]], role: str, counter: str) -> float | None:
    points = [
        (record["ts"], record["metrics"].get(counter))
        for record in records
        if record.get("role") == role and record.get("metrics", {}).get(counter) is not None
    ]
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        return None
    return max(0.0, points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])


def _histogram_delta(
    records: list[dict[str, Any]], role: str, histogram: str
) -> dict[str, float | int | None]:
    snapshots = [record["metrics"] for record in records if record.get("role") == role and "metrics" in record]
    sum_key, count_key = f"{histogram}_sum", f"{histogram}_count"
    snapshots = [snapshot for snapshot in snapshots if sum_key in snapshot and count_key in snapshot]
    if len(snapshots) < 2:
        return distribution([])
    first, last = snapshots[0], snapshots[-1]
    count = max(0.0, last[count_key] - first[count_key])
    total = max(0.0, last[sum_key] - first[sum_key])
    buckets = []
    prefix = f"{histogram}_bucket|le="
    for key, end_value in last.items():
        if key.startswith(prefix):
            boundary = key.removeprefix(prefix)
            if boundary != "+Inf":
                buckets.append((float(boundary), max(0.0, end_value - first.get(key, 0.0))))
    buckets.sort()

    def quantile(q: float) -> float | None:
        target = count * q
        return next((boundary for boundary, cumulative in buckets if cumulative >= target), None)

    return {
        "count": int(count),
        "mean": total / count if count else None,
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": None,
    }


def _gauge_max(records: list[dict[str, Any]], metric: str) -> float | None:
    values = [record["metrics"][metric] for record in records if metric in record.get("metrics", {})]
    return max(values) if values else None


def build_engine_timeseries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert cumulative SGLang counters into per-sampling-interval rates."""
    rows: list[dict[str, Any]] = []
    by_role: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if "metrics" in record:
            by_role.setdefault(record["role"], []).append(record)
    first_ts = min((record["ts"] for values in by_role.values() for record in values), default=0.0)
    for role, values in by_role.items():
        values.sort(key=lambda record: record["ts"])
        for previous, current in zip(values, values[1:]):
            elapsed = current["ts"] - previous["ts"]
            if elapsed <= 0:
                continue
            metrics = current["metrics"]
            old_metrics = previous["metrics"]
            prompt_rate = max(
                0.0,
                metrics.get("sglang_prompt_tokens_total", 0.0)
                - old_metrics.get("sglang_prompt_tokens_total", 0.0),
            ) / elapsed
            generation_rate = max(
                0.0,
                metrics.get("sglang_generation_tokens_total", 0.0)
                - old_metrics.get("sglang_generation_tokens_total", 0.0),
            ) / elapsed
            realtime_prefill_key = "sglang_realtime_tokens_total|mode=prefill_compute"
            realtime_decode_key = "sglang_realtime_tokens_total|mode=decode"
            realtime_prefill_rate = (
                max(
                    0.0,
                    metrics[realtime_prefill_key] - old_metrics[realtime_prefill_key],
                )
                / elapsed
                if realtime_prefill_key in metrics and realtime_prefill_key in old_metrics
                else None
            )
            realtime_decode_rate = (
                max(
                    0.0,
                    metrics[realtime_decode_key] - old_metrics[realtime_decode_key],
                )
                / elapsed
                if realtime_decode_key in metrics and realtime_decode_key in old_metrics
                else None
            )
            engine_prompt_rate = (
                realtime_prefill_rate if realtime_prefill_rate is not None else prompt_rate
            )
            engine_generation_rate = (
                realtime_decode_rate if realtime_decode_rate is not None else generation_rate
            )
            if role == "prefill":
                active_requests = sum(
                    metrics.get(name, 0.0)
                    for name in (
                        "sglang_num_running_reqs",
                        "sglang_num_queue_reqs",
                        "sglang_num_prefill_prealloc_queue_reqs",
                        "sglang_num_prefill_inflight_queue_reqs",
                    )
                )
            else:
                active_requests = sum(
                    metrics.get(name, 0.0)
                    for name in (
                        "sglang_num_running_reqs",
                        "sglang_num_queue_reqs",
                        "sglang_num_decode_prealloc_queue_reqs",
                        "sglang_num_decode_transfer_queue_reqs",
                    )
                )
            rows.append(
                {
                    "ts": current["ts"],
                    "elapsed_seconds": current["ts"] - first_ts,
                    "interval_seconds": elapsed,
                    "role": role,
                    "prompt_tokens_per_second": engine_prompt_rate,
                    "generation_tokens_per_second": engine_generation_rate,
                    "completion_accounted_prompt_tokens_per_second": prompt_rate,
                    "completion_accounted_generation_tokens_per_second": generation_rate,
                    "scheduler_gen_throughput": metrics.get("sglang_gen_throughput"),
                    "active_requests": active_requests,
                    "running_requests": metrics.get("sglang_num_running_reqs", 0.0),
                    "queued_requests": metrics.get("sglang_num_queue_reqs", 0.0),
                    "prefill_prealloc_requests": metrics.get(
                        "sglang_num_prefill_prealloc_queue_reqs", 0.0
                    ),
                    "prefill_inflight_requests": metrics.get(
                        "sglang_num_prefill_inflight_queue_reqs", 0.0
                    ),
                    "decode_prealloc_requests": metrics.get(
                        "sglang_num_decode_prealloc_queue_reqs", 0.0
                    ),
                    "decode_transfer_requests": metrics.get(
                        "sglang_num_decode_transfer_queue_reqs", 0.0
                    ),
                }
            )
    rows.sort(key=lambda row: (row["ts"], row["role"]))
    return rows


def _rolling_mean(values: list[float], window: int = 30) -> list[float]:
    result = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def _mean_std_cv(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance**0.5
    return mean, std, std / mean if mean > 0 else None


def _steady_state_summary(timeseries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the middle 60% of samples, excluding ramp-up and drain."""
    result: dict[str, Any] = {}
    for role in ("prefill", "decode"):
        rows = [row for row in timeseries if row["role"] == role]
        trim = len(rows) // 5
        steady = rows[trim : len(rows) - trim] if trim and len(rows) - 2 * trim >= 1 else rows
        throughput_key = (
            "prompt_tokens_per_second" if role == "prefill" else "generation_tokens_per_second"
        )
        raw_throughput = [row[throughput_key] for row in steady]
        smoothed_throughput = _rolling_mean(raw_throughput)
        throughput_mean, throughput_std, throughput_cv = _mean_std_cv(raw_throughput)
        smooth_mean, smooth_std, smooth_cv = _mean_std_cv(smoothed_throughput)
        result[role] = {
            "samples": len(steady),
            "window_start_seconds": steady[0]["elapsed_seconds"] if steady else None,
            "window_end_seconds": steady[-1]["elapsed_seconds"] if steady else None,
            "mean_active_requests": (
                sum(row["active_requests"] for row in steady) / len(steady) if steady else None
            ),
            "throughput_metric": throughput_key,
            "mean_tokens_per_second": throughput_mean,
            "std_tokens_per_second": throughput_std,
            "throughput_cv": throughput_cv,
            "rolling_60s_mean_tokens_per_second": smooth_mean,
            "rolling_60s_std_tokens_per_second": smooth_std,
            "rolling_60s_throughput_cv": smooth_cv,
            "idle_sample_fraction": (
                sum(value == 0 for value in raw_throughput) / len(raw_throughput)
                if raw_throughput
                else None
            ),
            "max_active_requests": max(
                (row["active_requests"] for row in steady), default=None
            ),
            "max_prefill_inflight_requests": max(
                (row["prefill_inflight_requests"] for row in steady), default=None
            ),
            "max_decode_transfer_requests": max(
                (row["decode_transfer_requests"] for row in steady), default=None
            ),
        }
    return result


def plot_engine_timeseries(timeseries: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = {"prefill": "#2563eb", "decode": "#dc2626"}

    for role in ("prefill", "decode"):
        rows = [row for row in timeseries if row["role"] == role]
        x = [row["elapsed_seconds"] for row in rows]
        throughput_key = (
            "prompt_tokens_per_second" if role == "prefill" else "generation_tokens_per_second"
        )
        throughput_axis = axes[0] if role == "prefill" else axes[1]
        throughput_values = [row[throughput_key] for row in rows]
        throughput_axis.plot(
            x,
            throughput_values,
            label="2-second raw",
            color=colors[role],
            linewidth=0.8,
            alpha=0.28,
        )
        throughput_axis.plot(
            x,
            _rolling_mean(throughput_values),
            label="60-second moving average",
            color=colors[role],
            linewidth=2.0,
        )
        sorted_values = sorted(throughput_values)
        if sorted_values:
            p99 = sorted_values[min(len(sorted_values) - 1, int(0.99 * len(sorted_values)))]
            if p99 > 0:
                throughput_axis.set_ylim(0, p99 * 1.15)
        axes[2].plot(
            x,
            [row["active_requests"] for row in rows],
            label=f"{role} active requests",
            color=colors[role],
            linewidth=1.6,
        )
    axes[0].set_ylabel("Prompt tokens/s")
    axes[0].set_title("Prefill scheduler throughput (2-second realtime token counter)")
    axes[1].set_ylabel("Generation tokens/s")
    axes[1].set_title("Decode scheduler throughput (2-second realtime token counter)")
    axes[2].set_xlabel("Elapsed time (seconds)")
    axes[2].set_ylabel("Active requests")
    axes[2].set_title("Per-node request occupancy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_summary(
    records: list[dict[str, Any]], engine_records: list[dict[str, Any]], request_rate: float
) -> dict[str, Any]:
    turns = [turn for record in records for turn in record["turn_metrics"]]
    prefill_rate = _role_throughput(engine_records, "prefill", "sglang_prompt_tokens_total")
    cached_rate = _role_throughput(engine_records, "prefill", "sglang_cached_tokens_total")
    domains = {}
    dataset_ids = sorted(
        {
            str(record.get("dataset_id") or record.get("task_type") or "unknown")
            for record in records
        }
    )
    for dataset_id in dataset_ids:
        subset = [
            record
            for record in records
            if str(record.get("dataset_id") or record.get("task_type") or "unknown")
            == dataset_id
        ]
        domains[dataset_id] = {
            "count": len(subset),
            "completed": sum(record["status"] == "completed" for record in subset),
            "agent_latency_seconds": distribution([record["agent_latency_seconds"] for record in subset]),
        }
    timeseries = build_engine_timeseries(engine_records)
    return {
        "configured_arrival_rate_rps": request_rate,
        "requests": len(records),
        "completed": sum(record["status"] == "completed" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "domains": domains,
        "queue_delay_seconds": distribution([record["queue_delay_seconds"] for record in records]),
        "agent_latency_seconds": distribution([record["agent_latency_seconds"] for record in records]),
        "first_turn_ttft_seconds": distribution(
            [record["first_turn_ttft_seconds"] for record in records if record["first_turn_ttft_seconds"] is not None]
        ),
        "all_turn_ttft_seconds": distribution(
            [turn["ttft_seconds"] for turn in turns if turn.get("ttft_seconds") is not None]
        ),
        "all_turn_tpot_seconds": distribution(
            [turn["tpot_seconds"] for turn in turns if turn.get("tpot_seconds") is not None]
        ),
        "engine_ttft_seconds": _histogram_delta(
            engine_records, "decode", "sglang_time_to_first_token_seconds"
        ),
        "engine_tpot_seconds": _histogram_delta(
            engine_records, "decode", "sglang_inter_token_latency_seconds"
        ),
        "prefill_prompt_tokens_per_second": prefill_rate,
        "prefill_uncached_tokens_per_second": (
            max(0.0, prefill_rate - (cached_rate or 0.0)) if prefill_rate is not None else None
        ),
        "decode_generation_tokens_per_second": _role_throughput(
            engine_records, "decode", "sglang_generation_tokens_total"
        ),
        "scheduler_prefill_compute_tokens_per_second": _role_throughput(
            engine_records,
            "prefill",
            "sglang_realtime_tokens_total|mode=prefill_compute",
        ),
        "scheduler_decode_tokens_per_second": _role_throughput(
            engine_records, "decode", "sglang_realtime_tokens_total|mode=decode"
        ),
        "max_engine_queue_requests": _gauge_max(engine_records, "sglang_num_queue_reqs"),
        "steady_state": _steady_state_summary(timeseries),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


async def run_closed_loop(
    cli: argparse.Namespace,
    args: Namespace,
    pristine_samples: list["Sample"],
    initial_dispatch_log: list[dict[str, Any]],
) -> None:
    """Keep a fixed number of end-to-end agents in flight, measuring only after churn."""
    if len(pristine_samples) < cli.max_inflight:
        raise RuntimeError(
            f"closed loop needs at least {cli.max_inflight} source samples, got {len(pristine_samples)}"
        )

    sampler = EngineSampler(
        {
            "prefill": [
                f"http://{cli.prefill_host}:{port}" for port in cli.prefill_ports
            ],
            "decode": [f"http://{cli.decode_host}:{port}" for port in cli.decode_ports],
            "local": [f"http://{cli.decode_host}:{port}" for port in cli.local_ports],
        },
        cli.metrics_interval,
    )
    sampler_task = asyncio.create_task(sampler.run())
    await asyncio.sleep(min(1.0, cli.metrics_interval))

    origin_mono, origin_wall = time.monotonic(), time.time()
    stop = asyncio.Event()
    pool_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(cli.max_inflight)
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    state = {"next_id": 0, "active": 0, "successes": 0, "failures": 0}

    async def next_sample() -> tuple[int, "Sample"]:
        async with pool_lock:
            admission_id = state["next_id"]
            state["next_id"] += 1
            source = pristine_samples[admission_id % len(pristine_samples)]
            sample = copy.deepcopy(source)
            sample.index = admission_id
            sample.group_index = admission_id
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["closed_loop_admission_id"] = admission_id
            return admission_id, sample

    async def worker(worker_id: int) -> None:
        while not stop.is_set():
            admission_id, sample = await next_sample()
            now_mono, now_wall = time.monotonic(), time.time()
            async with state_lock:
                state["active"] += 1
                events.append({
                    "ts": now_wall,
                    "elapsed_seconds": now_mono - origin_mono,
                    "event": "admit",
                    "worker_id": worker_id,
                    "admission_id": admission_id,
                    "task_type": sample.metadata.get("task_type"),
                    "active_agents": state["active"],
                })
            try:
                record = await run_one(args, sample, now_mono, now_wall, semaphore)
            except asyncio.CancelledError:
                raise
            records.append(record)
            async with state_lock:
                state["active"] -= 1
                success = not record.get("error")
                state["successes" if success else "failures"] += 1
                events.append({
                    "ts": time.time(),
                    "elapsed_seconds": time.monotonic() - origin_mono,
                    "event": "complete" if success else "fail",
                    "worker_id": worker_id,
                    "admission_id": admission_id,
                    "task_type": sample.metadata.get("task_type"),
                    "active_agents": state["active"],
                })

    workers = [asyncio.create_task(worker(worker_id)) for worker_id in range(cli.max_inflight)]

    def recent_counter_delta(role: str, key: str, seconds: float) -> float:
        valid = [row for row in sampler.records if row.get("role") == role and key in row.get("metrics", {})]
        if len(valid) < 2:
            return 0.0
        cutoff = valid[-1]["ts"] - seconds
        old = next((row for row in valid if row["ts"] >= cutoff), valid[0])
        return max(0.0, valid[-1]["metrics"][key] - old["metrics"][key])

    boundary_reason = "steady_churn"
    while True:
        await asyncio.sleep(cli.metrics_interval)
        elapsed = time.monotonic() - origin_mono
        p_recent = recent_counter_delta(
            "prefill", "sglang_realtime_tokens_total|mode=prefill_compute", cli.closed_loop_recent_seconds
        )
        d_recent = recent_counter_delta(
            "decode", "sglang_realtime_tokens_total|mode=decode", cli.closed_loop_recent_seconds
        )
        if (
            elapsed >= cli.closed_loop_warmup_min_seconds
            and state["successes"] >= cli.closed_loop_warmup_completions
            and p_recent > 0
            and d_recent > 0
        ):
            break
        if elapsed >= cli.closed_loop_max_warmup_seconds:
            boundary_reason = "warmup_timeout_without_steady_churn"
            break

    measurement_start_mono, measurement_start_wall = time.monotonic(), time.time()
    start_state = dict(state)
    if cli.cuda_profiler_range and boundary_reason == "steady_churn":
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://{cli.router_host}:{cli.router_port}/start_profile",
                json={"activities": ["CUDA_PROFILER"]},
            )
            response.raise_for_status()
    if boundary_reason == "steady_churn":
        await asyncio.sleep(cli.closed_loop_measurement_seconds)
    measurement_end_mono, measurement_end_wall = time.monotonic(), time.time()
    if cli.cuda_profiler_range and boundary_reason == "steady_churn":
        # Nsight Systems may need several minutes to flush a long CUDA trace
        # before /stop_profile returns.  Do not let a client timeout destroy
        # an otherwise complete capture.
        async with httpx.AsyncClient(timeout=900.0) as client:
            response = await client.post(
                f"http://{cli.router_host}:{cli.router_port}/stop_profile", json={}
            )
            response.raise_for_status()
    end_state = dict(state)

    stop.set()
    for task in workers:
        task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    sampler.stop()
    await sampler_task

    boundaries = {
        "status": boundary_reason,
        "origin_wall": origin_wall,
        "warmup_seconds": measurement_start_mono - origin_mono,
        "measurement_start_wall": measurement_start_wall,
        "measurement_end_wall": measurement_end_wall,
        "measurement_seconds": measurement_end_mono - measurement_start_mono,
        "state_at_measurement_start": start_state,
        "state_at_measurement_end": end_state,
        "recent_window_seconds": cli.closed_loop_recent_seconds,
        "recent_prefill_compute_tokens_at_boundary": recent_counter_delta(
            "prefill", "sglang_realtime_tokens_total|mode=prefill_compute", cli.closed_loop_recent_seconds
        ),
        "recent_decode_tokens_at_boundary": recent_counter_delta(
            "decode", "sglang_realtime_tokens_total|mode=decode", cli.closed_loop_recent_seconds
        ),
    }
    write_json(cli.output_dir / "closed_loop_boundaries.json", boundaries)
    write_jsonl(cli.output_dir / "closed_loop_events.jsonl", events)
    write_jsonl(cli.output_dir / "requests.jsonl", sorted(records, key=lambda row: row["sample_index"]))
    write_jsonl(cli.output_dir / "engine_metrics.jsonl", sampler.records)
    timeseries = build_engine_timeseries(sampler.records)
    write_jsonl(cli.output_dir / "engine_throughput_2s.jsonl", timeseries)
    plot_engine_timeseries(timeseries, cli.output_dir / "pd_throughput.png")
    write_json(cli.output_dir / "dispatch_sequence.json", initial_dispatch_log)
    if boundary_reason != "steady_churn":
        raise RuntimeError(json.dumps(boundaries, ensure_ascii=False))


async def async_main(cli: argparse.Namespace) -> None:
    from data.config import legacy_workload, load_workload
    from data.dispatch import select_samples
    from data.loading import load_samples
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import init_http_client
    from slime.utils.types import Sample

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    workload = (
        load_workload(cli.workload_config)
        if cli.workload_config is not None
        else legacy_workload(
            math_path=cli.math_data,
            qa_path=cli.qa_data,
            math_ratio=cli.math_ratio,
            policy=cli.dispatch_policy,
            seed=cli.seed,
            preserve_source_order=cli.preserve_source_order,
            schedule_file=str(cli.schedule_file) if cli.schedule_file else None,
        )
    )
    if cli.workload_config is not None:
        cli.seed = workload.sampling.seed
        cli.dispatch_policy = workload.sampling.policy
        cli.preserve_source_order = workload.sampling.preserve_source_order
        cli.schedule_file = (
            Path(workload.sampling.schedule_file)
            if workload.sampling.schedule_file is not None
            else None
        )
    args = make_runtime_args(cli, workload)
    source = load_samples(args, workload)
    profile_schedule = None
    if cli.dispatch_policy in {"profile_balanced", "fixed"}:
        if cli.schedule_file is None:
            raise ValueError(f"--schedule-file is required for {cli.dispatch_policy}")
        schedule_payload = json.loads(cli.schedule_file.read_text())
        if isinstance(schedule_payload, list):
            profile_schedule = schedule_payload
        elif "schedule" in schedule_payload:
            profile_schedule = schedule_payload["schedule"]
        elif schedule_payload.get("format") == "seeded_fixed_v1":
            measured_count = int(schedule_payload["measured_count"])
            math_count = int(schedule_payload["math_count"])
            qa_count = int(schedule_payload["qa_count"])
            sample_pool_seed = int(schedule_payload["sample_pool_seed"])
            if measured_count != cli.requests or math_count + qa_count != measured_count:
                raise ValueError("seeded fixed schedule does not match --requests")
            if sample_pool_seed != cli.seed:
                raise ValueError("seeded fixed schedule sample_pool_seed does not match --seed")
            labels = ["math"] * math_count + ["qa"] * qa_count
            random.Random(int(schedule_payload["task_order_seed"])).shuffle(labels)
            offsets = {"math": 0, "qa": 0}
            profile_schedule = []
            for position, task_type in enumerate(labels):
                profile_schedule.append(
                    {
                        "position": position,
                        "task_type": task_type,
                        "experiment_sample_id": f"{task_type}-{offsets[task_type]}",
                    }
                )
                offsets[task_type] += 1
        else:
            raise ValueError("unsupported schedule file format")
    samples, dispatch_log = select_samples(
        source.pools,
        workload,
        measured_count=cli.requests,
        warmup_count=cli.warmup_requests,
        schedule=profile_schedule,
    )
    if len(samples) != cli.warmup_requests + cli.requests:
        raise RuntimeError(f"requested {cli.warmup_requests + cli.requests} samples, got {len(samples)}")

    config = vars(cli) | {"output_dir": str(cli.output_dir)}
    config["workload_config"] = str(cli.workload_config) if cli.workload_config else None
    config["schedule_file"] = str(cli.schedule_file) if cli.schedule_file else None
    # Persist the effective serving/data-plane settings next to every result.
    # These values used to live only in transient shell exports, which made a
    # later rerun look identical in config.json even when Direct reserve or D
    # admission differed materially.
    runtime_keys = (
        "EXPERIMENT_CONFIG",
        "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY",
        "SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION",
        "SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD",
        "SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT",
        "SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB",
        "SGLANG_AGENTIC_KV_P2D_HOST_STAGING",
        "SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB",
        "SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT",
        "SGLANG_PREFILL_TRANSFER_CONSUMERS",
        "PD_INFERENCE_RETURN_LOGPROB",
        "MIN_P",
        "PREFILL_GPUS",
        "DECODE_GPUS",
        "PREFILL_TP_SIZE",
        "DECODE_TP_SIZE",
        "DECODE_MEM_FRACTION_STATICS",
    )
    config["serving_runtime"] = {
        key: os.environ[key] for key in runtime_keys if key in os.environ
    }
    write_json(cli.output_dir / "config.json", config)
    write_json(cli.output_dir / "resolved_workload.json", workload.to_dict())
    if cli.dry_run:
        write_json(cli.output_dir / "dispatch_sequence.json", dispatch_log)
        preview = [
            (sample.index, sample.metadata.get("dataset_id"), sample.metadata.get("harness_id"))
            for sample in samples
        ]
        write_json(cli.output_dir / "dispatch_preview.json", preview)
        print(json.dumps(preview, ensure_ascii=False))
        return

    init_http_client(args)
    GenerateState(args)
    if cli.closed_loop:
        await run_closed_loop(cli, args, samples[cli.warmup_requests :], dispatch_log)
        return
    for sample in samples[: cli.warmup_requests]:
        LOG.info("warmup sample=%s dataset=%s", sample.index, sample.metadata.get("dataset_id"))
        await run_one(args, sample, time.monotonic(), time.time(), asyncio.Semaphore(1))

    sampler = EngineSampler(
        {
            "prefill": [
                f"http://{cli.prefill_host}:{port}" for port in cli.prefill_ports
            ],
            "decode": [
                f"http://{cli.decode_host}:{port}"
                for port in cli.decode_ports
            ],
            "local": [
                f"http://{cli.decode_host}:{port}"
                for port in cli.local_ports
            ],
        },
        cli.metrics_interval,
    )
    sampler_task = asyncio.create_task(sampler.run())
    await asyncio.sleep(min(1.0, cli.metrics_interval))

    measured = samples[cli.warmup_requests :]
    offsets = arrival_offsets(cli.requests, cli.request_rate, cli.arrival_distribution, cli.seed + 1)
    if cli.measurement_duration_seconds is not None:
        offsets = [offset for offset in offsets if offset < cli.measurement_duration_seconds]
        measured = measured[: len(offsets)]
    start_monotonic = time.monotonic()
    start_wall = time.time()
    semaphore = asyncio.Semaphore(cli.max_inflight)
    selected_samples: dict[int, Sample] = {}
    if cli.dispatch_policy == "dynamic":
        if set(workload.dataset_ids) != {"math", "qa"}:
            raise ValueError(
                "the current pressure-feedback scheduler is defined only for the legacy math/qa mix"
            )
        pools = {
            task_type: [
                sample
                for sample in measured
                if sample.metadata.get("task_type") == task_type
            ]
            for task_type in ("math", "qa")
        }
        dispatch_log = []
        scheduler = DynamicScheduler(
            hysteresis=cli.dynamic_hysteresis,
            max_imbalance=cli.dynamic_max_imbalance,
            max_consecutive=cli.dynamic_max_consecutive,
            seed=cli.seed + 2,
        )
        pool_lock = asyncio.Lock()
        tasks = [
            asyncio.create_task(
                run_dynamic_slot(
                    args=args,
                    position=position,
                    scheduled_offset=offset,
                    scheduled_monotonic=start_monotonic + offset,
                    scheduled_wall=start_wall + offset,
                    pools=pools,
                    pool_lock=pool_lock,
                    scheduler=scheduler,
                    sampler=sampler,
                    semaphore=semaphore,
                    cli=cli,
                    dispatch_log=dispatch_log,
                    selected_samples=selected_samples,
                )
            )
            for position, offset in enumerate(offsets)
        ]
    else:
        tasks = [
            asyncio.create_task(
                run_one(
                    args,
                    sample,
                    start_monotonic + offset,
                    start_wall + offset,
                    semaphore,
                )
            )
            for sample, offset in zip(measured, offsets, strict=True)
        ]
        selected_samples = {position: sample for position, sample in enumerate(measured)}
    if cli.measurement_duration_seconds is None:
        records = await asyncio.gather(*tasks)
    else:
        done, pending = await asyncio.wait(tasks, timeout=cli.measurement_duration_seconds)
        window_finished_ts = time.time()
        records = [task.result() for task in done]
        task_context = {
            task: (position, offset)
            for position, (task, offset) in enumerate(zip(tasks, offsets, strict=True))
        }
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in pending:
            position, offset = task_context[task]
            sample = selected_samples.get(position)
            if sample is None:
                continue
            sample.status = Sample.Status.ABORTED
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["stop_reason"] = "measurement_window_end"
            scheduled_ts = start_wall + offset
            records.append(
                sample_record(
                    sample,
                    scheduled_ts=scheduled_ts,
                    arrival_ts=scheduled_ts,
                    started_ts=scheduled_ts,
                    finished_ts=window_finished_ts,
                    error="measurement_window_end",
                )
            )
    sampler.stop()
    await sampler_task

    records.sort(key=lambda record: record["sample_index"])
    dispatch_log.sort(key=lambda row: row["position"])
    write_json(cli.output_dir / "dispatch_sequence.json", dispatch_log)
    write_jsonl(cli.output_dir / "requests.jsonl", records)
    write_jsonl(cli.output_dir / "engine_metrics.jsonl", sampler.records)
    timeseries = build_engine_timeseries(sampler.records)
    write_jsonl(cli.output_dir / "engine_throughput_2s.jsonl", timeseries)
    plot_engine_timeseries(timeseries, cli.output_dir / "pd_throughput.png")
    summary = build_summary(records, sampler.records, cli.request_rate)
    summary["measurement_duration_seconds"] = cli.measurement_duration_seconds
    summary["measurement_elapsed_seconds"] = (
        sampler.records[-1]["ts"] - sampler.records[0]["ts"] if len(sampler.records) >= 2 else None
    )
    summary["scheduled_within_window"] = len(offsets)
    summary["window_end_aborted"] = sum(
        record.get("error") == "measurement_window_end" for record in records
    )
    write_json(cli.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(WORKSPACE_ROOT / "Qwen3-8B"))
    parser.add_argument(
        "--workload-config",
        type=Path,
        default=None,
        help="YAML/JSON dataset mixture; when set, replaces --math-data/--qa-data/--math-ratio",
    )
    parser.add_argument(
        "--math-data",
        default=str(WORKSPACE_ROOT / "data/dapo-math-17k/dapo-math-17k.jsonl"),
    )
    parser.add_argument(
        "--qa-data", default=str(WORKSPACE_ROOT / "data/browsecomp/bc_train.jsonl")
    )
    parser.add_argument("--math-ratio", type=float, default=0.5)
    parser.add_argument("--router-host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30002)
    parser.add_argument(
        "--router-request-timeout-seconds",
        type=float,
        default=3600.0,
        help=(
            "timeout for one /generate call; must exceed the longest valid "
            "queued+decode turn so a request-generation is never retried "
            "while its original execution is still active"
        ),
    )
    parser.add_argument(
        "--retool-local-router-port",
        type=int,
        default=None,
        help="route Retool turns after turn 1 to this colocated Prefill+Decode router",
    )
    parser.add_argument("--prefill-host", default="127.0.0.1")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument(
        "--prefill-ports",
        type=lambda value: [int(port) for port in value.split(",")],
        default=None,
        help="comma-separated prefill metrics ports; defaults to --prefill-port",
    )
    parser.add_argument("--decode-host", default="127.0.0.1")
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument(
        "--decode-ports",
        type=lambda value: [int(port) for port in value.split(",")],
        default=None,
        help="comma-separated decode metrics ports; defaults to --decode-port",
    )
    parser.add_argument(
        "--local-ports",
        type=lambda value: [int(port) for port in value.split(",")],
        default=[],
        help="comma-separated colocated Retool worker metrics ports",
    )
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--request-rate", type=float, default=0.05)
    parser.add_argument("--arrival-distribution", choices=("poisson", "fixed"), default="poisson")
    parser.add_argument(
        "--dispatch-policy",
        choices=("random", "alternating", "dynamic", "profile_balanced", "fixed"),
        default="random",
    )
    parser.add_argument("--schedule-file", type=Path, default=None)
    parser.add_argument(
        "--preserve-source-order",
        action="store_true",
        help="select samples in dataset order instead of seed-shuffling each task pool",
    )
    parser.add_argument("--dynamic-lookback-seconds", type=float, default=12.0)
    parser.add_argument("--dynamic-recent-seconds", type=float, default=10.0)
    parser.add_argument("--dynamic-history-start-seconds", type=float, default=20.0)
    parser.add_argument("--dynamic-history-end-seconds", type=float, default=60.0)
    parser.add_argument("--dynamic-prefill-capacity-tps", type=float, default=9000.0)
    parser.add_argument("--dynamic-decode-capacity-tps", type=float, default=1100.0)
    parser.add_argument("--dynamic-decode-target-active", type=float, default=30.0)
    parser.add_argument("--dynamic-hysteresis", type=float, default=0.12)
    parser.add_argument("--dynamic-max-imbalance", type=int, default=8)
    parser.add_argument("--dynamic-max-consecutive", type=int, default=3)
    parser.add_argument("--max-inflight", type=int, default=16)
    parser.add_argument("--metrics-interval", type=float, default=2.0)
    parser.add_argument(
        "--pd-max-transfer-inflight",
        type=int,
        default=0,
        help="metadata for the D-side tight-pairing transfer-window experiment",
    )
    parser.add_argument(
        "--pd-p-ready-dir",
        default="",
        help="metadata for the experimental same-host P-ready/JIT reservation mode",
    )
    parser.add_argument(
        "--pd-hicache-storage-backend",
        default="",
        help="metadata for the shared P/D HiCache storage backend",
    )
    parser.add_argument(
        "--pd-hicache-storage-dir",
        default="",
        help="metadata for the shared P/D HiCache storage directory",
    )
    parser.add_argument(
        "--pd-hicache-storage-prefetch-policy",
        default="best_effort",
        help="metadata for the P-side HiCache storage prefetch policy",
    )
    parser.add_argument(
        "--pd-hicache-prefetch-threshold",
        type=int,
        default=256,
        help="metadata for the minimum P-side HiCache prefetch length",
    )
    parser.add_argument(
        "--pd-enable-decode-offload-kvcache",
        action="store_true",
        help="metadata indicating D-side incremental KV offload is enabled",
    )
    parser.add_argument("--measurement-duration-seconds", type=float, default=None)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--closed-loop-warmup-min-seconds", type=float, default=300.0)
    parser.add_argument("--closed-loop-warmup-completions", type=int, default=64)
    parser.add_argument("--closed-loop-recent-seconds", type=float, default=120.0)
    parser.add_argument("--closed-loop-max-warmup-seconds", type=float, default=1800.0)
    parser.add_argument("--closed-loop-measurement-seconds", type=float, default=300.0)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-context-length", type=int, default=40960)
    parser.add_argument("--max-response-length", type=int, default=36864)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/manual"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.prefill_ports is None:
        args.prefill_ports = [args.prefill_port]
    if args.decode_ports is None:
        args.decode_ports = [args.decode_port]
    if not 0.0 <= args.math_ratio <= 1.0:
        parser.error("--math-ratio must be in [0, 1]")
    if args.requests <= 0 or args.request_rate <= 0 or args.max_inflight <= 0:
        parser.error("--requests, --request-rate and --max-inflight must be positive")
    if args.router_request_timeout_seconds <= 0:
        parser.error("--router-request-timeout-seconds must be positive")
    if args.measurement_duration_seconds is not None and args.measurement_duration_seconds <= 0:
        parser.error("--measurement-duration-seconds must be positive")
    if (
        args.dynamic_lookback_seconds <= 0
        or args.dynamic_prefill_capacity_tps <= 0
        or args.dynamic_decode_capacity_tps <= 0
        or args.dynamic_decode_target_active <= 0
        or args.dynamic_max_imbalance <= 0
        or args.dynamic_max_consecutive <= 0
    ):
        parser.error("dynamic scheduler capacities, lookback, target and max imbalance must be positive")
    if (
        args.dynamic_recent_seconds <= 0
        or args.dynamic_history_start_seconds < args.dynamic_recent_seconds
        or args.dynamic_history_end_seconds <= args.dynamic_history_start_seconds
    ):
        parser.error(
            "dynamic windows require 0 < recent <= history-start < history-end"
        )
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(async_main(parse_args()))
