#!/usr/bin/env python3
"""Aligned mixed-rollout admission benchmark: steady 32 vs one-shot 16."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from rollout_engine_ab_benchmark import (
    GenerateState,
    Recorder,
    _generator_args,
    _one_sample,
    _patch_posts,
    _read_jsonl,
    _workers,
)
from slime.utils.types import Sample


def _stats(values: list[float]) -> dict[str, float | int]:
    values = sorted(values)

    def q(p: float) -> float:
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)

    return {
        "count": len(values),
        "avg": statistics.mean(values),
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": max(values),
    }


def _make_scheduled_samples(args: argparse.Namespace) -> list[list[Sample]]:
    math_rows = _read_jsonl(Path(args.math_data))
    qa_rows = _read_jsonl(Path(args.qa_data))
    rng = random.Random(args.seed)
    math_rows = sorted(math_rows, key=lambda _: rng.random())
    qa_rows = sorted(qa_rows, key=lambda _: rng.random())
    if args.schedule == "block4":
        cycle = ["qa"] * 4 + ["math"] * 4
        domains = [cycle[i % len(cycle)] for i in range(args.groups)]
    else:
        math_count = round(args.groups * args.math_ratio)
        domains = ["math"] * math_count + ["qa"] * (args.groups - math_count)
        random.Random(args.seed).shuffle(domains)

    offsets = {"math": 0, "qa": 0}
    pools = {"math": math_rows, "qa": qa_rows}
    groups = []
    for group_index, domain in enumerate(domains):
        pool = pools[domain]
        row = pool[offsets[domain] % len(pool)]
        offsets[domain] += 1
        metadata = dict(row.get("metadata") or {})
        metadata.update({"task_type": domain, "benchmark_group_index": group_index})
        group = []
        for sample_in_group in range(args.samples_per_group):
            sample_id = f"g{group_index:03d}-s{sample_in_group:02d}"
            group.append(
                Sample(
                    group_index=group_index,
                    index=group_index * args.samples_per_group + sample_in_group,
                    prompt=row["prompt"],
                    label=row.get("label"),
                    metadata={**metadata, "benchmark_sample_id": sample_id},
                )
            )
        groups.append(group)
    return groups


class SystemMonitor:
    def __init__(self, workers: list[str], interval: float):
        self.workers = workers
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu_rows: list[dict] = []
        self.sglang_rows: list[dict] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval * 2))

    def _run(self):
        query = (
            "timestamp,index,utilization.gpu,utilization.memory,memory.used,"
            "power.draw,clocks.sm,clocks.mem"
        )
        while not self.stop_event.is_set():
            sampled_at = time.time()
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=max(5.0, self.interval),
                )
                for line in output.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) == 8:
                        self.gpu_rows.append(
                            {
                                "sampled_at": sampled_at,
                                "timestamp": values[0],
                                "gpu": int(values[1]),
                                "gpu_util": float(values[2]),
                                "memory_util": float(values[3]),
                                "memory_used_mib": float(values[4]),
                                "power_w": float(values[5]),
                                "sm_clock_mhz": float(values[6]),
                                "memory_clock_mhz": float(values[7]),
                            }
                        )
            except Exception:
                pass
            for engine_id, worker in enumerate(self.workers):
                try:
                    with urllib.request.urlopen(worker + "/metrics", timeout=2) as response:
                        text = response.read().decode("utf-8", errors="replace")
                    selected = {}
                    for line in text.splitlines():
                        if line.startswith("#") or " " not in line:
                            continue
                        metric, value = line.rsplit(" ", 1)
                        name = metric.split("{", 1)[0]
                        if any(
                            key in name
                            for key in (
                                "num_running_reqs",
                                "num_queue_reqs",
                                "token_usage",
                                "gen_throughput",
                                "cache_hit_rate",
                                "retracted",
                            )
                        ):
                            try:
                                selected[metric] = float(value)
                            except ValueError:
                                pass
                    self.sglang_rows.append(
                        {"sampled_at": sampled_at, "engine_id": engine_id, "metrics": selected}
                    )
                except Exception:
                    pass
            self.stop_event.wait(self.interval)


def _gpu_summary(rows: list[dict]) -> dict:
    result = {}
    for key in ("gpu_util", "memory_util", "memory_used_mib", "power_w"):
        values = [float(row[key]) for row in rows]
        result[key] = _stats(values) if values else {}
    return result


async def run(args: argparse.Namespace) -> None:
    workers = await _workers(None, args.worker_urls)
    args.engines = len(workers)
    groups = _make_scheduled_samples(args)
    gen_args = _generator_args(args)
    GenerateState(gen_args).aborted = False
    recorder = Recorder(workers)
    await recorder.start()
    _patch_posts(recorder)

    by_engine = [[] for _ in workers]
    by_id = {}
    for group in groups:
        for sample in group:
            engine = int(sample.index) % len(workers)
            by_engine[engine].append(sample)
            by_id[sample.metadata["benchmark_sample_id"]] = sample

    completed = []
    sample_rows = []
    sample_lock = asyncio.Lock()
    started = time.monotonic()
    started_wall = time.time()
    monitor = SystemMonitor(workers, args.monitor_interval)
    monitor.start()

    async def engine_slots(engine_id: int) -> None:
        queue = by_engine[engine_id]
        next_index = 0
        lock = asyncio.Lock()

        async def slot() -> None:
            nonlocal next_index
            while True:
                async with lock:
                    if next_index >= len(queue):
                        return
                    sample = queue[next_index]
                    next_index += 1
                sample_started = time.monotonic()
                sample_started_wall = time.time()
                await _one_sample(gen_args, sample, recorder, engine_id, 0)
                sample_completed_wall = time.time()
                sid = sample.metadata["benchmark_sample_id"]
                async with sample_lock:
                    completed.append(sid)
                    sample_rows.append(
                        {
                            "sample_id": sid,
                            "group_index": int(sample.group_index),
                            "engine_id": engine_id,
                            "domain": sample.metadata["task_type"],
                            "dispatch_time": sample_started_wall,
                            "completion_time": sample_completed_wall,
                            "end_to_end_time": time.monotonic() - sample_started,
                            "sample_time": float(getattr(sample, "sample_time", 0.0) or 0.0),
                            "tool_time": float(getattr(sample, "tool_time", 0.0) or 0.0),
                            "response_length": int(getattr(sample, "response_length", 0) or 0),
                            "status": sample.status.value,
                        }
                    )

        await asyncio.gather(*(slot() for _ in range(args.admission_per_engine)))

    try:
        await asyncio.gather(*(engine_slots(i) for i in range(len(workers))))
    finally:
        monitor.stop()
        await recorder.close()

    wall_time = time.monotonic() - started
    # For steady-32, later groups are replacement load only. Calls from the
    # initial 32 groups form the measured cohort and never see a drain phase.
    measurement_groups = args.measurement_groups or (args.groups - args.admission_per_engine)
    measured_ids = {
        sid
        for sid, sample in by_id.items()
        if int(sample.group_index) < measurement_groups
    }
    measured_calls = [row for row in recorder.calls if row["sample_id"] in measured_ids]
    domains = {sid: by_id[sid].metadata["task_type"] for sid in by_id}
    for row in recorder.calls:
        row["domain"] = domains[row["sample_id"]]
        row["measured_cohort"] = row["sample_id"] in measured_ids
    for row in sample_rows:
        row["measured_cohort"] = row["sample_id"] in measured_ids

    measured_samples = [row for row in sample_rows if row["measured_cohort"]]
    measurement_end = max(row["completion_time"] for row in measured_samples)
    measured_gpu_rows = [row for row in monitor.gpu_rows if started_wall <= row["sampled_at"] <= measurement_end]

    def summarize_calls(rows: list[dict]) -> dict:
        result = {}
        for domain in ("all", "math", "qa"):
            selected = rows if domain == "all" else [r for r in rows if domains[r["sample_id"]] == domain]
            result[domain] = {
                "request_time": _stats([float(r["request_time"]) for r in selected]),
                "queue_time": _stats([max(0.0, float(r["queue_time"])) for r in selected]),
                "completion_tokens": _stats([float(r["completion_tokens"]) for r in selected]),
                "generate_calls_per_sample": len(selected) / len({r["sample_id"] for r in selected}),
            }
        return result

    def summarize_samples(rows: list[dict]) -> dict:
        result = {}
        for domain in ("all", "math", "qa"):
            selected = rows if domain == "all" else [r for r in rows if r["domain"] == domain]
            result[domain] = {
                "end_to_end_time": _stats([float(r["end_to_end_time"]) for r in selected]),
                "sample_time": _stats([float(r["sample_time"]) for r in selected]),
                "tool_time": _stats([float(r["tool_time"]) for r in selected]),
                "response_length": _stats([float(r["response_length"]) for r in selected]),
            }
        return result

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": args.name,
        "schedule": args.schedule,
        "model": args.model,
        "model_precision": "bfloat16",
        "engines": len(workers),
        "rollout_tp": 1,
        "sglang_mem_fraction_static": args.mem_fraction_static,
        "context_length": args.context_length,
        "admission_per_engine": args.admission_per_engine,
        "samples_per_engine": args.groups,
        "continuous_replenishment": args.groups > args.admission_per_engine,
        "groups": args.groups,
        "samples_per_group": args.samples_per_group,
        "math_ratio": args.math_ratio,
        "mixed_generators": "generate_unified (ReTool + BrowseComp Search/Open)",
        "wall_time": wall_time,
        "measured_sample_count": len(measured_ids),
        "measured_generate_call_count": len(measured_calls),
        "measured": {
            "samples": summarize_samples(measured_samples),
            "calls": summarize_calls(measured_calls),
            "gpu": _gpu_summary(measured_gpu_rows),
            "measurement_wall_time": measurement_end - started_wall,
        },
        "all_including_replacement_drain": summarize_calls(recorder.calls),
    }
    (output / f"{args.name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / f"{args.name}.calls.jsonl").open("w") as handle:
        for row in recorder.calls:
            handle.write(json.dumps(row) + "\n")
    for suffix, rows in (
        ("samples", sample_rows),
        ("gpu", monitor.gpu_rows),
        ("sglang", monitor.sglang_rows),
    ):
        with (output / f"{args.name}.{suffix}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--worker-urls", nargs="+", required=True)
    p.add_argument("--output-dir", default=str(Path(__file__).parent / "debug" / "aligned_mixed_admission"))
    p.add_argument("--model", default="/workspace/Qwen3-8B")
    p.add_argument("--schedule", choices=("block4", "random"), required=True)
    p.add_argument("--math-data", default="/workspace/data/dapo-math-17k/dapo-math-17k.jsonl")
    p.add_argument("--qa-data", default="/workspace/data/browsecomp/bc_train.jsonl")
    p.add_argument("--groups", type=int, required=True)
    p.add_argument("--admission-per-engine", type=int, required=True)
    p.add_argument("--measurement-groups", type=int)
    p.add_argument("--samples-per-group", type=int, default=8)
    p.add_argument("--math-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-response-len", type=int, default=36864)
    p.add_argument("--context-length", type=int, default=40960)
    p.add_argument("--server-concurrency", type=int, default=32)
    p.add_argument("--mem-fraction-static", type=float, default=0.5)
    p.add_argument("--monitor-interval", type=float, default=1.0)
    p.add_argument("--router")
    p.add_argument("--partial", action="store_false", default=False)
    p.add_argument("--weight-label", default="base-qwen3-8b-bf16")
    p.add_argument("--topology-note", default="8 engines, TP1")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
