#!/usr/bin/env python3
"""Aligned mixed-rollout admission benchmark: steady 32 vs one-shot 16."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from rollout_engine_ab_benchmark import (
    GenerateState,
    Recorder,
    _generator_args,
    _make_samples,
    _one_sample,
    _patch_posts,
    _workers,
)


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


async def run(args: argparse.Namespace) -> None:
    workers = await _workers(None, args.worker_urls)
    args.engines = len(workers)
    groups = _make_samples(args)
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
    started = time.monotonic()

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
                await _one_sample(gen_args, sample, recorder, engine_id, 0)
                completed.append(sample.metadata["benchmark_sample_id"])

        await asyncio.gather(*(slot() for _ in range(args.admission_per_engine)))

    try:
        await asyncio.gather(*(engine_slots(i) for i in range(len(workers))))
    finally:
        await recorder.close()

    wall_time = time.monotonic() - started
    # For steady-32, later groups are replacement load only. Calls from the
    # initial 32 groups form the measured cohort and never see a drain phase.
    measured_ids = {
        sid
        for sid, sample in by_id.items()
        if int(sample.group_index) < args.admission_per_engine
    }
    measured_calls = [row for row in recorder.calls if row["sample_id"] in measured_ids]
    domains = {sid: by_id[sid].metadata["task_type"] for sid in by_id}
    for row in recorder.calls:
        row["domain"] = domains[row["sample_id"]]
        row["measured_cohort"] = row["sample_id"] in measured_ids

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

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": args.name,
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
        "measured": summarize_calls(measured_calls),
        "all_including_replacement_drain": summarize_calls(recorder.calls),
    }
    (output / f"{args.name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / f"{args.name}.calls.jsonl").open("w") as handle:
        for row in recorder.calls:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--worker-urls", nargs="+", required=True)
    p.add_argument("--output-dir", default=str(Path(__file__).parent / "debug" / "aligned_mixed_admission"))
    p.add_argument("--model", default="/workspace/Qwen3-8B")
    p.add_argument("--math-data", default="/workspace/data/dapo-math-17k/dapo-math-17k.jsonl")
    p.add_argument("--qa-data", default="/workspace/data/browsecomp/bc_train.jsonl")
    p.add_argument("--groups", type=int, required=True)
    p.add_argument("--admission-per-engine", type=int, required=True)
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
    p.add_argument("--router")
    p.add_argument("--partial", action="store_false", default=False)
    p.add_argument("--weight-label", default="base-qwen3-8b-bf16")
    p.add_argument("--topology-note", default="8 engines, TP1")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
