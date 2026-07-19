#!/usr/bin/env python3
"""Real SGLang rollout A/B benchmark for mixed ReTool + BrowseComp samples.

This intentionally does not train or update weights.  It reuses the production
mixed generators, pins each prompt group to a concrete SGLang worker, and can
abort/recycle unfinished groups with the same Sample.Status.ABORTED contract as
fully_async_rollout.py.

Typical use (routers must contain exactly the requested number of workers):

  python examples/mixed/rollout_engine_ab_benchmark.py run \
    --name e16_no_partial --router http://127.0.0.1:30000 --engines 16
  python examples/mixed/rollout_engine_ab_benchmark.py run \
    --name e8_no_partial --router http://127.0.0.1:30000 --engines 8
  python examples/mixed/rollout_engine_ab_benchmark.py run \
    --name e8_partial --router http://127.0.0.1:30000 --engines 8 \
    --partial --abort-after 60 --max-aborts 2
  python examples/mixed/rollout_engine_ab_benchmark.py summarize --output-dir DIR

SGLang workers must be started with metrics enabled (Slime does this by
default).  No API keys are accepted as command-line arguments or written to
the result files; BrowseComp reads its existing environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
for import_dir in (SCRIPT_DIR, SCRIPT_DIR.parent / "browsecomp"):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from generate_with_hybrid import generate_unified  # noqa: E402
from slime.rollout.sglang_rollout import GenerateState  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


CURRENT_SAMPLE: contextvars.ContextVar[str] = contextvars.ContextVar("bench_sample")
CURRENT_ENGINE: contextvars.ContextVar[int] = contextvars.ContextVar("bench_engine")
CURRENT_SEGMENT: contextvars.ContextVar[int] = contextvars.ContextVar("bench_segment")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "avg": statistics.mean(values) if values else 0.0,
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "max": max(values) if values else 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass
class Recorder:
    worker_urls: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)
    call_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tool_events: dict[str, list[tuple[int, float]]] = field(default_factory=lambda: defaultdict(list))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=20, read=3600, write=60, pool=60),
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512),
            trust_env=False,
        )

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()

    async def post(self, url: str, payload: Any, max_retries: int = 60, headers: Any = None) -> Any:
        assert self.client is not None
        sample_id = CURRENT_SAMPLE.get()
        engine_id = CURRENT_ENGINE.get()
        segment_index = CURRENT_SEGMENT.get()
        target = self.worker_urls[engine_id] + "/generate" if url.endswith("/generate") else url
        start_wall, start_mono = time.time(), time.monotonic()
        response = None
        for attempt in range(max_retries):
            try:
                response = await self.client.post(target, json=payload or {}, headers=headers)
                response.raise_for_status()
                output = response.json()
                break
            except Exception:
                if attempt + 1 == max_retries:
                    raise
                await asyncio.sleep(1)
        end_mono, end_wall = time.monotonic(), time.time()
        meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
        prefill_finished = meta.get("prefill_finished_ts")
        decode_finished = meta.get("decode_finished_ts")
        request_received = meta.get("request_received_ts")
        prefill_time = meta.get("prefill_launch_latency")
        if prefill_time is None and prefill_finished is not None and request_received is not None:
            prefill_time = max(0.0, prefill_finished - request_received - float(meta.get("queue_time") or 0.0))
        decode_time = None
        if decode_finished is not None and prefill_finished is not None:
            decode_time = max(0.0, decode_finished - prefill_finished)
        if decode_time is None and meta.get("decode_throughput"):
            decode_time = float(meta.get("completion_tokens") or 0) / float(meta["decode_throughput"])
        async with self.lock:
            call_index = self.call_counts[sample_id]
            self.call_counts[sample_id] += 1
            self.calls.append(
                {
                    "sample_id": sample_id,
                    "engine_id": engine_id,
                    "generate_call_index": call_index,
                    "segment_index": segment_index,
                    "request_start": start_wall,
                    "request_end": end_wall,
                    "request_time": end_mono - start_mono,
                    "queue_time": float(meta.get("queue_time") or 0.0),
                    "prefill_time": float(prefill_time or 0.0),
                    "decode_time": float(decode_time or 0.0),
                    "sglang_request_id": meta.get("id"),
                    "finish_reason": (meta.get("finish_reason") or {}).get("type"),
                    "prompt_tokens": int(meta.get("prompt_tokens") or 0),
                    "completion_tokens": int(meta.get("completion_tokens") or 0),
                }
            )
        return output


def _patch_posts(recorder: Recorder) -> None:
    # Both production generators import post into their module namespace.
    import browsecomp_agent
    import browsecomp_env
    import generate_with_retool

    browsecomp_agent.post = recorder.post
    generate_with_retool.post = recorder.post

    original_run_action = browsecomp_env.BrowseCompEnv.run_action

    async def timed_run_action(env: Any, response: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            return await original_run_action(env, response)
        finally:
            sid = CURRENT_SAMPLE.get()
            # run_action follows the generate call whose tool request it parses.
            call_index = max(0, recorder.call_counts[sid] - 1)
            recorder.tool_events[sid].append((call_index, time.monotonic() - started))

    browsecomp_env.BrowseCompEnv.run_action = timed_run_action


async def _workers(router: str | None, explicit_urls: list[str] | None = None) -> list[str]:
    if explicit_urls:
        return [url.rstrip("/") for url in explicit_urls]
    if not router:
        raise ValueError("Provide --router or --worker-urls")
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        for endpoint in ("/workers", "/list_workers"):
            response = await client.get(router.rstrip("/") + endpoint)
            if response.status_code != 200:
                continue
            body = response.json()
            urls = body.get("urls") or [item["url"] for item in body.get("workers", [])]
            if urls:
                return [url.rstrip("/") for url in urls]
    raise RuntimeError(f"No workers returned by {router}")


def _make_samples(opts: argparse.Namespace) -> list[list[Sample]]:
    # Mirror CustomDataSource._mix_samples exactly: independently randomized
    # domain pools plus a seeded stochastic task sequence.  Materialize this
    # once, then reuse it unchanged for every condition.
    math_rows = _read_jsonl(Path(opts.math_data))
    qa_rows = _read_jsonl(Path(opts.qa_data))
    rng = random.Random(opts.seed)
    math_rows = sorted(math_rows, key=lambda _: rng.random())
    qa_rows = sorted(qa_rows, key=lambda _: rng.random())
    total_len = max(math.ceil(len(math_rows) / opts.math_ratio), math.ceil(len(qa_rows) / (1 - opts.math_ratio)))
    target_math = round(total_len * opts.math_ratio)
    task_sequence = ["math"] * target_math + ["qa"] * (total_len - target_math)
    random.Random(opts.seed).shuffle(task_sequence)
    math_index = qa_index = 0
    rows = []
    for domain in task_sequence[: opts.groups]:
        if domain == "math":
            rows.append((domain, math_rows[math_index % len(math_rows)]))
            math_index += 1
        else:
            rows.append((domain, qa_rows[qa_index % len(qa_rows)]))
            qa_index += 1
    groups: list[list[Sample]] = []
    for group_index, (domain, row) in enumerate(rows):
        metadata = dict(row.get("metadata") or {})
        metadata.update({"task_type": domain, "benchmark_group_index": group_index})
        group = []
        for sample_in_group in range(opts.samples_per_group):
            sample_id = f"g{group_index:03d}-s{sample_in_group:02d}"
            sample = Sample(
                group_index=group_index,
                index=group_index * opts.samples_per_group + sample_in_group,
                prompt=row["prompt"],
                label=row.get("label"),
                metadata={**metadata, "benchmark_sample_id": sample_id},
            )
            group.append(sample)
        groups.append(group)
    return groups


def _generator_args(opts: argparse.Namespace) -> SimpleNamespace:
    parsed = urlparse(opts.router or "http://127.0.0.1:1")
    return SimpleNamespace(
        hf_checkpoint=opts.model,
        sglang_router_ip=parsed.hostname or "127.0.0.1",
        sglang_router_port=parsed.port or 80,
        sglang_server_concurrency=opts.server_concurrency,
        rollout_num_gpus=opts.engines,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=opts.temperature,
        rollout_top_p=opts.top_p,
        rollout_top_k=opts.top_k,
        rollout_max_response_len=opts.max_response_len,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        rollout_seed=opts.seed,
        n_samples_per_prompt=opts.samples_per_group,
        partial_rollout=opts.partial,
        mask_offpolicy_in_partial_rollout=False,
        mask_offpolicy_math=None,
        mask_offpolicy_qa=None,
        rollout_max_context_len=opts.context_length,
        sglang_context_length=opts.context_length,
        context_parallel_size=1,
        max_tokens_per_gpu=opts.context_length,
        max_seq_len=opts.context_length,
        sglang_enable_deterministic_inference=False,
        sglang_speculative_algorithm=None,
        enable_tool_delay=False,
        ci_test=False,
    )


async def _one_sample(args: Any, sample: Sample, recorder: Recorder, engine_id: int, segment: int) -> Sample:
    sample_id = sample.metadata["benchmark_sample_id"]
    tokens = [
        CURRENT_SAMPLE.set(sample_id),
        CURRENT_ENGINE.set(engine_id),
        CURRENT_SEGMENT.set(segment),
    ]
    try:
        return await generate_unified(args, sample, GenerateState(args).sampling_params.copy())
    finally:
        CURRENT_SAMPLE.reset(tokens[0])
        CURRENT_ENGINE.reset(tokens[1])
        CURRENT_SEGMENT.reset(tokens[2])


async def _one_group(args: Any, group: list[Sample], recorder: Recorder, engines: int, segment: int):
    # A recycled group can contain siblings that finished just before abort_all.
    # Keep those results and resume only PENDING/ABORTED members.
    active = [sample for sample in group if sample.status in (Sample.Status.PENDING, Sample.Status.ABORTED)]
    if active:
        await asyncio.gather(
            *[_one_sample(args, sample, recorder, int(sample.index) % engines, segment) for sample in active]
        )
    return group


async def _abort_workers(recorder: Recorder) -> None:
    assert recorder.client is not None
    await asyncio.gather(
        *[recorder.client.post(url + "/abort_request", json={"abort_all": True}) for url in recorder.worker_urls]
    )


async def run_experiment(opts: argparse.Namespace) -> None:
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = await _workers(opts.router, opts.worker_urls)
    if len(workers) != opts.engines:
        raise RuntimeError(f"Found {len(workers)} workers; experiment requires exactly {opts.engines}")

    groups = _make_samples(opts)
    args = _generator_args(opts)
    state = GenerateState(args)
    state.aborted = False
    recorder = Recorder(workers)
    await recorder.start()
    _patch_posts(recorder)
    experiment_start = time.time()
    sample_first_dispatch = {s.metadata["benchmark_sample_id"]: experiment_start for g in groups for s in g}
    sample_completed: dict[str, float] = {}
    buffer_wait = defaultdict(float)
    partial_count = defaultdict(int)
    segment_index = defaultdict(int)
    pending = list(groups)
    completed: list[list[Sample]] = []
    abort_count = 0
    try:
        while pending:
            tasks = {
                asyncio.create_task(
                    _one_group(args, group, recorder, opts.engines, segment_index[group[0].group_index])
                ): group
                for group in pending
            }
            abort_task = None
            if opts.partial and abort_count < opts.max_aborts:
                abort_task = asyncio.create_task(asyncio.sleep(opts.abort_after))
            if abort_task:
                done, _ = await asyncio.wait([*tasks, abort_task], return_when=asyncio.FIRST_COMPLETED)
                while abort_task not in done and tasks:
                    finished = {task for task in done if task in tasks}
                    for task in finished:
                        group = list(task.result())
                        completed.append(group)
                        finished_at = time.time()
                        for sample in group:
                            sample_completed[sample.metadata["benchmark_sample_id"]] = finished_at
                        tasks.pop(task)
                    if not tasks:
                        abort_task.cancel()
                        break
                    done, _ = await asyncio.wait([*tasks, abort_task], return_when=asyncio.FIRST_COMPLETED)
                if abort_task in done and tasks:
                    abort_count += 1
                    state.aborted = True
                    await _abort_workers(recorder)
            results = await asyncio.gather(*tasks.keys()) if tasks else []
            next_pending = []
            for result in results:
                group = list(result)
                aborted = [sample for sample in group if sample.status == Sample.Status.ABORTED]
                if aborted and opts.partial:
                    now = time.monotonic()
                    for sample in aborted:
                        sid = sample.metadata["benchmark_sample_id"]
                        partial_count[sid] += 1
                        sample.metadata["benchmark_buffer_enter"] = now
                    next_pending.append(group)
                else:
                    completed.append(group)
                    finished_at = time.time()
                    for sample in group:
                        sample_completed[sample.metadata["benchmark_sample_id"]] = finished_at
            if next_pending:
                await asyncio.sleep(opts.buffer_delay)
                now = time.monotonic()
                for group in next_pending:
                    segment_index[group[0].group_index] += 1
                    for sample in group:
                        entered = sample.metadata.pop("benchmark_buffer_enter", None)
                        if entered is not None:
                            buffer_wait[sample.metadata["benchmark_sample_id"]] += now - entered
                state.aborted = False
            pending = next_pending
    finally:
        await recorder.close()

    experiment_end = time.time()
    samples = [sample for group in completed for sample in group]
    by_id = {sample.metadata["benchmark_sample_id"]: sample for sample in samples}
    tool_by_call: dict[tuple[str, int], float] = defaultdict(float)
    for sid, events in recorder.tool_events.items():
        for call_index, duration in events:
            tool_by_call[(sid, call_index)] += duration

    for call in recorder.calls:
        sample = by_id[call["sample_id"]]
        call["domain"] = sample.metadata["task_type"]
        call["tool_time"] = tool_by_call[(call["sample_id"], call["generate_call_index"])]
        call["partial_count"] = partial_count[call["sample_id"]]
        call["buffer_wait_time"] = buffer_wait[call["sample_id"]]
        call["end_to_end_time"] = sample_completed[call["sample_id"]] - sample_first_dispatch[call["sample_id"]]

    calls_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in recorder.calls:
        calls_by_sample[call["sample_id"]].append(call)
    active_time: dict[str, float] = {}
    tool_time: dict[str, float] = {}
    for sid, calls in calls_by_sample.items():
        running = 0.0
        for call in sorted(calls, key=lambda row: row["generate_call_index"]):
            running += call["request_time"] + call["tool_time"]
            call["cumulative_sample_time"] = running
        # Both production generators record whole-segment active and tool time.
        sample_tool = float(getattr(by_id[sid], "tool_time", 0.0) or 0.0)
        attributed_tool = sum(call["tool_time"] for call in calls)
        if sample_tool > attributed_tool:
            residual_tool = sample_tool - attributed_tool
            calls[-1]["tool_time"] += residual_tool
            running += residual_tool
            calls[-1]["cumulative_sample_time"] = running
        authoritative_time = float(getattr(by_id[sid], "sample_time", 0.0) or running)
        calls[-1]["cumulative_sample_time"] = authoritative_time
        active_time[sid] = authoritative_time
        tool_time[sid] = sum(call["tool_time"] for call in calls)

    sample_rows = []
    for sid, sample in sorted(by_id.items()):
        sample_rows.append(
            {
                "experiment": opts.name,
                "sample_id": sid,
                "domain": sample.metadata["task_type"],
                "engine_id": int(sample.index) % opts.engines,
                "generate_count": recorder.call_counts[sid],
                "partial_count": partial_count[sid],
                "buffer_wait_time": buffer_wait[sid],
                "cumulative_sample_time": active_time[sid],
                "end_to_end_time": sample_completed[sid] - sample_first_dispatch[sid],
                "tool_time": tool_time[sid],
                "status": sample.status.value,
                "response_length": sample.response_length,
            }
        )
    prefix = output_dir / opts.name
    with (prefix.with_suffix(".calls.jsonl")).open("w", encoding="utf-8") as handle:
        for row in recorder.calls:
            handle.write(json.dumps({"experiment": opts.name, **row}, ensure_ascii=False) + "\n")
    with (prefix.with_suffix(".samples.jsonl")).open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for suffix, rows in (("calls.csv", recorder.calls), ("samples.csv", sample_rows)):
        with (output_dir / f"{opts.name}.{suffix}").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "name": opts.name,
        "model": opts.model,
        "weight_label": opts.weight_label,
        "topology_note": opts.topology_note,
        "workers": workers,
        "engine_count": opts.engines,
        "groups": opts.groups,
        "samples_per_group": opts.samples_per_group,
        "math_groups": sum(group[0].metadata["task_type"] == "math" for group in groups),
        "partial": opts.partial,
        "abort_after": opts.abort_after if opts.partial else None,
        "abort_count": abort_count,
        "sampling": {"temperature": opts.temperature, "top_p": opts.top_p, "top_k": opts.top_k},
        "wall_time": experiment_end - experiment_start,
    }
    _atomic_json(output_dir / f"{opts.name}.manifest.json", manifest)
    summarize(output_dir)


def summarize(output_dir: Path) -> None:
    result: dict[str, Any] = {}
    for path in sorted(output_dir.glob("*.samples.jsonl")):
        name = path.name.removesuffix(".samples.jsonl")
        rows = _read_jsonl(path)
        result[name] = {}
        for domain in ("math", "qa", "all"):
            selected = rows if domain == "all" else [row for row in rows if row["domain"] == domain]
            sample_times = [float(row["cumulative_sample_time"]) for row in selected]
            e2e_times = [float(row["end_to_end_time"]) for row in selected]
            result[name][domain] = {
                "sample_time": _stats(sample_times),
                "end_to_end_time": _stats(e2e_times),
                "generate_count": _stats([float(row["generate_count"]) for row in selected]),
                "partial_count": _stats([float(row["partial_count"]) for row in selected]),
                "buffer_wait_time": _stats([float(row["buffer_wait_time"]) for row in selected]),
                "time_share": {
                    "tool": sum(float(row["tool_time"]) for row in selected) / sum(sample_times) if sum(sample_times) else 0.0,
                    "buffer_over_e2e": sum(float(row["buffer_wait_time"]) for row in selected) / sum(e2e_times) if sum(e2e_times) else 0.0,
                },
            }
        calls_path = output_dir / f"{name}.calls.jsonl"
        calls = _read_jsonl(calls_path) if calls_path.exists() else []
        for domain in ("math", "qa", "all"):
            selected = calls if domain == "all" else [row for row in calls if row["domain"] == domain]
            denom = sum(float(row["request_time"]) for row in selected)
            result[name][domain]["request_time"] = _stats(
                [float(row["request_time"]) for row in selected]
            )
            result[name][domain]["time_share"].update(
                {
                    key: sum(float(row[key]) for row in selected) / denom if denom else 0.0
                    for key in ("queue_time", "prefill_time", "decode_time")
                }
            )
    names = list(result)
    comparisons = {}
    if "e16_no_partial" in result and "e8_no_partial" in result:
        comparisons["8_vs_16"] = {
            domain: {
                stat: result["e8_no_partial"][domain]["sample_time"][stat]
                / result["e16_no_partial"][domain]["sample_time"][stat]
                if result["e16_no_partial"][domain]["sample_time"][stat]
                else 0.0
                for stat in ("avg", "p50", "p90", "max")
            }
            for domain in ("math", "qa", "all")
        }
    if "e8_no_partial" in result and "e8_partial" in result:
        comparisons["partial_vs_8"] = {
            domain: {
                stat: result["e8_partial"][domain]["sample_time"][stat]
                / result["e8_no_partial"][domain]["sample_time"][stat]
                if result["e8_no_partial"][domain]["sample_time"][stat]
                else 0.0
                for stat in ("avg", "p50", "p90", "max")
            }
            for domain in ("math", "qa", "all")
        }
    if "e16_no_partial" in result and "e8_partial" in result:
        comparisons["request_time_32_partial_vs_16"] = {
            domain: {
                stat: result["e8_partial"][domain]["request_time"][stat]
                / result["e16_no_partial"][domain]["request_time"][stat]
                if result["e16_no_partial"][domain]["request_time"][stat]
                else 0.0
                for stat in ("avg", "p50", "p90", "max")
            }
            for domain in ("math", "qa", "all")
        }
    _atomic_json(output_dir / "summary.json", {"experiments": result, "comparisons": comparisons, "found": names})


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--router")
    run.add_argument("--worker-urls", nargs="+", help="Explicit Slime worker URLs; bypass router discovery")
    run.add_argument("--engines", type=int, required=True, choices=(8, 16))
    run.add_argument("--partial", action="store_true")
    run.add_argument("--abort-after", type=float, default=60.0)
    run.add_argument("--max-aborts", type=int, default=2)
    run.add_argument("--buffer-delay", type=float, default=1.0)
    run.add_argument("--output-dir", default=str(SCRIPT_DIR / "debug" / "rollout_engine_ab"))
    run.add_argument("--model", default="/workspace/Qwen3-8B")
    run.add_argument("--weight-label", default="", help="Slime REF_LOAD identity recorded for comparability")
    run.add_argument("--topology-note", default="")
    run.add_argument("--math-data", default="/workspace/data/dapo-math-17k/dapo-math-17k.jsonl")
    run.add_argument("--qa-data", default="/workspace/data/browsecomp/bc_train.jsonl")
    run.add_argument("--groups", type=int, default=32)
    run.add_argument("--math-ratio", type=float, default=0.5)
    run.add_argument("--samples-per-group", type=int, default=8)
    run.add_argument("--seed", type=int, default=47)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--top-k", type=int, default=-1)
    run.add_argument("--max-response-len", type=int, default=36864)
    run.add_argument("--context-length", type=int, default=40960)
    run.add_argument("--server-concurrency", type=int, default=32)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output-dir", required=True)
    return root


def main() -> None:
    opts = parser().parse_args()
    if opts.command == "run":
        if opts.partial and opts.name == "e8_no_partial":
            raise SystemExit("e8_no_partial must not use --partial")
        asyncio.run(run_experiment(opts))
    else:
        summarize(Path(opts.output_dir))


if __name__ == "__main__":
    main()
