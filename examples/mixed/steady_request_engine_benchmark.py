#!/usr/bin/env python3
"""Compare steady 32-request admission with a one-shot 16-request batch."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "avg": statistics.mean(values),
        "p50": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def load_prompts(path: Path, model: str, count: int) -> list[str]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["prompt"])
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = []
    for i in range(count):
        prompt = rows[i % len(rows)]
        if isinstance(prompt, list):
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


async def one_request(client: httpx.AsyncClient, url: str, prompt: str, index: int) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 8192,
        },
    }
    started = time.monotonic()
    response = await client.post(url + "/generate", json=payload)
    response.raise_for_status()
    elapsed = time.monotonic() - started
    body = response.json()
    meta = body.get("meta_info", {})
    return {
        "request_index": index,
        "request_time": elapsed,
        # Some SGLang retract paths expose a negative derived queue_time.
        # It is not a meaningful duration, so clamp it for aggregate metrics.
        "queue_time": max(0.0, float(meta.get("queue_time") or 0)),
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "completion_tokens": int(meta.get("completion_tokens") or 0),
        "finish_reason": (meta.get("finish_reason") or {}).get("type"),
    }


async def run_engine(
    client: httpx.AsyncClient,
    engine_id: int,
    url: str,
    prompts: list[str],
    concurrency: int,
    total: int,
) -> list[dict]:
    next_index = 0
    lock = asyncio.Lock()
    rows: list[dict] = []

    async def slot() -> None:
        nonlocal next_index
        while True:
            async with lock:
                if next_index >= total:
                    return
                index = next_index
                next_index += 1
            row = await one_request(client, url, prompts[index], index)
            row["engine_id"] = engine_id
            rows.append(row)

    await asyncio.gather(*(slot() for _ in range(concurrency)))
    return rows


async def run_condition(args: argparse.Namespace, name: str, concurrency: int, total: int) -> dict:
    prompt_count = total
    prompts = load_prompts(Path(args.prompt_data), args.model, prompt_count)
    timeout = httpx.Timeout(connect=30, read=3600, write=60, pool=60)
    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=256)
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        per_engine = await asyncio.gather(
            *(
                run_engine(client, engine_id, url.rstrip("/"), prompts, concurrency, total)
                for engine_id, url in enumerate(args.worker_urls)
            )
        )
    rows = [row for engine_rows in per_engine for row in engine_rows]
    # In the replenished condition, replacement requests exist to keep the
    # first cohort under steady load.  Do not include the final draining wave
    # in the headline latency; it would receive the same tail acceleration as
    # the one-shot baseline.
    measured_rows = [row for row in rows if row["request_index"] < concurrency]
    durations = [row["request_time"] for row in measured_rows]
    result = {
        "name": name,
        "engines": len(args.worker_urls),
        "concurrency_per_engine": concurrency,
        "requests_per_engine": total,
        "continuous_replenishment": total > concurrency,
        "wall_time": time.monotonic() - started,
        "measured_requests_per_engine": concurrency,
        "request_time": stats(durations),
        "queue_time": stats([row["queue_time"] for row in measured_rows]),
        "completion_tokens": stats([float(row["completion_tokens"]) for row in measured_rows]),
        "all_requests_including_drain": {
            "request_time": stats([row["request_time"] for row in rows]),
            "queue_time": stats([row["queue_time"] for row in rows]),
        },
        "rows": rows,
    }
    return result


async def main_async(args: argparse.Namespace) -> None:
    # 64 requests/engine gives the steady condition one full replacement wave.
    steady = await run_condition(args, "steady_32", 32, args.steady_requests_per_engine)
    one_shot = await run_condition(args, "one_shot_16", 16, 16)
    summary = {
        "model": args.model,
        "steady_32": {k: v for k, v in steady.items() if k != "rows"},
        "one_shot_16": {k: v for k, v in one_shot.items() if k != "rows"},
        "avg_ratio_steady32_over_oneshot16": (
            steady["request_time"]["avg"] / one_shot["request_time"]["avg"]
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for result in (steady, one_shot):
        path = output / f"{result['name']}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in result["rows"]))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-urls", nargs="+", required=True)
    parser.add_argument("--model", default="/workspace/Qwen3-8B-AWQ")
    parser.add_argument("--prompt-data", default="/workspace/data/dapo-math-17k/dapo-math-17k.jsonl")
    parser.add_argument("--steady-requests-per-engine", type=int, default=64)
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent / "debug" / "steady32_vs_oneshot16")
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
