#!/usr/bin/env python3
"""Preload SWE-bench instance images from GHCR into the local Docker cache.

The official harness expects Docker Hub-style local names.  Epoch Research
publishes all SWE-bench Verified images on GHCR, so this utility pulls that
public mirror and adds the exact local tag consumed by the PD harness.  A
JSONL audit log records cold-pull wall time separately from formal inference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any


def read_instances(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as source:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in source if line.strip()]
        else:
            value = json.load(source)
            rows = value if isinstance(value, list) else value["data"]
    result = [str(row["instance_id"]) for row in rows]
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate instance_id values in {path}")
    return result


def image_names(instance_id: str) -> tuple[str, str]:
    source = (
        "ghcr.io/epoch-research/"
        f"swe-bench.eval.x86_64.{instance_id.lower()}:latest"
    )
    local_id = instance_id.lower().replace("__", "_1776_")
    destination = f"docker.io/swebench/sweb.eval.x86_64.{local_id}:latest"
    return source, destination


async def command(*args: str, timeout: float) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise
    return process.returncode or 0, output.decode("utf-8", errors="replace")


async def image_exists(image: str) -> bool:
    code, _ = await command("docker", "image", "inspect", image, timeout=60)
    return code == 0


async def prefetch_one(
    instance_id: str,
    *,
    semaphore: asyncio.Semaphore,
    retries: int,
    timeout: float,
    keep_source_tag: bool,
) -> dict[str, Any]:
    source, destination = image_names(instance_id)
    started = time.monotonic()
    if await image_exists(destination):
        return {
            "instance_id": instance_id,
            "source": source,
            "destination": destination,
            "status": "already_present",
            "attempts": 0,
            "duration_seconds": time.monotonic() - started,
        }

    last_output = ""
    async with semaphore:
        for attempt in range(1, retries + 2):
            pull_started = time.monotonic()
            try:
                code, output = await command("docker", "pull", source, timeout=timeout)
            except TimeoutError:
                code, output = 124, f"docker pull exceeded {timeout:g}s"
            last_output = output
            if code == 0:
                tag_code, tag_output = await command(
                    "docker", "tag", source, destination, timeout=120
                )
                if tag_code != 0:
                    last_output = tag_output
                    break
                if not keep_source_tag:
                    await command("docker", "image", "rm", source, timeout=120)
                return {
                    "instance_id": instance_id,
                    "source": source,
                    "destination": destination,
                    "status": "downloaded",
                    "attempts": attempt,
                    "last_pull_seconds": time.monotonic() - pull_started,
                    "duration_seconds": time.monotonic() - started,
                }
            if attempt <= retries:
                await asyncio.sleep(min(60, 2 ** attempt))

    return {
        "instance_id": instance_id,
        "source": source,
        "destination": destination,
        "status": "failed",
        "attempts": retries + 1,
        "duration_seconds": time.monotonic() - started,
        "error_tail": last_output[-2000:],
    }


async def run(args: argparse.Namespace) -> int:
    instances = read_instances(args.dataset)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            prefetch_one(
                instance_id,
                semaphore=semaphore,
                retries=args.retries,
                timeout=args.timeout,
                keep_source_tag=args.keep_source_tag,
            )
        )
        for instance_id in instances
    ]
    counts: dict[str, int] = {}
    started = time.monotonic()
    with args.log.open("a", encoding="utf-8") as log:
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            status = str(result["status"])
            counts[status] = counts.get(status, 0) + 1
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            print(
                f"[{completed}/{len(tasks)}] {status}: {result['instance_id']} "
                f"({result['duration_seconds']:.1f}s)",
                flush=True,
            )
    summary = {
        "dataset": str(args.dataset),
        "total": len(instances),
        "counts": counts,
        "wall_seconds": time.monotonic() - started,
        "concurrency": args.concurrency,
        "log": str(args.log),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if counts.get("failed") else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--keep-source-tag", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
