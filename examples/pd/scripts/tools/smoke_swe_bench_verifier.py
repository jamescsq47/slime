#!/usr/bin/env python3
"""Run an empty-patch or Oracle-patch SWE-bench verifier smoke test."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


PD_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PD_DIR.parents[1]
for path in (str(PD_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.swe_bench.harness import DockerTask, _image_name  # noqa: E402
from data.swe_bench.verifier import (  # noqa: E402
    capture_repository_patch,
    prepare_repository_baseline,
    run_inline_verifier,
)


def _read_instance(path: Path, instance_id: str) -> dict:
    with path.open(encoding="utf-8-sig") as source:
        if path.suffix.lower() == ".jsonl":
            rows = (json.loads(line) for line in source if line.strip())
        else:
            value = json.load(source)
            rows = iter(value if isinstance(value, list) else value["data"])
        for row in rows:
            if str(row.get("instance_id")) == instance_id:
                return dict(row)
    raise ValueError(f"instance {instance_id!r} was not found in {path}")


async def _run(args: argparse.Namespace) -> dict:
    row = _read_instance(args.dataset, args.instance_id)
    image = args.image or _image_name(row, {})
    task = DockerTask(
        image=image,
        command_timeout=args.command_timeout,
        start_timeout=args.start_timeout,
        network=args.network,
    )
    try:
        await task.start()
        baseline = await prepare_repository_baseline(task, str(row["base_commit"]))
        if args.patch == "oracle":
            oracle = str(row.get("patch") or "")
            if not oracle:
                raise ValueError(f"{args.instance_id} has no Oracle patch")
            await task.upload_bytes(oracle.encode(), "/tmp/oracle.patch")
            exit_code, output = await task.execute(
                "git apply /tmp/oracle.patch", timeout=args.command_timeout
            )
            if exit_code != 0:
                raise RuntimeError(f"Oracle patch failed to apply: {output[-4000:]}")
        model_patch = await capture_repository_patch(task, baseline.image_commit)
        result = await run_inline_verifier(
            task,
            row,
            model_patch,
            timeout_seconds=args.verifier_timeout,
        )
        return {
            "instance_id": args.instance_id,
            "image": image,
            "patch": args.patch,
            "patch_chars": len(model_patch),
            "baseline": {
                "kind": baseline.kind,
                "official_base_commit": baseline.official_base_commit,
                "image_commit": baseline.image_commit,
                "fingerprint": baseline.fingerprint,
            },
            "verifier": result.to_metadata(),
        }
    finally:
        await task.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--patch", choices=("empty", "oracle"), default="empty")
    parser.add_argument("--image", default="")
    parser.add_argument("--network", default="none")
    parser.add_argument("--command-timeout", type=float, default=300)
    parser.add_argument("--start-timeout", type=float, default=600)
    parser.add_argument("--verifier-timeout", type=float, default=2400)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
