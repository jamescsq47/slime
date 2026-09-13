"""Run the existing evaluator, persisting every finished episode immediately.

No harness, request, sampling, or KV lifecycle changes. This entrypoint is opt-in
and does not change other experiments using inference.py directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import inference  # noqa: E402


def append_record(path: Path, record: dict, lock: threading.Lock) -> None:
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cli = inference.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    path = cli.output_dir / "requests.completed.jsonl"
    if path.exists():
        raise RuntimeError(f"Refusing to mix completed episode records: {path}")
    lock = threading.Lock()
    original = inference.run_one

    async def recorded_run_one(*args, **kwargs):
        record = await original(*args, **kwargs)
        await asyncio.to_thread(append_record, path, record, lock)
        return record

    inference.run_one = recorded_run_one
    asyncio.run(inference.async_main(cli))


if __name__ == "__main__":
    main()
