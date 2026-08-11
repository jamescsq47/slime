#!/usr/bin/env python3
"""Measure simultaneous P GPU<->Host copies on independent CUDA streams."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch


MIB = 1 << 20
GIB = 1 << 30


def _gpu_busy(gpu: int) -> bool:
    uuid = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    active = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return uuid in {line.strip() for line in active if line.strip()}


def _measure(size_bytes: int, d2h_stream, h2d_stream, *, serial: bool) -> float:
    del size_bytes
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.cuda.stream(d2h_stream):
        d2h_done = torch.cuda.Event()
        _measure.d2h_host.copy_(_measure.d2h_gpu, non_blocking=True)
        d2h_done.record(d2h_stream)

    if serial:
        h2d_stream.wait_event(d2h_done)
    with torch.cuda.stream(h2d_stream):
        _measure.h2d_gpu.copy_(_measure.h2d_host, non_blocking=True)

    d2h_stream.synchronize()
    h2d_stream.synchronize()
    return time.perf_counter() - started


def _measure_one(stream, copy_fn) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        copy_fn()
    stream.synchronize()
    return time.perf_counter() - started


def run(args) -> dict:
    if _gpu_busy(args.gpu) and not args.allow_busy:
        raise RuntimeError(f"refusing contaminated benchmark; GPU {args.gpu} is busy")
    size_bytes = args.size_mib * MIB
    with torch.cuda.device(args.gpu):
        _measure.d2h_gpu = torch.full(
            (size_bytes,), 17, dtype=torch.uint8, device=f"cuda:{args.gpu}"
        )
        _measure.h2d_gpu = torch.zeros(
            size_bytes, dtype=torch.uint8, device=f"cuda:{args.gpu}"
        )
        _measure.d2h_host = torch.zeros(size_bytes, dtype=torch.uint8, pin_memory=True)
        _measure.h2d_host = torch.full(
            (size_bytes,), 91, dtype=torch.uint8, pin_memory=True
        )
        serial_stream = torch.cuda.Stream(device=args.gpu, priority=0)
        d2h_stream = torch.cuda.Stream(device=args.gpu, priority=0)
        h2d_stream = torch.cuda.Stream(device=args.gpu, priority=-1)

    torch.cuda.synchronize(args.gpu)
    records = {"d2h": [], "h2d": [], "serial": [], "duplex": []}
    total_iterations = args.warmup_iterations + args.iterations
    for iteration in range(total_iterations):
        d2h_elapsed = _measure_one(
            d2h_stream,
            lambda: _measure.d2h_host.copy_(_measure.d2h_gpu, non_blocking=True),
        )
        h2d_elapsed = _measure_one(
            h2d_stream,
            lambda: _measure.h2d_gpu.copy_(_measure.h2d_host, non_blocking=True),
        )
        serial_elapsed = _measure(
            size_bytes, serial_stream, serial_stream, serial=True
        )
        duplex_elapsed = _measure(
            size_bytes, d2h_stream, h2d_stream, serial=False
        )
        if iteration >= args.warmup_iterations:
            records["d2h"].append(d2h_elapsed)
            records["h2d"].append(h2d_elapsed)
            records["serial"].append(serial_elapsed)
            records["duplex"].append(duplex_elapsed)

    torch.cuda.synchronize(args.gpu)
    valid = {
        "d2h": bool(torch.all(_measure.d2h_host == 17).item()),
        "h2d": bool(torch.all(_measure.h2d_gpu == 91).item()),
    }
    if not all(valid.values()):
        raise RuntimeError(f"bidirectional data validation failed: {valid}")

    def summarize(values, directions=1):
        elapsed = statistics.median(values)
        return {
            "median_seconds": elapsed,
            "payload_GiB_s": (directions * size_bytes / GIB) / elapsed,
            "samples_seconds": values,
        }

    d2h = summarize(records["d2h"])
    h2d = summarize(records["h2d"])
    serial = summarize(records["serial"], directions=2)
    duplex = summarize(records["duplex"], directions=2)
    return {
        "gpu": args.gpu,
        "size_MiB_per_direction": args.size_mib,
        "d2h_stream_priority": d2h_stream.priority,
        "h2d_stream_priority": h2d_stream.priority,
        "d2h_only": d2h,
        "h2d_only": h2d,
        "serial": serial,
        "duplex": duplex,
        "duplex_wall_time_speedup": (
            serial["median_seconds"] / duplex["median_seconds"]
        ),
        "duplex_overlap_efficiency": (
            max(d2h["median_seconds"], h2d["median_seconds"])
            / duplex["median_seconds"]
        ),
        "validation": valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
