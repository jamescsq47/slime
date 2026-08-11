#!/usr/bin/env python3
"""Measure the GPU-memory paths used by the agentic PD KV pipeline.

The reported rate is payload bytes divided by elapsed time.  For a local HBM
copy this is intentionally different from the DRAM pin bandwidth convention:
copying N bytes causes N bytes of reads and N bytes of writes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch


GB = 1_000_000_000
GIB = 1 << 30


def gpu_has_compute_process(gpu: int) -> bool:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    if not query:
        return False
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
    return uuid in {line.strip() for line in query if line.strip()}


def rate_record(path: str, size: int, repeats: int, elapsed_s: float, **extra):
    payload = size * repeats
    return {
        "path": path,
        "size_bytes": size,
        "repeats": repeats,
        "elapsed_s": elapsed_s,
        "payload_GB_s": payload / elapsed_s / GB,
        "payload_GiB_s": payload / elapsed_s / GIB,
        **extra,
    }


def cuda_timed_copy(dst, src, stream, repeats: int) -> float:
    with torch.cuda.stream(stream):
        for _ in range(3):
            dst.copy_(src, non_blocking=True)
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(repeats):
            dst.copy_(src, non_blocking=True)
        stop.record(stream)
    stop.synchronize()
    return start.elapsed_time(stop) / 1_000


def benchmark_size(src_gpu: int, dst_gpu: int, size: int, repeats: int):
    records = []

    with torch.cuda.device(src_gpu):
        src = torch.empty(size, dtype=torch.uint8, device=f"cuda:{src_gpu}")
        local_dst = torch.empty_like(src)
        local_stream = torch.cuda.Stream(device=src_gpu)
        elapsed = cuda_timed_copy(local_dst, src, local_stream, repeats)
        records.append(
            rate_record("HBM local copy", size, repeats, elapsed, gpu=src_gpu)
        )

        pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        elapsed = cuda_timed_copy(src, pinned, local_stream, repeats)
        records.append(rate_record("Host pinned -> HBM", size, repeats, elapsed, gpu=src_gpu))
        elapsed = cuda_timed_copy(pinned, src, local_stream, repeats)
        records.append(rate_record("HBM -> Host pinned", size, repeats, elapsed, gpu=src_gpu))

    with torch.cuda.device(dst_gpu):
        dst = torch.empty(size, dtype=torch.uint8, device=f"cuda:{dst_gpu}")
        reverse_src = torch.empty_like(dst)
        dst_stream = torch.cuda.Stream(device=dst_gpu)
        elapsed = cuda_timed_copy(dst, src, dst_stream, repeats)
        records.append(
            rate_record(
                "GPU P2P one-way",
                size,
                repeats,
                elapsed,
                src_gpu=src_gpu,
                dst_gpu=dst_gpu,
            )
        )

    with torch.cuda.device(src_gpu):
        reverse_dst = torch.empty_like(src)
        src_stream = torch.cuda.Stream(device=src_gpu)

    # Two independent copy engines, one transfer in each direction.  Wall time
    # is used because CUDA events live on different devices.
    for _ in range(3):
        with torch.cuda.stream(dst_stream):
            dst.copy_(src, non_blocking=True)
        with torch.cuda.stream(src_stream):
            reverse_dst.copy_(reverse_src, non_blocking=True)
    torch.cuda.synchronize(src_gpu)
    torch.cuda.synchronize(dst_gpu)
    start = time.perf_counter()
    for _ in range(repeats):
        with torch.cuda.stream(dst_stream):
            dst.copy_(src, non_blocking=True)
        with torch.cuda.stream(src_stream):
            reverse_dst.copy_(reverse_src, non_blocking=True)
    torch.cuda.synchronize(src_gpu)
    torch.cuda.synchronize(dst_gpu)
    elapsed = time.perf_counter() - start
    records.append(
        rate_record(
            "GPU P2P full-duplex aggregate",
            size * 2,
            repeats,
            elapsed,
            src_gpu=src_gpu,
            dst_gpu=dst_gpu,
        )
    )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-gpu", type=int, default=0)
    parser.add_argument("--dst-gpu", type=int, default=1)
    parser.add_argument("--sizes-mib", default="64,256,1024")
    parser.add_argument("--target-copy-gib", type=float, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not args.allow_busy:
        busy = [gpu for gpu in (args.src_gpu, args.dst_gpu) if gpu_has_compute_process(gpu)]
        if busy:
            raise SystemExit(f"Refusing a contaminated benchmark; GPUs are busy: {busy}")

    for src, dst in ((args.src_gpu, args.dst_gpu), (args.dst_gpu, args.src_gpu)):
        if not torch.cuda.can_device_access_peer(src, dst):
            raise SystemExit(f"CUDA peer access is unavailable: GPU {src} -> GPU {dst}")

    results = {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "src_gpu": args.src_gpu,
        "dst_gpu": args.dst_gpu,
        "devices": {
            str(i): torch.cuda.get_device_name(i) for i in (args.src_gpu, args.dst_gpu)
        },
        "records": [],
    }
    for size_mib in [int(x) for x in args.sizes_mib.split(",")]:
        size = size_mib << 20
        repeats = max(4, int(args.target_copy_gib * GIB / size))
        results["records"].extend(
            benchmark_size(args.src_gpu, args.dst_gpu, size, repeats)
        )

    encoded = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
