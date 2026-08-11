#!/usr/bin/env python3
"""Benchmark D-HBM -> P-HBM staging -> P pinned-host KV storage.

This is the same-node fallback when direct NIXL VRAM->remote DRAM selects a
slow transport.  A small fixed staging ring on P is filled over NVLink and
drained to P-owned pinned host memory using independent CUDA streams.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

GIB = 1 << 30
MIB = 1 << 20


def _numa_cpus(numa: int) -> set[int]:
    if numa == 0:
        wanted = set(range(0, 64)) | set(range(128, 192))
    elif numa == 1:
        wanted = set(range(64, 128)) | set(range(192, 256))
    else:
        raise ValueError(f"unsupported NUMA node {numa}")
    return wanted & os.sched_getaffinity(0)


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


def run(args) -> dict:
    d_gpus = [int(value) for value in args.d_gpus.split(",") if value]
    all_gpus = [args.p_gpu, *d_gpus]
    busy = [gpu for gpu in all_gpus if _gpu_busy(gpu)]
    if busy and not args.allow_busy:
        raise RuntimeError(f"refusing contaminated benchmark; busy GPUs={busy}")
    for gpu in d_gpus:
        if not torch.cuda.can_device_access_peer(args.p_gpu, gpu):
            raise RuntimeError(f"P GPU {args.p_gpu} cannot access D GPU {gpu}")

    cpus = _numa_cpus(args.p_host_numa)
    if cpus:
        os.sched_setaffinity(0, cpus)

    size_bytes = int(args.size_gib * GIB)
    chunk_bytes = args.chunk_mib * MIB
    if size_bytes % chunk_bytes:
        raise ValueError("size must be divisible by chunk size")
    chunks_per_writer = size_bytes // chunk_bytes

    sources = []
    for writer, gpu in enumerate(d_gpus):
        pattern = (17 + 31 * writer) % 251
        with torch.cuda.device(gpu):
            sources.append(
                torch.full(
                    (size_bytes,),
                    pattern,
                    dtype=torch.uint8,
                    device=f"cuda:{gpu}",
                )
            )
    with torch.cuda.device(args.p_gpu):
        targets = [
            torch.zeros(size_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in d_gpus
        ]
        staging = [
            torch.empty(
                chunk_bytes,
                dtype=torch.uint8,
                device=f"cuda:{args.p_gpu}",
            )
            for _ in range(args.staging_slots)
        ]
        nvlink_stream = torch.cuda.Stream(device=args.p_gpu)
        d2h_stream = torch.cuda.Stream(device=args.p_gpu)

    for gpu in all_gpus:
        torch.cuda.synchronize(gpu)

    records = []
    for iteration in range(args.iterations):
        slot_free_events = [None] * args.staging_slots
        sequence = 0
        started_at = time.perf_counter()
        for chunk in range(chunks_per_writer):
            start = chunk * chunk_bytes
            end = start + chunk_bytes
            for writer in range(len(d_gpus)):
                slot = sequence % args.staging_slots
                with torch.cuda.stream(nvlink_stream):
                    if slot_free_events[slot] is not None:
                        nvlink_stream.wait_event(slot_free_events[slot])
                    staging[slot].copy_(
                        sources[writer][start:end], non_blocking=True
                    )
                    nvlink_ready = torch.cuda.Event()
                    nvlink_ready.record(nvlink_stream)
                with torch.cuda.stream(d2h_stream):
                    d2h_stream.wait_event(nvlink_ready)
                    targets[writer][start:end].copy_(
                        staging[slot], non_blocking=True
                    )
                    slot_free = torch.cuda.Event()
                    slot_free.record(d2h_stream)
                    slot_free_events[slot] = slot_free
                sequence += 1
        d2h_stream.synchronize()
        nvlink_stream.synchronize()
        elapsed_s = time.perf_counter() - started_at
        records.append(
            {
                "iteration": iteration,
                "elapsed_s": elapsed_s,
                "aggregate_GiB_s": args.size_gib * len(d_gpus) / elapsed_s,
            }
        )

    # Model D releasing its request KV after the complete staging pipeline ACK.
    for gpu in d_gpus:
        torch.cuda.synchronize(gpu)
    del sources
    for gpu in d_gpus:
        with torch.cuda.device(gpu):
            torch.cuda.empty_cache()

    validation_started = time.perf_counter()
    validation = []
    for writer, target in enumerate(targets):
        pattern = (17 + 31 * writer) % 251
        validation.append(
            {
                "writer": writer,
                "pattern": pattern,
                "all_bytes_match_after_source_hbm_release": bool(
                    torch.all(target == pattern).item()
                ),
            }
        )
    validation_s = time.perf_counter() - validation_started
    if not all(item["all_bytes_match_after_source_hbm_release"] for item in validation):
        raise RuntimeError(f"data validation failed: {validation}")

    measured = records[args.warmup_iterations :]
    rates = [item["aggregate_GiB_s"] for item in measured]
    return {
        "p_gpu": args.p_gpu,
        "p_host_numa": args.p_host_numa,
        "d_gpus": d_gpus,
        "size_GiB_per_D": args.size_gib,
        "chunk_MiB": args.chunk_mib,
        "staging_slots": args.staging_slots,
        "p_hbm_staging_MiB": args.chunk_mib * args.staging_slots,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "aggregate_GiB_s_median": statistics.median(rates),
        "aggregate_GiB_s_min": min(rates),
        "aggregate_GiB_s_max": max(rates),
        "source_hbm_released_before_validation": True,
        "validation_s": validation_s,
        "validation": validation,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-gpu", type=int, default=0)
    parser.add_argument("--p-host-numa", type=int, default=0)
    parser.add_argument("--d-gpus", required=True)
    parser.add_argument("--size-gib", type=float, default=1.0)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--staging-slots", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_iterations >= args.iterations:
        raise ValueError("warmup_iterations must be smaller than iterations")
    result = run(args)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
