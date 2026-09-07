#!/usr/bin/env python3
"""Benchmark direct CUDA DMA to a registered memfd-backed Host extent.

The production Shared Host Arena uses anonymous memfd DRAM.  This benchmark
registers one already-populated extent with the current CUDA context and copies
directly between that final address and HBM.  It intentionally does not use a
pinned bounce or a CPU memcpy, so it measures the ceiling for the proposed
simple registered-extent backend.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import statistics
import threading
import time

import torch


GIB = 1 << 30
MFD_CLOEXEC = 0x0001


def create_memfd() -> int:
    create = getattr(os, "memfd_create", None)
    if create is not None:
        return int(create("agentic-registered-host-benchmark", MFD_CLOEXEC))
    libc = ctypes.CDLL(None, use_errno=True)
    memfd_create = getattr(libc, "memfd_create", None)
    if memfd_create is None:
        raise RuntimeError("memfd_create is unavailable")
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    descriptor = int(
        memfd_create(b"agentic-registered-host-benchmark", MFD_CLOEXEC)
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return descriptor


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def measure_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    iterations: int,
    *,
    stream: torch.cuda.Stream,
) -> list[float]:
    samples = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            begin.record(stream)
            result = torch.mm(a, b)
            end.record(stream)
        # Synchronize only the Forward fence.  A device-wide synchronize would
        # incorrectly charge every outstanding DMA chunk to this GEMM.
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)) / 1000.0)
        del result
    return samples


def cuda_success(result) -> None:
    cudart = torch.cuda.cudart()
    if result != cudart.cudaError.success:
        raise RuntimeError(f"CUDA runtime call failed: {result}")


def copy_async(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    stream: torch.cuda.Stream,
    chunk_bytes: int,
    chunk_gap_seconds: float,
) -> tuple[float, float]:
    started = time.perf_counter()
    gpu_seconds = 0.0
    for offset in range(0, source.numel(), chunk_bytes):
        end_offset = min(offset + chunk_bytes, source.numel())
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            begin.record(stream)
            destination[offset:end_offset].copy_(
                source[offset:end_offset], non_blocking=True
            )
            end.record(stream)
        # Production progress likewise exposes at most one bounded physical
        # chunk before returning to Forward/control work.
        end.synchronize()
        gpu_seconds += float(begin.elapsed_time(end)) / 1000.0
        if chunk_gap_seconds > 0.0 and end_offset < source.numel():
            time.sleep(chunk_gap_seconds)
    return gpu_seconds, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-gib", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--chunk-gap-ms", type=float, default=0.0)
    parser.add_argument(
        "--concurrent-direction", choices=("d2h", "h2d"), default="d2h"
    )
    parser.add_argument("--baseline-gemms", type=int, default=30)
    parser.add_argument("--gemm-size", type=int, default=4096)
    parser.add_argument("--min-bandwidth-gib-s", type=float, default=10.0)
    parser.add_argument("--max-forward-p50-regression", type=float, default=0.05)
    parser.add_argument("--max-forward-p95-regression", type=float, default=0.10)
    parser.add_argument("--output")
    args = parser.parse_args()

    byte_size = int(args.size_gib * GIB)
    chunk_bytes = int(args.chunk_mib) << 20
    chunk_gap_seconds = float(args.chunk_gap_ms) / 1000.0
    if byte_size <= 0 or byte_size % mmap.ALLOCATIONGRANULARITY:
        raise ValueError("size must be positive and mmap-page aligned")
    if chunk_bytes <= 0 or chunk_bytes > byte_size:
        raise ValueError("chunk-mib must be positive and no larger than the extent")
    if chunk_gap_seconds < 0.0:
        raise ValueError("chunk-gap-ms must be non-negative")

    torch.cuda.set_device(0)
    descriptor = create_memfd()
    mapping = None
    raw = None
    registered = False
    try:
        os.ftruncate(descriptor, byte_size)
        os.posix_fallocate(descriptor, 0, byte_size)
        mapping = mmap.mmap(descriptor, byte_size, access=mmap.ACCESS_WRITE)
        raw = torch.frombuffer(mapping, dtype=torch.uint8, count=byte_size)
        raw.zero_()

        register_started = time.perf_counter()
        cuda_success(
            torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), byte_size, 0)
        )
        registered = True
        registration_seconds = time.perf_counter() - register_started

        source = torch.full((byte_size,), 0x5A, dtype=torch.uint8, device="cuda")
        destination = torch.empty_like(source)
        stream = torch.cuda.Stream(priority=0)
        forward_stream = torch.cuda.Stream(priority=0)

        # Warm both directions and validate that the externally registered
        # tensor is accepted as a truly asynchronous CUDA copy endpoint.
        copy_async(
            raw,
            source,
            stream=stream,
            chunk_bytes=chunk_bytes,
            chunk_gap_seconds=chunk_gap_seconds,
        )
        copy_async(
            destination,
            raw,
            stream=stream,
            chunk_bytes=chunk_bytes,
            chunk_gap_seconds=chunk_gap_seconds,
        )
        if not bool(torch.all(destination == source).item()):
            raise RuntimeError("registered Host round-trip validation failed")

        d2h_gpu = []
        d2h_wall = []
        h2d_gpu = []
        h2d_wall = []
        for _ in range(args.repeats):
            gpu_s, wall_s = copy_async(
                raw,
                source,
                stream=stream,
                chunk_bytes=chunk_bytes,
                chunk_gap_seconds=chunk_gap_seconds,
            )
            d2h_gpu.append(gpu_s)
            d2h_wall.append(wall_s)
            gpu_s, wall_s = copy_async(
                destination,
                raw,
                stream=stream,
                chunk_bytes=chunk_bytes,
                chunk_gap_seconds=chunk_gap_seconds,
            )
            h2d_gpu.append(gpu_s)
            h2d_wall.append(wall_s)

        a = torch.randn(
            (args.gemm_size, args.gemm_size), dtype=torch.bfloat16, device="cuda"
        )
        b = torch.randn_like(a)
        measure_gemm(a, b, 50, stream=forward_stream)
        baseline_before = measure_gemm(
            a, b, args.baseline_gemms, stream=forward_stream
        )

        concurrent_copy_gpu = []

        def transfer_loop() -> None:
            for _ in range(args.repeats):
                copy_destination, copy_source = (
                    (raw, source)
                    if args.concurrent_direction == "d2h"
                    else (destination, raw)
                )
                gpu_s, _ = copy_async(
                    copy_destination,
                    copy_source,
                    stream=stream,
                    chunk_bytes=chunk_bytes,
                    chunk_gap_seconds=chunk_gap_seconds,
                )
                concurrent_copy_gpu.append(gpu_s)

        worker = threading.Thread(target=transfer_loop, daemon=True)
        worker.start()
        concurrent = []
        while worker.is_alive():
            concurrent.extend(measure_gemm(a, b, 1, stream=forward_stream))
        worker.join()
        if not concurrent:
            raise RuntimeError("registered D2H completed without a forward sample")

        baseline_after = measure_gemm(
            a, b, args.baseline_gemms, stream=forward_stream
        )
        # Use a before/after sandwich so normal clock and thermal settling is
        # not mislabeled as DMA interference.
        baseline = baseline_before + baseline_after

        baseline_p50 = statistics.median(baseline)
        baseline_p95 = percentile(baseline, 0.95)
        concurrent_p50 = statistics.median(concurrent)
        concurrent_p95 = percentile(concurrent, 0.95)
        p50_regression = concurrent_p50 / baseline_p50 - 1.0
        p95_regression = concurrent_p95 / baseline_p95 - 1.0

        def transfer_report(gpu: list[float], wall: list[float]) -> dict:
            return {
                "gpu_GiB_s": args.size_gib / statistics.mean(gpu),
                "gpu_p50_GiB_s": args.size_gib / statistics.median(gpu),
                "wall_GiB_s": args.size_gib / statistics.mean(wall),
                "wall_p50_GiB_s": args.size_gib / statistics.median(wall),
            }

        report = {
            "size_GiB": args.size_gib,
            "repeats": args.repeats,
            "chunk_MiB": args.chunk_mib,
            "chunk_gap_ms": args.chunk_gap_ms,
            "registration_ms": registration_seconds * 1000.0,
            "registration_inclusive_d2h_GiB_s": args.size_gib
            / (registration_seconds + statistics.mean(d2h_wall)),
            "registration_inclusive_h2d_GiB_s": args.size_gib
            / (registration_seconds + statistics.mean(h2d_wall)),
            "torch_reports_pinned": bool(raw.is_pinned()),
            "validation": "PASS",
            "d2h": transfer_report(d2h_gpu, d2h_wall),
            "h2d": transfer_report(h2d_gpu, h2d_wall),
            "concurrent_direction": args.concurrent_direction,
            "concurrent_gpu_GiB_s": args.size_gib
            / statistics.mean(concurrent_copy_gpu),
            "forward": {
                "baseline_before_p50_ms": statistics.median(baseline_before)
                * 1000.0,
                "baseline_after_p50_ms": statistics.median(baseline_after)
                * 1000.0,
                "baseline_p50_ms": baseline_p50 * 1000.0,
                "baseline_p95_ms": baseline_p95 * 1000.0,
                "concurrent_p50_ms": concurrent_p50 * 1000.0,
                "concurrent_p95_ms": concurrent_p95 * 1000.0,
                "p50_regression": p50_regression,
                "p95_regression": p95_regression,
                "samples": len(concurrent),
            },
        }
        encoded = json.dumps(report, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as output_file:
                output_file.write(encoded + "\n")
        print(encoded)
        failed_bandwidth = [
            direction
            for direction in ("d2h", "h2d")
            if report[direction]["wall_GiB_s"] < args.min_bandwidth_gib_s
        ]
        if failed_bandwidth:
            raise SystemExit(
                f"registered extent bandwidth below {args.min_bandwidth_gib_s:.1f} "
                f"GiB/s: {', '.join(failed_bandwidth)}"
            )
        if p50_regression > args.max_forward_p50_regression:
            raise SystemExit(
                f"forward p50 regression {p50_regression:.1%} exceeds "
                f"{args.max_forward_p50_regression:.1%}"
            )
        if p95_regression > args.max_forward_p95_regression:
            raise SystemExit(
                f"forward p95 regression {p95_regression:.1%} exceeds "
                f"{args.max_forward_p95_regression:.1%}"
            )
    finally:
        torch.cuda.synchronize()
        if registered and raw is not None:
            cuda_success(torch.cuda.cudart().cudaHostUnregister(raw.data_ptr()))
        raw = None
        if mapping is not None:
            mapping.close()
        os.close(descriptor)


if __name__ == "__main__":
    main()
