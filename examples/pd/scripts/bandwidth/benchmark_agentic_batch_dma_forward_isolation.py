#!/usr/bin/env python3
"""Validate multi-lane indexed registered-memory DMA against Forward.

The benchmark uses the production SharedMHAHostSnapshot batch-copy path.  It
warms registration first, then runs N same-GPU DMA lanes concurrently with a
small forward-like BF16 GEMM.  Both D2H and H2D are checked independently.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import threading
import time
import types

import numpy as np
import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    H2DLaunchFence,
    LayerFirstD2HStaging,
    SharedMHAHostSnapshot,
)


GIB = 1 << 30


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def make_pool(token_capacity: int):
    layer_num, head_num, head_dim = 36, 8, 128
    dtype = torch.bfloat16
    device = torch.device("cuda")
    k_buffer = [
        torch.empty((token_capacity, head_num, head_dim), dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    v_buffer = [torch.empty_like(value) for value in k_buffer]
    return types.SimpleNamespace(
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=head_dim,
        store_dtype=dtype,
        device=device,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=torch.tensor(
            [value.data_ptr() for value in k_buffer],
            dtype=torch.uint64,
            device=device,
        ),
        v_data_ptrs=torch.tensor(
            [value.data_ptr() for value in v_buffer],
            dtype=torch.uint64,
            device=device,
        ),
    )


def measure_forward(a, b, iterations: int, stream) -> list[float]:
    samples = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            begin.record(stream)
            result = torch.mm(a, b)
            end.record(stream)
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
        del result
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--baseline-forwards", type=int, default=80)
    parser.add_argument("--min-bandwidth-gib-s", type=float, default=10.0)
    parser.add_argument("--max-forward-p50-regression", type=float, default=0.05)
    parser.add_argument("--max-forward-p95-regression", type=float, default=0.10)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.lanes <= 0 or args.tokens <= 0 or args.repeats <= 0:
        raise ValueError("lanes, tokens and repeats must be positive")

    os.environ.setdefault("SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA", "1")
    os.environ.setdefault("SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB", "8")
    os.environ.setdefault("SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB", "64")
    torch.cuda.set_device(0)
    pool = make_pool(args.lanes * args.tokens + 64)
    byte_size = (
        2
        * args.tokens
        * pool.layer_num
        * pool.head_num
        * pool.head_dim
        * pool.store_dtype.itemsize
    )
    streams = [torch.cuda.Stream(priority=0) for _ in range(args.lanes)]
    forward_stream = torch.cuda.Stream(priority=0)
    staging = LayerFirstD2HStaging(pool, args.tokens)
    device_indices = []
    host_indices = []
    rng = np.random.default_rng(2026)
    for lane in range(args.lanes):
        start = lane * args.tokens
        # Shuffle allocator pages, while preserving the 64-token contiguity
        # inside each page.  This matches production req_to_token layout and
        # exercises descriptor coalescing without inventing a pathological
        # one-descriptor-per-token workload.
        if args.tokens % 64:
            raise ValueError("tokens must be a multiple of the 64-token page size")
        pages = np.arange(args.tokens, dtype=np.int64).reshape(-1, 64)
        host = (rng.permutation(pages, axis=0).reshape(-1) + start).copy()
        host_indices.append(host)
        device_indices.append(torch.from_numpy(host).to(device="cuda"))

    a = torch.randn((64, 4096), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
    measure_forward(a, b, 20, forward_stream)

    with tempfile.TemporaryDirectory(prefix="agentic-batch-dma-", dir="/dev/shm") as root:
        snapshots = []
        try:
            for lane in range(args.lanes):
                snapshot = SharedMHAHostSnapshot(
                    path=os.path.join(root, f"lane-{lane}.kv"),
                    token_count=args.tokens,
                    device_pool=pool,
                    byte_size=byte_size,
                    create=True,
                )
                if not snapshot.prepare_registered_host_dma():
                    raise RuntimeError("registered Host DMA is unavailable")
                snapshots.append(snapshot)

            def one_copy(lane: int, direction: str) -> float:
                snapshot = snapshots[lane]
                fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
                started = time.perf_counter()
                if direction == "d2h":
                    event, _ = snapshot.start_backup_range_from_device(
                        device_indices[lane],
                        destination_start=0,
                        stream=streams[lane],
                        staging=staging,
                        launch_fence=fence,
                        source_indices_host=host_indices[lane],
                    )
                else:
                    event, _ = snapshot.start_load_range_from_bounce_to_device(
                        device_indices[lane],
                        source_start=0,
                        stream=streams[lane],
                        staging=staging,
                        launch_fence=fence,
                        device_indices_host=host_indices[lane],
                    )
                event.synchronize()
                return time.perf_counter() - started

            # Warm both directions and the descriptor construction path.
            for direction in ("d2h", "h2d"):
                for lane in range(args.lanes):
                    one_copy(lane, direction)

            report = {}
            for direction in ("d2h", "h2d"):
                baseline_before = measure_forward(
                    a, b, args.baseline_forwards, forward_stream
                )
                lane_elapsed = [[] for _ in range(args.lanes)]
                start_barrier = threading.Barrier(args.lanes + 1)

                def transfer(lane: int) -> None:
                    start_barrier.wait()
                    for _ in range(args.repeats):
                        lane_elapsed[lane].append(one_copy(lane, direction))

                workers = [
                    threading.Thread(target=transfer, args=(lane,), daemon=True)
                    for lane in range(args.lanes)
                ]
                for worker in workers:
                    worker.start()
                start_barrier.wait()
                concurrent = []
                while any(worker.is_alive() for worker in workers):
                    concurrent.extend(measure_forward(a, b, 1, forward_stream))
                for worker in workers:
                    worker.join()
                baseline_after = measure_forward(
                    a, b, args.baseline_forwards, forward_stream
                )
                baseline = baseline_before + baseline_after
                lane_bandwidth = [
                    byte_size * args.repeats / GIB / sum(elapsed)
                    for elapsed in lane_elapsed
                ]
                baseline_p50 = statistics.median(baseline)
                baseline_p95 = percentile(baseline, 0.95)
                concurrent_p50 = statistics.median(concurrent)
                concurrent_p95 = percentile(concurrent, 0.95)
                report[direction] = {
                    "lane_GiB_s": lane_bandwidth,
                    "min_lane_GiB_s": min(lane_bandwidth),
                    "forward_baseline_p50_ms": baseline_p50,
                    "forward_baseline_p95_ms": baseline_p95,
                    "forward_concurrent_p50_ms": concurrent_p50,
                    "forward_concurrent_p95_ms": concurrent_p95,
                    "forward_p50_regression": concurrent_p50 / baseline_p50 - 1.0,
                    "forward_p95_regression": concurrent_p95 / baseline_p95 - 1.0,
                    "forward_samples": len(concurrent),
                }
        finally:
            for snapshot in snapshots:
                snapshot.close(unlink=True)

    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(encoded + "\n")
    for direction, result in report.items():
        if result["min_lane_GiB_s"] < args.min_bandwidth_gib_s:
            raise SystemExit(f"{direction} lane bandwidth is below the gate")
        if result["forward_p50_regression"] > args.max_forward_p50_regression:
            raise SystemExit(f"{direction} forward p50 regression exceeds the gate")
        if result["forward_p95_regression"] > args.max_forward_p95_regression:
            raise SystemExit(f"{direction} forward p95 regression exceeds the gate")


if __name__ == "__main__":
    main()
