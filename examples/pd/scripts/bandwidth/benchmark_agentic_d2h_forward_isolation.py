#!/usr/bin/env python3
"""Microbenchmark the shared D2H primitive while forward-like GEMMs run.

Both D->P and P->D use the same physical path: gather KV into bounded HBM,
DMA into a pinned bounce, then commit the bounce to pageable Shared Arena on
an independent CPU worker.  Direction-specific state-machine tests cover
ownership; this benchmark checks byte exactness, pipeline time and whether
bounded D2H work materially delays foreground GPU compute.
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

import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    H2DLaunchFence,
    HostCopyWorkerPool,
    LayerFirstD2HStaging,
    PinnedMHAHostBounce,
    SharedMHAHostSnapshot,
)

GIB = 1 << 30


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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def run_pipeline(
    snapshot,
    source_indices,
    *,
    stream,
    staging,
    bounces,
    host_pool,
    chunk_tokens: int,
    repeats: int,
) -> dict:
    wall_started = time.perf_counter()
    gpu_ms = 0.0
    host_seconds = 0.0
    for _ in range(repeats):
        futures = {}
        for chunk_index, start in enumerate(range(0, len(source_indices), chunk_tokens)):
            end = min(start + chunk_tokens, len(source_indices))
            bounce_index = chunk_index % len(bounces)
            prior = futures.pop(bounce_index, None)
            if prior is not None:
                host_seconds += float(prior.result())
            fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
            event, _ = snapshot.start_backup_range_from_device(
                source_indices[start:end],
                destination_start=start,
                stream=stream,
                staging=staging,
                host_bounce=bounces[bounce_index],
                launch_fence=fence,
            )
            start_event = snapshot._last_d2h_start_event
            event.synchronize()
            gpu_ms += float(start_event.elapsed_time(event))
            futures[bounce_index] = host_pool.submit(
                snapshot.commit_backup_range_from_bounce,
                bounces[bounce_index],
                destination_start=start,
                token_count=end - start,
            )
        for future in futures.values():
            host_seconds += float(future.result())
    elapsed = time.perf_counter() - wall_started
    return {
        "elapsed_s": elapsed,
        "gpu_s_sum": gpu_ms / 1000.0,
        "host_commit_s_sum": host_seconds,
        "payload_GiB_s": snapshot.byte_size * repeats / GIB / elapsed,
    }


def measure_gemm(a, b, iterations: int) -> list[float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        torch.mm(a, b)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("d2p", "p2d"), required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--bounce-depth", type=int, default=2)
    parser.add_argument("--host-workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-gemms", type=int, default=30)
    parser.add_argument("--gemm-size", type=int, default=4096)
    parser.add_argument("--max-forward-regression", type=float, default=0.10)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.tokens % args.chunk_tokens:
        raise ValueError("tokens must be divisible by chunk-tokens")

    torch.cuda.set_device(0)
    pool = make_pool(args.tokens + 64)
    source_indices = torch.arange(args.tokens, dtype=torch.int64, device="cuda")
    for layer_id in range(pool.layer_num):
        pool.k_buffer[layer_id].fill_(layer_id + 1)
        pool.v_buffer[layer_id].fill_(-(layer_id + 1))
    stream = torch.cuda.Stream(priority=0)
    staging = LayerFirstD2HStaging(pool, args.chunk_tokens)
    bounces = tuple(
        PinnedMHAHostBounce(pool, args.chunk_tokens)
        for _ in range(args.bounce_depth)
    )
    host_pool = HostCopyWorkerPool(
        f"agentic-{args.direction}-microbench", args.host_workers
    )
    byte_size = (
        2
        * args.tokens
        * pool.layer_num
        * pool.head_num
        * pool.head_dim
        * pool.store_dtype.itemsize
    )
    a = torch.randn(
        (args.gemm_size, args.gemm_size), dtype=torch.bfloat16, device="cuda"
    )
    b = torch.randn_like(a)
    measure_gemm(a, b, 5)
    baseline = measure_gemm(a, b, args.baseline_gemms)

    with tempfile.TemporaryDirectory(
        prefix=f"agentic-{args.direction}-d2h-", dir="/dev/shm"
    ) as directory:
        snapshot = SharedMHAHostSnapshot(
            path=os.path.join(directory, "snapshot.kv"),
            token_count=args.tokens,
            device_pool=pool,
            byte_size=byte_size,
            create=True,
        )
        try:
            snapshot.kv_buffer.zero_()
            result = {}

            def transfer():
                result.update(
                    run_pipeline(
                        snapshot,
                        source_indices,
                        stream=stream,
                        staging=staging,
                        bounces=bounces,
                        host_pool=host_pool,
                        chunk_tokens=args.chunk_tokens,
                        repeats=args.repeats,
                    )
                )

            worker = threading.Thread(target=transfer, daemon=True)
            worker.start()
            concurrent = []
            while worker.is_alive():
                concurrent.extend(measure_gemm(a, b, 1))
            worker.join()
            if not concurrent:
                raise RuntimeError("D2H completed before a forward sample was observed")
            for layer_id in range(pool.layer_num):
                if not torch.all(snapshot.k_buffer[layer_id] == layer_id + 1):
                    raise RuntimeError(f"K mismatch at layer {layer_id}")
                if not torch.all(snapshot.v_buffer[layer_id] == -(layer_id + 1)):
                    raise RuntimeError(f"V mismatch at layer {layer_id}")
        finally:
            snapshot.close(unlink=True)

    baseline_p50 = statistics.median(baseline)
    concurrent_p50 = statistics.median(concurrent)
    regression = concurrent_p50 / baseline_p50 - 1.0
    report = {
        "direction": args.direction,
        "tokens": args.tokens,
        "snapshot_GiB": byte_size / GIB,
        "chunk_tokens": args.chunk_tokens,
        "bounce_depth": args.bounce_depth,
        "host_workers": args.host_workers,
        "validation": "PASS",
        "pipeline": result,
        "forward": {
            "baseline_p50_ms": baseline_p50 * 1000.0,
            "baseline_p95_ms": percentile(baseline, 0.95) * 1000.0,
            "concurrent_p50_ms": concurrent_p50 * 1000.0,
            "concurrent_p95_ms": percentile(concurrent, 0.95) * 1000.0,
            "p50_regression": regression,
            "samples": len(concurrent),
        },
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file_obj:
            file_obj.write(encoded + "\n")
    print(encoded)
    if regression > args.max_forward_regression:
        raise SystemExit(
            f"forward p50 regression {regression:.1%} exceeds "
            f"{args.max_forward_regression:.1%}"
        )


if __name__ == "__main__":
    main()
