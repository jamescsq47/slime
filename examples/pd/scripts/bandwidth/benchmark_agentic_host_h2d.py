#!/usr/bin/env python3
"""Exactness and bandwidth check for the agentic shared-Host H2D path.

This deliberately uses the Qwen3-8B KV shape so both the PCIe DMA volume and
the scatter-kernel launch pattern match the serving experiment.  It queues
multiple snapshots on one stream with one reusable GPU staging buffer, which
is the same ownership rule used by ``AgenticPHostStagingManager``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import types

import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    LayerFirstD2HStaging,
    SharedMHAHostSnapshot,
)


def make_pool(token_capacity: int):
    layer_num = 36
    head_num = 8
    head_dim = 128
    dtype = torch.bfloat16
    device = torch.device("cuda")
    k_buffer = [
        torch.empty((token_capacity, head_num, head_dim), dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    v_buffer = [
        torch.empty((token_capacity, head_num, head_dim), dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
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
            [value.data_ptr() for value in k_buffer], dtype=torch.uint64, device=device
        ),
        v_data_ptrs=torch.tensor(
            [value.data_ptr() for value in v_buffer], dtype=torch.uint64, device=device
        ),
    )


def snapshot(pool, token_count: int, path: str):
    bytes_per_token = (
        2
        * pool.layer_num
        * pool.head_num
        * pool.head_dim
        * pool.store_dtype.itemsize
    )
    return SharedMHAHostSnapshot(
        path=path,
        token_count=token_count,
        device_pool=pool,
        byte_size=token_count * bytes_per_token,
        create=True,
    )


def exact_q4_test(pool, staging, directory: str) -> None:
    token_count = 512
    snapshots = []
    stream = torch.cuda.Stream()
    permutation = torch.randperm(token_count * 4, device="cuda", dtype=torch.int64)
    events = []
    try:
        for snapshot_index in range(4):
            value = snapshot(
                pool, token_count, os.path.join(directory, f"exact-{snapshot_index}.kv")
            )
            for layer in range(pool.layer_num):
                value.k_buffer[layer].fill_(snapshot_index * 40 + layer + 1)
                value.v_buffer[layer].fill_(-(snapshot_index * 40 + layer + 1))
            snapshots.append(value)
            destinations = permutation[
                snapshot_index * token_count : (snapshot_index + 1) * token_count
            ]
            event, refs = value.start_load_to_device(
                destinations, stream, chunk_tokens=staging.token_capacity, staging=staging
            )
            events.append((event, refs, destinations, snapshot_index))

        events[-1][0].synchronize()
        for _, _, destinations, snapshot_index in events:
            for layer in range(pool.layer_num):
                expected = snapshot_index * 40 + layer + 1
                if not torch.all(pool.k_buffer[layer][destinations] == expected):
                    raise AssertionError(
                        f"K mismatch: snapshot={snapshot_index} layer={layer}"
                    )
                if not torch.all(pool.v_buffer[layer][destinations] == -expected):
                    raise AssertionError(
                        f"V mismatch: snapshot={snapshot_index} layer={layer}"
                    )
        print("exact_q4=PASS snapshots=4 tokens_per_snapshot=512")
    finally:
        for value in snapshots:
            value.close(unlink=True)


def bandwidth_test(pool, staging, directory: str, token_count: int, repeats: int) -> None:
    value = snapshot(pool, token_count, os.path.join(directory, "bandwidth.kv"))
    try:
        # First-touch every tmpfs page on the GPU-local NUMA node selected by
        # the caller's numactl binding.
        value.kv_buffer.zero_()
        destinations = torch.randperm(
            pool.k_buffer[0].shape[0], device="cuda", dtype=torch.int64
        )[:token_count]
        stream = torch.cuda.Stream()

        event, refs = value.start_load_to_device(
            destinations, stream, chunk_tokens=staging.token_capacity, staging=staging
        )
        event.synchronize()
        rates = []
        for _ in range(repeats):
            event, refs = value.start_load_to_device(
                destinations, stream, chunk_tokens=staging.token_capacity, staging=staging
            )
            event.synchronize()
            elapsed_s = value._last_h2d_start_event.elapsed_time(event) / 1000.0
            rates.append(value.byte_size / elapsed_s / 1024**3)
        ordered = sorted(rates)
        median = ordered[len(ordered) // 2]
        print(
            "h2d_gib_s=" + ",".join(f"{rate:.3f}" for rate in rates)
            + f" median={median:.3f} snapshot_gib={value.byte_size / 1024**3:.3f}"
        )
        if median < 12.0:
            raise RuntimeError(f"H2D median bandwidth is unexpectedly low: {median:.3f} GiB/s")
    finally:
        value.close(unlink=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    pool = make_pool(max(args.tokens + 1024, 4096))
    staging = LayerFirstD2HStaging(pool, args.chunk_tokens)
    with tempfile.TemporaryDirectory(prefix="agentic-h2d-test-", dir="/dev/shm") as directory:
        exact_q4_test(pool, staging, directory)
        bandwidth_test(pool, staging, directory, args.tokens, args.repeats)


if __name__ == "__main__":
    main()
