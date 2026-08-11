#!/usr/bin/env python3
"""Measure the proposed agentic D-VRAM -> P-Host-HiCache NIXL path.

One process owns pre-registered pinned host buffers (the simulated P Host
HiCache).  Independent child processes own CUDA source buffers (simulated D
workers) and concurrently NIXL-WRITE them into disjoint P-host destinations.
The benchmark excludes allocation/registration and validates the complete
destination after the D workers have released their source HBM.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import statistics
import time
import uuid
from pathlib import Path

GIB = 1 << 30


def _numa_cpus(numa: int) -> set[int]:
    if numa == 0:
        wanted = set(range(0, 64)) | set(range(128, 192))
    elif numa == 1:
        wanted = set(range(64, 128)) | set(range(192, 256))
    else:
        raise ValueError(f"unsupported NUMA node {numa}")
    return wanted & os.sched_getaffinity(0)


def _gpu_numa(gpu: int) -> int:
    return 0 if gpu < 4 else 1


def _bind_numa(numa: int) -> None:
    cpus = _numa_cpus(numa)
    if cpus:
        os.sched_setaffinity(0, cpus)


def _decode_writer(
    *,
    writer_index: int,
    gpu: int,
    size_bytes: int,
    iterations: int,
    p_metadata: bytes,
    p_target_desc: bytes,
    barrier,
    result_queue,
    free_source_event,
    source_freed_queue,
    exit_event,
) -> None:
    try:
        _bind_numa(_gpu_numa(gpu))
        import torch
        from nixl._api import nixl_agent, nixl_agent_config

        torch.cuda.set_device(gpu)
        pattern = (17 + 31 * writer_index) % 251
        source = torch.full(
            (size_bytes,), pattern, dtype=torch.uint8, device=f"cuda:{gpu}"
        )
        torch.cuda.synchronize(gpu)

        agent_name = f"agentic-d{writer_index}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        agent = nixl_agent(
            agent_name,
            nixl_agent_config(
                backends=["UCX"],
                num_threads=8,
                capture_telemetry=True,
            ),
        )
        source_registration = agent.register_memory([source], backends=["UCX"])
        local_desc = agent.get_xfer_descs([source])
        remote_desc = agent.deserialize_descs(p_target_desc)
        p_agent_name = agent.add_remote_agent(p_metadata)
        agent.make_connection(p_agent_name, backends=["UCX"])

        for iteration in range(iterations):
            tag = f"d{writer_index}-i{iteration}".encode()
            handle = agent.initialize_xfer(
                "WRITE",
                local_desc,
                remote_desc,
                p_agent_name,
                tag,
                backends=["UCX"],
            )
            barrier.wait(timeout=30)
            started_at = time.monotonic()
            state = agent.transfer(handle)
            while state == "PROC":
                state = agent.check_xfer_state(handle)
            finished_at = time.monotonic()
            if state != "DONE":
                raise RuntimeError(f"NIXL transfer failed with state={state}")
            result_queue.put(
                {
                    "kind": "transfer",
                    "writer": writer_index,
                    "gpu": gpu,
                    "iteration": iteration,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_s": finished_at - started_at,
                    "backend": agent.query_xfer_backend(handle),
                }
            )
            agent.release_xfer_handle(handle)

        free_source_event.wait(timeout=120)
        agent.deregister_memory(source_registration, backends=["UCX"])
        del source
        torch.cuda.synchronize(gpu)
        torch.cuda.empty_cache()
        source_freed_queue.put({"writer": writer_index, "gpu": gpu})
        exit_event.wait(timeout=120)
    except Exception as exc:
        result_queue.put(
            {
                "kind": "error",
                "writer": writer_index,
                "gpu": gpu,
                "error": repr(exc),
            }
        )


def run(args) -> dict:
    import torch
    from nixl._api import nixl_agent, nixl_agent_config

    d_gpus = [int(value) for value in args.d_gpus.split(",") if value]
    if not d_gpus:
        raise ValueError("at least one D GPU is required")
    size_bytes = int(args.size_gib * GIB)
    _bind_numa(args.p_host_numa)

    if args.target_memory == "host":
        targets = [
            torch.zeros(size_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in d_gpus
        ]
    else:
        torch.cuda.set_device(args.p_gpu)
        targets = [
            torch.zeros(
                size_bytes,
                dtype=torch.uint8,
                device=f"cuda:{args.p_gpu}",
            )
            for _ in d_gpus
        ]
        torch.cuda.synchronize(args.p_gpu)
    p_agent_name = f"agentic-p-host-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    p_agent = nixl_agent(
        p_agent_name,
        nixl_agent_config(
            backends=["UCX"],
            num_threads=8,
            capture_telemetry=True,
        ),
    )
    target_registration = p_agent.register_memory(targets, backends=["UCX"])
    p_metadata = p_agent.get_agent_metadata()
    target_descs = [
        p_agent.get_serialized_descs(p_agent.get_xfer_descs([target]))
        for target in targets
    ]

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(d_gpus) + 1)
    result_queue = ctx.Queue()
    free_source_event = ctx.Event()
    source_freed_queue = ctx.Queue()
    exit_event = ctx.Event()
    processes = []
    for writer_index, gpu in enumerate(d_gpus):
        process = ctx.Process(
            target=_decode_writer,
            kwargs={
                "writer_index": writer_index,
                "gpu": gpu,
                "size_bytes": size_bytes,
                "iterations": args.iterations,
                "p_metadata": p_metadata,
                "p_target_desc": target_descs[writer_index],
                "barrier": barrier,
                "result_queue": result_queue,
                "free_source_event": free_source_event,
                "source_freed_queue": source_freed_queue,
                "exit_event": exit_event,
            },
        )
        process.start()
        processes.append(process)

    records = []
    try:
        for iteration in range(args.iterations):
            barrier.wait(timeout=60)
            iteration_records = []
            deadline = time.monotonic() + 120
            while len(iteration_records) < len(d_gpus):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for D transfer results")
                item = result_queue.get(timeout=min(5, remaining))
                if item["kind"] == "error":
                    raise RuntimeError(item)
                if item["iteration"] != iteration:
                    raise RuntimeError(f"out-of-order result: {item}")
                iteration_records.append(item)
            wall_start = min(item["started_at"] for item in iteration_records)
            wall_end = max(item["finished_at"] for item in iteration_records)
            wall_s = wall_end - wall_start
            records.append(
                {
                    "iteration": iteration,
                    "wall_s": wall_s,
                    "aggregate_GiB_s": args.size_gib * len(d_gpus) / wall_s,
                    "writers": sorted(
                        iteration_records, key=lambda item: item["writer"]
                    ),
                }
            )

        # This models D observing NIXL completion and releasing source HBM.
        free_source_event.set()
        freed = []
        deadline = time.monotonic() + 120
        while len(freed) < len(d_gpus):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for D HBM release")
            freed.append(source_freed_queue.get(timeout=min(5, remaining)))

        validation_started = time.monotonic()
        validation = []
        for writer_index, target in enumerate(targets):
            pattern = (17 + 31 * writer_index) % 251
            valid = bool(torch.all(target == pattern).item())
            validation.append(
                {
                    "writer": writer_index,
                    "pattern": pattern,
                    "all_bytes_match_after_source_hbm_release": valid,
                }
            )
        validation_s = time.monotonic() - validation_started
        if not all(item["all_bytes_match_after_source_hbm_release"] for item in validation):
            raise RuntimeError(f"destination validation failed: {validation}")
    finally:
        exit_event.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        p_agent.deregister_memory(target_registration, backends=["UCX"])

    measured = records[args.warmup_iterations :]
    aggregate_values = [item["aggregate_GiB_s"] for item in measured]
    return {
        "p_host_numa": args.p_host_numa,
        "p_gpu": args.p_gpu if args.target_memory == "vram" else None,
        "target_memory": args.target_memory,
        "d_gpus": d_gpus,
        "d_gpu_numas": [_gpu_numa(gpu) for gpu in d_gpus],
        "size_GiB_per_D": args.size_gib,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "backend": "UCX",
        "aggregate_GiB_s_median": statistics.median(aggregate_values),
        "aggregate_GiB_s_min": min(aggregate_values),
        "aggregate_GiB_s_max": max(aggregate_values),
        "source_hbm_released_before_validation": True,
        "validation_s": validation_s,
        "validation": validation,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-gpus", required=True)
    parser.add_argument("--p-host-numa", type=int, default=0)
    parser.add_argument("--p-gpu", type=int, default=0)
    parser.add_argument(
        "--target-memory", choices=("host", "vram"), default="host"
    )
    parser.add_argument("--size-gib", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--warmup-iterations", type=int, default=1)
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
