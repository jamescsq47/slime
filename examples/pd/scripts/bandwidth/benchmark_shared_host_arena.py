#!/usr/bin/env python3
"""Benchmark a shared pinned Host KV arena for agentic PD serving.

Each D process owns one GPU source and writes its next request-generation KV
snapshot directly into a disjoint shared-memory extent with CUDA D2H.  Each P
process maps the same physical pages and concurrently restores the previous
double-buffered snapshot with CUDA H2D.  This deliberately bypasses P-HBM
staging on the slow ingress path.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import multiprocessing as mp
import os
import queue
import statistics
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _cuda_runtime():
    lib = ctypes.CDLL("libcudart.so")
    lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.cudaHostRegister.restype = ctypes.c_int
    lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    lib.cudaHostUnregister.restype = ctypes.c_int
    lib.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    return lib


def _check_cuda(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {code}")


def _gpu_numa(gpu: int) -> int:
    return 0 if gpu < 4 else 1


def _numa_cpus(numa: int) -> set[int]:
    if numa == 0:
        wanted = set(range(0, 64)) | set(range(128, 192))
    elif numa == 1:
        wanted = set(range(64, 128)) | set(range(192, 256))
    else:
        raise ValueError(f"unsupported NUMA node {numa}")
    return wanted & os.sched_getaffinity(0)


def _bind_cpu_to_gpu_numa(gpu: int) -> None:
    cpus = _numa_cpus(_gpu_numa(gpu))
    if cpus:
        os.sched_setaffinity(0, cpus)


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


def _mapping(path: str, total_bytes: int):
    fd = os.open(path, os.O_RDWR)
    try:
        region = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_WRITE)
    finally:
        os.close(fd)
    return region


def _address(region: mmap.mmap, offset: int = 0) -> int:
    return ctypes.addressof(ctypes.c_ubyte.from_buffer(region, offset))


def _d_writer(
    *,
    writer: int,
    gpu: int,
    path: str,
    size_bytes: int,
    total_bytes: int,
    iterations: int,
    start_barrier,
    finish_barrier,
    ready_queue,
    result_queue,
) -> None:
    region = None
    registered = False
    try:
        _bind_cpu_to_gpu_numa(gpu)
        import torch

        torch.cuda.set_device(gpu)
        runtime = _cuda_runtime()
        region = _mapping(path, total_bytes)
        base_offset = writer * 2 * size_bytes
        host_ptr = _address(region, base_offset)
        _check_cuda(
            runtime.cudaHostRegister(host_ptr, 2 * size_bytes, 0),
            f"D{writer} cudaHostRegister",
        )
        registered = True

        pattern = (17 + 31 * writer) % 251
        source = torch.full(
            (size_bytes,), pattern, dtype=torch.uint8, device=f"cuda:{gpu}"
        )
        stream = torch.cuda.Stream(device=gpu)

        # Seed both slots so the first concurrent P read is valid.
        with torch.cuda.stream(stream):
            for slot in range(2):
                _check_cuda(
                    runtime.cudaMemcpyAsync(
                        host_ptr + slot * size_bytes,
                        source.data_ptr(),
                        size_bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        stream.cuda_stream,
                    ),
                    f"D{writer} initial D2H",
                )
        stream.synchronize()
        ready_queue.put({"kind": "ready", "role": "D", "worker": writer})

        for iteration in range(iterations):
            start_barrier.wait(timeout=120)
            slot = iteration & 1
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                _check_cuda(
                    runtime.cudaMemcpyAsync(
                        host_ptr + slot * size_bytes,
                        source.data_ptr(),
                        size_bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        stream.cuda_stream,
                    ),
                    f"D{writer} timed D2H",
                )
            stream.synchronize()
            finished = time.perf_counter()
            result_queue.put(
                {
                    "kind": "sample",
                    "role": "D",
                    "worker": writer,
                    "gpu": gpu,
                    "iteration": iteration,
                    "bytes": size_bytes,
                    "started": started,
                    "finished": finished,
                    "elapsed_s": finished - started,
                }
            )
            finish_barrier.wait(timeout=120)
    except Exception as exc:
        result_queue.put(
            {
                "kind": "error",
                "role": "D",
                "worker": writer,
                "gpu": gpu,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if registered:
            _check_cuda(runtime.cudaHostUnregister(host_ptr), "D cudaHostUnregister")
        if region is not None:
            region.close()


def _p_reader(
    *,
    reader: int,
    gpu: int,
    assigned_writers: list[int],
    path: str,
    size_bytes: int,
    total_bytes: int,
    iterations: int,
    start_barrier,
    finish_barrier,
    ready_queue,
    result_queue,
) -> None:
    region = None
    registrations: list[int] = []
    try:
        _bind_cpu_to_gpu_numa(gpu)
        import torch

        torch.cuda.set_device(gpu)
        runtime = _cuda_runtime()
        region = _mapping(path, total_bytes)
        host_ptrs = []
        for writer in assigned_writers:
            ptr = _address(region, writer * 2 * size_bytes)
            _check_cuda(
                runtime.cudaHostRegister(ptr, 2 * size_bytes, 0),
                f"P{reader} cudaHostRegister writer={writer}",
            )
            registrations.append(ptr)
            host_ptrs.append(ptr)

        target = torch.empty(
            (len(assigned_writers), size_bytes),
            dtype=torch.uint8,
            device=f"cuda:{gpu}",
        )
        stream = torch.cuda.Stream(device=gpu, priority=-1)
        ready_queue.put({"kind": "ready", "role": "P", "worker": reader})

        for iteration in range(iterations):
            start_barrier.wait(timeout=120)
            read_slot = 1 - (iteration & 1)
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                for local_index, host_ptr in enumerate(host_ptrs):
                    _check_cuda(
                        runtime.cudaMemcpyAsync(
                            target[local_index].data_ptr(),
                            host_ptr + read_slot * size_bytes,
                            size_bytes,
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                            stream.cuda_stream,
                        ),
                        f"P{reader} timed H2D",
                    )
            stream.synchronize()
            finished = time.perf_counter()
            validation = []
            if iteration == iterations - 1:
                for local_index, writer in enumerate(assigned_writers):
                    expected = (17 + 31 * writer) % 251
                    validation.append(
                        bool(torch.all(target[local_index] == expected).item())
                    )
            result_queue.put(
                {
                    "kind": "sample",
                    "role": "P",
                    "worker": reader,
                    "gpu": gpu,
                    "assigned_writers": assigned_writers,
                    "iteration": iteration,
                    "bytes": len(assigned_writers) * size_bytes,
                    "started": started,
                    "finished": finished,
                    "elapsed_s": finished - started,
                    "validation": validation,
                }
            )
            finish_barrier.wait(timeout=120)
    except Exception as exc:
        result_queue.put(
            {
                "kind": "error",
                "role": "P",
                "worker": reader,
                "gpu": gpu,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        for ptr in registrations:
            _check_cuda(runtime.cudaHostUnregister(ptr), "P cudaHostUnregister")
        if region is not None:
            region.close()


def _parse_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def run(args) -> dict:
    p_gpus = _parse_gpus(args.p_gpus)
    d_gpus = _parse_gpus(args.d_gpus)
    if not p_gpus or not d_gpus:
        raise ValueError("at least one P and one D GPU are required")
    if len(p_gpus) > len(d_gpus):
        raise ValueError("this balanced-flow benchmark requires P count <= D count")
    if len(set(p_gpus + d_gpus)) != len(p_gpus) + len(d_gpus):
        raise ValueError("P and D GPU sets must be disjoint")
    busy = [gpu for gpu in p_gpus + d_gpus if _gpu_busy(gpu)]
    if busy and not args.allow_busy:
        raise RuntimeError(f"refusing contaminated benchmark; busy GPUs={busy}")

    size_bytes = args.size_mib * MIB
    total_bytes = len(d_gpus) * 2 * size_bytes
    assignments = [[] for _ in p_gpus]
    for writer in range(len(d_gpus)):
        assignments[writer % len(p_gpus)].append(writer)

    fd, path = tempfile.mkstemp(prefix="agentic-pd-host-arena-", dir="/dev/shm")
    os.ftruncate(fd, total_bytes)
    os.close(fd)

    ctx = mp.get_context("spawn")
    worker_count = len(p_gpus) + len(d_gpus)
    start_barrier = ctx.Barrier(worker_count + 1)
    finish_barrier = ctx.Barrier(worker_count + 1)
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = []
    try:
        for writer, gpu in enumerate(d_gpus):
            process = ctx.Process(
                target=_d_writer,
                kwargs={
                    "writer": writer,
                    "gpu": gpu,
                    "path": path,
                    "size_bytes": size_bytes,
                    "total_bytes": total_bytes,
                    "iterations": args.iterations,
                    "start_barrier": start_barrier,
                    "finish_barrier": finish_barrier,
                    "ready_queue": ready_queue,
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append(process)
        for reader, (gpu, assigned) in enumerate(zip(p_gpus, assignments)):
            process = ctx.Process(
                target=_p_reader,
                kwargs={
                    "reader": reader,
                    "gpu": gpu,
                    "assigned_writers": assigned,
                    "path": path,
                    "size_bytes": size_bytes,
                    "total_bytes": total_bytes,
                    "iterations": args.iterations,
                    "start_barrier": start_barrier,
                    "finish_barrier": finish_barrier,
                    "ready_queue": ready_queue,
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append(process)

        ready = []
        deadline = time.monotonic() + 180
        while len(ready) < worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for worker initialization")
            try:
                item = ready_queue.get(timeout=min(2, remaining))
                ready.append(item)
            except queue.Empty:
                while True:
                    try:
                        item = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item.get("kind") == "error":
                        raise RuntimeError(item)

        records = []
        for iteration in range(args.iterations):
            start_barrier.wait(timeout=120)
            finish_barrier.wait(timeout=120)
            samples = []
            deadline = time.monotonic() + 120
            while len(samples) < worker_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for iteration samples")
                item = result_queue.get(timeout=min(2, remaining))
                if item.get("kind") == "error":
                    raise RuntimeError(item)
                if item["iteration"] != iteration:
                    raise RuntimeError(f"out-of-order sample: {item}")
                samples.append(item)
            d_samples = [item for item in samples if item["role"] == "D"]
            p_samples = [item for item in samples if item["role"] == "P"]
            d_wall = max(x["finished"] for x in d_samples) - min(
                x["started"] for x in d_samples
            )
            p_wall = max(x["finished"] for x in p_samples) - min(
                x["started"] for x in p_samples
            )
            all_wall = max(x["finished"] for x in samples) - min(
                x["started"] for x in samples
            )
            records.append(
                {
                    "iteration": iteration,
                    "d2h_GiB_s": sum(x["bytes"] for x in d_samples) / GIB / d_wall,
                    "h2d_GiB_s": sum(x["bytes"] for x in p_samples) / GIB / p_wall,
                    "duplex_payload_GiB_s": sum(x["bytes"] for x in samples)
                    / GIB
                    / all_wall,
                    "wall_s": all_wall,
                    "per_worker": sorted(
                        samples, key=lambda x: (x["role"], x["worker"])
                    ),
                }
            )

        for process in processes:
            process.join(timeout=60)
            if process.exitcode != 0:
                raise RuntimeError(
                    f"worker pid={process.pid} exited with {process.exitcode}"
                )
        measured = records[args.warmup_iterations :]
        validations = [
            value
            for record in records
            for worker in record["per_worker"]
            for value in worker.get("validation", [])
        ]
        if not validations or not all(validations):
            raise RuntimeError(f"data validation failed: {validations}")
        return {
            "p_gpus": p_gpus,
            "d_gpus": d_gpus,
            "ratio": f"{len(p_gpus)}P:{len(d_gpus)}D",
            "size_MiB_per_D_per_direction": args.size_mib,
            "double_buffered_shared_host_GiB": total_bytes / GIB,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "p_assignments": assignments,
            "d2h_GiB_s_median": statistics.median(x["d2h_GiB_s"] for x in measured),
            "h2d_GiB_s_median": statistics.median(x["h2d_GiB_s"] for x in measured),
            "duplex_payload_GiB_s_median": statistics.median(
                x["duplex_payload_GiB_s"] for x in measured
            ),
            "validation": validations,
            "records": records,
        }
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-gpus", required=True)
    parser.add_argument("--d-gpus", required=True)
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup-iterations", type=int, default=2)
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
