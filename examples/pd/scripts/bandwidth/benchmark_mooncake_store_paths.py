#!/usr/bin/env python3
"""Benchmark the exact Mooncake Store batch API used by SGLang HiCache.

Services (mooncake_master and one standalone mooncake_client) are expected to
be running.  Every logical Qwen3-8B/page-size-64 KV page is represented by two
4.5 MiB objects (K and V), matching SGLang's batch_put_from/get_into layout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mooncake.store import MooncakeDistributedStore
from sglang.srt.mem_cache.storage.mooncake_store.agentic_put_admission import (
    AgenticMooncakePutAdmission,
)


GB = 1_000_000_000
GIB = 1 << 30
DEFAULT_K_OR_V_PAGE_BYTES = int(4.5 * (1 << 20))


@dataclass
class RegisteredBuffer:
    array: np.ndarray
    ptr: int
    size: int


class BenchClient:
    def __init__(self, name, metadata, master, protocol, size):
        self.store = MooncakeDistributedStore()
        rc = self.store.setup(
            name,
            metadata,
            0,
            16 << 20,
            protocol,
            "",
            master,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"Mooncake setup failed for {name}: {rc}")
        array = np.empty(size, dtype=np.uint8)
        array.fill(0xA5)  # pre-fault pages outside the measurement
        ptr = int(array.ctypes.data)
        rc = self.store.register_buffer(ptr, size)
        if rc != 0:
            self.store.close()
            raise RuntimeError(f"Mooncake register_buffer failed for {name}: {rc}")
        self.buffer = RegisteredBuffer(array, ptr, size)

    def close(self):
        self.store.unregister_buffer(self.buffer.ptr)
        self.store.close()


def endpoint(tag: str, base_port: int) -> str:
    del tag
    return f"127.0.0.1:{base_port}"


def layout(prefix: str, ptr: int, snapshot_bytes: int, object_bytes: int):
    count = math.ceil(snapshot_bytes / object_bytes)
    sizes = [object_bytes] * count
    sizes[-1] = snapshot_bytes - object_bytes * (count - 1)
    keys = [f"{prefix}-{i:06d}" for i in range(count)]
    ptrs = [ptr + i * object_bytes for i in range(count)]
    return keys, ptrs, sizes


def measure_put(client, snapshot_bytes, object_bytes, tag):
    keys, ptrs, sizes = layout(tag, client.buffer.ptr, snapshot_bytes, object_bytes)
    t0 = time.perf_counter()
    result = client.store.batch_put_from(keys, ptrs, sizes)
    elapsed = time.perf_counter() - t0
    if len(result) != len(keys) or any(x != 0 for x in result):
        raise RuntimeError(f"batch_put_from failed: {result[:8]}")
    return keys, elapsed


def measure_get(client, keys, snapshot_bytes, object_bytes):
    _, ptrs, sizes = layout("unused", client.buffer.ptr, snapshot_bytes, object_bytes)
    t0 = time.perf_counter()
    result = client.store.batch_get_into(keys, ptrs, sizes)
    elapsed = time.perf_counter() - t0
    if len(result) != len(keys) or any(x != want for x, want in zip(result, sizes)):
        raise RuntimeError(f"batch_get_into failed: {result[:8]}")
    return elapsed


def record(path, payload, elapsed, concurrency=1):
    return {
        "path": path,
        "payload_bytes": payload,
        "elapsed_s": elapsed,
        "concurrency": concurrency,
        "payload_GB_s": payload / elapsed / GB,
        "payload_GiB_s": payload / elapsed / GIB,
    }


def run_size(args, size_gib):
    snapshot_bytes = int(size_gib * GIB)
    put_admission = None
    if args.max_concurrent_puts > 0:
        put_admission = AgenticMooncakePutAdmission(
            max_concurrent_puts=args.max_concurrent_puts,
            min_bytes=1,
            base_dir=args.put_admission_dir,
            store_identity=f"benchmark:{args.master}",
        )
    # One independent registered region per simulated D, and one for P.
    clients = [
        BenchClient(
            endpoint(f"d{i}", args.client_port_base + i),
            args.metadata,
            args.master,
            args.protocol,
            snapshot_bytes,
        )
        for i in range(args.decode_writers)
    ]
    p_client = BenchClient(
        endpoint("p", args.client_port_base + args.decode_writers),
        args.metadata,
        args.master,
        args.protocol,
        snapshot_bytes,
    )
    records = []
    created_keys = []
    try:
        # Exclude first-connection and metadata-cache setup from every measured
        # size.  Each D performs one small write and P performs one small read.
        warm_bytes = min(snapshot_bytes, args.object_bytes)
        warm_keys = []
        for i, client in enumerate(clients):
            keys, _ = measure_put(
                client, warm_bytes, args.object_bytes, f"warm-d{i}-{uuid.uuid4().hex}"
            )
            warm_keys.extend(keys)
        measure_get(p_client, warm_keys[:1], warm_bytes, args.object_bytes)
        created_keys.extend(warm_keys)

        keys, put_s = measure_put(
            clients[0], snapshot_bytes, args.object_bytes, f"warm-get-{uuid.uuid4().hex}"
        )
        created_keys.extend(keys)
        records.append(record("D Host -> Mooncake Store PUT", snapshot_bytes, put_s))

        try:
            get_s = measure_get(p_client, keys, snapshot_bytes, args.object_bytes)
            records.append(record("Mooncake Store -> P Host GET", snapshot_bytes, get_s))
        except Exception as exc:
            records.append(
                {"path": "Mooncake Store -> P Host GET", "error": str(exc)}
            )

        # Preload a second read-only snapshot, then overlap one GET with N PUTs.
        read_keys, _ = measure_put(
            clients[0], snapshot_bytes, args.object_bytes, f"duplex-get-{uuid.uuid4().hex}"
        )
        created_keys.extend(read_keys)

        def put_job(i):
            tag = f"duplex-put-d{i}-{uuid.uuid4().hex}"
            if put_admission is None:
                out_keys, elapsed = measure_put(
                    clients[i], snapshot_bytes, args.object_bytes, tag
                )
            else:
                with put_admission.admit(snapshot_bytes):
                    out_keys, elapsed = measure_put(
                        clients[i], snapshot_bytes, args.object_bytes, tag
                    )
            return out_keys, elapsed

        wall = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.decode_writers + 1
        ) as pool:
            put_futures = [pool.submit(put_job, i) for i in range(args.decode_writers)]
            get_future = pool.submit(
                measure_get, p_client, read_keys, snapshot_bytes, args.object_bytes
            )
            put_results = []
            put_errors = []
            for future in put_futures:
                try:
                    put_results.append(future.result())
                except Exception as exc:
                    put_errors.append(str(exc))
            try:
                duplex_get_s = get_future.result()
                get_error = None
            except Exception as exc:
                duplex_get_s = None
                get_error = str(exc)
        wall = time.perf_counter() - wall
        for out_keys, _ in put_results:
            created_keys.extend(out_keys)
        put_elapsed = [elapsed for _, elapsed in put_results]
        if not put_errors and get_error is None:
            records.append(
                record(
                    f"{args.decode_writers}D PUT + 1P GET aggregate",
                    snapshot_bytes * (args.decode_writers + 1),
                    wall,
                    concurrency=args.decode_writers + 1,
                )
            )
        else:
            records.append(
                {
                    "path": f"{args.decode_writers}D PUT + 1P GET aggregate",
                    "wall_elapsed_s": wall,
                    "put_errors": put_errors,
                    "get_error": get_error,
                }
            )
        records.append(
            {
                "path": "concurrent operation latencies",
                "D_put_elapsed_s": put_elapsed,
                "P_get_elapsed_s": duplex_get_s,
                "wall_elapsed_s": wall,
            }
        )
    finally:
        # GET creates a lease; forced cleanup makes the benchmark repeatable.
        for chunk_start in range(0, len(created_keys), 512):
            clients[0].store.batch_remove(
                created_keys[chunk_start : chunk_start + 512], force=True
            )
        p_client.close()
        for client in clients:
            client.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="http://127.0.0.1:48080/metadata")
    parser.add_argument("--master", default="127.0.0.1:48051")
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--client-port-base", type=int, default=48100)
    parser.add_argument("--decode-writers", type=int, default=3)
    parser.add_argument("--sizes-gib", default="0.6,1,2,5")
    parser.add_argument("--object-bytes", type=int, default=DEFAULT_K_OR_V_PAGE_BYTES)
    parser.add_argument(
        "--max-concurrent-puts",
        type=int,
        default=0,
        help="0 disables admission control; otherwise cap concurrent D PUTs",
    )
    parser.add_argument(
        "--put-admission-dir",
        default="/dev/shm/sglang-agentic-mooncake-put-benchmark",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = {
        "timestamp": time.time(),
        "host": socket.gethostname(),
        "metadata": args.metadata,
        "master": args.master,
        "protocol": args.protocol,
        "decode_writers": args.decode_writers,
        "max_concurrent_puts": args.max_concurrent_puts,
        "object_bytes": args.object_bytes,
        "logical_kv_page_bytes": args.object_bytes * 2,
        "records": [],
    }
    for size in [float(x) for x in args.sizes_gib.split(",")]:
        for item in run_size(args, size):
            item["snapshot_GiB"] = size
            output["records"].append(item)
    encoded = json.dumps(output, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
