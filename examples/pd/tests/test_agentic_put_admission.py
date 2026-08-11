#!/usr/bin/env python3
"""Focused tests for the same-node Mooncake PUT admission controller."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
import unittest
from contextlib import contextmanager

from sglang.srt.mem_cache.storage.mooncake_store.agentic_put_admission import (
    AgenticMooncakePutAdmission,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


class _FakeRawStore:
    def __init__(self) -> None:
        self.put_calls = 0
        self.get_calls = 0

    def batch_put_from(self, keys, ptrs, sizes):
        self.put_calls += 1
        return [0] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes):
        self.get_calls += 1
        return list(sizes)


class _CountingAdmission:
    def __init__(self) -> None:
        self.calls = []

    @contextmanager
    def admit(self, byte_count: int):
        self.calls.append(byte_count)
        yield 0.0


def _hold_slot(
    base_dir: str,
    identity: str,
    limit: int,
    started: mp.synchronize.Event,
    active: mp.sharedctypes.Synchronized,
    peak: mp.sharedctypes.Synchronized,
    counter_lock: mp.synchronize.Lock,
    hold_seconds: float,
) -> None:
    gate = AgenticMooncakePutAdmission(
        max_concurrent_puts=limit,
        min_bytes=1,
        base_dir=base_dir,
        store_identity=identity,
    )
    started.wait()
    with gate.admit(1024):
        with counter_lock:
            active.value += 1
            peak.value = max(peak.value, active.value)
        time.sleep(hold_seconds)
        with counter_lock:
            active.value -= 1


def _acquire_then_exit(base_dir: str, ready: mp.synchronize.Event) -> None:
    gate = AgenticMooncakePutAdmission(
        max_concurrent_puts=1,
        min_bytes=1,
        base_dir=base_dir,
        store_identity="crash-release",
    )
    with gate.admit(1024):
        ready.set()
        os._exit(0)


class AgenticMooncakePutAdmissionTest(unittest.TestCase):
    def test_mooncake_hook_limits_only_put(self) -> None:
        store = object.__new__(MooncakeStore)
        store.store = _FakeRawStore()
        store._agentic_put_admission = _CountingAdmission()

        self.assertEqual(store._put_batch_zero_copy_impl(["a"], [1], [4096]), [0])
        self.assertEqual(
            store._get_batch_zero_copy_impl(["a"], [2], [4096]),
            [4096],
        )
        self.assertEqual(store._agentic_put_admission.calls, [4096])
        self.assertEqual(store.store.put_calls, 1)
        self.assertEqual(store.store.get_calls, 1)

    def test_mooncake_hook_baseline_bypasses_admission(self) -> None:
        store = object.__new__(MooncakeStore)
        store.store = _FakeRawStore()
        store._agentic_put_admission = None
        self.assertEqual(store._put_batch_zero_copy_impl(["a"], [1], [16]), [0])
        self.assertEqual(store.store.put_calls, 1)

    def test_limit_is_shared_across_processes(self) -> None:
        ctx = mp.get_context("fork")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as base_dir:
            started = ctx.Event()
            active = ctx.Value("i", 0)
            peak = ctx.Value("i", 0)
            counter_lock = ctx.Lock()
            processes = [
                ctx.Process(
                    target=_hold_slot,
                    args=(
                        base_dir,
                        "shared-store",
                        2,
                        started,
                        active,
                        peak,
                        counter_lock,
                        0.08,
                    ),
                )
                for _ in range(6)
            ]
            for process in processes:
                process.start()
            started.set()
            for process in processes:
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(peak.value, 2)

    def test_small_put_does_not_wait_for_large_put_token(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as base_dir:
            gate = AgenticMooncakePutAdmission(
                max_concurrent_puts=1,
                min_bytes=1024,
                base_dir=base_dir,
                store_identity="small-bypass",
            )
            with gate.admit(4096):
                started_at = time.monotonic()
                with gate.admit(128) as wait_seconds:
                    self.assertEqual(wait_seconds, 0.0)
                self.assertLess(time.monotonic() - started_at, 0.05)

    def test_process_exit_releases_token(self) -> None:
        ctx = mp.get_context("fork")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as base_dir:
            ready = ctx.Event()
            process = ctx.Process(
                target=_acquire_then_exit,
                args=(base_dir, ready),
            )
            process.start()
            self.assertTrue(ready.wait(timeout=2))
            process.join(timeout=2)
            self.assertEqual(process.exitcode, 0)

            gate = AgenticMooncakePutAdmission(
                max_concurrent_puts=1,
                min_bytes=1,
                base_dir=base_dir,
                store_identity="crash-release",
            )
            started_at = time.monotonic()
            with gate.admit(1024):
                pass
            self.assertLess(time.monotonic() - started_at, 0.1)


if __name__ == "__main__":
    unittest.main()
