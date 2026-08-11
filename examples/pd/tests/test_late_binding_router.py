import asyncio
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import orjson

from late_binding_router import (
    DecodeLoad,
    DecodeReservation,
    LateBindingMiniLoadBalancer,
    _PrefillAdmissionGate,
    _PrefillAdmissionWaiter,
)


class LateBindingRouterTest(unittest.IsolatedAsyncioTestCase):
    def make_router(self, ready_dir: Path) -> LateBindingMiniLoadBalancer:
        router = object.__new__(LateBindingMiniLoadBalancer)
        router.p_ready_dir = ready_dir
        router.ready_timeout = 1.0
        router.prefill_accept_timeout = 1.0
        router.prefill_queue_timeout = 1.0
        router.ready_poll_interval = 0.001
        router.load_timeout = 1.0
        router.reservation_timeout = 1.0
        router.decode_headroom_tokens = 512
        router.max_decode_admission_tokens = 4096
        router.request_load_weight = 0.05
        router.transfer_request_weight = 2.0
        router.context_token_floor = 2048
        router.context_token_ceiling = 8192
        router.wait_for_feasible_decode = True
        router.no_capacity_poll_interval = 0.001
        router.soft_reservation_delay = 0.0
        router.soft_reservation_min_tokens = 0
        router.soft_reservation_force_after = 1.0
        router.load_cache_ttl = 0.05
        router.decode_urls = ["http://d0", "http://d1"]
        router._selection_lock = asyncio.Lock()
        router._p_ready_fifo_lock = asyncio.Lock()
        router._p_ready_submitted_sequences = set()
        router._reservations = {}
        router._last_loads = {}
        router._load_cache = []
        router._load_cache_at = 0.0
        router._load_refresh_task = None
        router._admitted_reservation_at = {}
        router._legacy_load_urls = set()
        return router

    def test_numa_domain_is_stable_and_decode_workers_are_partitioned(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.numa_domains = True
        router.prefill_urls = ["http://p0:1", "http://p1:2"]
        router.prefill_bootstrap_ports = [100, 200]
        router.decode_urls = [f"http://d{index}" for index in range(6)]
        metadata = types.SimpleNamespace(request_id="stable-agent")

        first = router._request_domain(metadata, (1,))
        second = router._request_domain(metadata, (999,))

        self.assertEqual(first, second)
        self.assertEqual(
            router._domain_decode_urls(0),
            {"http://d0", "http://d1", "http://d2"},
        )
        self.assertEqual(
            router._domain_decode_urls(1),
            {"http://d3", "http://d4", "http://d5"},
        )

    def test_bind_prefill_domain_rewrites_bootstrap_destination(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.prefill_urls = ["http://127.0.0.1:10", "http://127.0.0.2:20"]
        router.prefill_bootstrap_ports = [101, 202]
        request = {"bootstrap_room": 7}

        selected = router._bind_prefill_domain(request, 1)

        self.assertEqual(selected, "http://127.0.0.2:20")
        self.assertEqual(request["bootstrap_host"], "127.0.0.2")
        self.assertEqual(request["bootstrap_port"], 202)

    async def test_ready_marker_reports_exact_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            room = 17
            router._ready_path(room).write_bytes(
                orjson.dumps({"rid": "request-a", "num_kv_tokens": 7317})
            )
            unfinished = asyncio.create_task(asyncio.sleep(10))
            try:
                tokens = await router._wait_until_prefill_ready((room,), unfinished)
            finally:
                unfinished.cancel()
            self.assertEqual(tokens, 7317)

    async def test_p_ready_fifo_uses_p_completion_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router._ready_path(1).write_bytes(
                orjson.dumps({"num_kv_tokens": 10, "ready_sequence": 4})
            )
            router._ready_path(2).write_bytes(
                orjson.dumps({"num_kv_tokens": 10, "ready_sequence": 5})
            )

            later = asyncio.create_task(router._acquire_p_ready_fifo(5))
            await asyncio.sleep(0.01)
            self.assertFalse(later.done())

            await router._acquire_p_ready_fifo(4)
            router._p_ready_fifo_lock.release()

            await asyncio.wait_for(later, timeout=1)
            router._p_ready_fifo_lock.release()

    async def test_d_admission_wait_does_not_hold_global_p_ready_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 4
            router._prefill_admission = _PrefillAdmissionGate(
                limit=4, new_aging_seconds=10
            )
            router._wait_until_prefill_accepted = AsyncMock()
            router._wait_until_prefill_scheduled = AsyncMock()
            router._wait_until_prefill_ready = AsyncMock(return_value=1024)
            router._p_ready_sequence = lambda rooms: 1
            reservation = DecodeReservation(
                reservation_id="reservation-1",
                url="http://d0",
                prompt_tokens=1024,
                admission_tokens=1536,
                request_count=1,
                rooms=(7,),
                created_at=0.0,
            )
            router._select_and_reserve_decode = AsyncMock(return_value=reservation)
            admission_started = asyncio.Event()
            release_admission = asyncio.Event()

            async def wait_for_admission(_reservation, _sequence=None):
                admission_started.set()
                await release_admission.wait()

            router._release_reservation_when_admitted = wait_for_admission

            class Response:
                status = 200

            class Session:
                async def post(self, *args, **kwargs):
                    return Response()

            task = asyncio.create_task(
                router._late_dispatch(
                    Session(),
                    {"bootstrap_room": 7, "sampling_params": {}},
                    "http://p0",
                    "generate",
                    {},
                )
            )
            await asyncio.wait_for(admission_started.wait(), timeout=1)
            self.assertFalse(router._p_ready_fifo_lock.locked())
            release_admission.set()
            await asyncio.wait_for(task, timeout=1)

    async def test_accepted_marker_releases_router_admission_before_p_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            room = 23
            router._accepted_path(room).write_bytes(
                orjson.dumps({"rid": "request-b"})
            )
            unfinished = asyncio.create_task(asyncio.sleep(10))
            try:
                await router._wait_until_prefill_accepted((room,), unfinished)
            finally:
                unfinished.cancel()
            self.assertFalse(router._accepted_path(room).exists())

    async def test_scheduled_marker_starts_processing_timeout_after_queue_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            room = 24
            router._scheduled_path(room).write_bytes(
                orjson.dumps({"rid": "request-c"})
            )
            unfinished = asyncio.create_task(asyncio.sleep(10))
            try:
                await router._wait_until_prefill_scheduled((room,), unfinished)
            finally:
                unfinished.cancel()
            self.assertFalse(router._scheduled_path(room).exists())

    async def test_prefill_admission_is_bounded_and_parent_preempts_fresh_new(self):
        gate = _PrefillAdmissionGate(limit=1, new_aging_seconds=100)
        await gate.acquire(parent_turn=False)
        fresh_new = asyncio.create_task(gate.acquire(parent_turn=False))
        parent = asyncio.create_task(gate.acquire(parent_turn=True))
        await asyncio.sleep(0)

        await gate.release()
        await asyncio.wait_for(parent, timeout=1)
        self.assertFalse(fresh_new.done())
        self.assertEqual(gate.active, 1)

        await gate.release()
        await asyncio.wait_for(fresh_new, timeout=1)
        await gate.release()
        self.assertEqual(gate.active, 0)

    def test_prefill_admission_parent_preempts_even_aged_new(self):
        gate = _PrefillAdmissionGate(limit=1, new_aging_seconds=10)
        now = 100.0
        aged_new = _PrefillAdmissionWaiter(False, now - 11, 0)
        parent = _PrefillAdmissionWaiter(True, now, 1)
        gate._waiters = [parent, aged_new]

        self.assertIs(gate._next_waiter(now), parent)

    async def test_atomic_reservation_spreads_simultaneous_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            equal_loads = [
                DecodeLoad("http://d0", 10_000, 100_000, 10, 0, 0, 0, 100),
                DecodeLoad("http://d1", 10_000, 100_000, 10, 0, 0, 0, 100),
            ]

            async def loads(_session):
                return equal_loads

            router._all_decode_loads = loads
            first = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (1,), 20_000
            )
            second = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (2,), 20_000
            )
            self.assertEqual(first.url, "http://d0")
            self.assertEqual(second.url, "http://d1")

    async def test_admitted_reservation_remains_charged_until_new_load_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            reservation = DecodeReservation(
                reservation_id="admitted-1",
                url="http://d0",
                prompt_tokens=10_000,
                admission_tokens=12_000,
                request_count=1,
                rooms=(91,),
                created_at=1.0,
            )
            router._reservations[reservation.reservation_id] = reservation
            router._load_cache_at = 10.0
            router._admitted_reservation_at[reservation.reservation_id] = 11.0

            self.assertEqual(router._reserved_for("http://d0"), (10_000, 12_000, 1))

            router._load_cache_at = 12.0
            router._prune_accounted_reservations()
            self.assertEqual(router._reserved_for("http://d0"), (0, 0, 0))
            self.assertNotIn(reservation.reservation_id, router._reservations)

    async def test_least_running_dominates_kv_pressure_after_capacity_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad("http://d0", 90_000, 100_000, 5, 0, 0, 0, 100),
                DecodeLoad("http://d1", 40_000, 100_000, 30, 0, 0, 0, 100),
            ]

            async def loads(_session):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (3,), 5_000
            )
            self.assertEqual(selected.url, "http://d0")

    async def test_capacity_filter_dominates_least_running(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad("http://d0", 95_000, 100_000, 5, 0, 0, 0, 100),
                DecodeLoad("http://d1", 40_000, 100_000, 30, 0, 0, 0, 100),
            ]

            async def loads(_session):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (31,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_running_kv_work_can_outweigh_a_smaller_request_count(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad(
                    "http://d0", 50_000, 100_000, 10, 0, 0, 0, 100,
                    running_kv_tokens=80_000,
                ),
                DecodeLoad(
                    "http://d1", 30_000, 100_000, 12, 0, 0, 0, 100,
                    running_kv_tokens=24_000,
                ),
            ]

            async def loads(_session):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (311,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_transfer_gets_an_extra_interference_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad(
                    "http://d0", 30_000, 100_000, 10, 3, 0, 3, 100,
                    running_kv_tokens=30_000,
                    transfer_tokens=6_000,
                ),
                DecodeLoad(
                    "http://d1", 30_000, 100_000, 13, 0, 0, 0, 100,
                    running_kv_tokens=36_000,
                ),
            ]

            async def loads(_session):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (312,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_handoff_queue_counts_as_projected_decode_work(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad("http://d0", 20_000, 100_000, 10, 4, 4, 0, 100),
                DecodeLoad("http://d1", 20_000, 100_000, 10, 0, 0, 0, 100),
            ]

            async def loads(_session):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (32,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_waits_for_real_decode_capacity_instead_of_preallocating(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            attempts = 0

            async def loads(_session):
                nonlocal attempts
                attempts += 1
                used = 99_000 if attempts == 1 else 80_000
                return [
                    DecodeLoad("http://d0", used, 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", 99_000, 100_000, 50, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (4,), 5_000
            )
            self.assertEqual(selected.url, "http://d0")
            self.assertGreaterEqual(attempts, 2)

    async def test_large_request_soft_reserves_future_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            attempts = 0

            async def loads(_session):
                nonlocal attempts
                attempts += 1
                # Initially neither D can fit 31k.  Once a soft reservation is
                # installed on d0, Decode drain makes enough room there.
                d0_used = 95_000 if attempts < 3 else 60_000
                return [
                    DecodeLoad("http://d0", d0_used, 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", 95_000, 100_000, 50, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (5,), 30_000
            )
            self.assertEqual(selected.url, "http://d0")
            self.assertTrue(selected.draining)
            self.assertGreaterEqual(attempts, 3)


if __name__ == "__main__":
    unittest.main()
