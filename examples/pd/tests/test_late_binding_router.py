import asyncio
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
import orjson
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    RequestGeneration,
)
from sglang.srt.disaggregation.agentic_host_staging import (
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.p2d_host_staging import p2d_snapshot_id

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
        router.target_decode_kv_fraction = 1.0
        router.no_capacity_poll_interval = 0.001
        router.soft_reservation_delay = 0.0
        router.soft_reservation_min_tokens = 0
        router.soft_reservation_force_after = 1.0
        router.load_cache_ttl = 0.05
        router.decode_urls = ["http://d0", "http://d1"]
        router._selection_lock = asyncio.Lock()
        router._p_ready_fifo_lock = asyncio.Lock()
        router._p_ready_fifo_locks = {}
        router._p_ready_submitted_sequences = set()
        router._p_ready_monitor_task = None
        router._p_ready_waiters = {}
        router._p_ready_snapshot = {}
        router._reservations = {}
        router._last_loads = {}
        router._load_cache = []
        router._load_cache_at = 0.0
        router._load_refresh_task = None
        router._admitted_reservation_at = {}
        router._legacy_load_urls = set()
        router.prefill_urls = ["http://p0", "http://p1"]
        router._prefill_work_lock = asyncio.Lock()
        router._prefill_pending_tokens = [0, 0]
        router._prefill_pending_requests = [0, 0]
        router._prefill_work_tiebreak = 0
        return router

    async def test_dynamic_prefill_shared_watcher_wakes_ready_batch(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            router = self.make_router(Path(tmp))
            router.dynamic_prefill_domains = True
            unfinished = asyncio.get_running_loop().create_future()
            waits = [
                asyncio.create_task(
                    router._wait_until_prefill_ready_shared((room,), unfinished)
                )
                for room in (101, 102, 103)
            ]
            await asyncio.sleep(0)
            for sequence, room in enumerate((101, 102, 103)):
                (Path(tmp) / f"{room}.ready").write_bytes(
                    orjson.dumps(
                        {
                            "rid": f"r-{room}",
                            "num_kv_tokens": room,
                            "ready_sequence": sequence,
                            "prefill_domain": sequence % 2,
                        }
                    )
                )
            self.assertEqual(await asyncio.gather(*waits), [101, 102, 103])
            self.assertEqual(len(router._p_ready_snapshot), 3)
            router._p_ready_monitor_task.cancel()
            await asyncio.gather(
                router._p_ready_monitor_task, return_exceptions=True
            )

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

    def test_global_decode_mode_keeps_slow_path_local_but_opens_p_to_all_d(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.numa_domains = True
        router.global_decode = True
        router.prefill_urls = ["http://p0", "http://p1"]
        router.decode_urls = [f"http://d{index}" for index in range(6)]

        self.assertEqual(
            router._domain_decode_urls(0), set(router.decode_urls)
        )
        self.assertEqual(
            router._domain_decode_urls(1), set(router.decode_urls)
        )

    async def test_parent_direct_chooses_lower_token_backlog(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = True
            router.prefill_urls = ["http://p0", "http://p1"]
            router._prefill_pending_tokens = [9000, 1000]
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            metadata = AgenticRequestMetadata(
                request_id="fixed-numa",
                generation=2,
                parent_generation=1,
            )
            request = {
                "bootstrap_room": 7,
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "fixed-numa",
                        "agentic_generation": 2,
                        "agentic_parent_generation": 1,
                    }
                },
            }
            original = time.time() - 0.1
            router.early_claim_store.publish_route(
                RequestGeneration("fixed-numa", 1),
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=7000,
            )

            request["input_ids"] = list(range(7600))
            selection = asyncio.create_task(
                router._resolve_dynamic_prefill_work(request, metadata, original)
            )
            for _ in range(100):
                arrival = router.early_claim_store.read_arrival(
                    RequestGeneration("fixed-numa", 1),
                    not_before=0.0,
                    max_age_seconds=5.0,
                )
                if arrival is not None:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(arrival["target_prefill_domain"], 1)
            selected = await asyncio.wait_for(selection, timeout=1)
            self.assertTrue(selected.route_pending)
            router.early_claim_store.publish_route(
                RequestGeneration("fixed-numa", 1),
                route="direct_complete",
                prefill_domain=1,
                snapshot_tokens=7000,
            )

            self.assertEqual(selected.domain, 1)
            self.assertEqual(selected.tokens, 600)
            self.assertEqual(router._prefill_pending_tokens, [9000, 1600])
            arrival = router.early_claim_store.read_arrival(
                RequestGeneration("fixed-numa", 1),
                not_before=0.0,
                max_age_seconds=5.0,
            )
            self.assertEqual(arrival["target_prefill_domain"], 1)
            self.assertEqual(arrival["arrived_at"], original)
            await router._release_prefill_work(selected)
            self.assertEqual(router._prefill_pending_tokens, [9000, 1000])

    async def test_direct_fallback_moves_work_to_host_local_p(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = True
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            router._prefill_pending_tokens = [8000, 0]
            metadata = AgenticRequestMetadata(
                request_id="fallback", generation=1, parent_generation=0
            )
            request = {
                "bootstrap_room": 9,
                "input_ids": list(range(9000)),
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "fallback",
                        "agentic_generation": 1,
                        "agentic_parent_generation": 0,
                    }
                },
            }
            parent = RequestGeneration("fallback", 0)
            router.early_claim_store.publish_route(
                parent,
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=8000,
            )

            selection = asyncio.create_task(
                router._resolve_dynamic_prefill_work(
                    request, metadata, time.time()
                )
            )
            for _ in range(100):
                arrival = router.early_claim_store.read_arrival(
                    parent, not_before=0.0, max_age_seconds=5.0
                )
                if arrival is not None:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(arrival["target_prefill_domain"], 1)
            selected = await asyncio.wait_for(selection, timeout=1)
            watcher = asyncio.create_task(
                router._watch_dynamic_prefill_route(
                    request, metadata, selected
                )
            )
            router.early_claim_store.publish_route(
                parent,
                route="host_writing",
                prefill_domain=0,
                arena_numa_node=0,
                snapshot_tokens=8000,
            )
            outcome = await asyncio.wait_for(watcher, timeout=1)

            self.assertEqual(selected.domain, 1)
            self.assertEqual(outcome["action"], "redirect")
            self.assertEqual(outcome["route"], "host_writing")
            self.assertEqual(selected.tokens, 1000)
            self.assertEqual(router._prefill_pending_tokens, [8000, 1000])
            await router._move_prefill_work(selected, 0)
            router._publish_parent_arrival(
                request, target_prefill_domain=0, arrived_at=time.time()
            )
            self.assertEqual(router._prefill_pending_tokens, [9000, 0])
            arrival = router.early_claim_store.read_arrival(
                parent, not_before=0.0, max_age_seconds=5.0
            )
            self.assertEqual(arrival["target_prefill_domain"], 0)
            await router._release_prefill_work(selected)

    async def test_recompute_route_charges_full_prompt_tokens(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = True
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            metadata = AgenticRequestMetadata(
                request_id="recompute", generation=1, parent_generation=0
            )
            request = {
                "bootstrap_room": 10,
                "input_ids": list(range(9000)),
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "recompute",
                        "agentic_generation": 1,
                        "agentic_parent_generation": 0,
                    }
                },
            }
            parent = RequestGeneration("recompute", 0)
            router.early_claim_store.publish_route(
                parent,
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=8000,
            )
            selected = await router._resolve_dynamic_prefill_work(
                request, metadata, time.time()
            )
            self.assertEqual(selected.tokens, 1000)

            watcher = asyncio.create_task(
                router._watch_dynamic_prefill_route(
                    request, metadata, selected
                )
            )
            router.early_claim_store.publish_route(
                parent,
                route="recompute",
                prefill_domain=selected.domain,
                snapshot_tokens=8000,
            )
            outcome = await asyncio.wait_for(watcher, timeout=1)

            self.assertEqual(outcome["action"], "recompute")
            self.assertEqual(selected.tokens, 9000)
            self.assertEqual(
                router._prefill_pending_tokens[selected.domain], 9000
            )
            await router._release_prefill_work(selected)

    async def test_dynamic_prefill_fifo_is_independent_per_p(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._p_ready_fifo_locks = {}
            router._ready_path(1).write_bytes(
                orjson.dumps(
                    {
                        "num_kv_tokens": 10,
                        "ready_sequence": 0,
                        "prefill_domain": 0,
                    }
                )
            )
            router._ready_path(2).write_bytes(
                orjson.dumps(
                    {
                        "num_kv_tokens": 10,
                        "ready_sequence": 0,
                        "prefill_domain": 1,
                    }
                )
            )

            lock0, lock1 = await asyncio.gather(
                router._acquire_p_ready_fifo(0, 0),
                router._acquire_p_ready_fifo(0, 1),
            )
            self.assertIsNot(lock0, lock1)
            self.assertIn((0, 0), router._p_ready_submitted_sequences)
            self.assertIn((1, 0), router._p_ready_submitted_sequences)
            lock0.release()
            lock1.release()

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

    async def test_dynamic_dispatch_aborts_then_reposts_on_numa_host_fallback(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.dynamic_prefill_domains = True
            router.numa_domains = True
            router.global_decode = True
            router.prefill_urls = ["http://p0", "http://p1"]
            router.prefill_bootstrap_ports = [100, 200]
            router._prefill_pending_tokens = [8000, 0]
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            router.max_prefill_inflight = 8
            router._prefill_admission = _PrefillAdmissionGate(8, 10)
            parent = RequestGeneration("redirect", 0)
            router.early_claim_store.publish_route(
                parent,
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=8000,
            )
            request = {
                "bootstrap_room": 99,
                "input_ids": list(range(9000)),
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "redirect",
                        "agentic_generation": 1,
                        "agentic_parent_generation": 0,
                    }
                },
            }

            accepted_calls = 0

            async def accepted(_rooms, _task, _route_task=None):
                nonlocal accepted_calls
                accepted_calls += 1
                if accepted_calls == 1:
                    router.early_claim_store.publish_route(
                        parent,
                        route="host_writing",
                        prefill_domain=0,
                        arena_numa_node=0,
                        snapshot_tokens=8000,
                    )

            async def scheduled(_rooms, _task, route_task=None):
                if route_task is not None:
                    await asyncio.wait_for(
                        asyncio.shield(route_task), timeout=1
                    )
                    router._raise_prefill_redirect(route_task)

            router._wait_until_prefill_accepted = accepted
            router._wait_until_prefill_scheduled = scheduled
            router._wait_until_prefill_ready = AsyncMock(return_value=9000)
            router._p_ready_sequence = lambda _rooms: 0

            fifo_lock = asyncio.Lock()

            async def acquire_fifo(_sequence, _domain=0):
                await fifo_lock.acquire()
                return fifo_lock

            router._acquire_p_ready_fifo = acquire_fifo
            decode_reservation = DecodeReservation(
                reservation_id="redirect-d",
                url="http://d0",
                prompt_tokens=9000,
                admission_tokens=9512,
                request_count=1,
                rooms=(1,),
                created_at=0.0,
            )
            router._select_and_reserve_decode = AsyncMock(
                return_value=decode_reservation
            )
            router._release_reservation_when_admitted = AsyncMock()

            class Response:
                status = 200

                def release(self):
                    return None

            class Session:
                def __init__(self):
                    self.calls = []

                async def post(self, url, **kwargs):
                    self.calls.append(
                        (url, orjson.loads(orjson.dumps(kwargs.get("json"))))
                    )
                    return Response()

            session = Session()
            await asyncio.wait_for(
                router._late_dispatch(
                    session, request, "http://p0", "generate", {}
                ),
                timeout=2,
            )

            generate_calls = [call for call in session.calls if call[0].endswith("/generate")]
            self.assertEqual(
                [call[0] for call in generate_calls],
                ["http://p1/generate", "http://p0/generate", "http://d0/generate"],
            )
            self.assertIn(("http://p1/abort_request"), [call[0] for call in session.calls])
            self.assertNotEqual(
                generate_calls[0][1]["bootstrap_room"],
                generate_calls[1][1]["bootstrap_room"],
            )
            self.assertNotEqual(
                generate_calls[0][1]["rid"], generate_calls[1][1]["rid"]
            )

    async def test_same_generation_retries_join_one_detached_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            started = asyncio.Event()
            finish = asyncio.Event()
            calls = 0

            async def generate_once(request, prefill_server, endpoint):
                nonlocal calls
                calls += 1
                started.set()
                await finish.wait()
                return {"text": request["input_ids"]}, 200

            router._generate_once = generate_once
            request = {"input_ids": [1, 2, 3]}
            first = asyncio.create_task(
                router._generate_singleflight(
                    "agent:g2", request, "http://p0", "generate"
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            retry = asyncio.create_task(
                router._generate_singleflight(
                    "agent:g2", request, "http://p1", "generate"
                )
            )
            await asyncio.sleep(0)
            self.assertEqual(calls, 1)
            finish.set()
            self.assertEqual(await first, ({"text": [1, 2, 3]}, 200))
            self.assertEqual(await retry, ({"text": [1, 2, 3]}, 200))
            await asyncio.sleep(0)
            self.assertEqual(calls, 1)

    async def test_client_cancellation_does_not_cancel_generation_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            started = asyncio.Event()
            finish = asyncio.Event()
            calls = 0

            async def generate_once(_request, _prefill_server, _endpoint):
                nonlocal calls
                calls += 1
                started.set()
                await finish.wait()
                return {"ok": True}, 200

            router._generate_once = generate_once
            client = asyncio.create_task(
                router._generate_singleflight(
                    "agent:g3", {}, "http://p0", "generate"
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            client.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await client
            finish.set()
            retry = await router._generate_singleflight(
                "agent:g3", {}, "http://p1", "generate"
            )
            self.assertEqual(retry, ({"ok": True}, 200))
            self.assertEqual(calls, 1)

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

    async def test_prefill_admission_capacity_is_independent_per_p(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._prefill_admissions = [
            _PrefillAdmissionGate(limit=1, new_aging_seconds=10),
            _PrefillAdmissionGate(limit=1, new_aging_seconds=10),
        ]

        p0 = router._prefill_admission_for_domain(0)
        p1 = router._prefill_admission_for_domain(1)
        await p0.acquire(parent_turn=False)
        p0_waiter = asyncio.create_task(p0.acquire(parent_turn=True))
        await asyncio.sleep(0)

        # Saturating P0 does not consume P1's independent capacity.
        await asyncio.wait_for(p1.acquire(parent_turn=False), timeout=1)
        self.assertFalse(p0_waiter.done())
        self.assertEqual((p0.active, p1.active), (1, 1))

        await p0.release()
        await asyncio.wait_for(p0_waiter, timeout=1)
        await p0.release()
        await p1.release()

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

    async def test_target_kv_margin_buffers_p_ready_work(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.target_decode_kv_fraction = 0.90
            loads_now = [
                DecodeLoad("http://d0", 85_000, 100_000, 5, 0, 0, 0, 100),
                DecodeLoad("http://d1", 60_000, 100_000, 30, 0, 0, 0, 100),
            ]

            async def loads(_session):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (32,), 5_000
            )
            # d0 has fewer requests, but admitting this P-ready snapshot would
            # cross the 90% operating target.  Keep it buffered on P and use
            # the D that still has growth/egress headroom.
            self.assertEqual(selected.url, "http://d1")

    async def test_p_ready_admission_fills_available_d_capacity_continuously(self):
        """A prior P->D submission need not complete before the next one starts."""

        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.target_decode_kv_fraction = 0.90
            loads_now = [
                DecodeLoad("http://d0", 40_000, 100_000, 20, 0, 0, 0, 100)
            ]

            async def loads(_session):
                return loads_now

            router._all_decode_loads = loads
            first = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (321,), 5_000
            )
            second = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (322,), 5_000
            )

            self.assertEqual(first.url, "http://d0")
            self.assertEqual(second.url, "http://d0")
            self.assertEqual(len(router._reservations), 2)

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

    async def test_p2d_host_claim_locks_restore_to_same_numa(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.prefill_urls = ["http://p0", "http://p1"]
            router.decode_urls = [
                "http://d0",
                "http://d1",
                "http://d2",
                "http://d3",
            ]
            router.numa_domains = True
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            router.soft_reservation_delay = 100.0
            router.soft_reservation_force_after = 100.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            snapshot_id = p2d_snapshot_id(401)
            attempts = 0

            async def loads(_session):
                nonlocal attempts
                attempts += 1
                entry = ledger.get(snapshot_id)
                if entry is not None and entry["state"] == HostStageState.OFFERED.value:
                    claimed = ledger.claim(snapshot_id, "p2d-p:test")
                    self.assertIsNotNone(claimed)
                    self.assertTrue(
                        ledger.publish_grants(
                            snapshot_id,
                            "p2d-p:test",
                            [{"kind": "shared_host_extent"}],
                        )
                    )
                    self.assertTrue(ledger.ack_chunk(snapshot_id, "p2d-p:test", 0))
                    self.assertTrue(
                        ledger.mark_host_ready(snapshot_id, "p2d-p:test", 1)
                    )
                used = 99_000 if attempts == 1 else 20_000
                return [
                    DecodeLoad("http://d0", used, 100_000, 20, 0, 0, 0, 100),
                    DecodeLoad("http://d1", used, 100_000, 25, 0, 0, 0, 100),
                    # Remote NUMA is deliberately less loaded.  A staged
                    # snapshot must nevertheless stay with P0's local Ds.
                    DecodeLoad("http://d2", used, 100_000, 1, 0, 0, 0, 100),
                    DecodeLoad("http://d3", used, 100_000, 2, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (401,), 5_000, domain=0
            )
            self.assertEqual(selected.url, "http://d0")
            self.assertEqual(selected.p2d_host_snapshot_id, snapshot_id)
            self.assertEqual(selected.prefill_domain, 0)

    async def test_p2d_unclaimed_offer_is_cancelled_when_d_frees(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            router.soft_reservation_delay = 100.0
            router.soft_reservation_force_after = 100.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            attempts = 0

            async def loads(_session):
                nonlocal attempts
                attempts += 1
                used = 99_000 if attempts == 1 else 20_000
                return [
                    DecodeLoad("http://d0", used, 100_000, 10, 0, 0, 0, 100),
                    DecodeLoad("http://d1", used, 100_000, 11, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (402,), 5_000, domain=0
            )
            self.assertIsNone(selected.p2d_host_snapshot_id)
            self.assertEqual(
                ledger.get(p2d_snapshot_id(402))["state"],
                HostStageState.REJECTED.value,
            )

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
