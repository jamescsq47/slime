from __future__ import annotations

import json
import threading
import time
import types
from collections import deque
from pathlib import Path

import torch

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
    token_ids_digest,
)
from sglang.srt.managers.scheduler import (
    AgenticEarlyDirectReceive,
    Scheduler,
)
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache


class _Req:
    def __init__(self, rid: str, queue_class: str | None = None):
        self.rid = rid
        if queue_class is not None:
            self._agentic_kv_queue_class = queue_class


def test_new_method_forces_custom_storage_only_and_baseline_clears_it():
    pd_root = Path(__file__).resolve().parents[1]
    pipeline = (
        pd_root / "scripts/new_method/internal/run_agentic_pipeline.sh"
    ).read_text()
    baseline = (pd_root / "scripts/baseline/run_pd_case.sh").read_text()

    assert "export SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY=true" in pipeline
    assert "unset SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY" in baseline


def test_hiradix_write_back_falls_back_when_host_has_no_capacity():
    """Host pressure must evict an unlocked cache leaf, not crash serving."""

    cache = HiRadixCache.__new__(HiRadixCache)
    class Node:
        lock_ref = 0
        backuped = False
        evicted = False

    node = Node()
    node.parent = types.SimpleNamespace(children={})
    node.parent.children["leaf"] = node
    cache.evictable_leaves = {node}
    cache.eviction_strategy = types.SimpleNamespace(get_priority=lambda _node: 0)
    cache.cache_controller = types.SimpleNamespace(write_policy="write_back")
    cache.write_backup = lambda _node, write_back=False: 0
    regular = []
    cache._evict_regular = lambda victim: regular.append(victim) or 64
    cache.writing_check = lambda write_back=False: None
    cache.update_eviction_metrics = lambda *_args: None

    result = cache.evict(EvictParams(num_tokens=64))

    assert result.num_tokens_evicted == 64
    assert regular == [node]


def test_custom_storage_only_evicts_without_native_host_backup():
    """Custom slow paths own Host storage; generic Radix eviction drops GPU KV."""

    cache = HiRadixCache.__new__(HiRadixCache)

    class Node:
        lock_ref = 0
        backuped = False
        evicted = False

    node = Node()
    node.parent = types.SimpleNamespace(children={"leaf": node})
    cache.agentic_custom_storage_only = True
    cache.evictable_leaves = {node}
    cache.eviction_strategy = types.SimpleNamespace(get_priority=lambda _node: 0)
    cache.cache_controller = types.SimpleNamespace(write_policy="write_back")
    cache.write_backup = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("native HiCache backup must stay disabled")
    )
    regular = []
    cache._evict_regular = lambda victim: regular.append(victim) or 64
    cache.writing_check = lambda write_back=False: (_ for _ in ()).throw(
        AssertionError("native write-back drain must stay disabled")
    )
    cache.update_eviction_metrics = lambda *_args: None

    result = cache.evict(EvictParams(num_tokens=64))

    assert result.num_tokens_evicted == 64
    assert regular == [node]


def test_custom_storage_only_skips_ordinary_storage_prefetch():
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.agentic_custom_storage_only = True
    cache.enable_storage = True
    cache.is_eagle = False
    cache.page_size = 1
    cache.cache_controller = types.SimpleNamespace(
        prefetch_rate_limited=lambda: (_ for _ in ()).throw(
            AssertionError("ordinary requests must not reach Mooncake prefetch")
        )
    )

    assert (
        cache.prefetch_from_storage(
            "ordinary-request",
            types.SimpleNamespace(),
            [1, 2, 3, 4],
        )
        is None
    )


def test_custom_storage_only_keeps_agentic_mooncake_prefetch_available():
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.agentic_custom_storage_only = True
    cache.enable_storage = True
    cache.is_eagle = False
    cache.page_size = 1
    calls = []
    cache.cache_controller = types.SimpleNamespace(
        prefetch_rate_limited=lambda: calls.append("checked") or True
    )

    cache.prefetch_from_storage(
        "agentic-request",
        types.SimpleNamespace(),
        [1, 2, 3, 4],
        agentic_expected_tokens=4,
        agentic_extra_key="agentic-v1:snapshot:g1",
    )

    assert calls == ["checked"]


def test_p_agentic_transient_backup_stays_out_of_generic_storage():
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.ongoing_backup = {}
    cache.cache_controller = types.SimpleNamespace(
        write_storage=lambda *_args: (_ for _ in ()).throw(
            AssertionError("transient P KV must use the agentic slow path, not generic L3")
        )
    )
    node = types.SimpleNamespace(
        key=types.SimpleNamespace(extra_key="agentic-v1:snapshot:g1"),
    )

    cache.write_backup_storage(node)

    assert cache.ongoing_backup == {}


def test_early_direct_transport_progresses_off_scheduler_thread(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    request = RequestGeneration("trajectory-background-poll", 0)
    receiver = types.SimpleNamespace(poll=lambda: KVPoll.Success)
    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=types.SimpleNamespace(token_count=64),
        claim_id="claim",
        receiver=receiver,
        device_indices=torch.tensor([1]),
        started_at=time.monotonic(),
        arrived_at=time.time(),
    )
    scheduler.agentic_early_direct_receives = {request.snapshot_id: entry}
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_early_direct_progress_stop = threading.Event()
    scheduler._agentic_poll_early_direct_receives = types.MethodType(
        lambda self: setattr(entry, "transport_poll", entry.receiver.poll()),
        scheduler,
    )
    monkeypatch.setenv(
        "SGLANG_AGENTIC_KV_P_DIRECT_PROGRESS_INTERVAL_SECONDS", "0.001"
    )

    worker = threading.Thread(
        target=scheduler._agentic_early_direct_progress_worker, daemon=True
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    while entry.transport_poll is not KVPoll.Success and time.monotonic() < deadline:
        time.sleep(0.001)
    scheduler.agentic_early_direct_progress_stop.set()
    worker.join(timeout=1.0)

    assert entry.transport_poll is KVPoll.Success
    assert entry.completed_at is None


def test_scheduler_direct_gate_never_drives_transport_progress(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    parent = RequestGeneration("trajectory-nonblocking-gate", 0)
    manifest = types.SimpleNamespace(
        request=parent,
        state=SnapshotState.DIRECT_READY,
        created_at=time.time(),
    )
    req = _Req("tokenized-child")
    req._agentic_kv_gate_complete = False
    scheduler.agentic_host_staging_manager = None
    scheduler.agentic_early_direct_receives = {}
    scheduler.agentic_early_claim_store = types.SimpleNamespace(
        read_arrival=lambda *args, **kwargs: {"kind": "arrival"},
        read_final=lambda *args, **kwargs: None,
    )
    scheduler._agentic_snapshot_store = types.MethodType(
        lambda self: types.SimpleNamespace(
            load=lambda request, require_ready=False: manifest
        ),
        scheduler,
    )
    scheduler._agentic_poll_early_direct_receives = types.MethodType(
        lambda self, *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the GPU scheduler must not progress Direct transport")
        ),
        scheduler,
    )
    scheduler._agentic_bind_early_direct_receive = types.MethodType(
        lambda self, child, request: None,
        scheduler,
    )
    scheduler._agentic_start_direct_load = types.MethodType(
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy Req-owned Direct must not start")
        ),
        scheduler,
    )
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=parent),
    )

    assert scheduler._agentic_should_defer(
        req, time.monotonic(), allow_start_io=True
    ) is True


def test_blocking_direct_poll_does_not_hold_scheduler_state_lock(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    request = RequestGeneration("trajectory-blocking-poll", 0)
    poll_started = threading.Event()
    release_poll = threading.Event()

    def blocking_poll():
        poll_started.set()
        assert release_poll.wait(timeout=1.0)
        return KVPoll.Transferring

    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=types.SimpleNamespace(token_count=64),
        claim_id="claim",
        receiver=types.SimpleNamespace(poll=blocking_poll),
        device_indices=torch.tensor([1]),
        started_at=time.monotonic(),
        arrived_at=time.time(),
    )
    scheduler.agentic_early_direct_receives = {request.snapshot_id: entry}
    scheduler.agentic_early_direct_terminal = {}
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_early_direct_next_scan_at = float("inf")
    scheduler.agentic_early_claim_store = object()
    scheduler.agentic_direct_runtime = object()
    scheduler._agentic_snapshot_store = types.MethodType(
        lambda self: object(), scheduler
    )
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT", "120")

    worker = threading.Thread(
        target=scheduler._agentic_poll_early_direct_receives_once,
        daemon=True,
    )
    worker.start()
    assert poll_started.wait(timeout=1.0)

    acquired = scheduler.agentic_early_direct_poll_lock.acquire(timeout=0.05)
    if acquired:
        scheduler.agentic_early_direct_poll_lock.release()
    release_poll.set()
    worker.join(timeout=1.0)

    assert acquired
    assert not worker.is_alive()


def test_early_direct_batches_transport_progress_once_per_manager(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    manager = object()
    batch_calls = []

    class Receiver:
        def __init__(self):
            self.kv_mgr = manager

        def poll(self):
            raise AssertionError("individual poll must not repeat manager progress")

        @classmethod
        def poll_many(cls, receivers):
            batch_calls.append(tuple(receivers))
            return [KVPoll.Transferring] * len(receivers)

    entries = {}
    for index in range(3):
        request = RequestGeneration(f"trajectory-batched-poll-{index}", 0)
        entries[request.snapshot_id] = AgenticEarlyDirectReceive(
            request=request,
            manifest=types.SimpleNamespace(token_count=64),
            claim_id=f"claim-{index}",
            receiver=Receiver(),
            device_indices=torch.tensor([index + 1]),
            started_at=time.monotonic(),
            arrived_at=time.time(),
        )

    scheduler.agentic_early_direct_receives = entries
    scheduler.agentic_early_direct_terminal = {}
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_early_direct_next_scan_at = float("inf")
    scheduler.agentic_early_claim_store = object()
    scheduler.agentic_direct_runtime = object()
    scheduler._agentic_snapshot_store = types.MethodType(
        lambda self: object(), scheduler
    )
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT", "120")

    scheduler._agentic_poll_early_direct_receives_once()

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 3
    assert all(entry.transport_poll is KVPoll.Transferring for entry in entries.values())


def test_completed_early_direct_binds_only_when_tokenized_req_arrives():
    scheduler = Scheduler.__new__(Scheduler)
    request = RequestGeneration("trajectory-early", 0)
    tokens = [11, 12, 13, 14]
    manifest = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=len(tokens),
        byte_size=0,
        state=SnapshotState.DIRECT_LOADING,
        token_digest=token_ids_digest(tokens),
        direct_bootstrap_addr="127.0.0.1:1",
        direct_room=7,
        claim_id="claim",
    )
    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=manifest,
        claim_id="claim",
        receiver=object(),
        device_indices=torch.tensor([21, 22, 23, 24]),
        started_at=time.monotonic(),
        arrived_at=time.time(),
        completed_at=time.monotonic(),
    )
    scheduler.agentic_early_direct_receives = {request.snapshot_id: entry}
    scheduler.agentic_early_direct_terminal = {}
    inserted = []
    parent_node = object()
    locks = []
    scheduler.tree_cache = types.SimpleNamespace(
        insert=lambda params: (
            inserted.append(params) or types.SimpleNamespace(prefix_len=0)
        ),
        match_prefix=lambda _params: types.SimpleNamespace(
            device_indices=entry.device_indices,
            last_device_node=parent_node,
        ),
        inc_lock_ref=lambda node: locks.append(node),
    )
    scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(
        free=lambda indices: None
    )
    req = _Req("tokenized-child")
    req.origin_input_ids = tokens + [99]
    req.extra_key = "cache-salt"
    req.priority = 0

    assert scheduler._agentic_bind_early_direct_receive(req, request) is False
    assert inserted[0].key.token_ids == tokens
    assert inserted[0].value is entry.device_indices
    assert req._agentic_kv_direct_hit_tokens == len(tokens)
    assert req._agentic_direct_parent_pin_node is parent_node
    assert locks == [parent_node]
    assert request.snapshot_id not in scheduler.agentic_early_direct_receives


def test_completed_direct_uses_pre_reserved_ordinary_pages_for_ownership_swap():
    scheduler = Scheduler.__new__(Scheduler)
    request = RequestGeneration("trajectory-direct-transit", 0)
    tokens = list(range(64))
    manifest = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=64,
        byte_size=0,
        state=SnapshotState.DIRECT_LOADING,
        token_digest=token_ids_digest(tokens),
        direct_bootstrap_addr="127.0.0.1:1",
        direct_room=8,
        claim_id="claim",
    )
    allocation = types.SimpleNamespace(allocated_tokens=64, token_count=64)
    ordinary_replacement = torch.arange(640, 704, dtype=torch.int64)
    entry = AgenticEarlyDirectReceive(
        request=request,
        manifest=manifest,
        claim_id="claim",
        receiver=object(),
        device_indices=torch.arange(64, dtype=torch.int64),
        started_at=time.monotonic(),
        arrived_at=time.time(),
        credit_allocation=allocation,
        ordinary_replacement_indices=ordinary_replacement,
        completed_at=time.monotonic(),
    )
    scheduler.agentic_early_direct_receives = {request.snapshot_id: entry}
    scheduler.agentic_early_direct_terminal = {}
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_direct_ordinary_reserved_tokens = int(
        ordinary_replacement.numel()
    )
    inserted = []
    scheduler.tree_cache = types.SimpleNamespace(
        insert=lambda params: (
            inserted.append(params) or types.SimpleNamespace(prefix_len=0)
        )
    )
    scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(
        alloc=lambda _tokens: (_ for _ in ()).throw(
            AssertionError("ordinary pages must be reserved before Direct claim")
        ),
        free=lambda _indices: None,
    )
    parent_node = object()
    locks = []
    scheduler.tree_cache.match_prefix = lambda _params: types.SimpleNamespace(
        device_indices=entry.device_indices,
        last_device_node=parent_node,
    )
    scheduler.tree_cache.inc_lock_ref = lambda node: locks.append(node)
    promoted = []
    scheduler.agentic_direct_credit_pool = types.SimpleNamespace(
        promote_to_ordinary=lambda alloc, replacement: promoted.append(
            (alloc, replacement)
        ),
        mark_bound=lambda _alloc: (_ for _ in ()).throw(
            AssertionError("successful ownership swap must not stay reserve-owned")
        ),
        free_tokens=0,
    )
    req = _Req("tokenized-direct-transit")
    req.origin_input_ids = tokens + [999]
    req.extra_key = "cache-salt"
    req.priority = 0

    assert scheduler._agentic_bind_early_direct_receive(req, request) is False
    assert len(inserted) == 1
    assert promoted == [(allocation, ordinary_replacement)]
    assert locks == [parent_node]
    assert not hasattr(req, "_agentic_direct_credit_allocation")
    assert req._agentic_direct_parent_pin_node is parent_node
    assert request.snapshot_id not in scheduler.agentic_early_direct_receives


def test_native_prefill_lock_atomically_releases_direct_parent_pin():
    events = []
    native_node = object()
    direct_parent_node = object()
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.is_hybrid_swa = False
    adder.tree_cache = types.SimpleNamespace(
        inc_lock_ref=lambda node: events.append(("inc", node))
        or types.SimpleNamespace(),
        dec_lock_ref=lambda node: events.append(("dec", node)),
    )
    req = types.SimpleNamespace(
        last_node=native_node,
        _agentic_direct_parent_pin_node=direct_parent_node,
    )

    adder._req_inc_lock_ref(req)

    assert events == [("inc", native_node), ("dec", direct_parent_node)]
    assert not hasattr(req, "_agentic_direct_parent_pin_node")


def test_abort_releases_direct_parent_pin_and_request_generation_cache():
    scheduler = Scheduler.__new__(Scheduler)
    parent_node = object()
    events = []
    scheduler.tree_cache = types.SimpleNamespace(
        dec_lock_ref=lambda node: events.append(("unpin", node)),
        release_agentic_request_cache=lambda req, committed_len: events.append(
            ("release", req.rid, committed_len)
        ),
        release_aborted_request=lambda rid: events.append(("abort", rid)),
    )
    req = _Req("aborted-direct-parent")
    req._agentic_direct_parent_pin_node = parent_node
    req._agentic_direct_parent_token_count = 4096

    scheduler._agentic_abort_cleanup(req)

    assert events == [
        ("unpin", parent_node),
        ("release", req.rid, 4096),
        ("abort", req.rid),
    ]
    assert not hasattr(req, "_agentic_direct_parent_pin_node")
    assert not hasattr(req, "_agentic_direct_parent_token_count")


def test_inotify_arrivals_are_deduplicated_then_admitted_from_fifo(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    request = RequestGeneration("trajectory-event-admission", 0)
    payload = {
        "arrived_at": time.time(),
        "target_prefill_domain": 1,
    }
    scheduler.agentic_early_direct_arrival_watcher = types.SimpleNamespace(
        poll=lambda _timeout: [(request, payload), (request, payload)]
    )
    scheduler.agentic_early_direct_admission_queue = deque()
    scheduler.agentic_early_direct_admission_ids = set()
    scheduler.agentic_early_direct_reservation_queue = deque()
    scheduler.agentic_early_direct_reserved_queue = deque()
    scheduler.agentic_early_direct_receives = {}
    scheduler.agentic_early_direct_terminal = {}
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_direct_ordinary_reserved_tokens = 0
    allocation = types.SimpleNamespace(allocated_tokens=1024)
    scheduler.agentic_direct_credit_pool = types.SimpleNamespace(
        allocate=lambda tokens: allocation if tokens == 1024 else None,
        release=lambda _allocation: None,
    )
    scheduler.server_args = types.SimpleNamespace(page_size=64)
    ordinary = torch.arange(1024, dtype=torch.int64)
    scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(
        alloc=lambda tokens: ordinary if tokens == 1024 else None,
        free=lambda _indices: None,
    )
    manifest = types.SimpleNamespace(
        request=request,
        state=SnapshotState.DIRECT_READY,
        created_at=payload["arrived_at"],
        token_count=1024,
    )
    snapshot_store = types.SimpleNamespace(
        load=lambda _request, require_ready=False: manifest
    )
    admitted = []
    scheduler._agentic_start_early_direct_receive = types.MethodType(
        lambda self, observed, *_args, **kwargs: admitted.append(
            (observed, kwargs)
        )
        or True,
        scheduler,
    )
    monkeypatch.setenv("SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_PREFILL_DOMAIN", "1")

    scheduler._agentic_collect_direct_arrivals(
        scheduler.agentic_early_direct_poll_lock
    )
    assert len(scheduler.agentic_early_direct_admission_queue) == 1
    scheduler._agentic_admit_queued_direct_receives(
        snapshot_store,
        direct_timeout=2.0,
        poll_lock=scheduler.agentic_early_direct_poll_lock,
    )

    assert not scheduler.agentic_early_direct_admission_queue
    assert len(scheduler.agentic_early_direct_reservation_queue) == 1

    scheduler._agentic_reserve_queued_direct_pages()
    assert len(scheduler.agentic_early_direct_reserved_queue) == 1
    scheduler._agentic_start_reserved_direct_receives(snapshot_store)

    assert admitted[0][0] == request
    assert admitted[0][1]["credit_allocation"] is allocation
    assert admitted[0][1]["ordinary_replacement_indices"] is ordinary
    assert not scheduler.agentic_early_direct_reserved_queue
    assert request.snapshot_id not in scheduler.agentic_early_direct_admission_ids


def test_early_direct_ordinary_reservation_failure_never_claims():
    scheduler = Scheduler.__new__(Scheduler)
    request = RequestGeneration("trajectory-no-ordinary-space", 0)
    manifest = types.SimpleNamespace(token_count=1024)
    allocation = object()
    released = []
    scheduler.server_args = types.SimpleNamespace(page_size=64)
    scheduler.agentic_direct_credit_pool = types.SimpleNamespace(
        allocate=lambda _tokens: allocation,
        release=lambda value: released.append(value),
    )
    scheduler.token_to_kv_pool_allocator = types.SimpleNamespace(
        alloc=lambda _tokens: None
    )
    scheduler.agentic_early_direct_poll_lock = threading.RLock()
    scheduler.agentic_early_direct_reservation_queue = deque(
        [(request, manifest, time.time(), 0)]
    )
    scheduler.agentic_early_direct_reserved_queue = deque()
    scheduler.agentic_early_direct_admission_ids = {request.snapshot_id}

    scheduler._agentic_reserve_queued_direct_pages()

    assert released == [allocation]
    assert not scheduler.agentic_early_direct_reserved_queue
    assert request.snapshot_id not in scheduler.agentic_early_direct_admission_ids


def test_early_direct_pages_are_counted_as_transport_reservation():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.agentic_host_staging_manager = None
    scheduler.decode_offload_manager = None
    scheduler.agentic_early_direct_receives = {
        "a": types.SimpleNamespace(
            manifest=types.SimpleNamespace(token_count=640)
        ),
        "b": types.SimpleNamespace(
            manifest=types.SimpleNamespace(token_count=1024)
        ),
    }

    assert scheduler._agentic_reserved_tokens() == 1664


def test_arrival_marker_never_falls_back_to_req_owned_direct(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    parent = RequestGeneration("trajectory-marker", 0)
    manifest = types.SimpleNamespace(
        request=parent,
        state=SnapshotState.DIRECT_READY,
        created_at=time.time(),
    )
    req = _Req("tokenized-child")
    req._agentic_kv_gate_complete = False
    scheduler.agentic_host_staging_manager = None
    scheduler.agentic_early_direct_receives = {}
    scheduler.agentic_early_claim_store = types.SimpleNamespace(
        read_arrival=lambda *args, **kwargs: {"kind": "arrival"},
        read_final=lambda *args, **kwargs: None,
    )
    scheduler._agentic_snapshot_store = types.MethodType(
        lambda self: types.SimpleNamespace(
            load=lambda request, require_ready=False: manifest
        ),
        scheduler,
    )
    polls = []
    scheduler._agentic_poll_early_direct_receives = types.MethodType(
        lambda self: polls.append(True), scheduler
    )
    bind_calls = []

    def bind(self, child, request):
        bind_calls.append(request.snapshot_id)
        return None if len(bind_calls) == 1 else True

    scheduler._agentic_bind_early_direct_receive = types.MethodType(
        bind, scheduler
    )
    scheduler._agentic_start_direct_load = types.MethodType(
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy Req-owned Direct must not start")
        ),
        scheduler,
    )
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=parent),
    )

    assert scheduler._agentic_should_defer(
        req, time.monotonic(), allow_start_io=True
    ) is True
    assert polls == []
    assert bind_calls == [parent.snapshot_id, parent.snapshot_id]


def test_parent_final_marker_recomputes_without_waiting(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    parent = RequestGeneration("trajectory-final-marker", 0)
    req = _Req("repair-after-final")
    req._agentic_kv_gate_complete = False
    scheduler.agentic_early_claim_store = types.SimpleNamespace(
        read_final=lambda *args, **kwargs: {"kind": "final"}
    )
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=parent),
    )

    assert scheduler._agentic_should_defer(req, time.monotonic()) is False
    assert req._agentic_kv_gate_complete is True
    assert req._agentic_kv_fallback == "application_final"


def test_final_manifest_recomputes_without_ready_timeout(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    parent = RequestGeneration("trajectory-final-manifest", 0)
    manifest = types.SimpleNamespace(
        request=parent,
        state=SnapshotState.FINAL,
        created_at=time.time(),
    )
    req = _Req("repair-after-manifest")
    req._agentic_kv_gate_complete = False
    scheduler.agentic_host_staging_manager = None
    scheduler.agentic_early_claim_store = None
    scheduler.agentic_early_direct_receives = {}
    scheduler._agentic_bind_early_direct_receive = types.MethodType(
        lambda self, child, request: None, scheduler
    )
    scheduler._agentic_snapshot_store = types.MethodType(
        lambda self: types.SimpleNamespace(
            load=lambda request, require_ready=False: manifest
        ),
        scheduler,
    )
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=parent),
    )

    assert scheduler._agentic_should_defer(req, time.monotonic()) is False
    assert req._agentic_kv_gate_complete is True
    assert req._agentic_kv_fallback == "final"


def test_missing_parent_wait_uses_request_level_ready_timeout(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_READY_TIMEOUT", "120")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT", "2")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_HOST_TRANSITION_GRACE", "8")
    scheduler = Scheduler.__new__(Scheduler)
    parent = RequestGeneration("trajectory-transition", 0)
    scheduler.agentic_host_staging_manager = None
    scheduler.agentic_early_claim_store = None
    scheduler.agentic_early_direct_receives = {}
    scheduler._agentic_bind_early_direct_receive = types.MethodType(
        lambda self, child, request: None, scheduler
    )
    scheduler._agentic_snapshot_store = types.MethodType(
        lambda self: types.SimpleNamespace(
            load=lambda request, require_ready=False: None
        ),
        scheduler,
    )
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=parent),
    )

    waiting = _Req("transition-wait")
    waiting._agentic_kv_gate_complete = False
    assert scheduler._agentic_should_defer(
        waiting, time.monotonic() - 3.0
    ) is True

    transitioning = _Req("transition-still-publishing")
    transitioning._agentic_kv_gate_complete = False
    assert scheduler._agentic_should_defer(
        transitioning, time.monotonic() - 11.0
    ) is True

    expired = _Req("transition-expired")
    expired._agentic_kv_gate_complete = False
    assert scheduler._agentic_should_defer(
        expired, time.monotonic() - 121.0
    ) is False
    assert expired._agentic_kv_fallback == "timeout:missing"


def test_direct_loading_parent_waits_instead_of_recomputing(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_READY_TIMEOUT", "120")
    scheduler = Scheduler.__new__(Scheduler)
    parent = RequestGeneration("trajectory-direct-loading", 0)
    manifest = types.SimpleNamespace(
        request=parent,
        state=SnapshotState.DIRECT_LOADING,
        created_at=time.time(),
    )
    scheduler.agentic_host_staging_manager = None
    scheduler.agentic_early_claim_store = None
    scheduler.agentic_early_direct_receives = {}
    scheduler._agentic_bind_early_direct_receive = types.MethodType(
        lambda self, child, request: None, scheduler
    )
    scheduler._agentic_snapshot_store = types.MethodType(
        lambda self: types.SimpleNamespace(
            load=lambda request, require_ready=False: manifest
        ),
        scheduler,
    )
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=parent),
    )

    waiting = _Req("direct-loading-child")
    waiting._agentic_kv_gate_complete = False
    assert scheduler._agentic_should_defer(
        waiting, time.monotonic() - 10.0
    ) is True
    assert waiting._agentic_kv_gate_complete is False


def test_ready_merge_is_stable_fast_then_slow_then_new():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting_queue = [
        _Req("new-old"),
        _Req("slow-old", "slow"),
        _Req("fast-old", "fast"),
    ]
    scheduler._merge_disagg_prefill_ready(
        [_Req("new-new"), _Req("fast-new", "fast"), _Req("slow-new", "slow")]
    )
    assert [req.rid for req in scheduler.waiting_queue] == [
        "fast-old",
        "fast-new",
        "slow-old",
        "slow-new",
        "new-old",
        "new-new",
    ]


def test_priority_is_restored_after_generic_policy_sort():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting_queue = [
        _Req("new"),
        _Req("slow", "slow"),
        _Req("fast", "fast"),
    ]
    scheduler._prioritize_agentic_prefill_ready()
    assert [req.rid for req in scheduler.waiting_queue] == ["fast", "slow", "new"]


def test_prefill_scheduler_publishes_atomic_acceptance_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_PD_P_READY_DIR", str(tmp_path))
    req = _Req("accepted-request")
    req.bootstrap_room = 29

    Scheduler._agentic_publish_p_accepted(req)

    accepted = tmp_path / "29.accepted"
    assert json.loads(accepted.read_text()) == {"rid": "accepted-request"}
    assert req._agentic_p_accepted_notified is True
    assert list(tmp_path.iterdir()) == [accepted]


def test_initial_request_enters_metadata_only_new_queue(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = scheduler_module.DisaggregationMode.PREFILL
    scheduler.agentic_kv_waiting_queue = []
    req = _Req("initial-request")
    req._agentic_kv_gate_complete = False
    monkeypatch.delenv("SGLANG_PD_P_READY_DIR", raising=False)
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=None),
    )

    scheduler._add_request_to_queue(req)

    assert req._agentic_kv_queue_class == "new"
    assert req._agentic_kv_wait_enqueued is True
    assert scheduler.agentic_kv_waiting_queue[0][0] is req


def test_prefill_scheduler_publishes_scheduled_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_PD_P_READY_DIR", str(tmp_path))
    req = _Req("scheduled-request")
    req.bootstrap_room = 31

    Scheduler._agentic_publish_p_scheduled(req)

    scheduled = tmp_path / "31.scheduled"
    assert json.loads(scheduled.read_text()) == {"rid": "scheduled-request"}
    assert req._agentic_p_scheduled_notified is True


def test_waiting_drain_batches_io_and_does_not_hol_block_slow(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "1")
    scheduler = Scheduler.__new__(Scheduler)
    fast_missing = _Req("fast-missing", "fast")
    slow_ready = _Req("slow-ready", "slow")
    another_slow = _Req("slow-next", "slow")
    scheduler.agentic_kv_waiting_queue = [
        (fast_missing, time.monotonic()),
        (slow_ready, time.monotonic()),
        (another_slow, time.monotonic()),
    ]
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(loads={})
    calls = []

    def should_defer(self, req, started_at, *, allow_start_io=True):
        calls.append((req.rid, allow_start_io))
        if req is slow_ready and allow_start_io:
            self.agentic_host_staging_manager.loads[req.rid] = object()
        return True

    scheduler._agentic_should_defer = types.MethodType(should_defer, scheduler)
    scheduler._drain_agentic_kv_waiting_queue()

    # The unresolved fast item is inspected first, but it does not prevent a
    # runnable slow item from using this iteration's single admission. Once
    # that I/O starts, later entries remain metadata-only until the next tick.
    assert calls == [
        ("fast-missing", True),
        ("slow-ready", True),
    ]
    assert list(scheduler.agentic_host_staging_manager.loads) == ["slow-ready"]


def test_active_slow_io_consumes_only_slow_slot(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SELECTED_IO_CAP", "1")
    scheduler = Scheduler.__new__(Scheduler)
    active = _Req("active", "slow")
    waiting = _Req("waiting", "slow")
    scheduler.agentic_kv_waiting_queue = [
        (active, time.monotonic()),
        (waiting, time.monotonic()),
    ]
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(
        loads={active.rid: object()}
    )
    calls = []

    def should_defer(self, req, started_at, *, allow_start_io=True):
        calls.append((req.rid, allow_start_io))
        return True

    scheduler._agentic_should_defer = types.MethodType(should_defer, scheduler)
    scheduler._drain_agentic_kv_waiting_queue()
    assert calls == [("active", True), ("waiting", False)]


def test_direct_has_four_independent_page_credits(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SELECTED_IO_CAP", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "4")
    scheduler = Scheduler.__new__(Scheduler)
    active = [_Req(f"active-{index}", "fast") for index in range(4)]
    for req in active:
        req._agentic_direct_receiver = object()
    waiting = _Req("waiting-direct", "fast")
    scheduler.agentic_kv_waiting_queue = [
        *((req, time.monotonic()) for req in active),
        (waiting, time.monotonic()),
    ]
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(loads={})
    calls = []

    def should_defer(self, req, started_at, *, allow_start_io=True):
        calls.append((req.rid, allow_start_io))
        return True

    scheduler._agentic_should_defer = types.MethodType(should_defer, scheduler)
    scheduler._drain_agentic_kv_waiting_queue()

    assert calls[-1] == ("waiting-direct", False)


def test_active_slow_io_does_not_consume_direct_credit(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SELECTED_IO_CAP", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_IO_CAP", "4")
    scheduler = Scheduler.__new__(Scheduler)
    slow = _Req("active-slow", "slow")
    direct = _Req("waiting-direct", "fast")
    scheduler.agentic_kv_waiting_queue = [
        (slow, time.monotonic()),
        (direct, time.monotonic()),
    ]
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(
        loads={slow.rid: object()}
    )
    calls = []

    def should_defer(self, req, started_at, *, allow_start_io=True):
        calls.append((req.rid, allow_start_io))
        if req is direct and allow_start_io:
            req._agentic_direct_receiver = object()
        return True

    scheduler._agentic_should_defer = types.MethodType(should_defer, scheduler)
    scheduler._drain_agentic_kv_waiting_queue()

    assert calls == [("active-slow", True), ("waiting-direct", True)]


def test_mooncake_claim_consumes_selected_io_slot(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "8")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SELECTED_IO_CAP", "1")
    scheduler = Scheduler.__new__(Scheduler)
    claimed = _Req("claimed", "slow")
    waiting = _Req("waiting", "slow")
    scheduler.agentic_kv_waiting_queue = [
        (claimed, time.monotonic()),
        (waiting, time.monotonic()),
    ]
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(loads={})
    calls = []
    admitted = []

    def should_defer(self, req, started_at, *, allow_start_io=True):
        calls.append((req.rid, allow_start_io))
        if req is claimed and allow_start_io:
            req._agentic_kv_manifest = object()
            return False
        return True

    def add_request(self, req, is_retracted=False):
        admitted.append(req.rid)

    scheduler._agentic_should_defer = types.MethodType(should_defer, scheduler)
    scheduler._add_request_to_queue = types.MethodType(add_request, scheduler)
    scheduler._drain_agentic_kv_waiting_queue()

    assert calls == [("claimed", True), ("waiting", False)]
    assert admitted == ["claimed"]


def test_stale_shared_host_gate_falls_back_at_ready_timeout(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_READY_TIMEOUT", "0.01")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(
        loads={},
        gate_request=lambda req, parent, allow_start=True: True,
        snapshot_ready=lambda parent: False,
    )
    req = _Req("stale-shared-host", "slow")
    req._agentic_kv_gate_complete = False
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=object()),
    )

    deferred = scheduler._agentic_should_defer(
        req, time.monotonic() - 1.0, allow_start_io=True
    )

    assert deferred is False
    assert req._agentic_kv_gate_complete is True
    assert req._agentic_kv_fallback == "timeout:shared_host"


def test_ready_shared_host_snapshot_does_not_timeout_while_queued(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_READY_TIMEOUT", "0.01")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(
        loads={},
        gate_request=lambda req, parent, allow_start=True: True,
        snapshot_ready=lambda parent: True,
    )
    req = _Req("ready-shared-host", "slow")
    req._agentic_kv_gate_complete = False
    monkeypatch.setattr(
        scheduler_module.AgenticRequestMetadata,
        "from_req",
        lambda req: types.SimpleNamespace(parent=object()),
    )

    deferred = scheduler._agentic_should_defer(
        req, time.monotonic() - 1.0, allow_start_io=False
    )

    assert deferred is True
    assert req._agentic_kv_gate_complete is False


def test_fast_io_preempts_aged_slow_without_extra_concurrency(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_ADMISSION_BATCH", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SELECTED_IO_CAP", "1")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SLOW_AGING_SECONDS", "2")
    scheduler = Scheduler.__new__(Scheduler)
    fast = _Req("fresh-fast", "fast")
    slow = _Req("aged-slow", "slow")
    now = time.monotonic()
    scheduler.agentic_kv_waiting_queue = [(fast, now), (slow, now - 3)]
    scheduler.agentic_host_staging_manager = types.SimpleNamespace(loads={})
    calls = []

    def should_defer(self, req, started_at, *, allow_start_io=True):
        calls.append((req.rid, allow_start_io))
        if req is slow and allow_start_io:
            self.agentic_host_staging_manager.loads[req.rid] = object()
        return True

    scheduler._agentic_should_defer = types.MethodType(should_defer, scheduler)
    scheduler._drain_agentic_kv_waiting_queue()

    # The fast request is inspected first.  Because it does not start I/O,
    # the scheduler remains work-conserving and uses the slot for slow work.
    assert calls == [("fresh-fast", True), ("aged-slow", True)]


def test_compute_priority_remains_fast_slow_new_despite_age(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SLOW_AGING_SECONDS", "2")
    scheduler = Scheduler.__new__(Scheduler)
    aged = _Req("aged", "slow")
    aged._agentic_kv_wait_started_at = time.monotonic() - 3
    fast = _Req("fast", "fast")
    fresh = _Req("fresh", "slow")
    fresh._agentic_kv_wait_started_at = time.monotonic()
    new = _Req("new")
    scheduler.waiting_queue = [new, fresh, fast, aged]

    scheduler._prioritize_agentic_prefill_ready()

    assert [req.rid for req in scheduler.waiting_queue] == [
        "fast",
        "fresh",
        "aged",
        "new",
    ]


def test_new_compute_stays_after_fast_and_slow_despite_age(monkeypatch):
    monkeypatch.setenv("SGLANG_AGENTIC_KV_SLOW_AGING_SECONDS", "2")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_NEW_AGING_SECONDS", "10")
    scheduler = Scheduler.__new__(Scheduler)
    aged_new = _Req("aged-new")
    aged_new._agentic_kv_wait_started_at = time.monotonic() - 11
    fresh_new = _Req("fresh-new")
    fresh_new._agentic_kv_wait_started_at = time.monotonic()
    fast = _Req("fast", "fast")
    slow = _Req("slow", "slow")
    slow._agentic_kv_wait_started_at = time.monotonic()
    scheduler.waiting_queue = [fresh_new, slow, aged_new, fast]

    scheduler._prioritize_agentic_prefill_ready()

    assert [req.rid for req in scheduler.waiting_queue] == [
        "fast",
        "slow",
        "fresh-new",
        "aged-new",
    ]


def test_p_ready_soft_caps_count_completed_requests_and_tokens(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "continuous")
    monkeypatch.setenv("SGLANG_PD_P_READY_HBM_HIGH_WATERMARK", "0.85")
    monkeypatch.setenv("SGLANG_PD_P_READY_REQUEST_CAP", "12")
    monkeypatch.setenv("SGLANG_PD_P_READY_TOKEN_CAP_FRACTION", "0.25")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
        p_ready_dir="/dev/shm/test"
    )
    scheduler.chunked_req = None
    scheduler.max_total_num_tokens = 1000
    scheduler.disagg_prefill_inflight_queue = []
    scheduler._get_token_info = lambda: (100, 0.10, 900, 0)

    assert scheduler._should_throttle_p_ready_compute_ahead() is False

    ready = _Req("ready")
    ready.origin_input_ids = list(range(256))
    scheduler.disagg_prefill_inflight_queue = [ready]
    assert scheduler._should_throttle_p_ready_compute_ahead() is True


def test_disabled_p_ready_backpressure_uses_native_scheduler_capacity(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "disabled")
    monkeypatch.setenv("SGLANG_PD_P_READY_REQUEST_CAP", "12")
    monkeypatch.setenv("SGLANG_PD_P_READY_TOKEN_CAP_FRACTION", "0.25")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
        p_ready_dir="/dev/shm/test"
    )
    scheduler.chunked_req = None
    scheduler._p_ready_compute_ahead_throttled = True
    scheduler._p_ready_compute_credit_tokens = 1

    # Disabled means no synthetic request/token/HBM watermark throttle.  The
    # ordinary SGLang batch builder and KV allocator remain the safety limit.
    assert scheduler._should_throttle_p_ready_compute_ahead() is False
    assert scheduler._p_ready_compute_credit_tokens is None
    assert scheduler._p_ready_compute_ahead_throttled is False


def test_p_ready_hysteresis_does_not_resume_above_token_cap(monkeypatch):
    monkeypatch.setenv("SGLANG_PD_P_READY_BACKPRESSURE_MODE", "hysteresis")
    monkeypatch.setenv("SGLANG_PD_P_READY_HBM_HIGH_WATERMARK", "0.70")
    monkeypatch.setenv("SGLANG_PD_P_READY_HBM_LOW_WATERMARK", "0.55")
    monkeypatch.setenv("SGLANG_PD_P_READY_REQUEST_CAP", "12")
    monkeypatch.setenv("SGLANG_PD_P_READY_TOKEN_CAP_FRACTION", "0.25")
    monkeypatch.setenv("SGLANG_PD_P_READY_MAX_INFLIGHT", "48")
    monkeypatch.setenv("SGLANG_PD_P_READY_RESUME_INFLIGHT", "40")
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
        p_ready_dir="/dev/shm/test"
    )
    scheduler.chunked_req = None
    scheduler.max_total_num_tokens = 1000
    scheduler._p_ready_compute_ahead_throttled = True
    scheduler._get_token_info = lambda: (400, 0.40, 600, 0)
    ready = _Req("ready")
    ready.origin_input_ids = list(range(300))
    scheduler.disagg_prefill_inflight_queue = [ready]

    # HBM is below the low watermark, but the independent completed-token
    # credit is still exceeded, so ordinary New work must remain paused.
    assert scheduler._should_throttle_p_ready_compute_ahead() is True


def test_p_ready_soft_cap_holds_new_but_runs_direct(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running_batch = types.SimpleNamespace(batch_is_full=True)
    scheduler.chunked_req = None
    scheduler.waiting_queue = [_Req("new"), _Req("direct", "fast")]
    scheduler.process_prefill_chunk = lambda: None
    scheduler._should_throttle_p_ready_compute_ahead = lambda: True
    observed = []

    def get_batch():
        observed.append([req.rid for req in scheduler.waiting_queue])
        return None

    scheduler.get_new_batch_prefill = get_batch
    scheduler.maybe_prepare_mlp_sync_batch = lambda batch: batch

    batch = scheduler.get_next_disagg_prefill_batch_to_run()

    assert batch is None
    assert observed == [["direct"]]
    assert [req.rid for req in scheduler.waiting_queue] == ["direct", "new"]
