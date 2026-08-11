from __future__ import annotations

import json
import threading
import time
import types

import torch

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
    token_ids_digest,
)
from sglang.srt.managers.scheduler import AgenticEarlyDirectReceive, Scheduler


class _Req:
    def __init__(self, rid: str, queue_class: str | None = None):
        self.rid = rid
        if queue_class is not None:
            self._agentic_kv_queue_class = queue_class


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
    scheduler.tree_cache = types.SimpleNamespace(
        insert=lambda params: (
            inserted.append(params) or types.SimpleNamespace(prefix_len=0)
        )
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
    assert request.snapshot_id not in scheduler.agentic_early_direct_receives


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
    assert polls == [True]
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


def test_missing_parent_wait_covers_direct_to_host_transition(monkeypatch):
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

    expired = _Req("transition-expired")
    expired._agentic_kv_gate_complete = False
    assert scheduler._agentic_should_defer(
        expired, time.monotonic() - 11.0
    ) is False
    assert expired._agentic_kv_fallback == "timeout:missing"


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
