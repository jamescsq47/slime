"""CPU regression tests against the isolated pd_mamba source, no GPU startup."""
import ast
import logging
import os
from pathlib import Path
import queue
import threading
import time
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

SOURCE = Path(os.environ.get(
    "SGLANG_OVERLAY_ROOT", "/homes/siqic/sglang-agentic-mamba/python"
)) / "sglang/srt/disaggregation"


def method(filename, name):
    tree = ast.parse((SOURCE / filename).read_text())
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1
    unit = ast.Module(body=ast.parse("from __future__ import annotations").body + functions,
                      type_ignores=[])
    namespace = dict(os=os, queue=queue, time=time, nullcontext=nullcontext, HybridReqToTokenPool=HybridReqToTokenPool,
                     logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(unit), filename, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("native,installed", [(False, True), (True, True), (False, False)])
def test_release_progress_before_prealloc(native, installed):
    calls = []

    class ReachedPrealloc(Exception):
        pass

    def resume():
        calls.append("prealloc")
        raise ReachedPrealloc

    manager = SimpleNamespace(
        check_offload_progress=lambda: calls.append("release"),
        pop_ready_responses=lambda: [],
    ) if installed else None
    scheduler = SimpleNamespace(
        enable_decode_hicache=False,
        server_args=SimpleNamespace(disaggregation_decode_enable_offload_kvcache=native),
        decode_offload_manager=manager,
        disagg_decode_prealloc_queue=SimpleNamespace(resume_retracted_reqs=resume),
    )
    with pytest.raises(ReachedPrealloc):
        method("decode.py", "process_decode_queue")(scheduler)
    assert calls == (["release", "prealloc"] if installed else ["prealloc"])


def test_repeated_completion_returns_request_kv_and_three_mamba_slots_once():
    released_states, released_kv = [], []
    pool = object.__new__(HybridReqToTokenPool)
    pool.free_slots = []
    pool.req_to_token = torch.arange(8).reshape(2, 4)
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.tensor([[0, 0], [2, 3]])
    pool.mamba_allocator = SimpleNamespace(free=lambda ids: released_states.extend(ids.tolist()))
    manager = SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda ids: released_kv.extend(ids.tolist())),
        tree_cache=SimpleNamespace(disable=True, protected_size_=0),
        page_size=1, offloaded_state={},
        _decode_io_events=queue.SimpleQueue(), _decode_commit_ready_at=0,
        _decode_pending_release_tokens=0, _decode_scheduler_commit_events=0,
        _decode_scheduler_commit_seconds=0,
    )
    manager._release_finished_req = MethodType(method("agentic_decode_manager.py", "_release_finished_req"), manager)
    drain = method("agentic_decode_manager.py", "_drain_decode_io_events")
    for iteration in range(600):
        if iteration:
            assert pool.free_slots.pop() == 1
        req = SimpleNamespace(
            rid=str(iteration), req_pool_idx=1, mamba_pool_idx=torch.tensor(1),
            mamba_ping_pong_track_buffer=torch.tensor([2, 3]), mamba_next_track_idx=0,
            prefix_indices=[], pop_committed_kv_cache=lambda: 4,
            pop_overallocated_kv_cache=lambda: (4, 4),
        )
        # Same event twice represents duplicate completion/cancellation delivery.
        manager._decode_pending_release_tokens = 8
        for _ in range(2):
            manager._decode_io_events.put(("release_finished", req, 0, 4))
        drain(manager)
        assert req.req_pool_idx is None and req.mamba_pool_idx is None
        assert req.mamba_ping_pong_track_buffer is None
        assert pool.available_size() == 1
        assert manager._decode_pending_release_tokens == 0
    assert released_states == [1, 2, 3] * 600
    assert released_kv == [4, 5, 6, 7] * 600


@pytest.mark.parametrize("sentinel", [None, -1])
def test_terminal_completion_does_not_release_again(sentinel):
    events = queue.SimpleQueue()
    events.put(("release_finished", SimpleNamespace(req_pool_idx=sentinel), 0, 4))
    manager = SimpleNamespace(
        _decode_io_events=events, _decode_commit_ready_at=0,
        _decode_pending_release_tokens=4, _decode_scheduler_commit_events=0,
        _decode_scheduler_commit_seconds=0,
        _release_finished_req=lambda *_: pytest.fail("duplicate release"),
    )
    method("agentic_decode_manager.py", "_drain_decode_io_events")(manager)
    assert events.empty() and manager._decode_pending_release_tokens == 0


def handoff_fixture():
    req = SimpleNamespace(rid="one", req_pool_idx=1)
    manager = SimpleNamespace(
        agentic_direct_candidates={"one:0": {"req": req}},
        _agentic_candidates_lock=threading.RLock(),
        _decode_io_events=queue.SimpleQueue(),
        _agentic_pending_release_items=lambda: (),
        tp_world_size=1, _decode_io_async_enabled=True,
    )
    manager._enqueue_agentic_release = lambda req, start: manager._decode_io_events.put(
        ("release_finished", req, start, 22080)
    )
    retire = method("agentic_decode_manager.py", "_retire_agentic_candidate_for_release")
    count = method("agentic_decode_manager.py", "agentic_inflight_snapshot_count").fget
    return manager, req, retire, count


def test_idle_accounts_for_tp1_release_queue_until_commit():
    manager, req, retire, count = handoff_fixture()
    assert count(manager) == 1
    assert retire(manager, "one:0", req)
    assert not manager.agentic_direct_candidates
    assert count(manager) == 1  # Regression: old counter incorrectly returned 0.
    assert not retire(manager, "one:0", req)
    assert manager._decode_io_events.qsize() == 1
    manager._decode_io_events.get_nowait()
    req.req_pool_idx = None  # Scheduler frees on the same thread as idle check.
    assert count(manager) == 0
    # No catch-all "pool not empty => busy" guard: unowned leaked pages would
    # still reach the native checker once registered work is gone.
    manager.unowned_tokens = 22080
    assert count(manager) == 0


def test_observer_cannot_see_candidate_to_queue_gap():
    manager, req, retire, count = handoff_fixture()
    queued, finish, observing, observed = (threading.Event() for _ in range(4))
    readings = []

    def paused_enqueue(req, start):
        manager._decode_io_events.put(("release_finished", req, start, 22080))
        queued.set()
        assert finish.wait(3)

    def observe():
        observing.set()
        readings.append(count(manager))
        observed.set()

    manager._enqueue_agentic_release = paused_enqueue
    producer = threading.Thread(target=retire, args=(manager, "one:0", req))
    consumer = threading.Thread(target=observe)
    producer.start()
    try:
        assert queued.wait(3)
        consumer.start()
        assert observing.wait(3)
        assert not observed.wait(0.05)
    finally:
        finish.set()
        producer.join(3)
        if consumer.ident is not None:
            consumer.join(3)
    assert not producer.is_alive() and not consumer.is_alive()
    assert readings == [1]


def test_failed_or_mismatched_release_preserves_candidate():
    manager, req, retire, count = handoff_fixture()
    with pytest.raises(RuntimeError, match="identity"):
        retire(manager, "one:0", object())
    assert count(manager) == 1 and manager._decode_io_events.empty()

    def fail(*_):
        raise RuntimeError("injected publication failure")

    manager._enqueue_agentic_release = fail
    with pytest.raises(RuntimeError, match="publication failure"):
        retire(manager, "one:0", req)
    assert manager.agentic_direct_candidates["one:0"]["req"] is req
    assert count(manager) == 1 and manager._decode_io_events.empty()


def test_tp_pending_owner_remains_counted():
    manager, req, retire, count = handoff_fixture()
    pending = {}
    manager.tp_world_size = 2
    manager._agentic_pending_release_items = lambda: tuple(pending.items())
    manager._enqueue_agentic_release = lambda req, start: pending.setdefault("one:0", (req, start))
    assert retire(manager, "one:0", req)
    assert count(manager) == 1 and manager._decode_io_events.empty()
    pending.clear()
    assert count(manager) == 0


def test_legacy_sync_free_keeps_owner_without_holding_metadata_lock():
    manager, req, retire, count = handoff_fixture()
    manager._decode_io_async_enabled = False
    entered, finish = threading.Event(), threading.Event()
    frees = []

    def free(req, start):
        entered.set()
        assert finish.wait(3)
        frees.append(req.rid)
        req.req_pool_idx = None

    manager._enqueue_agentic_release = free
    worker = threading.Thread(target=retire, args=(manager, "one:0", req))
    worker.start()
    try:
        assert entered.wait(3)
        assert manager._agentic_candidates_lock.acquire(timeout=0.1)
        manager._agentic_candidates_lock.release()
        assert count(manager) == 1
        assert not retire(manager, "one:0", req)
    finally:
        finish.set()
        worker.join(3)
    assert not worker.is_alive() and frees == ["one"]
    assert count(manager) == 0


def test_legacy_sync_failure_keeps_owner_and_can_retry():
    manager, req, retire, count = handoff_fixture()
    manager._decode_io_async_enabled = False

    def fail(*_):
        raise RuntimeError("injected free failure")

    manager._enqueue_agentic_release = fail
    with pytest.raises(RuntimeError, match="free failure"):
        retire(manager, "one:0", req)
    assert count(manager) == 1
    assert not manager.agentic_direct_candidates["one:0"].get("release_committing")
    manager._enqueue_agentic_release = lambda req, start: setattr(req, "req_pool_idx", None)
    assert retire(manager, "one:0", req)
    assert count(manager) == 0
