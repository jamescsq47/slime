from __future__ import annotations

import json
import threading
import time
import types

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    SnapshotState,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator


class _CountingReceiver:
    require_staging = False

    def __init__(self):
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        return KVPoll.Transferring


def test_transfer_poll_runs_only_in_background_progress():
    receiver = _CountingReceiver()
    decode_req = types.SimpleNamespace(
        req=types.SimpleNamespace(rid="req-0"),
        kv_receiver=receiver,
    )
    transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    transfer_queue.queue = [decode_req]
    transfer_queue.enable_staging = False
    transfer_queue._async_progress_enabled = True
    transfer_queue._async_poll_lock = threading.Lock()

    transfer_queue.background_progress()
    assert receiver.poll_count == 1
    assert decode_req._async_transfer_poll == int(KVPoll.Transferring)

    # The scheduler-facing method consumes a cached status; it must not call
    # the receiver or any transport backend itself.
    assert transfer_queue.pop_transferred() == []
    assert receiver.poll_count == 1
    assert not hasattr(decode_req, "_async_transfer_poll")


def test_background_does_not_repoll_status_claimed_by_scheduler():
    receiver = _CountingReceiver()
    decode_req = types.SimpleNamespace(
        req=types.SimpleNamespace(rid="req-1"),
        kv_receiver=receiver,
        _async_transfer_poll=int(KVPoll.Success),
        _async_transfer_poll_claimed=True,
    )
    transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    transfer_queue.queue = [decode_req]
    transfer_queue.enable_staging = False
    transfer_queue._async_progress_enabled = True
    transfer_queue._async_poll_lock = threading.Lock()

    transfer_queue.background_progress()
    assert receiver.poll_count == 0


def test_prealloc_control_scan_is_paced_independently_from_metadata(monkeypatch):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue._async_progress_enabled = True
    queue._async_control_next_at = 0.0
    queue._async_control_interval = 0.02
    calls = {"resolve": 0, "handshake": 0, "ready": 0, "metadata": 0}

    queue._resolve_pending_reqs = lambda: calls.__setitem__(
        "resolve", calls["resolve"] + 1
    )
    queue._update_handshake_waiters = lambda: calls.__setitem__(
        "handshake", calls["handshake"] + 1
    )
    queue._background_update_p_ready = lambda: calls.__setitem__(
        "ready", calls["ready"] + 1
    )
    queue._background_prepare_metadata = lambda: calls.__setitem__(
        "metadata", calls["metadata"] + 1
    )

    now = [10.0]
    monkeypatch.setattr(
        "sglang.srt.disaggregation.decode.time.monotonic", lambda: now[0]
    )
    queue.background_progress()
    now[0] = 10.005
    queue.background_progress()
    now[0] = 10.021
    queue.background_progress()

    assert calls == {"resolve": 2, "handshake": 2, "ready": 2, "metadata": 3}


class _ProgressQueue:
    def __init__(self):
        self.queue = []
        self.calls = 0

    def enable_async_progress(self):
        pass

    def background_progress(self):
        self.calls += 1


def test_blocked_agentic_control_does_not_stop_transfer_progress():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._decode_io_async_enabled = True
    manager._decode_io_threads = {}
    manager._decode_io_wakeups = {
        name: threading.Event() for name in ("transfer", "prealloc", "agentic")
    }
    manager._decode_io_stop = threading.Event()
    manager._decode_io_intervals = {
        "transfer": 0.001,
        "prealloc": 0.005,
        "agentic": 0.001,
    }
    manager._decode_io_cuda_device = None
    manager._decode_io_error_count = 0
    manager._decode_io_last_error = None
    manager._decode_io_events = None
    manager.agentic_direct_candidates = {}
    manager.agentic_relay_worker = None

    prealloc = _ProgressQueue()
    transfer = _ProgressQueue()
    agentic_release = threading.Event()

    def blocked_agentic(*, progress_relay):
        assert progress_relay is False
        agentic_release.wait(0.25)

    manager._check_agentic_direct_progress = blocked_agentic
    manager.start_decode_io_progress_worker(prealloc, transfer)
    try:
        time.sleep(0.05)
        assert transfer.calls >= 10
        assert prealloc.calls >= 3
    finally:
        manager._decode_io_stop.set()
        agentic_release.set()
        for wakeup in manager._decode_io_wakeups.values():
            wakeup.set()
        for thread in manager._decode_io_threads.values():
            thread.join(timeout=1.0)


def test_blocked_isolated_relay_does_not_stop_p_to_d_progress():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._decode_io_async_enabled = True
    manager._decode_io_threads = {}
    manager._decode_io_wakeups = {
        name: threading.Event()
        for name in ("transfer", "prealloc", "agentic", "relay")
    }
    manager._decode_io_stop = threading.Event()
    manager._decode_io_intervals = {
        "transfer": 0.001,
        "prealloc": 0.005,
        "agentic": 0.005,
        "relay": 0.001,
    }
    manager._decode_io_cuda_device = None
    manager._decode_io_error_count = 0
    manager._decode_io_last_error = None
    manager._decode_io_events = None
    manager.agentic_direct_candidates = {}
    manager._agentic_relay_progress_isolated = True

    relay_release = threading.Event()
    relay_entered = threading.Event()

    def blocked_relay():
        relay_entered.set()
        relay_release.wait(0.25)

    manager.agentic_relay_worker = types.SimpleNamespace(
        poll=blocked_relay,
        active=None,
    )
    manager._check_agentic_direct_progress = lambda *, progress_relay: None
    prealloc = _ProgressQueue()
    transfer = _ProgressQueue()

    manager.start_decode_io_progress_worker(prealloc, transfer)
    try:
        assert relay_entered.wait(0.1)
        time.sleep(0.05)
        assert transfer.calls >= 10
        assert prealloc.calls >= 3
    finally:
        manager._decode_io_stop.set()
        relay_release.set()
        for wakeup in manager._decode_io_wakeups.values():
            wakeup.set()
        for thread in manager._decode_io_threads.values():
            thread.join(timeout=1.0)


class _PrefillHarness(SchedulerDisaggregationPrefillMixin):
    pass


def test_prefill_producer_enqueues_fifo_without_touching_sender():
    class Sender:
        def poll(self):
            raise AssertionError("producer must not poll transport")

    scheduler = _PrefillHarness()
    scheduler._prefill_ready_condition = threading.Condition()
    scheduler._prefill_ready_queue = __import__("collections").deque()
    scheduler._prefill_ready_queued_rids = set()
    first = types.SimpleNamespace(
        rid="first", disagg_kv_sender=Sender(),
        disagg_p_ready_deferred=True,
        _async_prefill_transfer_payload=(1, [1], None),
    )
    second = types.SimpleNamespace(
        rid="second", disagg_kv_sender=Sender(),
        disagg_p_ready_deferred=True,
        _async_prefill_transfer_payload=(1, [2], None),
    )

    assert scheduler._enqueue_deferred_prefill_transfer(first)
    assert scheduler._enqueue_deferred_prefill_transfer(second)
    assert list(scheduler._prefill_ready_queue) == [first, second]
    assert (first._p_ready_sequence, second._p_ready_sequence) == (0, 1)


def test_legacy_prefill_send_is_consumed_without_ready_marker():
    scheduler = _PrefillHarness()
    scheduler._prefill_ready_condition = threading.Condition()
    scheduler._prefill_ready_queue = __import__("collections").deque()
    scheduler._prefill_ready_queued_rids = set()
    req = types.SimpleNamespace(
        rid="warmup",
        disagg_p_ready_deferred=False,
        disagg_p_ready_transfer_started=True,
    )

    assert scheduler._enqueue_deferred_prefill_transfer(req)
    assert list(scheduler._prefill_ready_queue) == [req]
    assert not hasattr(req, "_p_ready_sequence")


def test_parallel_consumers_publish_ready_in_producer_fifo(tmp_path):
    scheduler = _PrefillHarness()
    scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
        p_ready_dir=str(tmp_path)
    )
    scheduler._prefill_ready_publish_condition = threading.Condition()
    scheduler._prefill_ready_next_publish_sequence = 0
    scheduler._prefill_transfer_stop = threading.Event()
    first = types.SimpleNamespace(
        rid="first", bootstrap_room=1, origin_input_ids=[1], _p_ready_sequence=0
    )
    second = types.SimpleNamespace(
        rid="second", bootstrap_room=2, origin_input_ids=[1, 2], _p_ready_sequence=1
    )

    later = threading.Thread(
        target=scheduler._publish_deferred_prefill_ready,
        args=(second,),
        daemon=True,
    )
    later.start()
    time.sleep(0.01)
    assert not (tmp_path / "2.ready").exists()

    scheduler._publish_deferred_prefill_ready(first)
    later.join(timeout=1.0)
    assert json.loads((tmp_path / "1.ready").read_text())["ready_sequence"] == 0
    assert json.loads((tmp_path / "2.ready").read_text())["ready_sequence"] == 1


def test_blocked_prefill_sender_does_not_block_another_consumer():
    entered = threading.Event()
    release = threading.Event()

    class BlockedSender:
        def poll(self):
            entered.set()
            release.wait(0.25)
            return KVPoll.Success

    class FastSender:
        def poll(self):
            return KVPoll.Success

    blocked = types.SimpleNamespace(
        rid="prefill-blocked", disagg_kv_sender=BlockedSender()
    )
    fast = types.SimpleNamespace(rid="prefill-fast", disagg_kv_sender=FastSender())
    scheduler = _PrefillHarness()
    scheduler.disagg_prefill_inflight_queue = [blocked, fast]
    scheduler._prefill_transfer_poll_lock = threading.Lock()
    scheduler._prefill_transfer_stop = threading.Event()
    scheduler._prefill_transfer_interval = 0.001
    scheduler._prefill_ready_condition = threading.Condition()
    scheduler._prefill_ready_queue = __import__("collections").deque([blocked, fast])
    scheduler._prefill_ready_queued_rids = {blocked.rid, fast.rid}
    scheduler._publish_deferred_prefill_ready = lambda _req: None

    workers = [
        threading.Thread(
            target=scheduler._prefill_transfer_consumer_worker,
            args=(index,),
            daemon=True,
        )
        for index in range(2)
    ]
    for worker in workers:
        worker.start()
    assert entered.wait(0.1)
    deadline = time.monotonic() + 0.1
    while not hasattr(fast, "_async_prefill_transfer_poll"):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert fast._async_prefill_transfer_poll == int(KVPoll.Success)

    release.set()
    deadline = time.monotonic() + 0.2
    while not hasattr(blocked, "_async_prefill_transfer_poll"):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    scheduler._prefill_transfer_stop.set()
    with scheduler._prefill_ready_condition:
        scheduler._prefill_ready_condition.notify_all()
    for worker in workers:
        worker.join(timeout=1.0)


def test_prefill_terminal_poll_survives_scheduler_snapshot_race():
    """A Success written after an old snapshot must remain level-triggered."""

    req = types.SimpleNamespace(rid="prefill-racy-terminal")
    scheduler = _PrefillHarness()
    scheduler.disagg_prefill_inflight_queue = [req]
    scheduler._prefill_transfer_poll_lock = threading.Lock()

    # Scheduler observes the default transient state because the consumer has
    # not published its terminal result yet.
    assert scheduler._prefill_transfer_cached_polls() == [int(KVPoll.Transferring)]

    # The consumer wins the race before scheduler cleanup.
    with scheduler._prefill_transfer_poll_lock:
        req._async_prefill_transfer_poll = int(KVPoll.Success)
        req._async_prefill_transfer_consumer_active = False

    scheduler._release_prefill_transfer_poll_claims([req])

    # The next scheduler pass must still observe Success and release P KV.
    assert scheduler._prefill_transfer_cached_polls() == [int(KVPoll.Success)]


def test_prefill_progress_worker_starts_deferred_p_to_d_transfer():
    calls = []

    class Sender:
        def poll(self):
            return KVPoll.WaitingForInput

        def init(self, num_pages, metadata_index):
            calls.append(("init", num_pages, metadata_index))

        def send(self, page_indices, state_indices):
            calls.append(("send", list(page_indices), state_indices))

    class TimeStats:
        def set_prefill_transfer_queue_entry_time(self):
            calls.append(("timestamp",))

    req = types.SimpleNamespace(
        rid="prefill-deferred",
        disagg_kv_sender=Sender(),
        disagg_p_ready_deferred=True,
        disagg_p_ready_transfer_started=False,
        metadata_buffer_index=7,
        _async_prefill_transfer_payload=(3, [1, 2, 3], None),
        time_stats=TimeStats(),
    )
    scheduler = _PrefillHarness()
    poll = scheduler._prefill_transfer_progress_req_once(req)

    assert calls == [
        ("init", 3, 7),
        ("send", [1, 2, 3], None),
        ("timestamp",),
    ]
    assert req.disagg_p_ready_transfer_started
    assert poll == int(KVPoll.Transferring)


def test_deferred_prefill_metadata_waits_for_matching_logprob_token_id():
    req = types.SimpleNamespace(
        return_logprob=True,
        output_ids=[42],
        output_token_logprobs_idx=[],
    )
    scheduler = _PrefillHarness()

    # Preparing here would snapshot output_ids=42 while the client-facing
    # logprob token id is still absent/stale.  The request must remain on the
    # scheduler retry path until add_logprob_return_values has populated it.
    assert not scheduler._prepare_deferred_prefill_transfer(req)

    req.output_token_logprobs_idx = [41]
    assert not scheduler._prepare_deferred_prefill_transfer(req)


def test_paged_allocator_frees_ordered_request_pages_without_global_sort():
    allocator = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
    allocator.page_size = 4
    allocator.is_not_in_free_group = True
    allocator.need_sort = False
    allocator.debug_mode = False
    allocator.free_pages = torch.empty((0,), dtype=torch.int64)
    allocator.release_pages = torch.empty((0,), dtype=torch.int64)
    allocator.free_group = []

    # Token locations for each request page are contiguous and ordered.  The
    # partial final page must still be released exactly once.
    allocator.free(torch.tensor([4, 5, 6, 7, 12, 13, 14], dtype=torch.int64))

    assert allocator.free_pages.tolist() == [1, 3]


class _CharacterTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class _FinishReason:
    def __init__(self, finish_type):
        self.finish_type = finish_type

    def to_json(self):
        return {"type": self.finish_type}


def _finished_agentic_req(text, finish_type="stop"):
    return types.SimpleNamespace(
        finished=lambda: True,
        output_ids=[ord(char) for char in text],
        origin_input_ids=[1] * 64,
        tokenizer=_CharacterTokenizer(),
        finished_reason=_FinishReason(finish_type),
    )


def _agentic_metadata():
    return AgenticRequestMetadata(
        request_id="terminal-output",
        generation=0,
        tool_suffix_strings=("</tool_call>",),
        terminal_marker_strings=(r"\boxed{",),
    )


def test_explicit_terminal_output_never_publishes_reverse_kv():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._publish_agentic_direct_candidate = lambda *_args: (_ for _ in ()).throw(
        AssertionError("terminal output attempted reverse-KV publication")
    )

    explicit_answer = _finished_agentic_req(r"therefore \boxed{42}")
    assert not manager._offload_agentic_finished_snapshot(
        explicit_answer, _agentic_metadata()
    )


def test_unknown_output_publishes_provisional_direct_candidate():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.page_size = 4
    manager.agentic_direct_runtime = object()
    published = []
    manager._publish_agentic_direct_candidate = (
        lambda req, metadata, tokens: published.append(list(tokens)) or True
    )

    ordinary = _finished_agentic_req("malformed output requiring repair")
    assert manager._offload_agentic_finished_snapshot(
        ordinary, _agentic_metadata()
    )
    assert len(published) == 1


def test_length_finish_wins_over_tool_like_suffix():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._publish_agentic_direct_candidate = lambda *_args: (_ for _ in ()).throw(
        AssertionError("truncated output attempted reverse-KV publication")
    )
    truncated = _finished_agentic_req(
        '{"name":"code_interpreter"}</tool_call>', finish_type="length"
    )
    assert not manager._offload_agentic_finished_snapshot(
        truncated, _agentic_metadata()
    )


def test_explicit_tool_continuation_can_publish_reverse_kv():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.page_size = 4
    manager.agentic_direct_runtime = object()
    published = []
    manager._publish_agentic_direct_candidate = (
        lambda req, metadata, tokens: published.append(
            (req, metadata, list(tokens))
        )
        or True
    )
    tool = _finished_agentic_req(
        '{"name":"code_interpreter"}</tool_call> '
    )
    assert manager._offload_agentic_finished_snapshot(tool, _agentic_metadata())
    assert len(published) == 1


def test_unconfirmed_tool_candidate_fails_without_becoming_final():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    snapshot_id = "repair-parent:0"
    manifest = types.SimpleNamespace(
        snapshot_id=snapshot_id,
        state=SnapshotState.DIRECT_READY,
    )
    failed = []
    manager.agentic_snapshot_store = types.SimpleNamespace(
        fail_direct_offer=lambda item, owner_id, reason: (
            failed.append((item, reason)) or item
        )
    )
    released = []
    manager._enqueue_agentic_release = lambda req, delay: released.append((req, delay))
    manager._cleanup_agentic_direct_sender = lambda _candidate: None
    manager._agentic_release_early_claim = lambda _candidate, reason: released.append(
        ("claim", reason)
    )
    manager._agentic_release_final_confirmation = lambda _candidate: None
    manager.agentic_direct_candidates = {snapshot_id: object()}
    candidate = {
        "req": object(),
        "manifest": manifest,
        "created_at": 10.0,
        "staging": False,
        "sent": False,
        "metadata": types.SimpleNamespace(
            current=types.SimpleNamespace(storage_id="repair-parent:0")
        ),
    }

    assert manager._agentic_fail_unconfirmed_tool_candidate(
        candidate, manifest, 12.0
    )
    assert failed == [(manifest, "application_tool_unconfirmed")]
    assert snapshot_id not in manager.agentic_direct_candidates
    assert ("claim", "unconfirmed_tool") in released
