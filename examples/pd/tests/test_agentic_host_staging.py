from __future__ import annotations

import os
import queue
import threading
import types
import uuid

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticDHostStagingClient,
    AgenticPHostStagingManager,
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)


def _ledger():
    path = f"/dev/shm/sglang-agentic-staging-test-{os.getpid()}-{uuid.uuid4().hex}.json"
    ledger = SharedHostStagingLedger(path)
    return ledger, path


def _offer(snapshot_id="req:0"):
    request_id, generation = snapshot_id.rsplit(":", 1)
    return {
        "snapshot_id": snapshot_id,
        "request_id": request_id,
        "generation": int(generation),
        "token_count": 192,
        "token_digest": "digest",
        "logical_hashes": ["a", "b", "c"],
        "byte_size": 300,
        "storage_namespace": "ns:",
        "d_bootstrap_addr": "127.0.0.1:1",
        "room_seed": 10,
    }


def test_host_ready_requires_every_d2h_chunk_ack():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer())
        assert ledger.claim("req:0", "p0") is not None
        assert ledger.publish_grants(
            "req:0",
            "p0",
            [
                {"seq": 0, "room": 11, "slot": 0, "start_page": 0, "num_pages": 1},
                {"seq": 1, "room": 12, "slot": 1, "start_page": 1, "num_pages": 1},
                {"seq": 2, "room": 13, "slot": 0, "start_page": 2, "num_pages": 1},
            ],
        )
        assert ledger.ack_chunk("req:0", "p0", 0)
        assert ledger.ack_chunk("req:0", "p0", 1)
        assert not ledger.mark_host_ready("req:0", "p0", 3)
        assert ledger.get("req:0")["state"] == HostStageState.HOST_WRITING.value
        assert ledger.ack_chunk("req:0", "p0", 2)
        assert ledger.mark_host_ready("req:0", "p0", 3)
        assert ledger.get("req:0")["state"] == HostStageState.HOST_READY.value
    finally:
        os.unlink(path)


def test_claim_is_atomic_across_competing_p_threads():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer())
        winners = []
        barrier = threading.Barrier(9)

        def claim(index):
            barrier.wait()
            if ledger.claim("req:0", f"p{index}") is not None:
                winners.append(index)

        threads = [threading.Thread(target=claim, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        assert len(winners) == 1
    finally:
        os.unlink(path)


def test_tp_host_h2d_ready_barrier_waits_for_every_rank():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer())
        assert ledger.claim("req:0", "p-group") is not None
        assert ledger.publish_grants(
            "req:0",
            "p-group",
            [{"seq": 0, "room": 11, "slot": 0, "start_page": 0, "num_pages": 3}],
        )
        assert ledger.ack_chunk("req:0", "p-group", 0)
        assert ledger.mark_host_ready("req:0", "p-group", 1)

        assert not ledger.mark_host_h2d_ready_rank(
            "req:0", "p-group", tp_rank=0, tp_size=2
        )
        assert ledger.mark_host_h2d_ready_rank(
            "req:0", "p-group", tp_rank=1, tp_size=2
        )
        assert ledger.get("req:0")["h2d_ready_ranks"] == [0, 1]
        assert not ledger.tp_host_followers_loaded("req:0", "p-group", tp_size=2)
        assert ledger.complete_host_load_rank(
            "req:0", "p-group", tp_rank=1, tp_size=2
        )
        assert ledger.tp_host_followers_loaded("req:0", "p-group", tp_size=2)
    finally:
        os.unlink(path)


def test_tp_host_load_selection_is_rank0_owned_and_group_atomic():
    ledger, path = _ledger()
    try:
        # A non-primary rank cannot independently select its local queue head.
        assert ledger.select_tp_host_load(
            "rank1-head:0", "p-group", tp_rank=1, tp_size=2
        ) == (None, False)
        assert ledger.select_tp_host_load(
            "rank0-head:0", "p-group", tp_rank=0, tp_size=2
        ) == ("rank0-head:0", False)
        # Rank 1 is redirected to rank 0's snapshot until it joins that exact
        # request-generation.
        assert ledger.select_tp_host_load(
            "rank1-head:0", "p-group", tp_rank=1, tp_size=2
        ) == ("rank0-head:0", False)
        assert ledger.select_tp_host_load(
            "rank0-head:0", "p-group", tp_rank=1, tp_size=2
        ) == ("rank0-head:0", True)
        assert ledger.active_tp_host_load("p-group", tp_size=2) == "rank0-head:0"
        assert not ledger.progress_tp_host_admission(
            "rank0-head:0", "p-group", tp_rank=0, tp_size=2
        )
        assert not ledger.progress_tp_host_admission(
            "rank0-head:0", "p-group", tp_rank=1, tp_size=2
        )
        assert not ledger.progress_tp_host_admission(
            "rank0-head:0", "p-group", tp_rank=0, tp_size=2
        )
        assert not ledger.progress_tp_host_admission(
            "rank0-head:0", "p-group", tp_rank=1, tp_size=2
        )
        assert not ledger.progress_tp_host_admission(
            "rank0-head:0", "p-group", tp_rank=0, tp_size=2
        )
        assert ledger.progress_tp_host_admission(
            "rank0-head:0", "p-group", tp_rank=0, tp_size=2
        )
        assert ledger.progress_tp_host_admission(
            "rank0-head:0", "p-group", tp_rank=1, tp_size=2
        )
        assert ledger.admit_tp_host_load(
            "rank0-head:0", "p-group", tp_rank=0, tp_size=2
        )
        assert ledger.active_tp_host_load("p-group", tp_size=2) == "rank0-head:0"
        assert ledger.admit_tp_host_load(
            "rank0-head:0", "p-group", tp_rank=1, tp_size=2
        )
        assert ledger.active_tp_host_load("p-group", tp_size=2) is None
    finally:
        os.unlink(path)


def test_p_only_claims_offers_for_its_numa_arena():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.arena_numa_node = 0
    manager.ledger = types.SimpleNamespace(
        claim=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wrong-NUMA offer must not be claimed")
        )
    )
    offer = _offer("numa-one:0")
    offer.update(
        state=HostStageState.OFFERED.value,
        arena_numa_node=1,
        created_at=1.0,
    )

    manager._admit_one({"numa-one:0": offer})


def test_p_only_claims_offers_for_its_domain_within_numa():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.arena_numa_node = 0
    manager.arena_domain = 0
    manager.ledger = types.SimpleNamespace(
        claim=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wrong-domain offer must not be claimed")
        )
    )
    offer = _offer("domain-one:0")
    offer.update(
        state=HostStageState.OFFERED.value,
        arena_numa_node=0,
        arena_domain=1,
        created_at=1.0,
    )

    assert manager._admit_one({"domain-one:0": offer}) is False


def test_p_accepts_legacy_numa_only_offer_without_domain():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.arena_numa_node = 0
    manager.arena_domain = 1

    offer = _offer("legacy:0")
    offer.update(arena_numa_node=0)

    assert manager._offer_targets_this_arena(offer) is True


def test_d_offer_carries_the_p_owned_arena_domain():
    offered = []
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.ledger = types.SimpleNamespace(
        offer=lambda payload: offered.append(payload) or payload
    )
    client.source_numa_node = 0
    client.arena_numa_node = 0
    client.arena_domain = 1
    client.direct_runtime = None
    metadata = AgenticRequestMetadata("domain-offer", 2, parent_generation=1)
    manifest = SnapshotManifest(
        request=metadata.current,
        page_keys=(),
        token_count=64,
        byte_size=0,
        state=SnapshotState.SLOW_FALLBACK,
    )

    client.offer(
        manifest=manifest,
        metadata=metadata,
        token_count=64,
        token_digest="digest",
        logical_hashes=["page"],
        byte_size=1024,
    )

    assert offered[0]["arena_numa_node"] == 0
    assert offered[0]["arena_domain"] == 1


def test_p_keeps_capacity_blocked_offer_pending():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.arena_numa_node = -1
    manager.page_size = 64
    claimed = []
    manager.ledger = types.SimpleNamespace(
        claim=lambda snapshot_id, owner: claimed.append((snapshot_id, owner))
    )
    manager.owner = "p0"
    manager._capacity_wait_timeout_seconds = 0.0
    manager._can_admit = lambda byte_size: False
    offer = _offer("capacity-wait:0")
    offer.update(state=HostStageState.OFFERED.value, created_at=1.0)

    assert manager._admit_one({"capacity-wait:0": offer}) is False
    assert claimed == []


def test_p_rejects_expired_capacity_blocked_offer_for_d_fail_open():
    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.arena_numa_node = -1
    manager.arena_domain = -1
    manager.page_size = 64
    manager.owner = "p0"
    manager._capacity_wait_timeout_seconds = 2.0
    transitions = []
    offer = _offer("capacity-expired:0")
    offer.update(state=HostStageState.OFFERED.value, created_at=1.0)
    manager.ledger = types.SimpleNamespace(
        claim=lambda snapshot_id, owner: dict(offer, owner=owner),
        transition=lambda snapshot_id, state, **kwargs: transitions.append(
            (snapshot_id, state, kwargs)
        ),
    )
    manager._can_admit = lambda byte_size: False

    assert manager._admit_one({"capacity-expired:0": offer}) is True
    assert transitions == [
        (
            "capacity-expired:0",
            HostStageState.REJECTED,
            {"owner": "p0", "reason": "p_host_capacity_wait_timeout"},
        )
    ]


def test_ledger_snapshot_reads_multiple_entries_once():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer("req-a:0"))
        ledger.offer(_offer("req-b:1"))
        entries = ledger.snapshot_entries()
        assert set(entries) == {"req-a:0", "req-b:1"}
        assert entries["req-a:0"]["state"] == HostStageState.OFFERED.value
        # The returned snapshot is detached from the ledger document.
        entries["req-a:0"]["state"] = "local-only"
        assert ledger.get("req-a:0")["state"] == HostStageState.OFFERED.value
    finally:
        os.unlink(path)


def test_consumed_entries_prune_earlier_than_failures():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer("consumed:0"))
        ledger.offer(_offer("failed:0"))

        def make_terminal(entries):
            now = __import__("time").time() - 10
            entries["consumed:0"].update(
                state=HostStageState.CONSUMED.value, updated_at=now
            )
            entries["failed:0"].update(
                state=HostStageState.FAILED.value, updated_at=now
            )
            return None, True

        ledger._mutate(make_terminal)
        ledger.prune(older_than_seconds=600, consumed_older_than_seconds=5)
        assert ledger.get("consumed:0") is None
        assert ledger.get("failed:0") is not None
    finally:
        os.unlink(path)


def _relay_ready_offer(snapshot_id, byte_size=1024**3, source_numa=1):
    value = _offer(snapshot_id)
    value.update(
        d_pid=os.getpid(),
        byte_size=byte_size,
        source_numa_node=source_numa,
        arena_numa_node=0,
        source_bootstrap_addr="127.0.0.1:10001",
        source_room=123,
    )
    return value


def _publish_extent(ledger, snapshot_id):
    ledger.claim(snapshot_id, "p0")
    ledger.publish_grants(
        snapshot_id,
        "p0",
        [
            {
                "kind": "shared_host_extent",
                "arena_path": "/dev/shm/fake.kv",
                "byte_size": 1024**3,
                "token_count": 192,
            }
        ],
    )


def test_relay_selection_balances_queued_bytes_and_falls_back_to_direct():
    ledger, path = _ledger()
    try:
        for relay_id in ("r0", "r1"):
            ledger.register_relay(
                relay_id=relay_id,
                pid=100 + int(relay_id[-1]),
                numa_node=0,
                slot_token_count=64,
                slot_count=2,
                d2h_gib_per_second=21.0,
            )
        modes = []
        relays = []
        # Once each relay has two GiB queued, a new 1-GiB job predicts
        # 3/21 s, slower than a direct cross-NUMA 1/7.45 s write.
        for index in range(5):
            snapshot_id = f"balance:{index}"
            ledger.offer(_relay_ready_offer(snapshot_id))
            _publish_extent(ledger, snapshot_id)
            selected = ledger.assign_transfer_path(
                snapshot_id,
                source_pid=os.getpid(),
                source_numa_node=1,
                arena_numa_node=0,
                direct_cross_numa_gib_per_second=7.45,
                nvlink_gib_per_second=220.0,
                relay_stale_seconds=5.0,
            )
            modes.append(selected["write_mode"])
            relays.append(selected.get("relay_id"))
        assert modes[:4] == ["relay"] * 4
        assert relays[:4] == ["r0", "r1", "r0", "r1"]
        assert modes[4] == "direct_cross_numa"
    finally:
        os.unlink(path)


def test_local_source_never_takes_an_extra_relay_hop():
    ledger, path = _ledger()
    try:
        ledger.register_relay(
            relay_id="r0",
            pid=123,
            numa_node=0,
            slot_token_count=64,
            slot_count=2,
            d2h_gib_per_second=21.0,
        )
        ledger.offer(_relay_ready_offer("local:0", source_numa=0))
        _publish_extent(ledger, "local:0")
        selected = ledger.assign_transfer_path(
            "local:0",
            source_pid=os.getpid(),
            source_numa_node=0,
            arena_numa_node=0,
            direct_cross_numa_gib_per_second=7.45,
            nvlink_gib_per_second=220.0,
            relay_stale_seconds=5.0,
        )
        assert selected["write_mode"] == "direct_local"
        assert selected.get("relay_id") is None
    finally:
        os.unlink(path)


def test_invalid_or_duplicate_transitions_fail_closed():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer())
        assert not ledger.transition("req:0", HostStageState.CONSUMED)
        assert ledger.claim("req:0", "p0") is not None
        assert not ledger.transition(
            "req:0", HostStageState.HOST_READY, owner="p0"
        )
        assert ledger.transition("req:0", HostStageState.REJECTED, owner="p0")
        # A stale D cannot resurrect a terminal request-generation.
        assert ledger.offer(_offer())["state"] == HostStageState.REJECTED.value
    finally:
        os.unlink(path)


class _FakeStagingClient:
    def __init__(self, outcome):
        self.outcome = outcome

    def progress(self, candidate, source_pages):
        return self.outcome


def _decode_manager_for_staging(outcome):
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    request = RequestGeneration("req", 0)
    manifest = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=64,
        byte_size=0,
        state=SnapshotState.SLOW_FALLBACK,
    )
    req = types.SimpleNamespace(req_pool_idx=1)
    metadata = AgenticRequestMetadata("req", 0)
    manager.agentic_fast_threshold = 0.2
    manager.agentic_host_staging_client = _FakeStagingClient(outcome)
    manager.agentic_direct_candidates = {
        request.snapshot_id: {
            "staging": True,
            "req": req,
            "metadata": metadata,
            "manifest": manifest,
            "source_token_indices": [1],
            "tokens": [1] * 64,
        }
    }
    releases = []
    manager._release_finished_req = lambda released_req, offset: releases.append(
        (released_req, offset)
    )
    return manager, req, releases


def test_d_hbm_is_not_released_while_p_host_is_partial():
    manager, _, releases = _decode_manager_for_staging("waiting")
    manager._check_agentic_direct_progress()
    assert releases == []
    assert manager.agentic_direct_candidates


def test_d_hbm_release_occurs_exactly_after_host_ready_ack():
    manager, req, releases = _decode_manager_for_staging("host_ready")
    manager._check_agentic_direct_progress()
    assert releases == [(req, 0)]
    assert manager.agentic_direct_candidates == {}
    manager._check_agentic_direct_progress()
    assert releases == [(req, 0)]


def test_async_d_hbm_release_is_committed_only_by_scheduler():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._decode_io_async_enabled = True
    manager._decode_io_events = queue.SimpleQueue()
    manager._decode_scheduler_commit_events = 0
    manager._decode_scheduler_commit_seconds = 0.0
    req = types.SimpleNamespace(req_pool_idx=7)
    releases = []
    manager._release_finished_req = lambda released_req, offset: releases.append(
        (released_req, offset)
    )

    # This call models the background transport thread.  It must never mutate
    # allocator/request-pool state directly.
    manager._enqueue_agentic_release(req, 0)
    assert releases == []

    # Only the Decode scheduler's bounded commit drain may release the pages.
    manager._drain_decode_io_events()
    assert releases == [(req, 0)]
    assert manager._decode_scheduler_commit_events == 1


def test_async_d_hbm_release_groups_allocator_frees():
    class _Allocator:
        def __init__(self):
            self.active = False
            self.begin_count = 0
            self.end_count = 0

        def free_group_begin(self):
            assert not self.active
            self.active = True
            self.begin_count += 1

        def free_group_end(self):
            assert self.active
            self.active = False
            self.end_count += 1

    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._decode_io_async_enabled = True
    manager._decode_io_events = queue.SimpleQueue()
    manager._decode_scheduler_commit_events = 0
    manager._decode_scheduler_commit_seconds = 0.0
    manager.token_to_kv_pool_allocator = _Allocator()
    released = []

    def release(req, offset):
        assert manager.token_to_kv_pool_allocator.active
        released.append((req, offset))

    manager._release_finished_req = release
    reqs = [types.SimpleNamespace(req_pool_idx=i) for i in (7, 8, 9)]
    for req in reqs:
        manager._enqueue_agentic_release(req, 0)

    manager._drain_decode_io_events()

    assert [req for req, _ in released] == reqs
    assert manager.token_to_kv_pool_allocator.begin_count == 1
    assert manager.token_to_kv_pool_allocator.end_count == 1
    assert manager._decode_scheduler_commit_events == 3


def test_async_d_hbm_release_waits_for_short_coalesce_window():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._decode_io_async_enabled = True
    manager._decode_io_events = queue.SimpleQueue()
    manager._decode_scheduler_commit_events = 0
    manager._decode_scheduler_commit_seconds = 0.0
    manager._decode_commit_interval = 0.02
    req = types.SimpleNamespace(req_pool_idx=7)
    releases = []
    manager._release_finished_req = lambda released_req, offset: releases.append(
        (released_req, offset)
    )

    manager._enqueue_agentic_release(req, 0)
    manager._drain_decode_io_events()
    assert releases == []

    manager._decode_commit_ready_at = 0.0
    manager._drain_decode_io_events()
    assert releases == [(req, 0)]


def test_pending_release_pages_remain_accounted_before_scheduler_commit():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager._decode_io_async_enabled = True
    manager._decode_io_events = queue.SimpleQueue()
    manager._decode_pending_release_tokens = 0
    manager._decode_scheduler_commit_events = 0
    manager._decode_scheduler_commit_seconds = 0.0
    manager._decode_commit_interval = 0.02
    manager.page_size = 64
    req = types.SimpleNamespace(
        req_pool_idx=7,
        kv_committed_len=742,
        kv_allocated_len=742,
    )
    manager._release_finished_req = lambda *_args: None

    manager._enqueue_agentic_release(req, 0)

    # The physical allocation occupies 12 pages until the scheduler applies
    # the queued free.  Idle memory checking must not report these as leaked.
    assert manager.agentic_pending_release_token_count == 768
    manager._decode_commit_ready_at = 0.0
    manager._drain_decode_io_events()
    assert manager.agentic_pending_release_token_count == 0


def test_d_accepts_every_state_after_complete_host_copy_as_release_ack():
    """P may advance past the short-lived HOST_READY state before D polls."""
    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    candidate = {"manifest": types.SimpleNamespace(snapshot_id="req:0")}
    for state in (
        HostStageState.HOST_READY,
        HostStageState.H2D_LOADING,
        HostStageState.SPILLING,
        HostStageState.MOONCAKE_READY,
        HostStageState.CONSUMED,
    ):
        client.ledger = types.SimpleNamespace(
            get=lambda _snapshot_id, value=state.value: {"state": value}
        )
        assert client.progress(candidate, []) == "host_ready"


def test_direct_ready_wait_is_extended_only_with_d_kv_headroom(monkeypatch):
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.agentic_fast_threshold = 2.0
    allocator = types.SimpleNamespace(
        size=100,
        available_size=lambda: 80,
    )
    manager.token_to_kv_pool_allocator = allocator
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_D_HBM_HIGH_WATERMARK", "0.70")

    timeout, usage = manager._agentic_direct_ready_timeout(10.0)
    assert timeout == 10.0
    assert usage == 0.20

    allocator.available_size = lambda: 20
    timeout, usage = manager._agentic_direct_ready_timeout(10.0)
    assert timeout == 2.0
    assert usage == 0.80


def test_direct_ready_watermark_is_validated(monkeypatch):
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.agentic_fast_threshold = 2.0
    manager.token_to_kv_pool_allocator = types.SimpleNamespace(
        size=100,
        available_size=lambda: 80,
    )
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_D_HBM_HIGH_WATERMARK", "1.0")

    try:
        manager._agentic_direct_ready_timeout(10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid D-HBM watermark must fail closed")


def test_direct_manifest_lookup_is_rate_limited_and_force_refreshes(monkeypatch):
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    initial = types.SimpleNamespace(state="ready")
    claimed = types.SimpleNamespace(state="loading")
    loads = []
    manager.agentic_snapshot_store = types.SimpleNamespace(
        load=lambda request, require_ready: loads.append(request) or claimed
    )
    metadata = types.SimpleNamespace(current="req:0")
    candidate = {
        "manifest": initial,
        "manifest_next_poll_at": 10.0,
    }
    monkeypatch.setenv("SGLANG_AGENTIC_KV_DIRECT_MANIFEST_POLL_INTERVAL", "0.1")

    assert manager._agentic_direct_manifest(candidate, metadata, 9.0) is initial
    assert loads == []
    assert manager._agentic_direct_manifest(candidate, metadata, 10.0) is claimed
    assert loads == ["req:0"]
    assert manager._agentic_direct_manifest(candidate, metadata, 10.05) is claimed
    assert loads == ["req:0"]
    assert (
        manager._agentic_direct_manifest(
            candidate, metadata, 10.05, force=True
        )
        is claimed
    )
    assert loads == ["req:0", "req:0"]


def test_d_waits_for_local_dma_before_acknowledging_abort():
    trace = []

    class Event:
        def __init__(self):
            self.ready = False

        def query(self):
            trace.append("event_query")
            return self.ready

        def synchronize(self):
            trace.append("event_sync")

    class Snapshot:
        def close(self, *, unlink):
            assert not unlink
            trace.append("mapping_close")

    client = AgenticDHostStagingClient.__new__(AgenticDHostStagingClient)
    client.ledger = types.SimpleNamespace(
        get=lambda _: {"state": HostStageState.ABORTING.value, "grants": []},
        mark_writer_drained=lambda snapshot_id, d_pid: trace.append("writer_drained"),
    )
    event = Event()
    candidate = {
        "manifest": types.SimpleNamespace(snapshot_id="req:0"),
        "arena_write": {
            "event": event,
            "snapshot": Snapshot(),
            "copy_refs": (),
        },
    }
    assert client.progress(candidate, []) == "waiting"
    assert trace == ["event_query"]
    assert "arena_write" in candidate
    event.ready = True
    assert client.progress(candidate, []) == "waiting"
    assert trace == [
        "event_query",
        "event_query",
        "mapping_close",
        "writer_drained",
    ]
    assert "arena_write" not in candidate


def test_d_hostless_hashing_does_not_require_hicache_controller():
    manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    manager.page_size = 4
    manager.cache_controller = None
    hashes = manager._compute_prefix_hash(list(range(8)))
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


class _TraceEvent:
    def __init__(self, trace):
        self.trace = trace

    def query(self):
        return True

    def synchronize(self):
        self.trace.append("h2d_complete")



class _TraceHostPool:
    def __init__(self, trace):
        self.trace = trace

    def free(self, indices):
        self.trace.append("host_free")


class _TraceArena:
    def __init__(self, trace):
        self.trace = trace

    def release(self, snapshot):
        self.trace.append("arena_free")


class _TraceTree:
    def __init__(self, trace):
        self.trace = trace

    def insert(self, params):
        self.trace.append("gpu_insert")
        return types.SimpleNamespace(prefix_len=0)

    def match_prefix(self, params):
        self.trace.append("gpu_match")
        return types.SimpleNamespace(
            device_indices=[0] * 192, last_device_node="node"
        )

    def inc_lock_ref(self, node):
        self.trace.append("gpu_pin")

    def dec_lock_ref(self, node):
        self.trace.append("gpu_unpin")


def test_p_host_release_is_after_h2d_completion_and_gpu_pin():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer())
        ledger.claim("req:0", "p0")
        ledger.publish_grants(
            "req:0",
            "p0",
            [{"seq": 0, "room": 1, "slot": 0, "start_page": 0, "num_pages": 3}],
        )
        ledger.ack_chunk("req:0", "p0", 0)
        ledger.mark_host_ready("req:0", "p0", 1)
        ledger.transition("req:0", HostStageState.H2D_LOADING, owner="p0")
        trace = []
        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager.owner = "p0"
        manager.ledger = ledger
        manager.host_pool = _TraceHostPool(trace)
        manager.tree_cache = _TraceTree(trace)
        manager.token_allocator = types.SimpleNamespace(free=lambda _: trace.append("gpu_free"))
        offer = _offer()
        manager.arena = _TraceArena(trace)
        record = {"offer": offer, "snapshot": object(), "loading": True}
        manager.host_ready = {"req:0": record}
        manager.loads = {
            "next": {
                "record": record,
                "device_indices": [7, 8, 9],
                "event": _TraceEvent(trace),
                "copy_refs": (),
            }
        }
        req = types.SimpleNamespace(
            rid="next",
            origin_input_ids=[0] * 192,
            extra_key="agentic-v1:req:g1",
            priority=0,
        )
        assert manager.gate_request(req, RequestGeneration("req", 0)) is False
        assert trace == [
            "h2d_complete",
            "gpu_insert",
            "gpu_match",
            "gpu_pin",
            "arena_free",
        ]
        assert ledger.get("req:0")["state"] == HostStageState.CONSUMED.value
        manager.release_request_pin(req)
        assert trace[-1] == "gpu_unpin"
    finally:
        os.unlink(path)


def test_p_host_h2d_admission_is_serialized():
    """A second demand restore waits without allocating while H2D is occupied."""

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.loads = {"busy": {"event": object()}}
    manager.max_h2d_inflight = 1
    manager.host_ready = {
        "next:0": {"offer": _offer("next:0"), "loading": False}
    }
    manager.ledger = types.SimpleNamespace(get=lambda _: None)
    req = types.SimpleNamespace(rid="next")

    assert manager.gate_request(req, RequestGeneration("next", 0)) is True


def test_host_ready_defers_cuda_materialization_until_request_selection():
    class LazySnapshot:
        def materialize(self):
            raise AssertionError("HOST_READY must not eagerly register P Host memory")

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.host_ready = {}
    manager.aborting = {}
    manager.active = {
        "req:0": {
            "offer": _offer(),
            "snapshot": LazySnapshot(),
        }
    }
    manager.ledger = types.SimpleNamespace()

    manager._poll_active(
        {"req:0": {"state": HostStageState.HOST_READY.value}}
    )

    assert manager.active == {}
    assert manager.host_ready["req:0"]["snapshot"].__class__ is LazySnapshot
    assert manager.host_ready["req:0"]["ready_at"] > 0


def test_selected_slow_recovery_maps_pageable_extent_once_before_h2d_admission():
    trace = []

    class LazySnapshot:
        _materialized = None

        def materialize(self):
            trace.append("materialize")
            self._materialized = object()
            return self

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager._state_lock = threading.RLock()
    manager.host_ready = {
        "next:0": {
            "offer": _offer("next:0"),
            "snapshot": LazySnapshot(),
            "loading": False,
        }
    }
    manager.loads = {}
    manager.max_h2d_inflight = 1
    manager._ledger_entries_cache = {
        "next:0": {"state": HostStageState.HOST_READY.value}
    }
    req = types.SimpleNamespace(rid="next")

    assert manager.gate_request(req, RequestGeneration("next", 0)) is True
    assert trace == ["materialize"]
    assert manager.host_ready["next:0"]["loading"] is False

    # An occupied H2D slot leaves the request queued without mapping again.
    manager.loads = {"busy": {"event": object()}}
    assert manager.gate_request(req, RequestGeneration("next", 0)) is True
    assert trace == ["materialize"]
    assert manager.host_ready["next:0"]["loading"] is False


def test_spill_does_not_free_host_before_mooncake_commit_result():
    ledger, path = _ledger()
    try:
        ledger.offer(_offer())
        ledger.claim("req:0", "p0")
        ledger.publish_grants(
            "req:0",
            "p0",
            [{"seq": 0, "room": 1, "slot": 0, "start_page": 0, "num_pages": 3}],
        )
        ledger.ack_chunk("req:0", "p0", 0)
        ledger.mark_host_ready("req:0", "p0", 1)
        ledger.transition("req:0", HostStageState.SPILLING, owner="p0")
        trace = []
        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager.owner = "p0"
        manager.ledger = ledger
        manager.host_pool = _TraceHostPool(trace)
        manager.arena = _TraceArena(trace)
        record = {"offer": _offer(), "snapshot": object(), "loading": False}
        manager.host_ready = {}
        manager.spills = {"req:0": record}
        manager._spill_threads = {}
        manager._progress_spills()
        assert trace == []
        assert ledger.get("req:0")["state"] == HostStageState.SPILLING.value
        record["spill_result"] = (True, object(), None)
        manager._progress_spills()
        assert trace == ["arena_free"]
        assert ledger.get("req:0")["state"] == HostStageState.MOONCAKE_READY.value
    finally:
        os.unlink(path)


def _spill_manager(trace, *, put_succeeds=True):
    request = RequestGeneration("req", 0)
    fallback = SnapshotManifest(
        request=request,
        page_keys=(),
        token_count=192,
        byte_size=300,
        state=SnapshotState.SLOW_FALLBACK,
    )

    class Store:
        def __init__(self):
            self.current = fallback
            self.store = types.SimpleNamespace(
                batch_remove=lambda keys, force=False: trace.append("remove_pages") or [0]
            )

        def load_request_generation(self, request_id, generation, require_ready=False):
            return self.current

        def load(self, request, require_ready=False):
            return self.current

        def update(self, manifest):
            trace.append(f"manifest:{manifest.state.value}")
            self.current = manifest

        def continue_slow_publish(self, manifest):
            self.update(manifest)

        def rollback_slow_publish(self, offloading, fallback_manifest):
            assert self.current is offloading
            self.update(fallback_manifest)

        def commit_publish(self, request):
            trace.append("publish")
            self.current = self.current.transition(SnapshotState.MOONCAKE_READY)
            return self.current

    store = Store()

    class Backend:
        def agentic_snapshot_layout(self, logical_hashes, indices, namespace):
            return tuple(f"physical-{index}" for index, _ in enumerate(logical_hashes)), 300

        def agentic_snapshot_store(self):
            return store

        def batch_set_v1(self, hashes, indices, extra):
            trace.append("put")
            return [put_succeeds] * len(hashes)

    class Eviction:
        def reserve(self, manifest):
            trace.append("reserve")
            return True

        def commit(self, manifest):
            trace.append("capacity_commit")

        def cancel(self, manifest):
            trace.append("capacity_cancel")

    manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
    manager.page_size = 64
    manager.host_pool = types.SimpleNamespace(
        alloc=lambda count: list(range(count)),
        free=lambda indices: trace.append("host_free"),
    )
    manager.cache_controller = types.SimpleNamespace(
        storage_backend=Backend(), storage_batch_size=16
    )
    manager.eviction_controller = Eviction()
    record = {
        "offer": _offer(),
        "snapshot": types.SimpleNamespace(
            materialize=lambda: None,
            copy_into_hicache=lambda pool, indices, page_size: trace.append("host_copy")
        ),
    }
    return manager, record


def test_p_host_spill_reserves_request_capacity_before_put_and_commits_after_publish():
    trace = []
    manager, record = _spill_manager(trace)
    manager._spill_worker("req:0", record)
    assert record["spill_result"][0] is True
    assert trace == [
        "host_copy",
        "reserve",
        "manifest:offloading",
        "put",
        "publish",
        "capacity_commit",
        "host_free",
    ]


def test_failed_p_host_spill_cancels_capacity_and_restores_fallback_manifest():
    trace = []
    manager, record = _spill_manager(trace, put_succeeds=False)
    manager._spill_worker("req:0", record)
    assert record["spill_result"][0] is False
    assert trace == [
        "host_copy",
        "reserve",
        "manifest:offloading",
        "put",
        "remove_pages",
        "manifest:slow_fallback",
        "capacity_cancel",
        "host_free",
    ]


def test_failed_snapshot_waits_for_d_writer_drained_before_arena_free():
    trace = []
    ledger, path = _ledger()
    try:
        offer = _offer()
        offer["d_pid"] = os.getpid()
        ledger.offer(offer)
        ledger.claim("req:0", "p0")
        ledger.publish_grants(
            "req:0",
            "p0",
            [{"kind": "shared_host_extent", "seq": 0, "arena_path": "/dev/shm/x"}],
        )
        manager = AgenticPHostStagingManager.__new__(AgenticPHostStagingManager)
        manager.owner = "p0"
        manager.aborting = {}
        manager.active = {
            "req:0": {"snapshot": object(), "offer": offer}
        }
        manager.arena = _TraceArena(trace)
        manager.ledger = ledger
        manager._fail_active("req:0", "injected_failure")
        assert ledger.get("req:0")["state"] == HostStageState.ABORTING.value
        manager._poll_aborting()
        assert trace == []
        assert "req:0" in manager.aborting
        ledger.mark_writer_drained("req:0", os.getpid())
        manager._poll_aborting()
        assert trace == ["arena_free"]
        assert ledger.get("req:0")["state"] == HostStageState.FAILED.value
        assert manager.aborting == {}
    finally:
        os.unlink(path)
