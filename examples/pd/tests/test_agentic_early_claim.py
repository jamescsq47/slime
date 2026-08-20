from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    RequestGeneration,
    SnapshotManifest,
    SnapshotState,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)

from late_binding_router import LateBindingMiniLoadBalancer


def _directory():
    return tempfile.mkdtemp(prefix="sglang-agentic-early-claim-", dir="/dev/shm")


def test_router_publishes_parent_arrival_before_scheduler_dispatch():
    directory = _directory()
    try:
        router = LateBindingMiniLoadBalancer.__new__(LateBindingMiniLoadBalancer)
        router.early_claim_store = AgenticEarlyClaimStore(directory)
        request = {
            "sampling_params": {
                "custom_params": {
                    "agentic_request_id": "trajectory-a",
                    "agentic_generation": 2,
                    "agentic_parent_generation": 1,
                }
            }
        }
        router._publish_parent_arrival(request)
        parent = RequestGeneration("trajectory-a", 1)
        marker = router.early_claim_store.read_arrival(
            parent, not_before=0.0, max_age_seconds=5.0
        )
        assert marker is not None
        assert marker["snapshot_id"] == "trajectory-a:1"
        assert marker["request_id"] == "trajectory-a"
        assert marker["generation"] == 1
        arrivals = router.early_claim_store.iter_arrivals(max_age_seconds=5.0)
        assert [(item.snapshot_id, payload["kind"]) for item, payload in arrivals] == [
            ("trajectory-a:1", "arrival")
        ]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_targeted_arrival_and_route_are_generation_scoped():
    directory = _directory()
    try:
        store = AgenticEarlyClaimStore(directory)
        current = RequestGeneration("trajectory-route", 2)
        previous = RequestGeneration("trajectory-route", 1)

        original_arrival = time.time() - 0.25
        arrival = store.publish_arrival(
            current,
            target_prefill_domain=1,
            arrived_at=original_arrival,
        )
        assert arrival["target_prefill_domain"] == 1
        assert arrival["arrived_at"] == original_arrival
        assert store.read_arrival(
            current, not_before=0.0, max_age_seconds=5.0
        )["target_prefill_domain"] == 1

        published = store.publish_route(
            current,
            route="direct_ready",
            prefill_domain=1,
            snapshot_tokens=8192,
        )
        assert published["prefill_domain"] == 1
        assert published["route"] == "direct_ready"
        assert published["snapshot_tokens"] == 8192
        assert store.read_route(current, max_age_seconds=5.0) == published
        assert store.read_route(previous, max_age_seconds=5.0) is None

        host_writing = store.publish_route(
            current,
            route="host_writing",
            prefill_domain=0,
            arena_numa_node=0,
            snapshot_tokens=8192,
        )
        assert host_writing["route"] == "host_writing"
        assert host_writing["arena_numa_node"] == 0
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_decode_observes_arrival_then_removes_marker_without_credit_ledger():
    directory = _directory()
    try:
        store = AgenticEarlyClaimStore(directory)
        request = RequestGeneration("trajectory-b", 0)
        manifest = SnapshotManifest(
            request=request,
            page_keys=(),
            token_count=8192,
            byte_size=0,
            state=SnapshotState.DIRECT_READY,
            token_digest="digest",
            direct_bootstrap_addr="127.0.0.1:1",
            direct_room=7,
        )
        store.publish_arrival(request)
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        manager.agentic_early_claim_store = store
        manager.agentic_early_claim_post_timeout = 1.0
        manager.agentic_early_claim_poll_interval = 0.01
        manager.agentic_fast_threshold = 1.0
        candidate = {
            "manifest": manifest,
            "created_at": time.monotonic(),
            "early_claim_next_poll_at": 0.0,
            "fast_arrival_seen": False,
            "fast_arrival_seen_at": None,
        }
        assert manager._agentic_try_early_claim(candidate, time.monotonic()) == "arrived"
        assert candidate["fast_arrival_seen"]
        assert not (store.directory / "credits.json").exists()
        manager._agentic_release_early_claim(candidate, "test")
        assert not store.marker_path(request).exists()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_decode_rejects_arrival_outside_fast_tool_window():
    directory = _directory()
    try:
        store = AgenticEarlyClaimStore(directory)
        request = RequestGeneration("trajectory-late", 0)
        manifest = SnapshotManifest(
            request=request,
            page_keys=(),
            token_count=4096,
            byte_size=0,
            state=SnapshotState.DIRECT_READY,
            token_digest="digest",
            direct_bootstrap_addr="127.0.0.1:1",
            direct_room=8,
            created_at=time.time() - 3.0,
        )
        store.publish_arrival(request)
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        manager.agentic_early_claim_store = store
        manager.agentic_early_claim_post_timeout = 2.0
        manager.agentic_early_claim_poll_interval = 0.01
        manager.agentic_fast_threshold = 2.0
        candidate = {
            "manifest": manifest,
            "created_at": time.monotonic() - 3.0,
            "early_claim_next_poll_at": 0.0,
            "fast_arrival_seen": False,
            "fast_arrival_seen_at": None,
        }
        assert manager._agentic_try_early_claim(candidate, time.monotonic()) == "absent"
        assert not candidate["fast_arrival_seen"]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_application_final_confirmation_is_generation_scoped():
    directory = _directory()
    try:
        store = AgenticEarlyClaimStore(directory)
        current = RequestGeneration("trajectory-final", 2)
        previous = RequestGeneration("trajectory-final", 1)
        published = store.publish_final(current)
        assert published["kind"] == "final"
        assert store.read_final(
            current, not_before=0.0, max_age_seconds=5.0
        ) is not None
        assert store.read_final(
            previous, not_before=0.0, max_age_seconds=5.0
        ) is None
        store.remove_final(current)
        assert not store.final_path(current).exists()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_valid_tool_confirmation_is_generation_scoped():
    directory = _directory()
    try:
        store = AgenticEarlyClaimStore(directory)
        current = RequestGeneration("trajectory-tool", 2)
        previous = RequestGeneration("trajectory-tool", 1)
        published = store.publish_tool(current)
        assert published["kind"] == "tool"
        assert store.read_tool(
            current, not_before=0.0, max_age_seconds=5.0
        ) is not None
        assert store.read_tool(
            previous, not_before=0.0, max_age_seconds=5.0
        ) is None
        store.remove_tool(current)
        assert not store.tool_path(current).exists()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_generation_producer_is_single_winner_and_generation_scoped():
    directory = _directory()
    try:
        first_process = AgenticEarlyClaimStore(directory)
        second_process = AgenticEarlyClaimStore(directory)
        generation_zero = RequestGeneration("trajectory-producer", 0)
        generation_one = RequestGeneration("trajectory-producer", 1)

        assert first_process.claim_generation_producer(generation_zero)
        assert not second_process.claim_generation_producer(generation_zero)
        assert second_process.claim_generation_producer(generation_one)
        assert first_process.producer_path(generation_zero).exists()
        assert second_process.producer_path(generation_one).exists()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_tp_generation_producer_publish_is_atomic_and_rank0_owned():
    directory = _directory()
    try:
        rank0 = AgenticEarlyClaimStore(directory)
        follower = AgenticEarlyClaimStore(directory)
        generation = RequestGeneration("trajectory-tp-producer", 3)
        owner = "decode-0:model-request"
        results = {}

        # Let the follower arrive first.  It must wait for rank 0 instead of
        # creating or independently electing a producer tombstone.
        thread = threading.Thread(
            target=lambda: results.setdefault(
                "follower",
                follower.wait_generation_producer(
                    generation, owner, timeout_seconds=1.0
                ),
            )
        )
        thread.start()
        time.sleep(0.01)
        assert rank0.claim_generation_producer(generation, producer_id=owner)
        thread.join(timeout=1.0)

        assert results == {"follower": True}
        assert rank0.producer_path(generation).read_text().strip() == owner
        assert not list(rank0.directory.glob(".producer-*.tmp"))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_arrival_watcher_observes_atomic_publish_without_directory_rescan():
    directory = _directory()
    try:
        store = AgenticEarlyClaimStore(directory)
        watcher = store.watch_arrivals(max_age_seconds=5.0)
        # Consume the one-time startup snapshot before publishing the marker.
        assert watcher.poll() == []
        store.iter_arrivals = lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("normal inotify delivery must not rescan the directory")
        )
        request = RequestGeneration("trajectory-inotify", 3)
        store.publish_arrival(request, target_prefill_domain=1)

        deadline = time.monotonic() + 1.0
        arrivals = []
        while not arrivals and time.monotonic() < deadline:
            arrivals = watcher.poll(0.05)
        watcher.close()

        assert len(arrivals) == 1
        observed, payload = arrivals[0]
        assert observed == request
        assert payload["target_prefill_domain"] == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)
