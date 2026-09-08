import inspect
import os
import queue
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticDHostStagingClient,
    HostStageState,
    SharedHostSnapshotArena,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostStagingManager,
    AgenticPToDHostLoadManager,
    AgenticPToDHostReceiver,
    P2D_CUSTOM_SNAPSHOT_ID,
    P2D_HOST_CAPACITY_LIMIT_DEFAULT,
    p2d_snapshot_from_req,
    p2d_snapshot_id,
)


def test_p2d_host_default_uses_full_arena_capacity():
    assert P2D_HOST_CAPACITY_LIMIT_DEFAULT == 1.0
    default = inspect.signature(AgenticPToDHostStagingManager).parameters[
        "hard_watermark"
    ].default
    assert default == P2D_HOST_CAPACITY_LIMIT_DEFAULT


class _TinyMHAPool:
    """Minimal real-CUDA pool used to verify the byte-moving data path."""

    def __init__(self, capacity=512):
        self.layer_num = 2
        self.head_num = 2
        self.head_dim = 8
        self.v_head_dim = 8
        self.store_dtype = torch.bfloat16
        self.device = torch.device("cuda:0")
        self.k_buffer = [
            torch.zeros(
                (capacity, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [torch.zeros_like(value) for value in self.k_buffer]
        self.k_data_ptrs = torch.tensor(
            [value.data_ptr() for value in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [value.data_ptr() for value in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )


def test_direction_specific_snapshot_id_and_req_metadata():
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            custom_params={P2D_CUSTOM_SNAPSHOT_ID: p2d_snapshot_id(17)}
        )
    )
    assert p2d_snapshot_from_req(req) == "p2d:17"


def test_receiver_commits_prefill_result_without_nixl_metadata():
    receiver = AgenticPToDHostReceiver(SimpleNamespace(), "p2d:19")
    receiver._grant = {
        "prefill_metadata": {
            "output_id": 42,
            "cached_tokens": 10,
            "cached_tokens_device": 8,
            "cached_tokens_host": 2,
            "cached_tokens_storage": 0,
        }
    }
    req = SimpleNamespace(output_ids=[])
    receiver.commit_req(req)
    assert req.output_ids == [42]
    assert req.cached_tokens == 10
    assert req.cached_tokens_device == 8
    assert req.cached_tokens_host == 2
    assert req.cached_tokens_storage == 0


def test_p2d_ledger_has_atomic_complete_snapshot_lifecycle(tmp_path):
    # tmp_path may live outside tmpfs; the production ledger deliberately
    # rejects that.  Keep this control-plane test node-local as well.
    ledger_path = "/dev/shm/sglang-p2d-test-lifecycle.json"
    try:
        ledger = SharedHostStagingLedger(ledger_path)
        snapshot_id = p2d_snapshot_id(23)
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 23,
                "prefill_domain": 0,
            }
        )
        assert ledger.claim(snapshot_id, "p2d-p:test") is not None
        assert ledger.publish_grants(
            snapshot_id,
            "p2d-p:test",
            [
                {
                    "kind": "shared_host_extent",
                    "arena_path": "/dev/shm/fake.kv",
                    "token_count": 8,
                    "byte_size": 1024,
                    "arena_numa_node": 0,
                }
            ],
        )
        assert ledger.ack_chunk(snapshot_id, "p2d-p:test", 0)
        assert ledger.mark_host_ready(snapshot_id, "p2d-p:test", 1)
        assert ledger.transition(
            snapshot_id, HostStageState.H2D_LOADING, owner="p2d-p:test"
        )
        assert ledger.transition(
            snapshot_id, HostStageState.CONSUMED, owner="p2d-p:test"
        )
    finally:
        import os

        try:
            os.unlink(ledger_path)
        except FileNotFoundError:
            pass


def test_d_loader_accepts_cross_numa_snapshot(tmp_path):
    ledger_path = "/dev/shm/sglang-p2d-test-numa.json"
    try:
        ledger = SharedHostStagingLedger(ledger_path)
        snapshot_id = p2d_snapshot_id(29)
        ledger.offer({"snapshot_id": snapshot_id})
        ledger.claim(snapshot_id, "p2d-p:test")
        ledger.publish_grants(
            snapshot_id,
            "p2d-p:test",
            [
                {
                    "kind": "shared_host_extent",
                    "arena_path": "/dev/shm/fake.kv",
                    "token_count": 8,
                    "byte_size": 1024,
                    "arena_numa_node": 1,
                }
            ],
        )
        ledger.ack_chunk(snapshot_id, "p2d-p:test", 0)
        ledger.mark_host_ready(snapshot_id, "p2d-p:test", 1)

        manager = AgenticPToDHostLoadManager.__new__(
            AgenticPToDHostLoadManager
        )
        manager.ledger = ledger
        manager.numa_node = 0
        manager.tp_rank = 0
        manager.tp_size = 1
        manager._work = queue.SimpleQueue()
        receiver = AgenticPToDHostReceiver(manager, snapshot_id)
        manager.submit(receiver, list(range(8)))
        assert receiver._submitted
        assert receiver._cross_numa
        queued_receiver, queued_indices = manager._work.get_nowait()
        assert queued_receiver is receiver
        assert queued_indices == list(range(8))
        assert ledger.get(snapshot_id)["state"] == HostStageState.H2D_LOADING.value
    finally:
        import os

        try:
            os.unlink(ledger_path)
        except FileNotFoundError:
            pass


def test_tp_host_load_failure_releases_source_only_after_all_ranks_drain():
    ledger_path = f"/dev/shm/sglang-p2d-load-fence-{time.time_ns()}.json"
    snapshot_id = p2d_snapshot_id(31)
    owner = "p2d-p:tp-group"
    released = []
    try:
        ledger = SharedHostStagingLedger(ledger_path)
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "tp_size": 2,
                "tp_rank": 0,
                "control_offer": True,
            }
        )
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=0, tp_size=2)
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=1, tp_size=2)
        for rank in range(2):
            assert ledger.publish_rank_grant(
                snapshot_id,
                owner,
                {
                    "kind": "shared_host_extent",
                    "arena_path": f"/dev/shm/fake-rank-{rank}.kv",
                    "token_count": 8,
                    "byte_size": 1024,
                    "arena_numa_node": rank,
                },
                tp_rank=rank,
                tp_size=2,
            )
            assert ledger.complete_p2d_host_write_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=2
            )
        assert ledger.begin_host_load_rank(
            snapshot_id, owner, tp_rank=0, tp_size=2
        )
        assert ledger.begin_host_load_rank(
            snapshot_id, owner, tp_rank=1, tp_size=2
        )

        producer = AgenticPToDHostStagingManager.__new__(
            AgenticPToDHostStagingManager
        )
        producer.ledger = ledger
        producer._lock = threading.RLock()
        producer._active = {}
        producer._records = {snapshot_id: {"snapshot": "rank-extent"}}
        producer.arena = SimpleNamespace(release=released.append)

        assert ledger.request_host_load_failure(
            snapshot_id, owner, reason="injected_rank0_failure"
        )
        assert ledger.mark_host_load_rank_drained(
            snapshot_id, owner, tp_rank=0, tp_size=2
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.ABORTING.value
        producer._cleanup_consumed()
        assert released == []

        assert ledger.mark_host_load_rank_drained(
            snapshot_id, owner, tp_rank=1, tp_size=2
        )
        assert ledger.get(snapshot_id)["state"] == HostStageState.FAILED.value
        producer._cleanup_consumed()
        assert released == ["rank-extent"]
    finally:
        import os

        try:
            os.unlink(ledger_path)
        except FileNotFoundError:
            pass


def test_p2d_cleanup_retries_extent_release_before_dropping_record():
    snapshot_id = "p2d:unregister-retry"
    snapshot = object()
    release_results = iter((False, True))
    release_calls = []

    def release(value):
        release_calls.append(value)
        return next(release_results)

    producer = AgenticPToDHostStagingManager.__new__(
        AgenticPToDHostStagingManager
    )
    producer.ledger = SimpleNamespace(
        get=lambda _snapshot_id: {"state": HostStageState.CONSUMED.value}
    )
    producer._lock = threading.RLock()
    producer._active = {}
    producer._records = {snapshot_id: {"snapshot": snapshot}}
    producer.arena = SimpleNamespace(release=release)

    producer._cleanup_consumed()
    assert producer._records[snapshot_id]["snapshot"] is snapshot

    producer._cleanup_consumed()
    assert snapshot_id not in producer._records
    assert release_calls == [snapshot, snapshot]


def test_p2d_h2d_completion_retries_unregister_before_terminal():
    snapshot_id = "p2d:h2d-unregister-retry"
    terminal = []

    class Snapshot:
        def __init__(self):
            self.close_calls = 0

        def close(self, *, unlink):
            assert unlink is False
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected cudaHostUnregister failure")

    receiver = SimpleNamespace(
        snapshot_id=snapshot_id,
        _grant={"arena_numa_node": 0},
        _cross_numa=False,
        mark_terminal=lambda result: terminal.append(result),
    )
    snapshot = Snapshot()
    completion = {
        "receiver": receiver,
        "snapshot": snapshot,
        "started_at": time.monotonic(),
        "token_count": 1,
        "byte_size": 128,
        "worker_id": 0,
        "host_copy_seconds": 0.0,
        "gpu_elapsed_ms": 0.0,
    }
    loader = AgenticPToDHostLoadManager.__new__(AgenticPToDHostLoadManager)
    loader._completion_lock = threading.RLock()
    loader._group_pending = {}
    loader._group_wakeup = threading.Event()
    loader.decode_domain = 0
    loader.numa_node = 0

    assert loader._finish_h2d_success(completion) is False
    assert loader._group_pending[snapshot_id] is completion
    assert completion["snapshot"] is snapshot
    assert terminal == []

    assert loader._finish_h2d_success(completion) is True
    assert snapshot_id not in loader._group_pending
    assert "snapshot" not in completion
    assert terminal == [KVPoll.Success]


def test_tp_host_load_failure_does_not_count_unfenced_peer_as_drained():
    ledger_path = f"/dev/shm/sglang-p2d-unfenced-{time.time_ns()}.json"
    snapshot_id = p2d_snapshot_id(32)
    owner = "p2d-p:tp-group"
    try:
        ledger = SharedHostStagingLedger(ledger_path)
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "tp_size": 2,
                "tp_rank": 0,
                "control_offer": True,
            }
        )
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=0, tp_size=2)
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=1, tp_size=2)
        for rank in range(2):
            assert ledger.publish_rank_grant(
                snapshot_id,
                owner,
                {
                    "kind": "shared_host_extent",
                    "arena_path": f"/dev/shm/fake-rank-{rank}.kv",
                    "token_count": 8,
                    "byte_size": 1024,
                    "arena_numa_node": rank,
                },
                tp_rank=rank,
                tp_size=2,
            )
            assert ledger.complete_p2d_host_write_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=2
            )
        assert ledger.begin_host_load_rank(
            snapshot_id, owner, tp_rank=0, tp_size=2
        )
        assert ledger.begin_host_load_rank(
            snapshot_id, owner, tp_rank=1, tp_size=2
        )
        assert ledger.request_host_load_failure(
            snapshot_id, owner, reason="lost_rank1_fence"
        )
        assert ledger.mark_host_load_rank_drained(
            snapshot_id, owner, tp_rank=0, tp_size=2
        )
        entry = ledger.get(snapshot_id)
        assert entry["state"] == HostStageState.ABORTING.value
        assert entry["loader_drained_ranks"] == [0]
    finally:
        import os

        try:
            os.unlink(ledger_path)
        except FileNotFoundError:
            pass


def test_p2d_source_ready_failure_quarantines_without_publishing_terminal():
    """An unprovable Prefill producer fence must retain P-HBM ownership."""

    snapshot_id = "p2d:source-ready-failure"
    transitions = []

    class Snapshot:
        def materialize(self):
            return None

    class FailedReadyEvent:
        def synchronize(self):
            raise RuntimeError("injected source-ready fence failure")

    source_indices = object()
    ready_event = FailedReadyEvent()
    record = {
        "snapshot": Snapshot(),
        "source_indices": source_indices,
        "source_ready_event": ready_event,
    }
    producer = AgenticPToDHostStagingManager.__new__(
        AgenticPToDHostStagingManager
    )
    producer._stop = threading.Event()
    producer._work = queue.SimpleQueue()
    producer._work.put((snapshot_id, record))
    producer._lock = threading.RLock()
    producer._dma_quarantine = []
    producer._active = {snapshot_id: record}
    producer._results = {}
    producer.ledger = SimpleNamespace(
        transition=lambda *args, **kwargs: transitions.append((args, kwargs))
    )

    producer._worker(0, object(), object(), ())

    assert transitions == []
    assert producer._active[snapshot_id] is record
    assert record["source_indices"] is source_indices
    assert record["source_ready_event"] is ready_event
    assert len(producer._dma_quarantine) == 1


def test_tp_receiver_cancel_before_submit_is_group_fenced():
    ledger_path = f"/dev/shm/sglang-p2d-cancel-fence-{time.time_ns()}.json"
    snapshot_id = p2d_snapshot_id(33)
    owner = "p2d-p:tp-group"
    try:
        ledger = SharedHostStagingLedger(ledger_path)
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "tp_size": 2,
                "tp_rank": 0,
                "control_offer": True,
            }
        )
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=0, tp_size=2)
        assert ledger.claim_rank(snapshot_id, owner, tp_rank=1, tp_size=2)
        for rank in range(2):
            assert ledger.publish_rank_grant(
                snapshot_id,
                owner,
                {
                    "kind": "shared_host_extent",
                    "arena_path": f"/dev/shm/fake-rank-{rank}.kv",
                    "token_count": 8,
                    "byte_size": 1024,
                    "arena_numa_node": rank,
                },
                tp_rank=rank,
                tp_size=2,
            )
            assert ledger.complete_p2d_host_write_rank(
                snapshot_id, owner, tp_rank=rank, tp_size=2
            )

        receivers = []
        for rank in range(2):
            manager = AgenticPToDHostLoadManager.__new__(
                AgenticPToDHostLoadManager
            )
            manager.ledger = ledger
            manager.tp_rank = rank
            manager.tp_size = 2
            manager._completion_lock = threading.RLock()
            manager._group_pending = {}
            manager._group_wakeup = threading.Event()
            receivers.append(AgenticPToDHostReceiver(manager, snapshot_id))

        receivers[0].abort()
        entry = ledger.get(snapshot_id)
        assert entry["state"] == HostStageState.ABORTING.value
        assert entry["loader_drained_ranks"] == [0]

        receivers[1].abort()
        entry = ledger.get(snapshot_id)
        assert entry["state"] == HostStageState.FAILED.value
        assert entry["loader_drained_ranks"] == [0, 1]
    finally:
        import os

        try:
            os.unlink(ledger_path)
        except FileNotFoundError:
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_p2d_host_round_trip():
    """P D2H -> tmpfs -> D H2D preserves every K/V element."""

    root = tempfile.mkdtemp(prefix="sglang-p2d-cuda-", dir="/dev/shm")
    producer = None
    loader = None
    try:
        torch.cuda.set_device(0)
        source = _TinyMHAPool()
        destination = _TinyMHAPool()
        token_count = 128
        source_indices = torch.arange(
            17, 17 + token_count, dtype=torch.int64, device="cuda"
        )
        destination_indices = torch.arange(
            211, 211 + token_count, dtype=torch.int64, device="cuda"
        )
        expected_k = []
        expected_v = []
        for layer_id in range(source.layer_num):
            values = torch.arange(
                token_count * source.head_num * source.head_dim,
                dtype=torch.float32,
                device="cuda",
            ).reshape(token_count, source.head_num, source.head_dim)
            values += layer_id * 5000
            source.k_buffer[layer_id][source_indices] = values.to(torch.bfloat16)
            source.v_buffer[layer_id][source_indices] = (-values - 7).to(
                torch.bfloat16
            )
            expected_k.append(source.k_buffer[layer_id][source_indices].clone())
            expected_v.append(source.v_buffer[layer_id][source_indices].clone())

        ledger = SharedHostStagingLedger(f"{root}/ledger.json")
        snapshot_id = p2d_snapshot_id(77)
        ledger.offer(
            {
                "snapshot_id": snapshot_id,
                "bootstrap_room": 77,
                "prefill_domain": 0,
            }
        )
        producer = AgenticPToDHostStagingManager(
            ledger=ledger,
            device_pool=source,
            page_size=1,
            arena_directory=f"{root}/arena",
            arena_capacity_bytes=64 * 1024 * 1024,
            prefill_domain=0,
            numa_node=0,
        )
        req = SimpleNamespace(
            bootstrap_room=77,
            origin_input_ids=list(range(token_count)),
            output_ids=[123],
            return_logprob=False,
            cached_tokens=9,
            cached_tokens_device=9,
            cached_tokens_host=0,
            cached_tokens_storage=0,
        )
        assert producer.try_submit(req, source_indices.clone())
        deadline = time.monotonic() + 30
        while producer.poll(req) != KVPoll.Success and time.monotonic() < deadline:
            time.sleep(0.01)
        assert producer.poll(req) == KVPoll.Success

        loader = AgenticPToDHostLoadManager(
            ledger=ledger,
            device_pool=destination,
            page_size=1,
            decode_domain=0,
            numa_node=int(os.getenv("SGLANG_TEST_P2D_DECODE_NUMA", "0")),
        )
        receiver = AgenticPToDHostReceiver(loader, snapshot_id)
        receiver.bind(destination_indices.clone())
        deadline = time.monotonic() + 30
        while receiver.poll() != KVPoll.Success and time.monotonic() < deadline:
            time.sleep(0.01)
        receiver.failure_exception()
        assert receiver.poll() == KVPoll.Success
        torch.cuda.synchronize()

        for layer_id in range(destination.layer_num):
            torch.testing.assert_close(
                destination.k_buffer[layer_id][destination_indices],
                expected_k[layer_id],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                destination.v_buffer[layer_id][destination_indices],
                expected_v[layer_id],
                rtol=0,
                atol=0,
            )
        assert ledger.get(snapshot_id)["state"] == HostStageState.CONSUMED.value
    finally:
        if loader is not None:
            loader.close()
        if producer is not None:
            producer.close()
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_d2p_pipelined_d2h_survives_source_release(monkeypatch):
    """Double-bounce D2H publishes only after every CPU commit is durable."""

    root = tempfile.mkdtemp(prefix="sglang-d2p-cuda-", dir="/dev/shm")
    arena = None
    try:
        torch.cuda.set_device(0)
        monkeypatch.setenv("SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS", "16")
        monkeypatch.setenv("SGLANG_AGENTIC_KV_D2H_STAGING_TOKENS", "16")
        monkeypatch.setenv("SGLANG_AGENTIC_KV_D2H_INFLIGHT", "1")
        monkeypatch.setenv("SGLANG_AGENTIC_KV_D2H_BOUNCE_DEPTH", "2")
        monkeypatch.setenv("SGLANG_AGENTIC_KV_D2H_HOST_COPY_WORKERS", "1")
        source = _TinyMHAPool()
        token_count = 64
        source_indices = torch.arange(
            23, 23 + token_count, dtype=torch.int64, device="cuda"
        )
        expected_k = []
        expected_v = []
        for layer_id in range(source.layer_num):
            values = torch.arange(
                token_count * source.head_num * source.head_dim,
                dtype=torch.float32,
                device="cuda",
            ).reshape(token_count, source.head_num, source.head_dim)
            values += layer_id * 7000
            source.k_buffer[layer_id][source_indices] = values.to(torch.bfloat16)
            source.v_buffer[layer_id][source_indices] = (-values - 11).to(
                torch.bfloat16
            )
            expected_k.append(source.k_buffer[layer_id][source_indices].cpu())
            expected_v.append(source.v_buffer[layer_id][source_indices].cpu())

        byte_size = (
            2
            * token_count
            * source.layer_num
            * source.head_num
            * source.head_dim
            * source.store_dtype.itemsize
        )
        arena = SharedHostSnapshotArena(
            f"{root}/arena", 16 * 1024 * 1024, backend="memfd"
        )
        host_snapshot = arena.create(
            "d2p-real:0", token_count, source, byte_size
        )
        ledger = SharedHostStagingLedger(f"{root}/ledger.json")
        ledger.offer(
            {
                "snapshot_id": "d2p-real:0",
                "request_id": "d2p-real",
                "generation": 0,
                "token_count": token_count,
                "token_digest": "digest",
                "byte_size": byte_size,
                "storage_namespace": "d2p-real:0:",
                "d_pid": os.getpid(),
                "source_numa_node": 0,
                "arena_numa_node": 0,
                "arena_domain": 0,
                "tp_rank": 0,
                "tp_size": 1,
            }
        )
        assert ledger.claim("d2p-real:0", "p0") is not None
        assert ledger.publish_grants(
            "d2p-real:0",
            "p0",
            [
                {
                    "kind": "shared_host_extent",
                    "arena_path": host_snapshot.path,
                    "arena_offset": host_snapshot.offset,
                    "byte_size": byte_size,
                    "token_count": token_count,
                    "tp_rank": 0,
                }
            ],
        )
        client = AgenticDHostStagingClient(
            ledger,
            source,
            1,
            source_numa_node=0,
            arena_numa_node=0,
            arena_domain=0,
        )
        candidate = {
            "manifest": SimpleNamespace(snapshot_id="d2p-real:0"),
        }
        deadline = time.monotonic() + 30
        outcome = "waiting"
        while outcome == "waiting" and time.monotonic() < deadline:
            outcome = client.progress(candidate, source_indices)
            time.sleep(0.001)
        assert outcome == "host_ready"
        assert ledger.get("d2p-real:0")["state"] == HostStageState.HOST_READY.value

        # Model the scheduler releasing/reusing D HBM only after HOST_READY.
        for layer_id in range(source.layer_num):
            source.k_buffer[layer_id][source_indices] = 0
            source.v_buffer[layer_id][source_indices] = 0
        torch.cuda.synchronize()

        host_snapshot.materialize()
        for layer_id in range(source.layer_num):
            torch.testing.assert_close(
                host_snapshot.k_buffer[layer_id], expected_k[layer_id], rtol=0, atol=0
            )
            torch.testing.assert_close(
                host_snapshot.v_buffer[layer_id], expected_v[layer_id], rtol=0, atol=0
            )
    finally:
        if arena is not None:
            arena.close()
        shutil.rmtree(root, ignore_errors=True)
