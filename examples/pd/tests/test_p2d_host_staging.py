import shutil
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.agentic_host_staging import (
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.p2d_host_staging import (
    AgenticPToDHostStagingManager,
    AgenticPToDHostLoadManager,
    AgenticPToDHostReceiver,
    P2D_CUSTOM_SNAPSHOT_ID,
    p2d_snapshot_from_req,
    p2d_snapshot_id,
)


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


def test_d_loader_rejects_cross_numa_snapshot(tmp_path):
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
        receiver = AgenticPToDHostReceiver(manager, snapshot_id)
        with pytest.raises(RuntimeError, match="crossed NUMA"):
            manager.submit(receiver, list(range(8)))
    finally:
        import os

        try:
            os.unlink(ledger_path)
        except FileNotFoundError:
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_p2d_host_round_trip():
    """P D2H -> tmpfs -> same-NUMA D H2D preserves every K/V element."""

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
            numa_node=0,
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
