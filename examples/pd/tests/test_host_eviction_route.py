"""Late consumers discover fenced Host eviction without a pre-existing route."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from late_binding_router import LateBindingMiniLoadBalancer
from sglang.srt.disaggregation.agentic_control_rpc import ControlRPCClient, ControlRPCServer
from sglang.srt.disaggregation.agentic_control_store import ControlKV, MemoryControlStore
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_host_rpc import HostLedgerService, RemoteHostStagingLedger
from sglang.srt.disaggregation.agentic_kv_lifecycle import AgenticRequestMetadata


@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("old_route", [None, "direct_ready", "host_ready"])
def test_pruned_eviction_notifies_unrouted_and_already_submitted_consumers(monkeypatch, size, old_route):
    server = ControlRPCServer("eviction-test", "secret")
    HostLedgerService(server, "d2p")
    memory = MemoryControlStore(server.publish)
    server.register_service("records", memory.methods())
    client = ControlRPCClient(server.address, run_id="eviction-test", token="secret")
    client.wait_ready()
    try:
        records = ControlKV("early", client=client)
        import sglang.srt.disaggregation.agentic_control_store as controls
        monkeypatch.setenv("SGLANG_AGENTIC_CONTROL_ENDPOINT", "enabled")
        monkeypatch.setattr(controls, "control_kv", lambda *args: records)
        early = AgenticEarlyClaimStore("/must-not-use-nfs/eviction")
        ledger = RemoteHostStagingLedger(client, "d2p")
        metadata = AgenticRequestMetadata(request_id="late-tool", generation=1, parent_generation=0)
        parent = metadata.parent
        sid, owner = parent.snapshot_id, "source-host"

        def forbidden(*args, **kwargs):
            raise AssertionError("socket Host eviction/Router attempted file IO")

        with monkeypatch.context() as trap:
            trap.setattr("builtins.open", forbidden)
            trap.setattr(Path, "open", forbidden)
            trap.setattr("fcntl.flock", forbidden)
            early.publish_arrival(parent, prompt_token_count=32)
            if old_route is not None:
                early.publish_route(parent, route=old_route, prefill_domain=0, snapshot_tokens=16)
            for rank in range(size):
                ledger.offer({"snapshot_id": sid, "request_id": "late-tool", "generation": 0,
                              "tp_rank": rank, "tp_size": size, "token_count": 16,
                              "byte_size": 128, "source_host_node": "D", "d_pid": 100 + rank})
            for rank in range(size):
                assert ledger.claim_rank(sid, owner, tp_rank=rank, tp_size=size)
                assert ledger.publish_rank_grant(sid, owner, {"kind": "shared_host_extent", "byte_size": 128},
                                                 tp_rank=rank, tp_size=size)
            for rank in range(size):
                assert ledger.complete_host_write(sid, 100 + rank, tp_rank=rank, tp_size=size)
            assert ledger.begin_host_eviction(sid, owner, tp_size=size, reason="source_d_host_pressure")
            for rank in range(size):
                assert ledger.complete_host_eviction_rank(sid, owner, tp_rank=rank, tp_size=size)
                assert ledger.complete_source_host_release_rank(sid, owner, tp_rank=rank, tp_size=size)
            ledger.prune(0, 0)
            assert ledger.get(sid)["terminal_receipt"]
            assert ledger.snapshot_entries()[sid]["state"] == "recompute_required"

            async def run():
                request = {"input_ids": list(range(32))}
                router = object.__new__(LateBindingMiniLoadBalancer)
                router.early_claim_store, router._d2p_host_ledger = early, ledger
                router.ready_timeout, router.ready_poll_interval = 0.1, 0.001
                reservation = SimpleNamespace(domain=0)
                router._reserve_prefill_work = AsyncMock(return_value=reservation)
                router._release_prefill_work = AsyncMock()
                selected = await router._resolve_dynamic_prefill_work(request, metadata, None)
                assert selected is reservation
                router._reserve_prefill_work.assert_awaited_once_with(32)
                assert early.read_route(parent)["route"] == "recompute"

                # The outcome watcher also must not depend on a pre-existing
                # Host route: simulate an already-submitted Direct request.
                early.publish_route(parent, route="direct_ready", prefill_domain=0)
                router._settle_direct_workset = AsyncMock()
                router._resize_prefill_work = AsyncMock()
                result = await router._watch_dynamic_prefill_route(request, metadata, reservation)
                assert result == {"action": "recompute", "route": "host_evicted"}
                router._settle_direct_workset.assert_awaited_once_with(reservation)
                router._resize_prefill_work.assert_awaited_once_with(reservation, 32)

            asyncio.run(run())
            # New subscribers get the terminal receipt in their initial
            # snapshot too; this is not merely a transient local notification.
            another = ControlRPCClient(server.address, run_id="eviction-test", token="secret")
            try:
                another.wait_ready()
                assert RemoteHostStagingLedger(another, "d2p").get(sid)["terminal_receipt"]
            finally:
                another.close()
    finally:
        client.close()
        server.close()
