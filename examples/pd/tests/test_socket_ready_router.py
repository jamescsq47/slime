"""Exercise the real Router event consumer, without model or control files."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from late_binding_router import LateBindingMiniLoadBalancer
from sglang.srt.disaggregation.agentic_control_rpc import ControlRPCClient, ControlRPCServer
from sglang.srt.disaggregation.agentic_control_store import ControlKV, MemoryControlStore


def test_router_consumes_ready_and_delete_events_without_scans(tmp_path, monkeypatch):
    server = ControlRPCServer("router-test", "secret")
    state = MemoryControlStore(server.publish)
    server.register_service("records", state.methods())
    client = ControlRPCClient(server.address, run_id="router-test", token="secret")
    client.wait_ready()
    records = ControlKV("ready", client=client)
    router = object.__new__(LateBindingMiniLoadBalancer)
    router.p_ready_dir = tmp_path / "no-control-files"
    router.ready_signals = SimpleNamespace(records=records)
    router._p_ready_snapshot = {}
    router._p_ready_waiters = {}
    router._p_ready_fifo_events = {}
    router._p_ready_monitor_task = None

    def forbidden(*args, **kwargs):
        raise AssertionError("event Router attempted filesystem access")

    async def run():
        router._signal_changed = asyncio.Event()
        router._p_ready_broker_event = asyncio.Event()
        future = asyncio.get_running_loop().create_future()
        router._p_ready_waiters[123] = {future}
        task = asyncio.create_task(router._p_ready_monitor_loop())
        try:
            records.notify("upsert", "123.ready", {"ready_sequence": 1, "num_kv_tokens": 500})
            result = await asyncio.wait_for(future, timeout=3)
            assert result["num_kv_tokens"] == 500
            assert router._has_signal(router._ready_path(123))
            assert router._p_ready_sequence((123,)) == 1
            # Eight true D admission ACKs, not one rank deleting others' latch.
            for rank in range(8):
                await asyncio.wrap_future(records.notify("admit_ready", "123.ready", "D", rank, 8))
            assert not router._has_signal(router._ready_path(123))
            # Event stream is ordered; another ready acts as a consumer barrier.
            barrier = asyncio.get_running_loop().create_future()
            router._p_ready_waiters[124] = {barrier}
            records.notify("upsert", "124.ready", {"ready_sequence": 2})
            await asyncio.wait_for(barrier, timeout=3)
            assert 123 not in router._p_ready_snapshot
            assert 124 in router._p_ready_snapshot
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", forbidden)
            patch.setattr(Path, "glob", forbidden)
            patch.setattr(Path, "unlink", forbidden)
            asyncio.run(run())
        assert not router.p_ready_dir.exists()
    finally:
        client.close()
        server.close()


def test_router_waits_for_arrival_ack_without_blocking_event_loop():
    """A delayed control server cannot stop unrelated request dispatch."""
    entered = threading.Event()
    release = threading.Event()
    server = ControlRPCServer("router-ack", "secret", start=False)
    state = MemoryControlStore(server.publish)
    methods = state.methods()
    original = methods["upsert"]

    def delayed(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    methods["upsert"] = delayed
    server.register_service("records", methods)
    server.start()
    client = ControlRPCClient(server.address, run_id="router-ack", token="secret")
    client.wait_ready()
    records = ControlKV("arrival", client=client)
    router = object.__new__(LateBindingMiniLoadBalancer)
    router.early_claim_store = SimpleNamespace(_records=records)

    async def run():
        task = asyncio.create_task(router._early_claim_write(
            records.call, "upsert", "generation", {"arrived": True},
        ))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert not task.done()  # Still requires the actual server ACK.
            # The event loop is running even while the server is paused.
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(task, 3)
            assert records.get("generation") == {"arrived": True}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        release.set()
        client.close()
        server.close()
