"""A cancelled HTTP coroutine cannot leave a late control writer behind."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from late_binding_router import LateBindingMiniLoadBalancer


def router_with_records(records):
    router = object.__new__(LateBindingMiniLoadBalancer)
    router.early_claim_store = SimpleNamespace(_records=records)
    return router


@pytest.mark.parametrize("cancel_count", [1, 3])
def test_cancelled_router_write_drains_same_worker_before_cleanup(cancel_count):
    router = router_with_records(object())
    entered, release = threading.Event(), threading.Event()
    order = []

    def write():
        entered.set()
        assert release.wait(5)
        order.append("write-ack")

    async def run():
        async def dispatch():
            try:
                await router._early_claim_write(write)
            finally:
                order.append("request-cleanup")

        task = asyncio.create_task(dispatch())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            for _ in range(cancel_count):
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done(), "HTTP cleanup detached an in-flight control write"
                assert order == []
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert order == ["write-ack", "request-cleanup"]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_router_write_error_propagates_without_fabricating_ack():
    router = router_with_records(object())
    calls = []

    def write():
        calls.append(1)
        raise ValueError("ownership rejected")

    async def run():
        with pytest.raises(ValueError, match="ownership rejected"):
            await router._early_claim_write(write)

    asyncio.run(run())
    assert calls == [1]


def test_legacy_router_write_remains_inline():
    router = router_with_records(None)
    caller = threading.get_ident()

    async def run():
        result = await router._early_claim_write(threading.get_ident)
        assert result == caller

    asyncio.run(run())
