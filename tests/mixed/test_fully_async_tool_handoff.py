from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from slime.utils.types import Sample
from slime.rollout import sglang_rollout


MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))

fully_async_rollout = importlib.import_module("fully_async_rollout")


class FakeDataBuffer:
    def __init__(self):
        self.samples = []

    def add_samples(self, samples):
        self.samples.extend(samples)


class FakeGenerateState:
    def __init__(self):
        self.sampling_params = {}
        self.abort_epoch = 0
        self.reset()

    def reset(self):
        self.pendings = set()
        self.pending_groups = {}
        self.aborted = False


def make_args(**overrides):
    values = {
        "sglang_server_concurrency": 32,
        "rollout_batch_size": 32,
        "n_samples_per_prompt": 8,
        "partial_rollout": True,
        "fully_async_buffer_policy": "window_evict",
        "fully_async_version_window": 100,
        "fully_async_max_partial_span": 100,
        "fully_async_eviction_policy": "drop_oldest_version",
        "fully_async_max_completed_samples": 32,
        "staleness_threshold": None,
        "update_weights_interval": 1,
        "current_policy_version": 0,
        "current_rollout_id": 0,
        "use_slime_dashboard": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tool_group_finishing_during_weight_update_is_deferred_and_recycled_once(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        data_buffer = FakeDataBuffer()
        release_tool = asyncio.Event()
        sample = Sample(
            prompt="question",
            group_index=17,
            metadata={"fully_async_sample_id": 17, "task_type": "math"},
        )
        group = [sample]

        async def running_tool_group():
            await release_tool.wait()
            sample.response = "<tool_call>...</tool_call><interpreter>result</interpreter>"
            sample.status = Sample.Status.ABORTED
            return group

        async def fake_abort(args, rollout_id, *, drain_timeout=None):
            assert drain_timeout == fully_async_rollout.FULLY_ASYNC_ABORT_DRAIN_TIMEOUT
            state.aborted = True
            state.abort_epoch += 1
            return []

        monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda args: state)
        monkeypatch.setattr(fully_async_rollout, "abort", fake_abort)
        monkeypatch.setattr(fully_async_rollout, "dashboard_event", lambda *args, **kwargs: None)

        worker = fully_async_rollout.AsyncRolloutWorker(make_args(), data_buffer)
        worker.task_lock = asyncio.Lock()
        task = asyncio.create_task(running_tool_group())
        state.pendings.add(task)
        worker.task_sample_ids[task] = 17
        worker.inflight_groups[task] = group

        result = await worker._prepare_for_weight_update_async(policy_version=1)

        assert result["deferred_groups"] == 1
        assert not state.pendings
        assert task in worker.deferred_groups
        assert data_buffer.samples == []

        # Simulate the tool returning while model weights are still syncing.
        release_tool.set()
        await task
        await worker._push_finished_deferred_tasks_to_data_buffer()
        await worker._push_finished_deferred_tasks_to_data_buffer()

        assert task not in worker.deferred_groups
        assert data_buffer.samples == [group]
        assert "<interpreter>result</interpreter>" in sample.response
        assert sample.status == Sample.Status.ABORTED

    asyncio.run(scenario())


def test_deferred_group_still_counts_against_group_concurrency(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(make_args(rollout_batch_size=2), FakeDataBuffer())

    pending = object()
    group = [Sample(group_index=1, metadata={"fully_async_sample_id": 1})]
    worker.deferred_groups[pending] = fully_async_rollout.DeferredGroupRecord(
        sample_id=1,
        group=group,
        policy_version=0,
        abort_epoch=1,
    )

    assert len(state.pendings) + len(worker.deferred_groups) == 1
    assert worker.max_concurrent_tasks == 2


def test_worker_uses_oversampling_for_group_concurrency_only(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda args: state)
    args = make_args(rollout_batch_size=32, over_sampling_batch_size=36)

    worker = fully_async_rollout.AsyncRolloutWorker(args, FakeDataBuffer())

    assert worker.max_concurrent_tasks == 36
    assert args.rollout_batch_size == 32


def test_sample_level_group_is_reactivated_as_soon_as_new_weights_are_ready(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda args: state)
        monkeypatch.setattr(fully_async_rollout, "dashboard_event", lambda *args, **kwargs: None)

        async def fake_abort(args, rollout_id, *, drain_timeout=None):
            state.aborted = True
            state.abort_epoch += 1
            return []

        monkeypatch.setattr(fully_async_rollout, "abort", fake_abort)
        worker = fully_async_rollout.AsyncRolloutWorker(make_args(), FakeDataBuffer())
        worker.task_lock = asyncio.Lock()
        worker.generation_resume_event = asyncio.Event()
        worker.generation_resume_event.set()
        group = [Sample(group_index=7, metadata={"fully_async_sample_id": 7})]

        async def persistent_group():
            await asyncio.Event().wait()

        task = asyncio.create_task(persistent_group())
        worker._track_inflight_task(task, 7, group, sample_level_handoff=True)

        before = await worker._prepare_for_weight_update_async(policy_version=1)
        assert before["deferred_groups"] == 1
        assert not worker.generation_resume_event.is_set()
        assert task in worker.deferred_groups
        assert task not in state.pendings

        after = await worker._finish_weight_update_async(policy_version=1)
        assert after["reactivated_groups"] == 1
        assert worker.generation_resume_event.is_set()
        assert task not in worker.deferred_groups
        assert task in state.pendings
        assert state.pending_groups[task] is group
        assert worker.task_sample_ids[task] == 7
        assert worker.inflight_groups[task] is group

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_worker_tracks_pending_group_for_sglang_abort_drain(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda args: state)
        worker = fully_async_rollout.AsyncRolloutWorker(make_args(), FakeDataBuffer())
        group = [Sample(group_index=3, metadata={"fully_async_sample_id": 3})]

        async def pending_group():
            await asyncio.Event().wait()

        task = asyncio.create_task(pending_group())
        worker._track_inflight_task(task, 3, group)

        assert task in state.pendings
        assert state.pending_groups[task] is group
        assert worker.task_sample_ids[task] == 3
        assert worker.inflight_groups[task] is group

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_sglang_abort_has_bounded_drain_for_running_tools(monkeypatch):
    async def scenario():
        state = FakeGenerateState()

        async def never_finishes():
            await asyncio.Event().wait()

        async def fake_get(url):
            return {"workers": []}

        task = asyncio.create_task(never_finishes())
        state.pendings.add(task)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
        monkeypatch.setattr(sglang_rollout, "get", fake_get)
        monkeypatch.setattr(sglang_rollout.sglang_router, "__version__", "0.3.0")

        loop = asyncio.get_running_loop()
        started = loop.time()
        groups = await sglang_rollout.abort(
            SimpleNamespace(partial_rollout=True, sglang_router_ip="127.0.0.1", sglang_router_port=30000),
            rollout_id=0,
            drain_timeout=0.01,
        )
        elapsed = loop.time() - started

        assert groups == []
        assert elapsed < 0.2
        assert state.abort_epoch == 1
        assert task in state.pendings

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
