from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from slime.rollout import sglang_rollout
from slime.utils.types import Sample


MIXED_DIR = Path(__file__).resolve().parents[2] / "examples" / "mixed"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))

fully_async_rollout = importlib.import_module("fully_async_rollout")


class FakeGenerateState:
    def __init__(self):
        self.sampling_params = {}
        self.abort_epoch = 0
        self.group_child_tasks = {}
        self.reset()

    def reset(self):
        self.pendings = set()
        self.pending_groups = {}
        self.aborted = False


class EndlessDataBuffer:
    def __init__(self, group_size: int):
        self.group_size = group_size
        self.next_group_id = 0

    def get_samples(self, count: int):
        assert count == 1
        group_id = self.next_group_id
        self.next_group_id += 1
        return [
            [
                Sample(
                    prompt=f"group-{group_id}",
                    group_index=group_id,
                    metadata={"fully_async_sample_id": group_id, "task_type": "math"},
                )
                for _ in range(self.group_size)
            ]
        ]


def make_worker_args(*, backfill: bool, **overrides) -> SimpleNamespace:
    values = dict(
        sglang_server_concurrency=4,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        rollout_sample_completion_backfill=backfill,
        partial_rollout=True,
        fully_async_buffer_policy="window_evict",
        fully_async_version_window=100,
        fully_async_max_partial_span=100,
        fully_async_eviction_policy="drop_oldest_version",
        fully_async_max_completed_samples=32,
        staleness_threshold=None,
        update_weights_interval=1,
        current_policy_version=0,
        current_rollout_id=0,
        use_slime_dashboard=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mixed_script_gates_decoupled_gpu_tool_scheduling():
    script = (MIXED_DIR / "hybrid_qwen3_4b_multi.sh").read_text()

    assert "--rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}" in script
    assert "DECOUPLED_GPU_TOOL_SCHEDULING=${DECOUPLED_GPU_TOOL_SCHEDULING:-0}" in script
    assert 'if [ "${DECOUPLED_GPU_TOOL_SCHEDULING}" = "1" ]; then' in script
    assert "--decoupled-gpu-tool-scheduling" in script
    assert '--gpu-generation-slots "${GPU_GENERATION_SLOTS:-256}"' in script
    assert '--terminal-live-session-limit "${TERMINAL_LIVE_SESSION_LIMIT:-224}"' in script
    assert '--terminal-concurrent-resets "${TERMINAL_CONCURRENT_RESETS:-32}"' in script
    assert "--inflight-group-soft-limit" not in script
    assert "--max-inflight-groups" not in script
    assert "CUSTOM_ARGS+=(--rollout-sample-completion-backfill)" in script
    assert "--over-sampling-batch-size" not in script
    assert "--elastic-group-window" not in script
    assert "--adaptive-group-oversampling" not in script
    assert "--tool-resume-admission" not in script


def test_decoupled_backfill_is_not_limited_by_group_count_by_default(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=False,
            decoupled_gpu_tool_scheduling=True,
            gpu_generation_slots=4,
        ),
        EndlessDataBuffer(group_size=2),
    )

    state.pendings.update(object() for _ in range(100))

    assert worker.max_inflight_groups is None
    assert worker.inflight_group_soft_limit is None
    assert worker._decoupled_groups_needed() == 2


def test_disabled_elastic_window_keeps_fixed_rollout_batch_target(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(backfill=True),
        EndlessDataBuffer(group_size=2),
    )

    assert worker.adaptive_group_oversampling is False
    assert worker.base_concurrent_tasks == 2
    assert worker.max_concurrent_tasks == 2
    assert worker.current_concurrent_tasks == 2
    assert worker._update_adaptive_group_oversampling({}, now=15) == {}
    assert worker.current_concurrent_tasks == 2


def test_decoupled_backfill_respects_max_inflight_groups(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=False,
            decoupled_gpu_tool_scheduling=True,
            gpu_generation_slots=4,
            max_inflight_groups=3,
        ),
        EndlessDataBuffer(group_size=2),
    )

    assert worker._decoupled_groups_needed() == 2
    state.pendings.update({object(), object()})
    assert worker._decoupled_groups_needed() == 1
    worker.deferred_groups[object()] = fully_async_rollout.DeferredGroupRecord(
        sample_id=1,
        group=[],
        policy_version=0,
        abort_epoch=0,
    )
    assert worker._decoupled_groups_needed() == 0


def test_decoupled_backfill_uses_soft_limit_then_one_emergency_container(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=False,
            decoupled_gpu_tool_scheduling=True,
            gpu_generation_slots=4,
            inflight_group_soft_limit=2,
            max_inflight_groups=3,
        ),
        EndlessDataBuffer(group_size=2),
    )

    assert worker._decoupled_groups_needed() == 2
    state.pendings.update({object(), object()})
    assert worker._decoupled_groups_needed() == 1
    state.pendings.add(object())
    assert worker._decoupled_groups_needed() == 0


def test_resumable_group_releases_each_sample_slot_before_group_finishes(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        second_sample_release = asyncio.Event()
        first_sample_finished = asyncio.Event()
        first_credit_released = asyncio.Event()
        credits = 0

        async def fake_generate(_args, sample, _sampling_params, **_kwargs):
            if sample.metadata["member"] == 1:
                await second_sample_release.wait()
            sample.status = Sample.Status.COMPLETED
            sample.reward = 1
            if sample.metadata["member"] == 0:
                first_sample_finished.set()
            return sample

        def on_sample_done():
            nonlocal credits
            credits += 1
            first_credit_released.set()

        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate)
        args = SimpleNamespace(group_rm=False, sglang_enable_deterministic_inference=False)
        group = [
            Sample(prompt="p", metadata={"member": 0}),
            Sample(prompt="p", metadata={"member": 1}),
        ]
        resume_event = asyncio.Event()
        resume_event.set()

        parent = asyncio.create_task(
            sglang_rollout.generate_and_rm_group(
                args,
                group,
                {},
                resume_event=resume_event,
                sample_done_callback=on_sample_done,
            )
        )
        await asyncio.wait_for(first_sample_finished.wait(), timeout=1)
        await asyncio.wait_for(first_credit_released.wait(), timeout=1)

        assert credits == 1
        assert not parent.done()

        second_sample_release.set()
        result = await asyncio.wait_for(parent, timeout=1)
        assert result == group
        assert credits == 2

    asyncio.run(scenario())


def test_weight_boundary_partial_resume_does_not_double_release_slot(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        first_attempt_returned = asyncio.Event()
        resume_event = asyncio.Event()
        resume_event.set()
        attempts = 0
        credits = 0

        async def fake_generate(_args, sample, _sampling_params, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                resume_event.clear()
                state.aborted = True
                state.abort_epoch += 1
                sample.status = Sample.Status.ABORTED
                first_attempt_returned.set()
                return sample
            sample.status = Sample.Status.COMPLETED
            sample.reward = 1
            return sample

        def on_sample_done():
            nonlocal credits
            credits += 1

        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate)
        args = SimpleNamespace(group_rm=False, sglang_enable_deterministic_inference=False)
        group = [Sample(prompt="p", metadata={})]

        parent = asyncio.create_task(
            sglang_rollout.generate_and_rm_group(
                args,
                group,
                {},
                resume_event=resume_event,
                sample_done_callback=on_sample_done,
            )
        )
        await asyncio.wait_for(first_attempt_returned.wait(), timeout=1)
        await asyncio.sleep(0)

        assert credits == 0
        assert not parent.done()

        state.aborted = False
        resume_event.set()
        await asyncio.wait_for(parent, timeout=1)

        assert attempts == 2
        assert credits == 1

    asyncio.run(scenario())


def test_default_disabled_worker_keeps_legacy_group_level_backfill(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        release_groups = asyncio.Event()
        submitted_groups = []

        # Deliberately omit sample_done_callback from the signature. The
        # disabled path must not pass the new keyword at all.
        async def fake_group(
            args,
            group,
            sampling_params,
            evaluation=False,
            *,
            resume_event=None,
            on_sample_dispatch=None,
        ):
            submitted_groups.append(group[0].group_index)
            await release_groups.wait()
            for sample in group:
                sample.status = Sample.Status.COMPLETED
            return group

        monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_group)
        monkeypatch.setattr(fully_async_rollout, "dashboard_event", lambda *_args, **_kwargs: None)
        worker = fully_async_rollout.AsyncRolloutWorker(
            make_worker_args(backfill=False),
            EndlessDataBuffer(group_size=2),
        )

        loop_task = asyncio.create_task(worker.continuous_worker_loop())
        await asyncio.sleep(0.08)
        assert submitted_groups == [0, 1]

        worker.running = False
        release_groups.set()
        await asyncio.wait_for(loop_task, timeout=1)

    asyncio.run(scenario())


def test_enabled_worker_backfills_after_cross_group_sample_credits(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        release_groups = asyncio.Event()
        third_group_submitted = asyncio.Event()
        submitted_groups = []

        async def fake_group(
            args,
            group,
            sampling_params,
            evaluation=False,
            *,
            resume_event=None,
            on_sample_dispatch=None,
            sample_done_callback=None,
        ):
            submitted_groups.append(group[0].group_index)
            if len(submitted_groups) == 3:
                third_group_submitted.set()
            assert sample_done_callback is not None
            # One member from each initial group finishes. Across two groups,
            # those two credits are enough to submit a new two-member group,
            # even though neither initial group coordinator has returned.
            sample_done_callback()
            await release_groups.wait()
            sample_done_callback()
            for sample in group:
                sample.status = Sample.Status.COMPLETED
            return group

        monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_group)
        monkeypatch.setattr(fully_async_rollout, "dashboard_event", lambda *_args, **_kwargs: None)
        worker = fully_async_rollout.AsyncRolloutWorker(
            make_worker_args(backfill=True),
            EndlessDataBuffer(group_size=2),
        )

        loop_task = asyncio.create_task(worker.continuous_worker_loop())
        await asyncio.wait_for(third_group_submitted.wait(), timeout=1)

        assert submitted_groups[:3] == [0, 1, 2]
        assert len(state.pendings) == 3

        worker.running = False
        release_groups.set()
        await asyncio.wait_for(loop_task, timeout=1)

    asyncio.run(scenario())


def test_decoupled_worker_refills_gpu_slot_while_sample_waits_in_tool(monkeypatch):
    async def scenario():
        state = FakeGenerateState()
        release_one_generation = asyncio.Event()
        shutdown = asyncio.Event()
        initial_slots_full = asyncio.Event()
        replacement_started = asyncio.Event()
        submitted_groups = []

        async def fake_group(
            args,
            group,
            sampling_params,
            evaluation=False,
            *,
            resume_event=None,
            on_sample_dispatch=None,
        ):
            group_id = group[0].group_index
            submitted_groups.append(group_id)

            async def run_sample(index, sample):
                await args._sglang_sample_activation_acquire(sample)
                async with sglang_rollout.sglang_generation_slot(args, sample):
                    if group_id == 0 and index == 0:
                        await release_one_generation.wait()
                    else:
                        if args._sglang_generation_slot_acquire.__self__.active == 4:
                            initial_slots_full.set()
                        if group_id == 2:
                            replacement_started.set()
                        await shutdown.wait()
                # The released request is now doing tool work without a GPU slot.
                await shutdown.wait()
                sample.status = Sample.Status.COMPLETED

            await asyncio.gather(
                *(run_sample(i, sample) for i, sample in enumerate(group)),
                return_exceptions=True,
            )
            return group

        monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_group)
        monkeypatch.setattr(fully_async_rollout, "dashboard_event", lambda *_args, **_kwargs: None)
        worker = fully_async_rollout.AsyncRolloutWorker(
            make_worker_args(
                backfill=False,
                decoupled_gpu_tool_scheduling=True,
                gpu_generation_slots=4,
            ),
            EndlessDataBuffer(group_size=2),
        )

        loop_task = asyncio.create_task(worker.continuous_worker_loop())
        await asyncio.wait_for(initial_slots_full.wait(), timeout=1)
        await asyncio.sleep(0)
        assert submitted_groups == [0, 1]
        assert worker.generation_slots.active == 4
        assert worker.generation_slots._pending(worker.generation_slots.fresh_waiters) == 0

        release_one_generation.set()
        await asyncio.wait_for(replacement_started.wait(), timeout=1)
        assert worker.generation_slots.active == 4
        await asyncio.sleep(0.02)
        assert submitted_groups == [0, 1, 2]
        assert worker.generation_slots._pending(worker.generation_slots.fresh_waiters) == 0
        assert len(worker.sample_activation_pool.waiters) == 1

        worker.running = False
        shutdown.set()
        await asyncio.wait_for(loop_task, timeout=1)

    asyncio.run(scenario())


def test_elastic_group_window_smooths_low_load_and_adds_whole_groups(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=True,
            over_sampling_batch_size=6,
            adaptive_group_oversampling=True,
            adaptive_group_oversampling_running_threshold=180,
            adaptive_group_oversampling_queue_threshold=4,
            adaptive_group_oversampling_window_seconds=15,
            adaptive_group_oversampling_cooldown_seconds=15,
            adaptive_group_oversampling_step_groups=2,
        ),
        EndlessDataBuffer(group_size=2),
    )
    worker.sample_backfill_initialized = True
    low_load = {
        "sglang/total_running_requests": 170,
        "sglang/total_queued_requests": 1,
        "sglang/avg_kv_cache_usage": 0.5,
    }

    for now in (0, 5, 10):
        worker._update_adaptive_group_oversampling(low_load, now=now)
    assert worker.current_concurrent_tasks == 2
    assert worker.completed_sample_credits == 0

    metrics = worker._update_adaptive_group_oversampling(low_load, now=15)
    assert worker.current_concurrent_tasks == 4
    assert metrics["fully_async/live/adaptive_oversampling_adjustment_groups"] == 2
    assert metrics["fully_async/live/elastic_group_window_ready"] == 1

    # Two added prompt groups release exactly two group-sized credits.
    assert worker._reserve_sample_backfill_group()
    assert worker._reserve_sample_backfill_group()
    assert not worker._reserve_sample_backfill_group()


def test_elastic_group_window_can_drain_below_rollout_batch_size(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=True,
            over_sampling_batch_size=4,
            adaptive_group_oversampling=True,
            adaptive_group_oversampling_min_groups=1,
            adaptive_group_oversampling_window_seconds=5,
            adaptive_group_oversampling_cooldown_seconds=15,
            adaptive_group_oversampling_recovery_seconds=10,
            adaptive_group_oversampling_step_groups=2,
        ),
        EndlessDataBuffer(group_size=2),
    )
    worker.sample_backfill_initialized = True
    worker.current_concurrent_tasks = 4
    pressure = {
        "sglang/total_running_requests": 190,
        "sglang/total_queued_requests": 13,
        "sglang/avg_kv_cache_usage": 0.70,
    }

    for now in (0, 5, 15):
        worker._update_adaptive_group_oversampling(pressure, now=now)
    assert worker.current_concurrent_tasks == 2
    for now in (20, 25, 30):
        worker._update_adaptive_group_oversampling(pressure, now=now)
    assert worker.current_concurrent_tasks == 1
    # Existing excess groups are not cancelled; three groups drain naturally.
    assert worker.completed_sample_credits == -6
    for _ in range(6):
        worker._on_sample_done()
    assert worker.completed_sample_credits == 0
    assert not worker._reserve_sample_backfill_group()


def test_elastic_group_window_suppresses_expansion_after_weight_resume(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=True,
            over_sampling_batch_size=6,
            adaptive_group_oversampling=True,
            adaptive_group_oversampling_window_seconds=5,
            adaptive_group_oversampling_cooldown_seconds=1,
            adaptive_group_oversampling_post_resume_expansion_grace_seconds=30,
        ),
        EndlessDataBuffer(group_size=2),
    )
    low_load = {
        "sglang/total_running_requests": 100,
        "sglang/total_queued_requests": 0,
        "sglang/avg_kv_cache_usage": 0.2,
    }

    worker._set_pause_requested(True, now=0)
    worker._set_pause_requested(False, now=10)
    for now in (10, 15, 20, 25, 35):
        metrics = worker._update_adaptive_group_oversampling(low_load, now=now)

    assert worker.current_concurrent_tasks == 2
    assert metrics["fully_async/live/elastic_group_window_expansion_grace_remaining_seconds"] == 5

    metrics = worker._update_adaptive_group_oversampling(low_load, now=40)
    assert worker.current_concurrent_tasks == 4
    assert metrics["fully_async/live/adaptive_oversampling_adjustment_groups"] == 2
    assert metrics["fully_async/live/elastic_group_window_expansion_grace_remaining_seconds"] == 0


def test_elastic_group_window_global_hard_kv_requires_global_queue(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=True,
            over_sampling_batch_size=8,
            adaptive_group_oversampling=True,
            adaptive_group_oversampling_min_groups=1,
            adaptive_group_oversampling_window_seconds=5,
            adaptive_group_oversampling_cooldown_seconds=1,
            adaptive_group_oversampling_pressure_queue_threshold=12,
            adaptive_group_oversampling_hard_pressure_seconds=5,
            adaptive_group_oversampling_hard_step_groups=4,
        ),
        EndlessDataBuffer(group_size=2),
    )
    worker.sample_backfill_initialized = True
    worker.current_concurrent_tasks = 8
    high_kv_low_queue = {
        "sglang/total_running_requests": 205,
        "sglang/total_queued_requests": 1,
        "sglang/avg_kv_cache_usage": 0.80,
    }

    for now in (0, 5, 10):
        metrics = worker._update_adaptive_group_oversampling(high_kv_low_queue, now=now)
    assert worker.current_concurrent_tasks == 8
    assert metrics["fully_async/live/adaptive_oversampling_hard_pressure_condition"] == 0

    high_kv_high_queue = {**high_kv_low_queue, "sglang/total_queued_requests": 13}
    for now in (15, 20, 25):
        metrics = worker._update_adaptive_group_oversampling(high_kv_high_queue, now=now)
    assert worker.current_concurrent_tasks == 4
    assert metrics["fully_async/live/adaptive_oversampling_adjustment_groups"] == -4


def test_elastic_group_window_hard_local_engine_pressure_drains_faster(monkeypatch):
    state = FakeGenerateState()
    monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda _args: state)
    worker = fully_async_rollout.AsyncRolloutWorker(
        make_worker_args(
            backfill=True,
            over_sampling_batch_size=8,
            adaptive_group_oversampling=True,
            adaptive_group_oversampling_min_groups=1,
            adaptive_group_oversampling_window_seconds=5,
            adaptive_group_oversampling_cooldown_seconds=1,
            adaptive_group_oversampling_hard_pressure_seconds=5,
            adaptive_group_oversampling_hard_step_groups=4,
        ),
        EndlessDataBuffer(group_size=2),
    )
    worker.sample_backfill_initialized = True
    worker.current_concurrent_tasks = 8
    hard_pressure = {
        "sglang/total_running_requests": 205,
        "sglang/total_queued_requests": 8,
        "sglang/avg_kv_cache_usage": 0.6,
        "sglang/max_kv_cache_usage": 0.96,
        "sglang/max_kv_cache_engine_queued_requests": 5,
    }

    worker._update_adaptive_group_oversampling(hard_pressure, now=0)
    worker._update_adaptive_group_oversampling(hard_pressure, now=5)
    metrics = worker._update_adaptive_group_oversampling(hard_pressure, now=10)

    assert worker.current_concurrent_tasks == 4
    assert metrics["fully_async/live/adaptive_oversampling_adjustment_groups"] == -4
    assert worker.completed_sample_credits == -8


def test_generation_slot_pool_prioritizes_tool_fifo_over_fresh():
    async def scenario():
        pool = fully_async_rollout.GenerationSlotPool(1)
        holder_release = await pool.acquire(None, resume=False)
        order = []
        gates = {name: asyncio.Event() for name in ("fresh", "resume-1", "resume-2")}
        samples = {
            name: Sample(prompt=name, metadata={})
            for name in ("fresh", "resume-1", "resume-2")
        }

        async def wait(name, *, resume):
            release = await pool.acquire(samples[name], resume=resume)
            order.append(name)
            await gates[name].wait()
            release()

        async def wait_for_order(size):
            for _ in range(10):
                if len(order) >= size:
                    return
                await asyncio.sleep(0)
            raise AssertionError(f"only admitted {order}")

        fresh = asyncio.create_task(wait("fresh", resume=False))
        await asyncio.sleep(0)
        # Tool completion order defines FIFO, even if result 2 reaches the
        # actual generation context before result 1.
        pool.prioritize_resume(samples["resume-1"])
        pool.prioritize_resume(samples["resume-2"])
        resume_2 = asyncio.create_task(wait("resume-2", resume=True))
        await asyncio.sleep(0)
        resume_1 = asyncio.create_task(wait("resume-1", resume=True))
        await asyncio.sleep(0)

        assert pool.fresh_demand == 0
        holder_release()
        await wait_for_order(1)
        assert order == ["resume-1"]

        gates["resume-1"].set()
        await wait_for_order(2)
        assert order == ["resume-1", "resume-2"]

        gates["resume-2"].set()
        await wait_for_order(3)
        assert order == ["resume-1", "resume-2", "fresh"]

        gates["fresh"].set()
        await asyncio.gather(fresh, resume_1, resume_2)
        assert pool.active == 0
        assert pool.fresh_demand == 1

    asyncio.run(scenario())


def test_generation_slot_pool_skips_unready_resume_ticket_without_idling():
    async def scenario():
        pool = fully_async_rollout.GenerationSlotPool(1)
        holder_release = await pool.acquire(None, resume=False)
        stalled = Sample(prompt="stalled resume", metadata={})
        ready = Sample(prompt="ready resume", metadata={})
        fresh = Sample(prompt="fresh", metadata={})
        order = []

        pool.prioritize_resume(stalled)
        pool.prioritize_resume(ready)

        async def wait(name, sample, *, resume):
            release = await pool.acquire(sample, resume=resume)
            order.append(name)
            release()

        fresh_task = asyncio.create_task(wait("fresh", fresh, resume=False))
        ready_task = asyncio.create_task(wait("ready", ready, resume=True))
        await asyncio.sleep(0)

        holder_release()
        await asyncio.gather(ready_task, fresh_task)

        assert order == ["ready", "fresh"]
        assert pool.active == 0
        pool.discard_resume(stalled)
        assert not pool.resume_waiters

    asyncio.run(scenario())


def test_generation_slot_pool_does_not_create_supply_during_weight_resume_gap():
    async def scenario():
        activated = 0

        def activate(count):
            nonlocal activated
            activated += count

        pool = fully_async_rollout.GenerationSlotPool(
            2,
            activate_samples=activate,
            initial_sample_supply=0,
        )
        pool.sample_activation_enabled = False
        first = await pool.acquire(Sample(prompt="first", metadata={}), resume=False)
        second = await pool.acquire(Sample(prompt="second", metadata={}), resume=False)
        pool.sample_activation_enabled = True

        pool.pause()
        first()
        second()
        pool.resume(activate_new_samples=False)
        assert activated == 0

        resumed = [
            asyncio.create_task(pool.acquire(Sample(prompt=str(i), metadata={}), resume=True))
            for i in range(2)
        ]
        releases = await asyncio.gather(*resumed)
        pool.enable_sample_activation()
        assert activated == 0

        for release in releases:
            release()
        assert activated == 2

    asyncio.run(scenario())


def test_sample_activation_pool_fills_one_gpu_gap_with_one_sample():
    async def scenario():
        pool = fully_async_rollout.SampleActivationPool(0)
        samples = [Sample(prompt=str(i), metadata={}) for i in range(3)]
        tasks = [asyncio.create_task(pool.acquire(sample)) for sample in samples]
        await asyncio.sleep(0)

        # The surrounding data container may be a full group, but a one-slot
        # GPU gap activates exactly one member and leaves its siblings dormant.
        pool.release(1)
        await asyncio.sleep(0)
        assert sum(task.done() for task in tasks) == 1
        assert (
            sum(
                sample.metadata.get(fully_async_rollout.SAMPLE_ACTIVATED_KEY, False)
                for sample in samples
            )
            == 1
        )
        assert len(pool.waiters) == 2

        pool.release(2)
        await asyncio.gather(*tasks)

    asyncio.run(scenario())


def test_generation_slot_pool_retracts_unused_supply_when_old_requests_arrive_late():
    async def scenario():
        activation_pool = fully_async_rollout.SampleActivationPool(0)
        pool = fully_async_rollout.GenerationSlotPool(
            2,
            activate_samples=activation_pool.release,
            retract_samples=activation_pool.retract_available,
            initial_sample_supply=0,
        )

        pool.enable_sample_activation()
        assert pool.provisional_fresh_supply == 2
        assert activation_pool.available == 2

        first = asyncio.create_task(
            pool.acquire(Sample(prompt="old-1", metadata={}), resume=True)
        )
        second = asyncio.create_task(
            pool.acquire(Sample(prompt="old-2", metadata={}), resume=True)
        )
        releases = await asyncio.gather(first, second)

        assert pool.active == 2
        assert pool.provisional_fresh_supply == 0
        assert activation_pool.available == 0
        assert pool.retracted_fresh_supply == 2

        for release in releases:
            release()

    asyncio.run(scenario())


def test_generation_slot_pool_replaces_terminal_live_waiter_sample_exactly():
    activated = 0

    def activate(count):
        nonlocal activated
        activated += count

    pool = fully_async_rollout.GenerationSlotPool(
        2,
        activate_samples=activate,
        initial_sample_supply=2,
    )
    sample = Sample(prompt="terminal", metadata={})
    sample.metadata[fully_async_rollout.SAMPLE_ACTIVATION_RESERVATION_KEY] = True

    assert pool.relinquish_fresh_reservation(sample)
    assert not pool.relinquish_fresh_reservation(sample)
    assert activated == 1
    assert pool.provisional_fresh_supply == 2
    assert pool.relinquished_fresh_supply == 1


def test_aborted_sample_discards_unready_resume_ticket(monkeypatch):
    async def scenario():
        pool = fully_async_rollout.GenerationSlotPool(1)
        sample = Sample(prompt="aborted resume", metadata={})
        pool.prioritize_resume(sample)
        args = SimpleNamespace(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            custom_generate_function_path=None,
            group_rm=True,
            _sglang_generation_slot_acquire=pool.acquire,
            _sglang_generation_slot_discard=pool.discard_resume,
        )
        state = SimpleNamespace(aborted=True, abort_epoch=1)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)

        result = await sglang_rollout.generate_and_rm(
            args,
            sample,
            sampling_params={},
            expected_abort_epoch=0,
        )

        assert result.status == Sample.Status.ABORTED
        assert not pool.resume_waiters
        assert not pool.resume_tickets

    asyncio.run(scenario())


def test_tool_return_marks_next_generation_as_resume_priority():
    async def scenario():
        pool = fully_async_rollout.GenerationSlotPool(1)
        args = SimpleNamespace(
            _sglang_generation_slot_acquire=pool.acquire,
            _sglang_generation_slot_prioritize=pool.prioritize_resume,
        )
        sample = Sample(prompt="tool sample", metadata={})

        await sglang_rollout.prioritize_tool_resume(args, sample)
        assert sample.metadata[sglang_rollout.PARTIAL_ROLLOUT_TOOL_RESUME_PRIORITY_KEY]

        async with sglang_rollout.sglang_generation_slot(args, sample):
            assert pool.active == 1
            assert sglang_rollout.PARTIAL_ROLLOUT_TOOL_RESUME_PRIORITY_KEY not in sample.metadata
        assert pool.active == 0

    asyncio.run(scenario())
