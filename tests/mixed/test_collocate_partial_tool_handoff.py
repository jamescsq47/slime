from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from slime.rollout import sglang_rollout
from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample


class FakeState:
    def __init__(self):
        self.pendings = set()
        self.pending_groups = {}
        self.deferred_tool_tasks = {}
        self.group_child_tasks = {}
        self.handoff_completed_since_log = 0
        self.handoff_failed_since_log = 0


def test_running_tool_is_deferred_and_recycled_exactly_once():
    async def scenario():
        state = FakeState()
        release_tool = asyncio.Event()
        sample = Sample(
            prompt="question",
            metadata={sglang_rollout.PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY: True},
        )
        group = [sample]

        async def running_tool_group():
            await release_tool.wait()
            sample.response = "<interpreter>42</interpreter>"
            sample.status = Sample.Status.ABORTED
            sample.metadata.pop(sglang_rollout.PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY, None)
            return group

        task = asyncio.create_task(running_tool_group())
        state.pendings.add(task)
        state.pending_groups[task] = group

        deferred, recycled = await sglang_rollout._handoff_pending_tool_tasks(state)
        assert deferred == 1
        assert recycled == []
        assert task in state.deferred_tool_tasks
        assert not state.pendings

        release_tool.set()
        await task
        buffer = []
        add_samples = buffer.extend
        assert await sglang_rollout._drain_deferred_tool_tasks(state, add_samples) == (1, 0)
        assert await sglang_rollout._drain_deferred_tool_tasks(state, add_samples) == (0, 0)

        assert buffer == [group]
        assert sample.response == "<interpreter>42</interpreter>"
        assert sample.status == Sample.Status.ABORTED
        assert task not in state.deferred_tool_tasks

        metrics = sglang_rollout._collect_tool_handoff_metrics(state, deferred_groups=1)
        assert metrics == {
            "tool/partial_deferred_groups": 1,
            "tool/partial_deferred_inflight_groups": 0,
            "tool/partial_deferred_completed_groups": 1,
            "tool/partial_deferred_failed_groups": 0,
        }

    asyncio.run(scenario())


def test_non_tool_cleanup_is_deferred_until_it_commits_partial_state():
    async def scenario():
        state = FakeState()
        release_cleanup = asyncio.Event()
        sample = Sample(prompt="queued generation", metadata={})
        group = [sample]

        async def queued_generation():
            await release_cleanup.wait()
            sample.status = Sample.Status.ABORTED
            return group

        task = asyncio.create_task(queued_generation())
        state.pendings.add(task)
        state.pending_groups[task] = group

        deferred, recycled = await sglang_rollout._handoff_pending_tool_tasks(state)

        assert deferred == 1
        assert recycled == []
        assert not task.done()
        assert task in state.deferred_tool_tasks

        release_cleanup.set()
        await task
        buffer = []
        assert await sglang_rollout._drain_deferred_tool_tasks(state, buffer.extend) == (1, 0)
        assert buffer == [group]

    asyncio.run(scenario())


def test_group_child_waiting_on_semaphore_cannot_generate_after_epoch_reset(monkeypatch):
    async def scenario():
        state = SimpleNamespace(
            semaphore=asyncio.Semaphore(0),
            aborted=False,
            abort_epoch=0,
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)

        sample = Sample(prompt="queued child", metadata={})
        args = SimpleNamespace(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
        )
        task = asyncio.create_task(
            sglang_rollout.generate_and_rm(
                args,
                sample,
                sampling_params={},
                expected_abort_epoch=0,
            )
        )
        await asyncio.sleep(0)

        # The boolean is reset after the rollout returns, but the monotonic
        # epoch proves that this child belongs to the pre-update group.
        state.abort_epoch = 1
        state.aborted = False
        state.semaphore.release()
        result = await task

        assert result is sample
        assert sample.status == Sample.Status.ABORTED

    asyncio.run(scenario())


def test_completed_sample_with_cancelled_reward_retries_only_reward(monkeypatch):
    async def scenario():
        state = SimpleNamespace(
            semaphore=asyncio.Semaphore(1),
            aborted=False,
            abort_epoch=1,
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
        reward_calls = []

        async def fake_async_rm(args, sample):
            reward_calls.append(sample.response)
            return 0.75

        monkeypatch.setattr(sglang_rollout, "async_rm", fake_async_rm)
        sample = Sample(
            prompt="question",
            response="final answer",
            status=Sample.Status.COMPLETED,
            reward=None,
            metadata={},
        )
        args = SimpleNamespace(
            partial_rollout=True,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=False,
        )

        result = await sglang_rollout.generate_and_rm(args, sample, sampling_params={})

        assert result is sample
        assert result.response == "final answer"
        assert result.status == Sample.Status.COMPLETED
        assert result.reward == 0.75
        assert reward_calls == ["final answer"]

    asyncio.run(scenario())


def test_group_generation_preserves_terminal_members_and_original_positions(monkeypatch):
    async def scenario():
        state = FakeState()
        state.aborted = False
        state.abort_epoch = 3
        state.group_sampling_seeds = [100, 101, 102, 103]
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)

        completed = Sample(
            prompt="completed",
            response="keep completed response",
            status=Sample.Status.COMPLETED,
            reward={"score": 0.25},
        )
        pending = Sample(prompt="pending")
        aborted = Sample(
            prompt="aborted",
            response="committed partial prefix",
            status=Sample.Status.ABORTED,
        )
        truncated = Sample(
            prompt="truncated",
            response="keep truncated response",
            status=Sample.Status.TRUNCATED,
            reward={"score": 0.5},
        )
        group = [completed, pending, aborted, truncated]
        generated = []

        async def fake_generate_and_rm(args, sample, sampling_params, **kwargs):
            generated.append((sample.prompt, sampling_params["sampling_seed"], kwargs["expected_abort_epoch"]))
            sample.response += "|generated"
            sample.status = Sample.Status.COMPLETED
            return sample

        async def fake_group_rm(args, samples):
            assert samples == group
            return [{"score": float(i)} for i in range(len(samples))]

        monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
        monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_group_rm)
        args = SimpleNamespace(sglang_enable_deterministic_inference=True, group_rm=True)

        result = await sglang_rollout.generate_and_rm_group(args, group, {})

        assert result == group
        assert generated == [("pending", 101, 3), ("aborted", 102, 3)]
        assert completed.response == "keep completed response"
        assert truncated.response == "keep truncated response"
        assert pending.response == "|generated"
        assert aborted.response == "committed partial prefix|generated"
        assert [sample.reward for sample in result] == [
            {"score": 0.0},
            {"score": 1.0},
            {"score": 2.0},
            {"score": 3.0},
        ]
        assert not state.group_child_tasks

    asyncio.run(scenario())


def test_failed_deferred_tool_recycles_last_committed_group():
    async def scenario():
        state = FakeState()
        sample = Sample(
            prompt="question",
            metadata={sglang_rollout.PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY: True},
        )
        group = [sample]

        async def failed_tool_group():
            raise RuntimeError("tool backend failed")

        task = asyncio.create_task(failed_tool_group())
        await asyncio.sleep(0)
        state.deferred_tool_tasks[task] = group
        buffer = []

        assert await sglang_rollout._drain_deferred_tool_tasks(state, buffer.extend) == (0, 1)
        assert buffer == [group]
        assert sample.status == Sample.Status.ABORTED
        assert sglang_rollout.PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY not in sample.metadata

    asyncio.run(scenario())


def test_handoff_allows_group_siblings_to_commit_consistent_partial_state(monkeypatch):
    async def scenario():
        state = FakeState()
        state.aborted = False
        state.abort_epoch = 0
        tool_release = asyncio.Event()
        sibling_release = asyncio.Event()
        tool_sample = Sample(prompt="tool", metadata={})
        sibling_sample = Sample(prompt="generation", metadata={})
        group = [tool_sample, sibling_sample]

        async def fake_generate_and_rm(args, sample, sampling_params, **kwargs):
            if sample is tool_sample:
                sample.metadata[sglang_rollout.PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY] = True
                await tool_release.wait()
                sample.metadata.pop(sglang_rollout.PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY, None)
                sample.status = Sample.Status.ABORTED
                return sample
            await sibling_release.wait()
            sample.response = "committed prior tool observation"
            sample.status = Sample.Status.ABORTED
            return sample

        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
        monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
        args = SimpleNamespace(sglang_enable_deterministic_inference=False, group_rm=False)
        parent = asyncio.create_task(sglang_rollout.generate_and_rm_group(args, group, {}))
        state.pendings.add(parent)
        state.pending_groups[parent] = group

        for _ in range(10):
            await asyncio.sleep(0)
            if tool_sample.metadata.get(sglang_rollout.PARTIAL_ROLLOUT_TOOL_INFLIGHT_KEY):
                break

        deferred, recycled = await sglang_rollout._handoff_pending_tool_tasks(state)
        assert deferred == 1
        assert recycled == []
        assert not parent.done()

        tool_release.set()
        sibling_release.set()
        assert await parent == group
        assert sibling_sample.response == "committed prior tool observation"
        buffer = []
        assert await sglang_rollout._drain_deferred_tool_tasks(state, buffer.extend) == (1, 0)
        assert buffer == [group]

    asyncio.run(scenario())


def test_abort_drain_recycles_group_when_task_raises(monkeypatch):
    async def scenario():
        state = FakeState()
        state.aborted = False
        state.abort_epoch = 0
        sample = Sample(prompt="question", metadata={})
        group = [sample]

        async def fail():
            raise RuntimeError("reward backend failed")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)
        state.pendings.add(task)
        state.pending_groups[task] = group
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)

        async def fake_get(url):
            return {"workers": []}

        monkeypatch.setattr(sglang_rollout, "get", fake_get)
        args = SimpleNamespace(
            partial_rollout=True,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )
        recycled = await sglang_rollout.abort(args, rollout_id=7, drain_timeout=0.1)

        assert recycled == [group]
        assert sample.status == Sample.Status.ABORTED
        assert state.handoff_failed_since_log == 1

    asyncio.run(scenario())


def test_abort_repeats_until_late_sglang_request_has_returned(monkeypatch):
    async def scenario():
        state = FakeState()
        state.aborted = False
        state.abort_epoch = 0
        release_generation = asyncio.Event()
        sample = Sample(
            prompt="question",
            metadata={sglang_rollout.PARTIAL_ROLLOUT_SGLANG_INFLIGHT_KEY: True},
        )
        group = [sample]

        async def late_generation():
            await release_generation.wait()
            sample.status = Sample.Status.ABORTED
            return group

        task = asyncio.create_task(late_generation())
        state.pendings.add(task)
        state.pending_groups[task] = group
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)

        async def fake_get(url):
            return {"workers": [{"url": "http://worker"}]}

        abort_calls = 0

        async def fake_post(url, payload):
            nonlocal abort_calls
            abort_calls += 1
            if abort_calls == 2:
                sample.metadata.pop(sglang_rollout.PARTIAL_ROLLOUT_SGLANG_INFLIGHT_KEY, None)
                release_generation.set()
            return {}

        monkeypatch.setattr(sglang_rollout, "get", fake_get)
        monkeypatch.setattr(sglang_rollout, "post", fake_post)
        args = SimpleNamespace(
            partial_rollout=True,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )

        recycled = await sglang_rollout.abort(args, rollout_id=7, drain_timeout=0.1)

        assert abort_calls == 2
        assert recycled == [group]
        assert sglang_rollout.PARTIAL_ROLLOUT_SGLANG_INFLIGHT_KEY not in sample.metadata

    asyncio.run(scenario())


def test_sglang_generation_marker_is_cleared_on_error():
    sample = Sample(prompt="question", metadata={})

    try:
        with sglang_rollout.track_sglang_generation(sample):
            assert sample.metadata[sglang_rollout.PARTIAL_ROLLOUT_SGLANG_INFLIGHT_KEY]
            raise RuntimeError("request failed")
    except RuntimeError:
        pass

    assert sglang_rollout.PARTIAL_ROLLOUT_SGLANG_INFLIGHT_KEY not in sample.metadata


def test_flush_cache_waits_between_pending_request_responses(monkeypatch):
    from slime.backends.sglang_utils import sglang_engine

    engine = sglang_engine.SGLangEngine.__new__(sglang_engine.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    responses = iter([SimpleNamespace(status_code=400), SimpleNamespace(status_code=200)])
    sleeps = []

    monkeypatch.setattr(sglang_engine.requests, "get", lambda url: next(responses))
    monkeypatch.setattr(sglang_engine.time, "sleep", sleeps.append)

    engine.flush_cache()

    assert sleeps == [1]


def test_collection_failure_recycles_group_and_keeps_rollout_alive():
    async def scenario():
        state = FakeState()
        state.remaining_batch_size = 40
        sample = Sample(prompt="question", metadata={})
        group = [sample]

        async def fail():
            raise RuntimeError("reward backend failed")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)
        state.pending_groups[task] = group
        buffer = []

        result = sglang_rollout._consume_finished_rollout_task(state, task, buffer.extend)

        assert result is None
        assert buffer == [group]
        assert sample.status == Sample.Status.ABORTED
        assert state.remaining_batch_size == 39
        assert state.handoff_failed_since_log == 1

    asyncio.run(scenario())


def test_submission_budget_charges_deferred_groups_to_oversampling_headroom():
    assert sglang_rollout._submission_batch_size(32, 40, 0, 0) == 40
    assert sglang_rollout._submission_batch_size(32, 40, 0, 4) == 36
    assert sglang_rollout._submission_batch_size(32, 40, 0, 8) == 32
    assert sglang_rollout._submission_batch_size(32, 40, 28, 4) == 8


def test_submission_target_replenishes_completed_groups_only_when_enabled():
    assert sglang_rollout._submission_target(32, 40, 0, False) == 32
    assert sglang_rollout._submission_target(32, 40, 0, True) == 40
    assert sglang_rollout._submission_target(32, 40, 4, True) == 36
    assert sglang_rollout._submission_target(32, 40, 12, True) == 32


def test_successful_group_releases_a_slot_when_replenishment_is_enabled():
    async def scenario(replenish_completed_groups):
        state = FakeState()
        state.remaining_batch_size = 40
        sample = Sample(prompt="question", response="answer", metadata={})
        sample.status = Sample.Status.COMPLETED
        group = [sample]

        async def finish():
            return group

        task = asyncio.create_task(finish())
        await asyncio.sleep(0)
        state.pending_groups[task] = group

        result = sglang_rollout._consume_finished_rollout_task(
            state,
            task,
            None,
            release_slot_on_completion=replenish_completed_groups,
        )
        return result, state.remaining_batch_size

    assert asyncio.run(scenario(False))[1] == 40
    result, remaining_batch_size = asyncio.run(scenario(True))
    assert result[0].response == "answer"
    assert remaining_batch_size == 39


def test_replenishment_stops_when_training_batch_is_full():
    async def scenario():
        state = FakeState()
        state.remaining_batch_size = 40
        total_submitted = 40

        for completed_groups in range(1, 33):
            sample = Sample(
                prompt=f"question-{completed_groups}",
                response="answer",
                status=Sample.Status.COMPLETED,
                metadata={},
            )
            group = [sample]

            async def finish(completed_group=group):
                return completed_group

            task = asyncio.create_task(finish())
            await asyncio.sleep(0)
            state.pending_groups[task] = group
            assert sglang_rollout._consume_finished_rollout_task(
                state,
                task,
                None,
                release_slot_on_completion=True,
            ) == group

            # generate_rollout_async exits immediately after the 32nd valid
            # group, so only the first 31 completions trigger replacements.
            if completed_groups < 32:
                submission_target = sglang_rollout._submission_target(32, 40, 0, True)
                submit_count = sglang_rollout._submission_batch_size(
                    32,
                    40,
                    state.remaining_batch_size,
                    0,
                )
                assert state.remaining_batch_size < submission_target
                assert submit_count == 1
                state.remaining_batch_size += submit_count
                total_submitted += submit_count

        return total_submitted, state.remaining_batch_size

    # 40 initial groups + 31 replacements; the final 39 become the existing
    # partial/abort handoff set while the 32 completed groups train.
    assert asyncio.run(scenario()) == (71, 39)


def test_train_time_mask_preserves_current_policy_suffix():
    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = manager_class.__new__(manager_class)
    manager.args = SimpleNamespace(
        current_policy_version=1,
        mask_offpolicy_math=32,
        mask_offpolicy_qa=32,
    )
    manager.data_source = SimpleNamespace(version_task_counts={0: {"math": 32, "qa": 32}})

    for task_type in ("math", "qa"):
        sample = Sample(
            prompt="question",
            response="abcde",
            response_length=5,
            loss_mask=[1, 1, 1, 1, 1],
            metadata={
                "task_type": task_type,
                "dispatch_version": 0,
                "partial_rollout": True,
                "partial_rollout_prefix_length": 3,
            },
        )
        assert manager._apply_train_time_offpolicy_mask(sample)
        assert sample.loss_mask == [0, 0, 0, 1, 1]


def test_mixed_launch_script_uses_collocate_partial_architecture():
    repo_root = Path(__file__).resolve().parents[2]
    script = (repo_root / "examples/mixed/hybrid_qwen3_4b_multi_async.sh").read_text()

    assert "-- python3 train.py" in script
    assert "--colocate" in script
    assert "--partial-rollout" in script
    assert "--partial-rollout-tool-handoff" in script
    assert "--over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE:-" in script
    assert '"${ROLLOUT_REPLENISH_COMPLETED_GROUPS:-1}" = "1"' in script
    assert "--rollout-replenish-completed-groups" in script
    assert "slime.rollout.sglang_rollout.generate_rollout" in script
    assert "fully_async_rollout" not in script
    assert "train_async.py" not in script
    assert "--fully-async-" not in script
