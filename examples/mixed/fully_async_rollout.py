import asyncio
import atexit
import math
import queue
import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, abort, eval_rollout, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.types import Sample
import wandb

_global_worker = None
_worker_lock = threading.Lock()
_wandb_metric_defined = False

def _sample_text_preview(sample: Sample) -> str:
    return str(sample.prompt) + str(sample.response)


def _extract_sample_id(group: list[Sample]) -> int | None:
    if not group:
        return None
    sample = group[0]
    for value in (
        sample.metadata.get("fully_async_sample_id"),
        sample.metadata.get("fully_async_group_id"),
        sample.group_index,
    ):
        if value is not None:
            return value
    return None


def _derive_max_stale_samples(args) -> int | None:
    staleness_threshold = getattr(args, "staleness_threshold", None)
    if staleness_threshold is None:
        return None
    return max(
        0,
        math.ceil(args.rollout_batch_size * args.update_weights_interval * (1 + staleness_threshold)),
    )


def _derive_max_completed_samples(args, max_stale_samples: int | None) -> int:
    configured_max_completed_samples = getattr(args, "fully_async_max_completed_samples", None)
    if configured_max_completed_samples is not None:
        return max(1, configured_max_completed_samples)
    return max(1000, (max_stale_samples or 0) + args.rollout_batch_size)


def _normalize_policy_version(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _extract_trajectory_versions_with_source(sample: Sample) -> tuple[str, list[int]]:
    versions = [_normalize_policy_version(version) for version in sample.weight_versions]
    versions = [version for version in versions if version is not None]
    if versions:
        return "weight_versions", versions

    scheduled_versions = [
        _normalize_policy_version(version) for version in sample.metadata.get("fully_async_schedule_versions", [])
    ]
    scheduled_versions = [version for version in scheduled_versions if version is not None]
    if scheduled_versions:
        return "fully_async_schedule_versions", scheduled_versions

    fallback_version = _normalize_policy_version(sample.metadata.get("policy_version"))
    return ("policy_version", [fallback_version]) if fallback_version is not None else ("none", [])


def _weight_version_to_policy_version(version: int) -> int:
    # Rollout engine weight_version starts from 1 after the initial sync while
    # training policy_version starts from 0, so align the fallback path here.
    return max(version - 1, 0)


def _extract_staleness_versions_with_source(sample: Sample) -> tuple[str, list[int]]:
    scheduled_versions = [
        _normalize_policy_version(version) for version in sample.metadata.get("fully_async_schedule_versions", [])
    ]
    scheduled_versions = [version for version in scheduled_versions if version is not None]
    if scheduled_versions:
        return "fully_async_schedule_versions", scheduled_versions

    fallback_version = _normalize_policy_version(sample.metadata.get("policy_version"))
    if fallback_version is not None:
        return "policy_version", [fallback_version]

    versions = [_normalize_policy_version(version) for version in sample.weight_versions]
    versions = [_weight_version_to_policy_version(version) for version in versions if version is not None]
    if versions:
        return "weight_versions", versions

    return ("none", [])


def _extract_trajectory_versions(sample: Sample) -> list[int]:
    _, versions = _extract_staleness_versions_with_source(sample)
    return versions


def _summarize_processed_group(group: list[Sample], current_policy_version: int) -> dict[str, object]:
    source_counts = Counter()
    staleness_source_counts = Counter()
    group_mins: list[int] = []
    group_maxs: list[int] = []
    stale_trajectory_count = 0
    trajectory_summaries = []

    for idx, sample in enumerate(group):
        source, raw_versions = _extract_trajectory_versions_with_source(sample)
        staleness_source, versions = _extract_staleness_versions_with_source(sample)
        source_counts[source] += 1
        staleness_source_counts[staleness_source] += 1
        max_version = max(versions) if versions else None
        min_version = min(versions) if versions else None
        is_stale = max_version is not None and current_policy_version - max_version >= 1
        if min_version is not None:
            group_mins.append(min_version)
        if max_version is not None:
            group_maxs.append(max_version)
        if is_stale:
            stale_trajectory_count += 1
        trajectory_summaries.append(
            {
                "trajectory_index": idx,
                "source": source,
                "versions": raw_versions,
                "staleness_source": staleness_source,
                "staleness_versions": versions,
                "weight_versions": list(sample.weight_versions),
                "schedule_versions": list(sample.metadata.get("fully_async_schedule_versions", [])),
                "policy_version": sample.metadata.get("policy_version"),
                "is_stale": is_stale,
            }
        )

    group_min_version = min(group_mins) if group_mins else None
    group_max_version = max(group_maxs) if group_maxs else None
    return {
        "sample_id": _extract_sample_id(group),
        "group_size": len(group),
        "group_min_version": group_min_version,
        "group_max_version": group_max_version,
        "stale_sample": group_max_version is not None and current_policy_version - group_max_version >= 1,
        "stale_trajectory_count": stale_trajectory_count,
        "partial_span": (
            max(0, group_max_version - group_min_version)
            if group_min_version is not None and group_max_version is not None
            else 0
        ),
        "source_counts": dict(source_counts),
        "staleness_source_counts": dict(staleness_source_counts),
        "trajectory_summaries": trajectory_summaries,
    }


def _log_processed_group_debug(
    args,
    groups: list[list[Sample]],
    *,
    current_policy_version: int,
    rollout_id: int,
    drained_group_count: int,
    leftover_group_count: int,
) -> None:
    if not getattr(args, "fully_async_debug_version_tracking", False):
        return

    summaries = [_summarize_processed_group(group, current_policy_version) for group in groups]
    stale_sample_count = sum(1 for summary in summaries if summary["stale_sample"])
    stale_trajectory_count = sum(int(summary["stale_trajectory_count"]) for summary in summaries)
    partial_group_count = sum(1 for summary in summaries if int(summary["partial_span"]) > 0)
    group_max_counter = Counter(
        summary["group_max_version"] for summary in summaries if summary["group_max_version"] is not None
    )
    source_counter = Counter()
    staleness_source_counter = Counter()
    for summary in summaries:
        source_counter.update(summary["source_counts"])
        staleness_source_counter.update(summary["staleness_source_counts"])

    print(
        "[fully_async_debug] "
        f"rollout_id={rollout_id}, current_policy_version={current_policy_version}, "
        f"drained_groups={drained_group_count}, selected_groups={len(groups)}, "
        f"leftover_completed_groups={leftover_group_count}, "
        f"stale_samples_in_selected={stale_sample_count}, "
        f"stale_trajectories_in_selected={stale_trajectory_count}, "
        f"partial_groups_in_selected={partial_group_count}, "
        f"group_max_versions={dict(sorted(group_max_counter.items()))}, "
        f"version_sources={dict(sorted(source_counter.items()))}, "
        f"staleness_version_sources={dict(sorted(staleness_source_counter.items()))}",
        flush=True,
    )

    for summary in summaries[:3]:
        print(
            "[fully_async_debug] "
            f"sample_id={summary['sample_id']}, group_size={summary['group_size']}, "
            f"group_min_version={summary['group_min_version']}, "
            f"group_max_version={summary['group_max_version']}, "
            f"stale_sample={summary['stale_sample']}, "
            f"stale_trajectory_count={summary['stale_trajectory_count']}, "
            f"partial_span={summary['partial_span']}, "
            f"source_counts={summary['source_counts']}, "
            f"staleness_source_counts={summary['staleness_source_counts']}, "
            f"trajectories={summary['trajectory_summaries']}",
            flush=True,
        )


def get_global_worker(args, data_buffer):
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def get_existing_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            return None
        return _global_worker


def stop_global_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


@dataclass
class CompletedSampleRecord:
    sample_id: int | None
    group: list[Sample]
    policy_version: int | None


class _CompletedStoreAdapter:
    def __init__(self, worker: "AsyncRolloutWorker"):
        self.worker = worker

    def put(self, item: tuple[int | None, list[Sample]]):
        sample_id, group = item
        self.worker._put_completed_sample(sample_id, group)

    def put_nowait(self, item: tuple[int | None, list[Sample]]):
        self.put(item)

    def qsize(self) -> int:
        return self.worker.get_queue_size()


class AsyncRolloutWorker:
    """
    Background rollout worker with weight-sync hooks.

    Compared with the original example, this version adds:
    - pause / resume around parameter synchronization
    - optional partial rollout recycling before weight updates
    - stale backlog accounting controlled by `staleness_threshold`
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.max_stale_samples = _derive_max_stale_samples(args)
        self.buffer_policy = getattr(args, "fully_async_buffer_policy", "legacy_backpressure")
        self.version_window = max(0, getattr(args, "fully_async_version_window", 1))
        self.max_partial_span = getattr(args, "fully_async_max_partial_span", None)
        self.eviction_policy = getattr(args, "fully_async_eviction_policy", "drop_oldest_version")
        self.max_completed_samples = _derive_max_completed_samples(args, self.max_stale_samples)
        self.completed_lock = threading.Lock()
        self.completed_records: list[CompletedSampleRecord] = []
        self.output_queue = _CompletedStoreAdapter(self)
        self.worker_thread = None
        self.state = GenerateState(args)
        self.loop = None
        self.task_lock: asyncio.Lock | None = None

        self.max_concurrent_tasks = self.args.rollout_batch_size
        self.sample_id_counter = -1
        self.task_sample_ids: dict[asyncio.Task, int | None] = {}
        self.inflight_groups: dict[asyncio.Task, list[Sample]] = {}
        self.policy_version = getattr(args, "current_policy_version", 0)
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0
        self.consumed_samples = 0
        self.recycled_samples = 0
        self.dropped_samples = 0
        self.evicted_samples = 0
        self.evicted_by_version = 0
        self.held_for_partial_span = 0
        self._window_total_samples = 0
        self._window_partial_samples = 0
        self._window_max_partial_span = 0

        self.control_lock = threading.Lock()
        self.pause_requested = False
        # Staleness budget for the current version window. It is reset from the
        # outstanding old-version snapshot on weight updates, then grows as new
        # samples are pulled under the current version. Trainer consumption
        # refunds this budget, while recycled partial samples remain counted.
        self.stale_sample_ids: set[int] = set()
        self.recycled_sample_ids: set[int] = set()

    def monitor_queue_size(self):
        while self.running:
            try:
                queue_size = self.get_queue_size()
                wandb.log({
                    "queue/output_queue_size": queue_size,
                    "queue/queue_utilization": queue_size / max(1, self.max_completed_samples)
                })
                time.sleep(5)  
            except Exception as e:
                print(f"Error monitoring queue: {e}")
                time.sleep(5)

    def _uses_window_evict_policy(self) -> bool:
        return self.buffer_policy == "window_evict"

    def _version_window_bounds(self, current_policy_version: int | None = None) -> tuple[int | None, int | None]:
        if not self._uses_window_evict_policy():
            return None, None
        max_policy_version = self.policy_version if current_policy_version is None else current_policy_version
        return max_policy_version - self.version_window, max_policy_version

    def _extract_group_policy_version(self, group: list[Sample]) -> int | None:
        group_max_version = None
        for sample in group:
            versions = _extract_trajectory_versions(sample)
            if not versions:
                continue
            sample_max_version = max(versions)
            if group_max_version is None:
                group_max_version = sample_max_version
            else:
                group_max_version = max(group_max_version, sample_max_version)
        return group_max_version

    def _extract_group_partial_span(self, group: list[Sample]) -> int:
        group_mins = []
        group_maxs = []
        for sample in group:
            versions = _extract_trajectory_versions(sample)
            if not versions:
                continue
            group_mins.append(min(versions))
            group_maxs.append(max(versions))
        if group_mins and group_maxs:
            return max(0, max(group_maxs) - min(group_mins))
        return 0

    def _has_inflight_exceeded_partial_span(self) -> bool:
        if self.max_partial_span is None:
            return False
        for group in list(self.inflight_groups.values()):
            if self._extract_group_partial_span(group) > self.max_partial_span:
                return True
        return False

    def _snapshot_completed_records(self) -> list[CompletedSampleRecord]:
        with self.completed_lock:
            return list(self.completed_records)

    def _drop_sample_tracking_locked(self, sample_id: int | None):
        if sample_id is None:
            return
        self.stale_sample_ids.discard(sample_id)
        self.recycled_sample_ids.discard(sample_id)

    def _record_evicted_records_locked(
        self,
        records: list[CompletedSampleRecord],
        *,
        version_eviction: bool,
    ) -> None:
        if not records:
            return
        with self.control_lock:
            self.evicted_samples += len(records)
            if version_eviction:
                self.evicted_by_version += len(records)
            for record in records:
                self._drop_sample_tracking_locked(record.sample_id)

    def _evict_records_outside_window_locked(self, current_policy_version: int | None = None) -> int:
        if not self._uses_window_evict_policy():
            return 0
        min_policy_version, max_policy_version = self._version_window_bounds(current_policy_version)
        retained_records = []
        evicted_records = []
        for record in self.completed_records:
            if (
                record.policy_version is not None
                and min_policy_version is not None
                and max_policy_version is not None
                and (record.policy_version < min_policy_version or record.policy_version > max_policy_version)
            ):
                evicted_records.append(record)
            else:
                retained_records.append(record)
        if evicted_records:
            self.completed_records = retained_records
            self._record_evicted_records_locked(evicted_records, version_eviction=True)
        return len(evicted_records)

    def _select_overflow_eviction_index_locked(self) -> int:
        if self.eviction_policy == "drop_oldest_fifo":
            return 0
        oldest_index = 0
        oldest_version = None
        for idx, record in enumerate(self.completed_records):
            candidate_version = -1 if record.policy_version is None else record.policy_version
            if oldest_version is None or candidate_version < oldest_version:
                oldest_index = idx
                oldest_version = candidate_version
        return oldest_index

    def _trim_completed_records_locked(self) -> int:
        if self.max_completed_samples <= 0:
            return 0
        trimmed_records = []
        while len(self.completed_records) > self.max_completed_samples:
            eviction_index = self._select_overflow_eviction_index_locked()
            trimmed_records.append(self.completed_records.pop(eviction_index))
        if trimmed_records:
            self._record_evicted_records_locked(trimmed_records, version_eviction=False)
        return len(trimmed_records)

    def _put_completed_sample(self, sample_id: int | None, group: list[Sample]) -> None:
        record = CompletedSampleRecord(
            sample_id=sample_id,
            group=group,
            policy_version=self._extract_group_policy_version(group),
        )
        with self.completed_lock:
            if not self._uses_window_evict_policy() and len(self.completed_records) >= self.max_completed_samples:
                with self.control_lock:
                    self.dropped_samples += 1
                raise queue.Full
            self.completed_records.append(record)
            if self._uses_window_evict_policy():
                self._evict_records_outside_window_locked(self.policy_version)
                self._trim_completed_records_locked()

    def _set_pause_requested(self, value: bool):
        with self.control_lock:
            self.pause_requested = value

    def _is_pause_requested(self) -> bool:
        with self.control_lock:
            return self.pause_requested

    def _set_stale_sample_ids(self, sample_ids: Iterable[int | None]):
        with self.control_lock:
            self.stale_sample_ids = {sample_id for sample_id in sample_ids if sample_id is not None}

    def _add_stale_sample_id(self, sample_id: int | None):
        if sample_id is None:
            return
        with self.control_lock:
            self.stale_sample_ids.add(sample_id)

    def _add_recycled_sample_ids(self, sample_ids: Iterable[int | None]):
        with self.control_lock:
            self.recycled_sample_ids.update(sample_id for sample_id in sample_ids if sample_id is not None)

    def _discard_recycled_sample_id(self, sample_id: int | None):
        if sample_id is None:
            return
        with self.control_lock:
            self.recycled_sample_ids.discard(sample_id)

    def _snapshot_recycled_sample_ids(self) -> set[int]:
        with self.control_lock:
            return set(self.recycled_sample_ids)

    def _get_stale_sample_count_locked(self) -> int:
        return len(self.stale_sample_ids | self.recycled_sample_ids)

    def get_stale_sample_count(self) -> int:
        with self.control_lock:
            return self._get_stale_sample_count_locked()

    def _should_pause_for_staleness(self) -> bool:
        if self._uses_window_evict_policy():
            return False
        return self.max_stale_samples is not None and self.get_stale_sample_count() >= self.max_stale_samples

    def _buffered_sample_ids(self) -> list[int | None]:
        return [record.sample_id for record in self._snapshot_completed_records()]

    def _current_stale_sample_ids(self) -> set[int]:
        return (
            {sample_id for sample_id in self._buffered_sample_ids() if sample_id is not None}
            | {sample_id for sample_id in self.task_sample_ids.values() if sample_id is not None}
            | self._snapshot_recycled_sample_ids()
        )

    def _mark_sample_consumed(self, sample_id: int | None):
        if sample_id is None:
            return
        with self.control_lock:
            self.consumed_samples += 1
            self._drop_sample_tracking_locked(sample_id)

    def _mark_sample_recycled(self, sample_id: int | None):
        if sample_id is None:
            return
        with self.control_lock:
            self.stale_sample_ids.discard(sample_id)
            self.recycled_sample_ids.add(sample_id)
            self.recycled_samples += 1

    def _enqueue_sample(self, sample_id: int | None, group: list[Sample]) -> bool:
        try:
            self.output_queue.put_nowait((sample_id, group))
            return True
        except queue.Full:
            print(f"WARNING: output queue full, dropping sample {sample_id}")
            return False

    def _collect_task_result(self, task: asyncio.Task) -> bool:
        sample_id = self.task_sample_ids.pop(task, None)
        self.inflight_groups.pop(task, None)
        try:
            group = task.result()
        except Exception as exc:
            print(f"Task failed with exception: {exc}")
            return False
        if sample_id is None:
            sample_id = _extract_sample_id(group)
        self._discard_recycled_sample_id(sample_id)
        self._enqueue_sample(sample_id, group)
        return True

    def _annotate_sample(self, sample_id: int | None, group: list[Sample]):
        current_rollout_id = getattr(self.args, "current_rollout_id", -1)
        for sample in group:
            schedule_versions = sample.metadata.setdefault("fully_async_schedule_versions", [])
            if not schedule_versions or schedule_versions[-1] != self.policy_version:
                schedule_versions.append(self.policy_version)
            if sample_id is not None:
                sample.metadata["fully_async_sample_id"] = sample_id
                sample.metadata["fully_async_group_id"] = sample_id
            sample.metadata["policy_version"] = self.policy_version
            sample.metadata.setdefault("start_rollout_id", current_rollout_id)

    async def _push_finished_samples_to_output_queue(self):
        assert self.task_lock is not None
        done_tasks = {task for task in self.state.pendings if task.done()}
        for task in done_tasks:
            self.state.pendings.remove(task)
            self._collect_task_result(task)

    async def _wait_for_all_active_tasks(self):
        while self.state.pendings:
            done_tasks, pending = await asyncio.wait(self.state.pendings, return_when=asyncio.FIRST_COMPLETED)
            self.state.pendings = pending
            for task in done_tasks:
                self._collect_task_result(task)

    def _snapshot_completed_store_metrics(self) -> dict[str, int]:
        records = self._snapshot_completed_records()
        min_policy_version, max_policy_version = self._version_window_bounds()
        policy_versions = [record.policy_version for record in records if record.policy_version is not None]
        eligible_samples = 0
        for record in records:
            if (
                not self._uses_window_evict_policy()
                or record.policy_version is None
                or (
                    min_policy_version is not None
                    and max_policy_version is not None
                    and min_policy_version <= record.policy_version <= max_policy_version
                )
            ):
                eligible_samples += 1
        return {
            "fully_async/window/completed_store_size": len(records),
            "fully_async/window/eligible_samples": eligible_samples,
            "fully_async/window/version_span": (max(policy_versions) - min(policy_versions) if policy_versions else 0),
        }

    def _reset_interval_metrics_locked(self) -> None:
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0
        self.consumed_samples = 0
        self.recycled_samples = 0
        self.dropped_samples = 0
        self.evicted_samples = 0
        self.evicted_by_version = 0
        self._window_total_samples = 0
        self._window_partial_samples = 0
        self._window_max_partial_span = 0

    def _has_interval_metrics(self) -> bool:
        with self.control_lock:
            return any(
                [
                    self.stale_samples_processed,
                    self.stale_trajectory_processed,
                    self.consumed_samples,
                    self.recycled_samples,
                    self.dropped_samples,
                    self.evicted_samples,
                    self.evicted_by_version,
                    self._window_total_samples,
                    self._window_partial_samples,
                    self._window_max_partial_span,
                ]
            )

    def _snapshot_processed_metrics(self) -> dict[str, float | int]:
        with self.control_lock:
            total = self._window_total_samples
            partial = self._window_partial_samples
            metrics = {
                "fully_async/count/stale_samples_processed": self.stale_samples_processed,
                "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                "fully_async/count/consumed_samples": self.consumed_samples,
                "fully_async/count/recycled_samples": self.recycled_samples,
                "fully_async/count/dropped_samples": self.dropped_samples,
                "fully_async/partial/total_partial_num": partial,
                "fully_async/partial/partial_ratio": partial / total if total else 0.0,
                "fully_async/partial/max_partial_span": self._window_max_partial_span,
                "fully_async/window/evicted_samples": self.evicted_samples,
                "fully_async/window/evicted_by_version": self.evicted_by_version,
            }
        metrics.update(self._snapshot_completed_store_metrics())
        return metrics

    def record_processed_samples(self, groups: list[list[Sample]]) -> None:
        current_policy_version = self.policy_version
        with self.control_lock:
            for group in groups:
                trajectory_mins: list[int] = []
                trajectory_maxs: list[int] = []
                stale_trajectory_count = 0

                for sample in group:
                    versions = _extract_trajectory_versions(sample)
                    if not versions:
                        continue
                    trajectory_mins.append(min(versions))
                    trajectory_maxs.append(max(versions))
                    if current_policy_version - max(versions) >= 1:
                        stale_trajectory_count += 1

                sample_min_version = min(trajectory_mins) if trajectory_mins else None
                sample_max_version = max(trajectory_maxs) if trajectory_maxs else None
                if sample_max_version is not None and current_policy_version - sample_max_version >= 1:
                    self.stale_samples_processed += 1
                self.stale_trajectory_processed += stale_trajectory_count

                partial_span = 0
                if sample_min_version is not None and sample_max_version is not None:
                    partial_span = max(0, sample_max_version - sample_min_version)

                self._window_total_samples += 1
                if partial_span > 0:
                    self._window_partial_samples += 1
                    self._window_max_partial_span = max(self._window_max_partial_span, partial_span)

    async def _prepare_for_weight_update_async(self, policy_version: int):
        assert self.task_lock is not None
        async with self.task_lock:
            await self._push_finished_samples_to_output_queue()
            active_samples_before = len(self.state.pendings)

            if active_samples_before == 0:
                stale_samples = len(self._current_stale_sample_ids())
                return {
                    "policy_version": policy_version,
                    "active_samples": 0,
                    "stale_samples": stale_samples,
                }

            if self.args.partial_rollout:
                aborted_groups = await abort(self.args, getattr(self.args, "current_rollout_id", -1))
                if aborted_groups:
                    self.data_buffer.add_samples(aborted_groups)
                    self._add_recycled_sample_ids(_extract_sample_id(group) for group in aborted_groups)
                self.task_sample_ids.clear()
                self.inflight_groups.clear()
                self.state.reset()
            else:
                await self._wait_for_all_active_tasks()

            stale_samples = len(self._current_stale_sample_ids())
            return {
                "policy_version": policy_version,
                "active_samples": active_samples_before,
                "stale_samples": stale_samples,
            }

    def _evict_completed_records_outside_window(self, current_policy_version: int | None = None) -> int:
        with self.completed_lock:
            return self._evict_records_outside_window_locked(current_policy_version)

    async def _finish_weight_update_async(self, policy_version: int):
        assert self.task_lock is not None
        async with self.task_lock:
            await self._push_finished_samples_to_output_queue()
            with self.control_lock:
                self.policy_version = policy_version
                self.state.aborted = False
            self._evict_completed_records_outside_window(policy_version)
            self._set_stale_sample_ids(self._current_stale_sample_ids())
            self._set_pause_requested(False)
            interval_metrics = self._snapshot_processed_metrics()
            with self.control_lock:
                self._reset_interval_metrics_locked()
            return {
                "policy_version": policy_version,
                "stale_samples": self.get_stale_sample_count(),
                "max_stale_samples": self.max_stale_samples,
                **interval_metrics,
            }

    async def run_eval(self, args, rollout_id):
        async with self.task_lock:
            await self._push_finished_samples_to_output_queue()
            await self._wait_for_all_active_tasks()
        # Temporarily reduce semaphore to prevent OOM during eval.
        # eval_rollout creates per-sample tasks (prompt × n_samples_per_eval_prompt)
        # that each acquire the semaphore individually.  Unlike training where
        # groups start/finish at staggered times, eval tasks launch simultaneously
        # and generate very long sequences, so we need a much lower limit.
        eval_concurrency = self.max_concurrent_tasks * self.args.n_samples_per_prompt

        original_semaphore = self.state.semaphore
        self.state.semaphore = asyncio.Semaphore(eval_concurrency)
        try:
            output, _ = await eval_rollout(args, rollout_id)
        finally:
            self.state.semaphore = original_semaphore
        return output

    async def _shutdown_async(self):
        assert self.task_lock is not None
        async with self.task_lock:
            await self._push_finished_samples_to_output_queue()
            if self.state.pendings:
                try:
                    await abort(self.args, getattr(self.args, "current_rollout_id", -1))
                except Exception as exc:
                    print(f"Failed to abort pending rollout tasks during shutdown: {exc}")
            self.task_sample_ids.clear()
            self.inflight_groups.clear()
            self.state.reset()

    async def continuous_worker_loop(self):
        print("Continuous async rollout worker started")
        self.loop = asyncio.get_running_loop()
        print(f"Continuous async rollout worker event loop: {type(self.loop).__module__}.{type(self.loop).__name__}")
        self.task_lock = asyncio.Lock()

        while self.running:
            try:
                async with self.task_lock:
                    await self._push_finished_samples_to_output_queue()

                if self._is_pause_requested() or self._should_pause_for_staleness():
                    await asyncio.sleep(0.05)
                    continue

                while (
                    len(self.state.pendings) < self.max_concurrent_tasks
                    and self.running
                    and not self._is_pause_requested()
                    and not self._should_pause_for_staleness()
                    and not self._has_inflight_exceeded_partial_span()
                ):
                    samples = self.data_buffer.get_samples(1)
                    if not samples:
                        break

                    group = samples[0]
                    sample_id = _extract_sample_id(group)
                    if sample_id is None:
                        sample_id = self.sample_id_counter
                        self.sample_id_counter -= 1
                    self._add_stale_sample_id(sample_id)
                    self._annotate_sample(sample_id, group)

                    task = asyncio.create_task(
                        generate_and_rm_group(
                            self.args,
                            group,
                            sampling_params=self.state.sampling_params.copy(),
                            evaluation=False,
                        )
                    )
                    self.state.pendings.add(task)
                    self.task_sample_ids[task] = sample_id
                    self.inflight_groups[task] = group
                    self._discard_recycled_sample_id(sample_id)

                await asyncio.sleep(0.01)

            except Exception as exc:
                print(f"Error in continuous worker loop: {exc}")
                await asyncio.sleep(0.1)

        if self.task_lock is not None:
            async with self.task_lock:
                await self._wait_for_all_active_tasks()

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.continuous_worker_loop())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            self.loop = None
            asyncio.set_event_loop(None)
            loop.close()

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print(
                "Started continuous async worker thread "
                f"(max_stale_samples={self.max_stale_samples}, partial_rollout={self.args.partial_rollout})"
            )
            self.monitor_thread = threading.Thread(target=self.monitor_queue_size, daemon=True)
            self.monitor_thread.start()
            print("Started queue monitoring thread")

    def stop(self):
        self._set_pause_requested(True)
        if self.loop is not None and self.worker_thread and self.worker_thread.is_alive():
            try:
                future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self.loop)
                future.result(timeout=30)
            except Exception as exc:
                print(f"Failed to shutdown async worker cleanly: {exc}")
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5)
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
        print("Stopped async worker thread")

    def _needs_weight_update_sync(self) -> bool:
        return self.max_stale_samples is not None or self.args.partial_rollout or self._uses_window_evict_policy()

    def before_weight_update(self, policy_version: int):
        if not self._needs_weight_update_sync():
            return {
                "policy_version": policy_version,
                "active_samples": len(self.state.pendings) if self.state else 0,
                "stale_samples": self.get_stale_sample_count(),
                "skipped_sync": True,
            }
        self._set_pause_requested(True)
        if self.loop is None:
            stale_samples = len(self._current_stale_sample_ids())
            return {
                "policy_version": policy_version,
                "active_samples": 0,
                "stale_samples": stale_samples,
            }
        future = asyncio.run_coroutine_threadsafe(self._prepare_for_weight_update_async(policy_version), self.loop)
        return future.result()

    def after_weight_update(self, policy_version: int):
        if not self._needs_weight_update_sync():
            with self.control_lock:
                self.policy_version = policy_version
            interval_metrics = self._snapshot_processed_metrics()
            with self.control_lock:
                self._reset_interval_metrics_locked()
            return {
                "policy_version": policy_version,
                "stale_samples": self.get_stale_sample_count(),
                "max_stale_samples": self.max_stale_samples,
                "skipped_sync": True,
                **interval_metrics,
            }
        if self.loop is None:
            with self.control_lock:
                self.policy_version = policy_version
                self.state.aborted = False
            self._evict_completed_records_outside_window(policy_version)
            self._set_stale_sample_ids(self._current_stale_sample_ids())
            self._set_pause_requested(False)
            interval_metrics = self._snapshot_processed_metrics()
            with self.control_lock:
                self._reset_interval_metrics_locked()
            stale_samples = self.get_stale_sample_count()
            return {
                "policy_version": policy_version,
                "stale_samples": stale_samples,
                "max_stale_samples": self.max_stale_samples,
                **interval_metrics,
            }
        future = asyncio.run_coroutine_threadsafe(self._finish_weight_update_async(policy_version), self.loop)
        return future.result()

    def get_completed_samples(
        self,
        limit: int | None = None,
        *,
        current_policy_version: int | None = None,
    ) -> list[tuple[int | None, list[Sample]]]:
        if limit is not None and limit <= 0:
            return []
        with self.completed_lock:
            self._evict_records_outside_window_locked(current_policy_version)
            if limit is None:
                selected_records = list(self.completed_records)
                self.completed_records.clear()
            else:
                selected_records = self.completed_records[:limit]
                del self.completed_records[:limit]
        return [(record.sample_id, record.group) for record in selected_records]

    def get_completed_groups(self) -> list[tuple[int | None, list[Sample]]]:
        return self.get_completed_samples()

    def get_queue_size(self) -> int:
        with self.completed_lock:
            return len(self.completed_records)


def before_weight_update(args, data_buffer, policy_version: int):
    worker = get_existing_worker()
    if worker is None:
        return {
            "policy_version": policy_version,
            "active_samples": 0,
            "stale_samples": 0,
        }
    return worker.before_weight_update(policy_version)


def after_weight_update(args, data_buffer, policy_version: int):
    worker = get_existing_worker()
    if worker is None:
        max_stale_samples = _derive_max_stale_samples(args)
        return {
            "policy_version": policy_version,
            "stale_samples": 0,
            "max_stale_samples": max_stale_samples,
        }
    return worker.after_weight_update(policy_version)


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset

    worker = get_global_worker(args, data_buffer)
    target_sample_count = args.rollout_batch_size

    data = []
    completed_samples = {}
    drained_group_count = 0
    do_print = True

    print(f"Starting async rollout generation for {target_sample_count} samples")
    print("Global worker queue size: " f"{worker.get_queue_size()}, stale_samples={worker.get_stale_sample_count()}")

    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0

    while len(data) < target_sample_count:
        pending_capacity = max(0, target_sample_count - len(data) - len(completed_samples))
        completed = worker.get_completed_samples(limit=pending_capacity, current_policy_version=worker.policy_version)
        drained_group_count += len(completed)

        made_progress = False
        for sample_id, group in completed:
            completed_samples[sample_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        processed_any = False
        for sample_id in list(completed_samples.keys()):
            if len(data) >= target_sample_count:
                break

            group = completed_samples.pop(sample_id)

            try:
                any_aborted = any(sample.status == Sample.Status.ABORTED for sample in group)
            except Exception:
                any_aborted = False

            # 修改
            if any_aborted:
                if getattr(args, "partial_rollout", False):
                    try:
                        data_buffer.add_samples([group])
                        worker._mark_sample_recycled(sample_id)
                        print(f"Returned aborted sample {sample_id} to data buffer", flush=True)
                    except Exception as exc:
                        print(f"Failed to return aborted sample {sample_id} to buffer: {exc}", flush=True)
                    continue
                else:
                    worker._mark_sample_consumed(sample_id)
                    print(f"Dropped aborted sample {sample_id} directly because partial_rollout is False", flush=True)
                    continue

            if do_print:
                print(
                    f"First rollout sample: {[_sample_text_preview(group[0])]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            data.append(group)
            worker._mark_sample_consumed(sample_id)
            processed_any = True

        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Stale samples: {worker.get_stale_sample_count()}, "
                f"Collected: {len(data)}/{target_sample_count}"
            )
            last_progress_time = current_time

        if not processed_any:
            await asyncio.sleep(0.01)

    try:
        print("record tool call counts for analysis")
        tool_time_counts = []
        sample_time_counts = []
        metrics_to_log = {}
        task_types = []
        math_samples = []  
        qa_samples = [] 

        tool_time_ratios = []

        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_time'):
                    tool_time_counts.append(sample.tool_time)
                if hasattr(sample, 'sample_time'):
                    sample_time_counts.append(sample.sample_time)
                if hasattr(sample, 'tool_time') and hasattr(sample, 'sample_time'):
                    tool_time_ratios.append(sample.tool_time / sample.sample_time if sample.sample_time > 0 else 0.0)

                if hasattr(sample, 'metadata'):
                    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
                    task_type = metadata.get("task_type", "error")
                    task_types.append(task_type)
                    if task_type == "math":
                        math_samples.append(sample)
                    elif task_type == "qa":
                        qa_samples.append(sample)

        if task_types:
            total_math = len(math_samples)
            total_qa = len(qa_samples)

            def _get_reward_score(r):
                if r is None: return 0.0
                if isinstance(r, dict): return r.get("score", 0.0)
                if isinstance(r, (float, int)): return r
                if hasattr(r, "item"): return r.item()
                return 0.0

            def _get_reward_acc(r):
                if r is None:
                    return None
                if isinstance(r, dict):
                    if "acc" in r:
                        return float(r["acc"])
                    if "score" in r:
                        return 1.0 if r["score"] >= 0.8 else 0.0
                    return None
                if isinstance(r, (float, int)):
                    return 1.0 if r >= 0.8 else 0.0
                if hasattr(r, "item"):
                    return 1.0 if r.item() >= 0.8 else 0.0
                return None

            def _average_acc(samples):
                accs = [_get_reward_acc(s.reward) for s in samples]
                accs = [acc for acc in accs if acc is not None]
                return sum(accs) / len(accs) if accs else 0

            math_reward_avg = sum(_get_reward_score(s.reward) for s in math_samples) / total_math if total_math > 0 else 0
            qa_reward_avg = sum(_get_reward_score(s.reward) for s in qa_samples) / total_qa if total_qa > 0 else 0
            math_acc_avg = _average_acc(math_samples)
            qa_acc_avg = _average_acc(qa_samples)

            math_tool_call_counts = [getattr(s, "tool_call_count", 0) or 0 for s in math_samples]
            qa_tool_call_counts = [getattr(s, "tool_call_count", 0) or 0 for s in qa_samples]
            math_tool_calls_ratio = (
                sum(1 for count in math_tool_call_counts if count > 0) / total_math if total_math > 0 else 0.0
            )
            qa_tool_calls_ratio = (
                sum(1 for count in qa_tool_call_counts if count > 0) / total_qa if total_qa > 0 else 0.0
            )

            # Calculate math task sample_time stats
            math_sample_times = [s.sample_time for s in math_samples if hasattr(s, 'sample_time')]
            math_sample_time_max = max(math_sample_times) if math_sample_times else 0
            math_sample_time_avg = sum(math_sample_times) / len(math_sample_times) if math_sample_times else 0
            
            # Calculate qa task sample_time stats
            qa_sample_times = [s.sample_time for s in qa_samples if hasattr(s, 'sample_time')]
            qa_sample_time_max = max(qa_sample_times) if qa_sample_times else 0
            qa_sample_time_avg = sum(qa_sample_times) / len(qa_sample_times) if qa_sample_times else 0
            
            # Calculate math task response_length stats
            math_response_lengths = [s.response_length for s in math_samples if hasattr(s, 'response_length')]
            math_response_length_max = max(math_response_lengths) if math_response_lengths else 0
            math_response_length_avg = sum(math_response_lengths) / len(math_response_lengths) if math_response_lengths else 0
            
            # Calculate qa task response_length stats
            qa_response_lengths = [s.response_length for s in qa_samples if hasattr(s, 'response_length')]
            qa_response_length_max = max(qa_response_lengths) if qa_response_lengths else 0
            qa_response_length_avg = sum(qa_response_lengths) / len(qa_response_lengths) if qa_response_lengths else 0
            
            metrics_to_log.update({
                "tool/math_count": total_math,
                "tool/qa_count": total_qa,
                "tool/math_reward": math_reward_avg,
                "tool/qa_reward": qa_reward_avg,
                "tool/math_acc": math_acc_avg,
                "tool/qa_acc": qa_acc_avg,
                "tool/avg_code_call_count": sum(math_tool_call_counts) / total_math if total_math > 0 else 0.0,
                "tool/avg_search_call_count": sum(qa_tool_call_counts) / total_qa if total_qa > 0 else 0.0,
                "tool/math_tool_calls_ratio": math_tool_calls_ratio,
                "tool/qa_tool_calls_ratio": qa_tool_calls_ratio,
                "tool/math_sample_time_max": math_sample_time_max,
                "tool/math_sample_time_avg": math_sample_time_avg,
                "tool/qa_sample_time_max": qa_sample_time_max,
                "tool/qa_sample_time_avg": qa_sample_time_avg,
                "tool/math_response_length_max": math_response_length_max,
                "tool/math_response_length_avg": math_response_length_avg,
                "tool/qa_response_length_max": qa_response_length_max,
                "tool/qa_response_length_avg": qa_response_length_avg,
            })

        if tool_time_counts:
            avg_tool_times = sum(tool_time_counts) / len(tool_time_counts)
            avg_sample_times = sum(sample_time_counts) / len(sample_time_counts) if sample_time_counts else 0.0

            metrics_to_log.update({
                "tool/avg_tool_calls_time": avg_tool_times,
                "tool/avg_sample_time": avg_sample_times,
                "tool/avg_tool_time_ratio_per_sample": sum(tool_time_ratios) / len(tool_time_ratios) if tool_time_ratios else 0.0,
            })
            
        
        tool_call_counts = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_call_count'):
                    tool_call_counts.append(sample.tool_call_count)
        
        if tool_call_counts:
            avg_tool_calls = sum(tool_call_counts) / len(tool_call_counts)
            samples_with_tool_calls = sum(1 for count in tool_call_counts if count > 0)

            metrics_to_log.update({
                "tool/avg_tool_calls_per_sample": avg_tool_calls,
                "tool/total_tool_calls": sum(tool_call_counts),
                "tool/samples_with_tool_calls": samples_with_tool_calls,
            })

        tool_token_counts = []
        response_lengths = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'tool_token_count'):
                    tool_token_counts.append(sample.tool_token_count)
                    response_lengths.append(sample.response_length)

        if tool_token_counts:
            total_tool_tokens = sum(tool_token_counts)
            avg_tool_tokens = total_tool_tokens / len(tool_token_counts)
            per_sample_ratios = [
                t / r if r > 0 else 0.0
                for t, r in zip(tool_token_counts, response_lengths)
            ]
            tool_token_ratio = sum(per_sample_ratios) / len(per_sample_ratios)
            metrics_to_log.update({
                "tool/avg_tool_tokens_per_sample": avg_tool_tokens,
                "tool/tool_token_ratio_in_response": tool_token_ratio,
            })
        
        mismatch_counts = []
        for group in data:
            for sample in group:
                if hasattr(sample, 'mismatch'):
                    mismatch_counts.append(sample.mismatch)
         

        if mismatch_counts:
            total_mismatches = sum(mismatch_counts)
            avg_mismatches = total_mismatches / len(mismatch_counts)
            samples_with_mismatches = sum(1 for m in mismatch_counts if m > 0)

            metrics_to_log.update({
                "debug/total_mismatches": total_mismatches,
                "debug/avg_mismatches_per_sample": avg_mismatches,
                "debug/samples_with_mismatches": samples_with_mismatches,
            })

        # Use a dedicated rollout step axis to avoid conflicts with other threads logging wandb step.
        if metrics_to_log:
            global _wandb_metric_defined
            if not _wandb_metric_defined:
                wandb.define_metric("rollout/step")
                wandb.define_metric("tool/*", step_metric="rollout/step")
                wandb.define_metric("debug/*", step_metric="rollout/step")
                _wandb_metric_defined = True

            metrics_to_log["rollout/step"] = rollout_id
            wandb.log(metrics_to_log)
    except Exception as e:
        import traceback
        print(f"[WARNING] Failed to log rollout metrics to wandb: {e}")
        traceback.print_exc()
    
    duration = time.time() - start_time
    print(
        f"Rollout completed in {duration:.2f}s! "
        f"Global worker queue size: {worker.get_queue_size()}, "
        f"stale_samples={worker.get_stale_sample_count()}"
    )

    if data:
        print(
            f"Finish rollout: {[_sample_text_preview(data[-1][0])]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    _log_processed_group_debug(
        args,
        data,
        current_policy_version=worker.policy_version,
        rollout_id=rollout_id,
        drained_group_count=drained_group_count,
        leftover_group_count=len(completed_samples),
    )
    worker.record_processed_samples(data)
    return RolloutFnTrainOutput(samples=data, metrics={})


def flush_metrics(args, data_buffer):
    worker = get_existing_worker()
    if worker is None:
        return None
    if not worker._has_interval_metrics():
        return None
    return worker._snapshot_processed_metrics()


def shutdown_worker(args, data_buffer):
    stop_global_worker()


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        worker = get_existing_worker()
        if worker is not None and worker.loop is not None:
            worker._set_pause_requested(True)
            try:
                future = asyncio.run_coroutine_threadsafe(worker.run_eval(args, rollout_id), worker.loop)
                output = future.result()
            finally:
                worker._set_pause_requested(False)
        else:
            output, _ = run(eval_rollout(args, rollout_id))
        return output

    return run(generate_rollout_async(args, rollout_id, data_buffer))


atexit.register(stop_global_worker)
