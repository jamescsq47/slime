#!/usr/bin/env python3
"""Event simulation calibrated against a real fully-async W&B run.

The completed-store unit is a prompt group (8 trajectories in lgdzo8cx),
matching ``CompletedSampleRecord`` in fully_async_rollout.py.
"""

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE = HERE / "debug" / "wandb_lgdzo8cx_sim_profile.json"
CALIBRATED_GROUP_TIME_SCALE = 1.0
CALIBRATED_DISPERSION = 1.0
CALIBRATED_TAIL_THRESHOLD = 1.5
CALIBRATED_TAIL_SCALE = 1.0
CALIBRATED_FAST_FRACTION = 0.0
CALIBRATED_FAST_SCALE = 1.0
CALIBRATED_SLOW_SCALE = 1.0
# W&B completed-store overflow is an observable marker for fast rollout
# regimes. These two factors jointly match both queue eviction and wall time;
# using only a global mean cannot reproduce the real burst/long-tail mixture.
CALIBRATED_BURST_SCALE = 0.96
CALIBRATED_QUIET_SCALE = 1.23
CALIBRATED_PHASE_SCALES = ((0, 0.95), (20, 0.90), (40, 1.15), (60, 1.0))
SAMPLE_FILES = {
    "math": HERE / "debug" / "profile" / "profile_math_samples_rollout0.jsonl",
    # Consolidated 192-group BrowseComp profile (8 trajectories per group).
    "qa": HERE.parent / "mixed" / "debug" / "profile_browsecomp_0_192_samples.jsonl",
}


@dataclass
class Group:
    seq: int
    task: str
    version: int
    origin_version: int
    started_at: float
    finishes_at: float


class DynamicTimePool:
    def __init__(
        self,
        profile,
        rng,
        radius=0,
        group_time_scale=CALIBRATED_GROUP_TIME_SCALE,
        dispersion=CALIBRATED_DISPERSION,
        tail_threshold=CALIBRATED_TAIL_THRESHOLD,
        tail_scale=CALIBRATED_TAIL_SCALE,
        fast_fraction=CALIBRATED_FAST_FRACTION,
        fast_scale=CALIBRATED_FAST_SCALE,
        slow_scale=CALIBRATED_SLOW_SCALE,
        burst_scale=CALIBRATED_BURST_SCALE,
        quiet_scale=CALIBRATED_QUIET_SCALE,
        train_time_profile=None,
    ):
        self.steps = profile["steps"]
        self.train_time_steps = (train_time_profile or profile)["steps"]
        self.rng = rng
        self.radius = radius
        self.group_time_scale = group_time_scale
        self.dispersion = dispersion
        self.tail_threshold = tail_threshold
        self.tail_scale = tail_scale
        self.fast_fraction = fast_fraction
        self.fast_scale = fast_scale
        self.slow_scale = slow_scale
        self.burst_scale = burst_scale
        self.quiet_scale = quiet_scale
        self.base = {task: self._load_samples(path) for task, path in SAMPLE_FILES.items()}
        self.base_mean = {task: float(np.mean(values)) for task, values in self.base.items()}
        self.base_moment = {
            task: float(np.mean((values / self.base_mean[task]) ** self.dispersion))
            for task, values in self.base.items()
        }
        self.step_row_cache = {}

    @staticmethod
    def _load_samples(path):
        values = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                value = float(row.get("sample_time", 0))
                if value > 0:
                    values.append(value)
        if not values:
            raise ValueError(f"empty sample-time profile: {path}")
        return np.asarray(values)

    def _nearby(self, step, field):
        candidates = [
            row[field]
            for row in self.steps
            if abs(row["step"] - step) <= self.radius and field in row
        ]
        if not candidates:
            candidates = [row[field] for row in self.steps if field in row]
        return candidates

    def _step_row(self, step, required_fields):
        key = (step, tuple(required_fields))
        if key not in self.step_row_cache:
            candidates = [
                row
                for row in self.steps
                if abs(row["step"] - step) <= self.radius and all(field in row for field in required_fields)
            ]
            if not candidates:
                candidates = [row for row in self.steps if all(field in row for field in required_fields)]
            self.step_row_cache[key] = candidates[int(self.rng.integers(len(candidates)))]
        return self.step_row_cache[key]

    def group_time(self, task, step, n_samples):
        avg_field = f"tool/{task}_sample_time_avg"
        max_field = f"tool/{task}_sample_time_max"
        # Pick one observed operating regime per simulated policy step. All
        # groups dispatched in that step share it, preserving real fast/slow
        # bursts instead of averaging them away group by group.
        source_row = self._step_row(step, (avg_field, max_field))
        target_avg = float(source_row[avg_field])
        target_max = float(source_row[max_field])
        raw = self.rng.choice(self.base[task], size=n_samples, replace=True)
        normalized = (raw / self.base_mean[task]) ** self.dispersion
        scaled = target_avg * normalized / self.base_moment[task]
        # The W&B max is a batch-level observed ceiling, not a group duration.
        duration = max(0.01, min(float(np.max(scaled)), target_max))
        if duration > target_avg * self.tail_threshold:
            duration *= self.tail_scale
        duration *= self.fast_scale if self.rng.random() < self.fast_fraction else self.slow_scale
        duration *= (
            self.burst_scale
            if source_row.get("fully_async/window/evicted_samples", 0) > 0
            else self.quiet_scale
        )
        phase_scale = CALIBRATED_PHASE_SCALES[0][1]
        for start_step, candidate_scale in CALIBRATED_PHASE_SCALES:
            if step >= start_step:
                phase_scale = candidate_scale
        duration *= phase_scale
        return self.group_time_scale * duration

    def train_time(self, step):
        key = ("train_time", step)
        if key not in self.step_row_cache:
            candidates = [
                row
                for row in self.train_time_steps
                if abs(row["step"] - step) <= self.radius and "perf/train_time" in row
            ]
            if not candidates:
                candidates = [row for row in self.train_time_steps if "perf/train_time" in row]
            self.step_row_cache[key] = candidates[int(self.rng.integers(len(candidates)))]
        return float(self.step_row_cache[key]["perf/train_time"])


class FullyAsyncReplay:
    def __init__(
        self,
        profile,
        seed=42,
        radius=0,
        group_time_scale=CALIBRATED_GROUP_TIME_SCALE,
        dispersion=CALIBRATED_DISPERSION,
        tail_threshold=CALIBRATED_TAIL_THRESHOLD,
        tail_scale=CALIBRATED_TAIL_SCALE,
        fast_fraction=CALIBRATED_FAST_FRACTION,
        fast_scale=CALIBRATED_FAST_SCALE,
        slow_scale=CALIBRATED_SLOW_SCALE,
        burst_scale=CALIBRATED_BURST_SCALE,
        quiet_scale=CALIBRATED_QUIET_SCALE,
        scheduling="fixed",
        phase_aware_math_groups=16,
        phase_aware_qa_groups=16,
        phase_aware_train_task="qa",
        phase_aware_post_update_task="math",
        phase_aware_flip_each_cycle=False,
        phase_aware_adaptive=False,
        phase_aware_adaptive_inverse=False,
        phase_aware_lag_margin=0.0,
        count_dynamic_threshold=2,
        microblock_size=1,
        phase_debt_bias=0.0,
        lag_debt_beta=0.0,
        lag_debt_scale=256.0,
        lag_debt_cap=16.0,
        completed_cap=None,
        train_time_profile=None,
    ):
        self.profile = profile
        self.config = profile["config"]
        self.rng = np.random.default_rng(seed)
        self.pool = DynamicTimePool(
            profile,
            self.rng,
            radius,
            group_time_scale,
            dispersion,
            tail_threshold,
            tail_scale,
            fast_fraction,
            fast_scale,
            slow_scale,
            burst_scale,
            quiet_scale,
            train_time_profile,
        )
        self.parallel = int(self.config.get("rollout_batch_size") or 32)
        self.batch_size = self.parallel
        self.n_samples = int(self.config.get("n_samples_per_prompt") or 8)
        self.cap = int(
            completed_cap
            if completed_cap is not None
            else (self.config.get("fully_async_max_completed_samples") or self.batch_size)
        )
        self.math_ratio = float(self.config.get("math_ratio") or 0.5)
        self.partial = bool(self.config.get("partial_rollout"))
        self.scheduling = scheduling
        self.phase_aware_quotas = {
            "math": max(0, int(phase_aware_math_groups)),
            "qa": max(0, int(phase_aware_qa_groups)),
        }
        self.phase_aware_cycle_counts = {"math": 0, "qa": 0}
        self.phase_aware_cycle_index = 0
        self.phase_aware_train_task = phase_aware_train_task
        self.phase_aware_post_update_task = phase_aware_post_update_task
        self.phase_aware_flip_each_cycle = phase_aware_flip_each_cycle
        self.phase_aware_adaptive = phase_aware_adaptive
        self.phase_aware_adaptive_inverse = phase_aware_adaptive_inverse
        self.phase_aware_lag_margin = max(0.0, float(phase_aware_lag_margin))
        self.count_dynamic_threshold = max(0, int(count_dynamic_threshold))
        self.microblock_size = max(1, int(microblock_size))
        self.phase_debt_bias = float(phase_debt_bias)
        self.lag_debt_beta = float(lag_debt_beta)
        self.lag_debt_scale = max(1.0, float(lag_debt_scale))
        self.lag_debt_cap = max(0.0, float(lag_debt_cap))
        # A policy is selected once after each 32-group training batch is
        # collected.  Initial lane filling is random because no previous batch
        # exists yet from which to make a dynamic decision.
        self.active_step_schedule = (
            scheduling if scheduling in ("fixed", "all_math", "all_qa") else "fixed"
        )
        self.pending_step_schedules = []
        self.schedule_cycle_started = scheduling in ("fixed", "all_math", "all_qa")
        self.step_schedule_counts = {"math": 0, "qa": 0}
        self.step_schedule_history = []
        self.phase = "post_update"
        self.now = 0.0
        self.version = 0
        self.seq = 0
        self.inflight = []
        self.completed = []
        self.evicted = 0
        self.recycled = 0
        self.new_dispatch_counts = {"math": 0, "qa": 0}
        self.batch_history = []

    def _lag_pressure(self):
        """Current per-task lag_sample pressure over active and completed groups."""
        pressure = {"math": 0, "qa": 0}
        groups = [entry[2] for entry in self.inflight] + list(self.completed)
        for group in groups:
            for history in self.batch_history[group.origin_version:self.version]:
                pressure[group.task] += history[group.task]
        return pressure

    def _select_step_schedule(self, batch):
        """Select the dispatch policy for the just-starting train/update step."""
        schedule = self.scheduling
        if schedule in ("all_math", "all_qa", "global_drr", "phase_debt", "phase_lag_debt"):
            self.active_step_schedule = schedule
            self.schedule_cycle_started = True
            self.step_schedule_history.append(schedule)
            return
        if schedule in ("count_dynamic", "count_block_math", "count_stale_hybrid"):
            math_count = sum(group.task == "math" for group in batch)
            qa_count = len(batch) - math_count
            imbalance = math_count - qa_count
            if imbalance >= self.count_dynamic_threshold:
                schedule = "adaptive_math"
            elif imbalance <= -self.count_dynamic_threshold:
                schedule = "adaptive_qa"
            elif self.scheduling == "count_block_math":
                schedule = "block_math"
            elif self.scheduling == "count_stale_hybrid":
                pressure = self._lag_pressure()
                schedule = "block_qa" if pressure["math"] > pressure["qa"] else "block_math"
            else:
                schedule = self.active_step_schedule if self.active_step_schedule.startswith("adaptive_") else "adaptive_math"
        elif schedule == "staleness_dynamic":
            pressure = self._lag_pressure()
            if pressure["math"] > pressure["qa"]:
                schedule = "block_qa"
            elif pressure["qa"] > pressure["math"]:
                schedule = "block_math"
            else:
                schedule = self.active_step_schedule if self.active_step_schedule.startswith("block_") else "block_qa"
        elif schedule in ("batch_lag_adaptive", "batch_lag_block"):
            lag_by_task = {"math": [], "qa": []}
            for group in batch:
                lag_by_task[group.task].append(
                    sum(history[group.task] for history in self.batch_history[group.origin_version:self.version])
                )
            math_lag = float(np.mean(lag_by_task["math"])) if lag_by_task["math"] else 0.0
            qa_lag = float(np.mean(lag_by_task["qa"])) if lag_by_task["qa"] else 0.0
            if self.scheduling == "batch_lag_adaptive":
                schedule = "adaptive_math" if math_lag > qa_lag else "adaptive_qa"
            else:
                schedule = "block_qa" if math_lag > qa_lag else "block_math"

        self.pending_step_schedules.append(schedule)
        if not self.schedule_cycle_started:
            self.active_step_schedule = self.pending_step_schedules.pop(0)
            self.step_schedule_counts = {"math": 0, "qa": 0}
            self.schedule_cycle_started = True
        self.step_schedule_history.append(schedule)

    def _new_task(self):
        if (
            self.active_step_schedule != "fixed"
            and sum(self.step_schedule_counts.values()) >= sum(self.phase_aware_quotas.values())
        ):
            if self.pending_step_schedules:
                self.active_step_schedule = self.pending_step_schedules.pop(0)
            self.step_schedule_counts = {"math": 0, "qa": 0}

        schedule = self.active_step_schedule
        if schedule in ("all_math", "all_qa"):
            task = "math" if schedule == "all_math" else "qa"
            self.new_dispatch_counts[task] += 1
            return task
        if schedule in ("global_drr", "phase_debt", "phase_lag_debt"):
            total = self.new_dispatch_counts["math"] + self.new_dispatch_counts["qa"]
            # Positive debt means Math is behind the long-run 1:1 target.
            debt = (total + 1) * self.math_ratio - self.new_dispatch_counts["math"]
            score = debt
            if schedule in ("phase_debt", "phase_lag_debt"):
                score += self.phase_debt_bias if self.phase == "post_update" else -self.phase_debt_bias
            if schedule == "phase_lag_debt" and self.lag_debt_beta != 0:
                pressure = self._lag_pressure()
                lag_delta = (pressure["math"] - pressure["qa"]) / self.lag_debt_scale
                lag_term = max(-self.lag_debt_cap, min(self.lag_debt_cap, self.lag_debt_beta * lag_delta))
                # Positive beta avoids adding work to the task with more stale pressure.
                score -= lag_term
            task = "math" if score > 0 else "qa"
            self.new_dispatch_counts[task] += 1
            return task
        if schedule == "fixed":
            task = "math" if self.rng.random() < self.math_ratio else "qa"
            self.new_dispatch_counts[task] += 1
            return task

        if schedule in ("block_qa", "block_math", "microblock_qa", "microblock_math"):
            first = "qa" if schedule == "block_qa" else "math"
            if schedule.startswith("microblock_"):
                first = "qa" if schedule == "microblock_qa" else "math"
                block_index = sum(self.step_schedule_counts.values()) // self.microblock_size
                task = first if block_index % 2 == 0 else ("math" if first == "qa" else "qa")
            else:
                task = first if sum(self.step_schedule_counts.values()) < 16 else ("math" if first == "qa" else "qa")
            self.step_schedule_counts[task] += 1
            self.new_dispatch_counts[task] += 1
            return task

        if schedule in ("adaptive_qa", "adaptive_math", "phase_aware"):
            # Match custom_data_source._choose_phase_aware_task exactly:
            # quotas persist across phase changes and reset only after both
            # tasks have exhausted their quota for the current cycle.
            if sum(self.step_schedule_counts.values()) == 0:
                self.phase_aware_cycle_index += 1
                if self.phase_aware_adaptive:
                    pressure = self._lag_pressure()
                    delta = pressure["math"] - pressure["qa"]
                    if delta > self.phase_aware_lag_margin:
                        self.phase_aware_train_task = "math" if self.phase_aware_adaptive_inverse else "qa"
                        self.phase_aware_post_update_task = "qa" if self.phase_aware_adaptive_inverse else "math"
                    elif delta < -self.phase_aware_lag_margin:
                        self.phase_aware_train_task = "qa" if self.phase_aware_adaptive_inverse else "math"
                        self.phase_aware_post_update_task = "math" if self.phase_aware_adaptive_inverse else "qa"

            if schedule == "adaptive_qa":
                train_task, post_update_task = "math", "qa"
            elif schedule == "adaptive_math":
                train_task, post_update_task = "qa", "math"
            else:
                train_task, post_update_task = self.phase_aware_train_task, self.phase_aware_post_update_task
            preferred = train_task if self.phase == "training" else post_update_task
            if self.phase_aware_flip_each_cycle and self.phase_aware_cycle_index % 2 == 1:
                preferred = "qa" if preferred == "math" else "math"
            fallback = "math" if preferred == "qa" else "qa"
            for task in (preferred, fallback):
                if self.step_schedule_counts[task] < self.phase_aware_quotas[task]:
                    self.step_schedule_counts[task] += 1
                    self.new_dispatch_counts[task] += 1
                    return task

            # Same degenerate zero-quota fallback as production.
            task = "math" if self.rng.random() < self.math_ratio else "qa"
            self.new_dispatch_counts[task] += 1
            return task
        task = "math" if self.rng.random() < self.math_ratio else "qa"
        self.new_dispatch_counts[task] += 1
        return task

    def dispatch(self, task=None, origin_version=None):
        task = task or self._new_task()
        duration = self.pool.group_time(task, self.version, self.n_samples)
        if origin_version is None:
            origin_version = self.version
        group = Group(self.seq, task, self.version, origin_version, self.now, self.now + duration)
        self.seq += 1
        heapq.heappush(self.inflight, (group.finishes_at, group.seq, group))

    def fill_lanes(self):
        while len(self.inflight) < self.parallel:
            self.dispatch()

    def _complete_next(self):
        finish, _, group = heapq.heappop(self.inflight)
        self.now = max(self.now, finish)
        self.completed.append(group)
        if len(self.completed) > self.cap:
            # Production drop_oldest_version; seq breaks ties like FIFO.
            idx = min(range(len(self.completed)), key=lambda i: (self.completed[i].version, self.completed[i].seq))
            self.completed.pop(idx)
            self.evicted += 1
        self.dispatch()

    def drain_until(self, end_time):
        while self.inflight and self.inflight[0][0] <= end_time:
            self._complete_next()
        self.now = end_time

    def _abort_and_recycle(self):
        if not self.partial:
            return
        groups = [entry[2] for entry in self.inflight]
        self.inflight.clear()
        self.recycled += len(groups)
        for group in groups:
            self.dispatch(group.task, origin_version=group.origin_version)

    def run(self, steps):
        self.fill_lanes()
        records = []
        previous_end = 0.0
        batch_history = []
        for step in range(steps):
            evicted_before = self.evicted
            wait_start = self.now
            while len(self.completed) < self.batch_size:
                self._complete_next()
            train_wait = self.now - wait_start
            batch = sorted(self.completed, key=lambda group: group.seq)[: self.batch_size]
            consumed = {group.seq for group in batch}
            self.completed = [group for group in self.completed if group.seq not in consumed]
            self._select_step_schedule(batch)
            self.phase = "training"
            train_time = self.pool.train_time(step)
            lag_by_task = {"math": [], "qa": []}
            for group in batch:
                lag_by_task[group.task].append(
                    sum(history[group.task] for history in batch_history[group.origin_version:step])
                )
            train_start = self.now
            self.drain_until(train_start + train_time)
            self.version += 1
            self.phase = "post_update"
            self._abort_and_recycle()
            self.fill_lanes()
            records.append(
                {
                    "step": step,
                    "train_time": train_time,
                    "train_wait_time": train_wait,
                    "step_time": self.now - previous_end,
                    "math_groups": sum(group.task == "math" for group in batch),
                    "qa_groups": sum(group.task == "qa" for group in batch),
                    "completed_store_size": len(self.completed),
                    "evicted_total": self.evicted,
                    "evicted_samples": self.evicted - evicted_before,
                    "recycled_total": self.recycled,
                    "new_math_dispatched_total": self.new_dispatch_counts["math"],
                    "new_qa_dispatched_total": self.new_dispatch_counts["qa"],
                    "phase_aware_train_task": self.phase_aware_train_task,
                    "step_schedule": self.active_step_schedule,
                    "batch_mean_version_lag": float(np.mean([step - group.version for group in batch])),
                    "lag_sample_math_average": float(np.mean(lag_by_task["math"])) if lag_by_task["math"] else 0.0,
                    "lag_sample_math_max": max(lag_by_task["math"], default=0),
                    "lag_sample_qa_average": float(np.mean(lag_by_task["qa"])) if lag_by_task["qa"] else 0.0,
                    "lag_sample_qa_max": max(lag_by_task["qa"], default=0),
                }
            )
            batch_history.append({
                "math": sum(group.task == "math" for group in batch) * self.n_samples,
                "qa": sum(group.task == "qa" for group in batch) * self.n_samples,
            })
            self.batch_history = batch_history
            previous_end = self.now
        return records


def summarize(records, profile, train_time_profile=None):
    def mean(field):
        return float(np.mean([row[field] for row in records]))

    real = profile["steps"][: len(records)]
    train_real = (train_time_profile or profile)["steps"][: len(records)]
    real_train = [row["perf/train_time"] for row in train_real if "perf/train_time" in row]
    real_wait = [row["perf/train_wait_time"] for row in real if "perf/train_wait_time" in row]
    real_step = [row["perf/step_time"] for row in real if "perf/step_time" in row]
    real_math_groups = [
        row["tool/math_count"] / int(profile["config"].get("n_samples_per_prompt") or 8)
        for row in real
        if "tool/math_count" in row
    ]
    eviction_steps = [i for i, row in enumerate(real) if "fully_async/window/evicted_samples" in row]
    real_store = [row["fully_async/window/completed_store_size"] for row in real if "fully_async/window/completed_store_size" in row]
    orientation_switches = sum(
        records[i]["phase_aware_train_task"] != records[i - 1]["phase_aware_train_task"]
        for i in range(1, len(records))
    )
    return {
        "steps": len(records),
        "sim_train_time": mean("train_time"),
        "real_train_time": float(np.mean(real_train)),
        "sim_train_wait_time": mean("train_wait_time"),
        "real_train_wait_time": float(np.mean(real_wait)),
        "sim_step_time": mean("step_time"),
        "real_step_time": float(np.mean(real_step)),
        "sim_math_groups": mean("math_groups"),
        "real_math_groups": float(np.mean(real_math_groups)),
        "sim_completed_store_size": mean("completed_store_size"),
        "real_completed_store_size": float(np.mean(real_store)),
        "sim_evicted_total": records[-1]["evicted_total"],
        "sim_evicted_on_logged_steps": float(sum(records[i]["evicted_samples"] for i in eviction_steps)),
        "real_evicted_on_logged_steps": float(
            sum(real[i]["fully_async/window/evicted_samples"] for i in eviction_steps)
        ),
        "sim_recycled_total": records[-1]["recycled_total"],
        "sim_lag_sample_math_average": mean("lag_sample_math_average"),
        "sim_lag_sample_math_max": float(max(row["lag_sample_math_max"] for row in records)),
        "sim_lag_sample_qa_average": mean("lag_sample_qa_average"),
        "sim_lag_sample_qa_max": float(max(row["lag_sample_qa_max"] for row in records)),
        "sim_new_math_dispatch_ratio": (
            records[-1]["new_math_dispatched_total"]
            / (records[-1]["new_math_dispatched_total"] + records[-1]["new_qa_dispatched_total"])
        ),
        "sim_orientation_switches": float(orientation_switches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--train-time-profile",
        type=Path,
        help="Use perf/train_time from this profile while leaving rollout sampling on --profile.",
    )
    parser.add_argument("--steps", type=int, default=89)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--pool-radius",
        type=int,
        default=0,
        help="0 replays the matching W&B step; >0 samples a nearby-step operating regime",
    )
    parser.add_argument("--group-time-scale", type=float, default=CALIBRATED_GROUP_TIME_SCALE)
    parser.add_argument("--dispersion", type=float, default=CALIBRATED_DISPERSION)
    parser.add_argument("--tail-threshold", type=float, default=CALIBRATED_TAIL_THRESHOLD)
    parser.add_argument("--tail-scale", type=float, default=CALIBRATED_TAIL_SCALE)
    parser.add_argument("--fast-fraction", type=float, default=CALIBRATED_FAST_FRACTION)
    parser.add_argument("--fast-scale", type=float, default=CALIBRATED_FAST_SCALE)
    parser.add_argument("--slow-scale", type=float, default=CALIBRATED_SLOW_SCALE)
    parser.add_argument("--burst-scale", type=float, default=CALIBRATED_BURST_SCALE)
    parser.add_argument("--quiet-scale", type=float, default=CALIBRATED_QUIET_SCALE)
    parser.add_argument(
        "--scheduling",
        choices=(
            "fixed",
            "all_math",
            "all_qa",
            "block_qa",
            "block_math",
            "adaptive_qa",
            "adaptive_math",
            "count_dynamic",
            "staleness_dynamic",
            "count_block_math",
            "count_stale_hybrid",
            "microblock_math",
            "microblock_qa",
            "batch_lag_adaptive",
            "batch_lag_block",
            "global_drr",
            "phase_debt",
            "phase_lag_debt",
            "phase_aware",  # legacy experimental mode
        ),
        default="fixed",
    )
    parser.add_argument("--phase-aware-math-groups", type=int, default=16)
    parser.add_argument("--phase-aware-qa-groups", type=int, default=16)
    parser.add_argument("--phase-aware-train-task", choices=("math", "qa"), default="qa")
    parser.add_argument("--phase-aware-post-update-task", choices=("math", "qa"), default="math")
    parser.add_argument("--phase-aware-flip-each-cycle", action="store_true")
    parser.add_argument("--phase-aware-adaptive", action="store_true")
    parser.add_argument("--phase-aware-adaptive-inverse", action="store_true")
    parser.add_argument("--count-dynamic-threshold", type=int, default=2)
    parser.add_argument("--microblock-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--phase-debt-bias", type=float, default=0.0)
    parser.add_argument("--lag-debt-beta", type=float, default=0.0)
    parser.add_argument("--lag-debt-scale", type=float, default=256.0)
    parser.add_argument("--lag-debt-cap", type=float, default=16.0)
    parser.add_argument(
        "--phase-aware-lag-margin",
        type=float,
        default=0.0,
        help="Keep the previous orientation while abs(math_lag-qa_lag) is within this margin.",
    )
    parser.add_argument(
        "--completed-cap",
        type=int,
        help="Override fully_async_max_completed_samples from the replay profile.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    train_time_profile = (
        json.loads(args.train_time_profile.read_text(encoding="utf-8"))
        if args.train_time_profile
        else None
    )
    summaries = []
    all_records = []
    for seed in range(args.seeds):
        records = FullyAsyncReplay(
            profile,
            seed=seed,
            radius=args.pool_radius,
            group_time_scale=args.group_time_scale,
            dispersion=args.dispersion,
            tail_threshold=args.tail_threshold,
            tail_scale=args.tail_scale,
            fast_fraction=args.fast_fraction,
            fast_scale=args.fast_scale,
            slow_scale=args.slow_scale,
            burst_scale=args.burst_scale,
            quiet_scale=args.quiet_scale,
            scheduling=args.scheduling,
            phase_aware_math_groups=args.phase_aware_math_groups,
            phase_aware_qa_groups=args.phase_aware_qa_groups,
            phase_aware_train_task=args.phase_aware_train_task,
            phase_aware_post_update_task=args.phase_aware_post_update_task,
            phase_aware_flip_each_cycle=args.phase_aware_flip_each_cycle,
            phase_aware_adaptive=args.phase_aware_adaptive,
            phase_aware_adaptive_inverse=args.phase_aware_adaptive_inverse,
            phase_aware_lag_margin=args.phase_aware_lag_margin,
            count_dynamic_threshold=args.count_dynamic_threshold,
            microblock_size=args.microblock_size,
            phase_debt_bias=args.phase_debt_bias,
            lag_debt_beta=args.lag_debt_beta,
            lag_debt_scale=args.lag_debt_scale,
            lag_debt_cap=args.lag_debt_cap,
            completed_cap=args.completed_cap,
            train_time_profile=train_time_profile,
        ).run(args.steps)
        summaries.append(summarize(records, profile, train_time_profile))
        if seed == 0:
            all_records = records
    aggregate = {key: float(np.mean([row[key] for row in summaries])) for key in summaries[0] if key != "steps"}
    aggregate["steps"] = args.steps
    aggregate["seeds"] = args.seeds
    aggregate["group_time_scale"] = args.group_time_scale
    aggregate["dispersion"] = args.dispersion
    aggregate["tail_threshold"] = args.tail_threshold
    aggregate["tail_scale"] = args.tail_scale
    aggregate["fast_fraction"] = args.fast_fraction
    aggregate["fast_scale"] = args.fast_scale
    aggregate["slow_scale"] = args.slow_scale
    aggregate["burst_scale"] = args.burst_scale
    aggregate["quiet_scale"] = args.quiet_scale
    aggregate["scheduling"] = args.scheduling
    aggregate["train_time_source"] = (
        train_time_profile.get("source_run") if train_time_profile else profile.get("source_run")
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps({"summary": aggregate, "seed0_steps": all_records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
