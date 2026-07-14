import heapq
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import test as sim

SEED = 42
NUM_TRAIN_POINTS = 20
OUT_DIR = Path(__file__).resolve().parent / "debug" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DISPATCH_MODE = os.environ.get("DISPATCH_MODE", "block16")
OUT_PATH = OUT_DIR / f"{DISPATCH_MODE}_real_timeline_seed{SEED}_{NUM_TRAIN_POINTS}train.png"

MATH_COLOR = "#d64a4a"
QA_COLOR = "#3478c6"
TRAIN_COLOR = "#222222"


def choose_task(dispatched_count, rng):
    if DISPATCH_MODE == "block16":
        block_idx = dispatched_count // 16
        return "math" if block_idx % 2 == 0 else "qa"
    if DISPATCH_MODE == "fixed":
        return "math" if rng.random() < 0.5 else "qa"
    raise ValueError(f"Unknown DISPATCH_MODE: {DISPATCH_MODE}")


def sample_duration(task_type, rng):
    return sim.sample_group_time(
        sim.MATH_MEAN_TIME if task_type == "math" else sim.QA_MEAN_TIME,
        sim.MATH_STD_TIME if task_type == "math" else sim.QA_STD_TIME,
        rng,
        task_type,
    )


def simulate():
    sim.DISTRIBUTION = "real"
    sim.MATH_RATIO = 0.5
    sim.BLOCK_ALTERNATION_SIZE = 16

    rng = np.random.default_rng(SEED)
    heap = []
    segments = []
    train_points = []
    completed_batch = []
    dispatched_count = 0
    job_id = 0

    def dispatch(lane, start_time):
        nonlocal dispatched_count, job_id
        task_type = choose_task(dispatched_count, rng)
        duration = sample_duration(task_type, rng)
        end_time = start_time + duration
        segments.append(
            {
                "lane": lane,
                "start": start_time,
                "end": end_time,
                "task": task_type,
                "job_id": job_id,
                "dispatch_idx": dispatched_count,
            }
        )
        heapq.heappush(heap, (end_time, lane, task_type, job_id))
        dispatched_count += 1
        job_id += 1

    for lane in range(sim.PARALLEL):
        dispatch(lane, 0.0)

    while len(train_points) < NUM_TRAIN_POINTS:
        end_time, lane, task_type, finished_job_id = heapq.heappop(heap)
        completed_batch.append((task_type, lane, finished_job_id, end_time))
        if len(completed_batch) == sim.PARALLEL:
            math_count = sum(1 for task, *_ in completed_batch if task == "math")
            qa_count = sim.PARALLEL - math_count
            train_points.append(
                {
                    "time": end_time,
                    "idx": len(train_points) + 1,
                    "math": math_count,
                    "qa": qa_count,
                }
            )
            completed_batch = []
        dispatch(lane, end_time)

    return segments, train_points


def plot(segments, train_points):
    end_time = train_points[-1]["time"]
    fig, ax = plt.subplots(figsize=(22, 11))

    for seg in segments:
        if seg["start"] > end_time:
            continue
        visible_end = min(seg["end"], end_time)
        width = max(visible_end - seg["start"], 0.001)
        color = MATH_COLOR if seg["task"] == "math" else QA_COLOR
        alpha = 0.92 if seg["end"] <= end_time else 0.35
        ax.barh(
            seg["lane"],
            width,
            left=seg["start"],
            height=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=alpha,
        )

    ymax = sim.PARALLEL - 0.25
    for tp in train_points:
        ax.axvline(tp["time"], color=TRAIN_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(
            tp["time"],
            ymax + 0.8,
            f"T{tp['idx']}\nM{tp['math']}/Q{tp['qa']}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
            color=TRAIN_COLOR,
        )

    ax.set_title(
        f"Real Distribution + {DISPATCH_MODE} Dispatch Timeline\n"
        "32 lanes are concurrent rollout groups; red=math, blue=QA; dashed lines mark train when 32 completions are collected",
        fontsize=14,
        pad=28,
    )
    ax.set_xlabel("simulation time (seconds)")
    ax.set_ylabel("concurrent rollout lane")
    ax.set_yticks(range(sim.PARALLEL))
    ax.set_ylim(-1, sim.PARALLEL + 3.0)
    ax.set_xlim(0, end_time * 1.01)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(
        handles=[
            Patch(facecolor=MATH_COLOR, label="math"),
            Patch(facecolor=QA_COLOR, label="QA"),
            Patch(facecolor="#999999", alpha=0.35, label="continues past plotted window"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180)
    return OUT_PATH


if __name__ == "__main__":
    segments, train_points = simulate()
    out_path = plot(segments, train_points)
    print(out_path)
    print("train_points:")
    for tp in train_points:
        print(f"T{tp['idx']:02d}: time={tp['time']:.2f}s math={tp['math']} qa={tp['qa']}")
