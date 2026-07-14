#!/usr/bin/env python3
"""Export sparse W&B history into one compact record per rollout step."""

import argparse
import json
from pathlib import Path

import wandb


DEFAULT_RUN = "hanlab-dev/mixed-qwen3-8b-sync/lgdzo8cx"
FIELDS = (
    "perf/train_time",
    "perf/actor_train_time",
    "perf/rollout_time",
    "perf/step_time",
    "perf/train_wait_time",
    "tool/math_sample_time_avg",
    "tool/math_sample_time_max",
    "tool/qa_sample_time_avg",
    "tool/qa_sample_time_max",
    "tool/math_response_length_avg",
    "tool/qa_response_length_avg",
    "tool/math_count",
    "tool/qa_count",
    "tool/lag_sample_math_average",
    "tool/lag_sample_qa_average",
    "fully_async/window/completed_store_size",
    "fully_async/window/evicted_samples",
    "fully_async/window/evicted_by_version",
    "fully_async/count/dropped_samples",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "debug" / "wandb_lgdzo8cx_sim_profile.json",
    )
    args = parser.parse_args()

    run = wandb.Api(timeout=90).run(args.run)
    by_step = {}
    last_rollout_step = None
    for row in run.scan_history(page_size=5000):
        rollout_step = row.get("rollout/step")
        if isinstance(rollout_step, (int, float)):
            last_rollout_step = int(rollout_step)
        if last_rollout_step is None:
            continue
        record = by_step.setdefault(last_rollout_step, {"step": last_rollout_step})
        for field in FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)):
                record[field] = float(value)

    records = [by_step[step] for step in sorted(by_step)]
    payload = {
        "source_run": args.run,
        "run_name": run.name,
        "created_at": run.created_at,
        "config": {
            key: run.config.get(key)
            for key in (
                "fully_async_buffer_policy",
                "fully_async_max_completed_samples",
                "fully_async_eviction_policy",
                "fully_async_version_window",
                "partial_rollout",
                "rollout_batch_size",
                "n_samples_per_prompt",
                "math_ratio",
            )
        },
        "steps": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} steps to {args.output}")


if __name__ == "__main__":
    main()
