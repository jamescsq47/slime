"""Incrementally mirror the 8B-baseline wandb run (miles-browsecomp/adq6lio8)
into the browsecomp-b300 project so all four experiment cells live in one
project. Idempotent: resumes the fixed destination run id and only logs
history rows with _step greater than what was already synced. Re-run any time.
"""

import sys

import wandb

SRC_PATH = "ys-2020/miles-browsecomp/adq6lio8"
DST_PROJECT = "browsecomp-b300"
DST_RUN_ID = "bc8b-baseline-mirror"
DST_NAME = "browsecomp-b300-qwen3-8b-grpo-baseline"
TAGS = ["cluster=b300", "model=qwen3-8b", "mode=baseline", "length_penalty_enabled=false", "mirror-of-adq6lio8"]

api = wandb.Api()
src = api.run(SRC_PATH)

# Find how far the destination already got.
last_synced = -1
try:
    dst_existing = api.run(f"ys-2020/{DST_PROJECT}/{DST_RUN_ID}")
    last_synced = dst_existing.summary.get("_step", -1)
except Exception:
    pass  # first sync

dst = wandb.init(
    project=DST_PROJECT,
    id=DST_RUN_ID,
    name=DST_NAME,
    resume="allow",
    tags=TAGS,
    config=dict(src.config),
    settings=wandb.Settings(init_timeout=120),
)

n = 0
for row in src.scan_history():
    step = row.pop("_step", None)
    if step is None or step <= last_synced:
        continue
    row = {k: v for k, v in row.items() if not k.startswith("_")}
    if row:
        wandb.log(row, step=int(step))
        n += 1

wandb.finish()
print(f"synced {n} new history rows (dest was at _step={last_synced})")
sys.exit(0)
