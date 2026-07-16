#!/bin/bash
# Keep 25-epoch SFT milestones plus safe rolling checkpoints while training.
set -euo pipefail

CHECKPOINT_DIR=${CHECKPOINT_DIR:-/workspace/Qwen3-8B-browsecomp-sft}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-34}
KEEP_EVERY_EPOCHS=${KEEP_EVERY_EPOCHS:-25}
POLL_SECONDS=${POLL_SECONDS:-60}
DRY_RUN=${DRY_RUN:-0}
RUN_ONCE=${RUN_ONCE:-0}

MILESTONE_STEPS=$((STEPS_PER_EPOCH * KEEP_EVERY_EPOCHS))

cleanup_once() {
  [[ -d "${CHECKPOINT_DIR}" ]] || return 0

  local latest=-1
  if [[ -s "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    latest=$(<"${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")
  fi

  # Keep the two numerically newest directories as a race guard. Megatron
  # creates a new iter directory before updating latest_checkpointed_iteration.
  mapfile -t checkpoints < <(
    find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'iter_[0-9]*' -printf '%f\n' | sort
  )
  local count=${#checkpoints[@]}
  local newest="" second_newest=""
  (( count >= 1 )) && newest=${checkpoints[count-1]}
  (( count >= 2 )) && second_newest=${checkpoints[count-2]}

  local name iteration completed_steps
  for name in "${checkpoints[@]}"; do
    iteration=$((10#${name#iter_}))
    completed_steps=$((iteration + 1))

    # A .keep marker provides an explicit escape hatch for manually pinned
    # recovery points and protects them across cleaner restarts/config changes.
    if [[ -e "${CHECKPOINT_DIR}/${name}/.keep" ]]; then
      continue
    fi

    # Milestones are zero-based iterations: epoch 25 ends at step 849 when
    # there are 34 optimizer steps per epoch.
    if (( completed_steps % MILESTONE_STEPS == 0 )); then
      continue
    fi
    if (( iteration == latest )) || [[ "${name}" == "${newest}" || "${name}" == "${second_newest}" ]]; then
      continue
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "would delete ${CHECKPOINT_DIR}/${name}"
    else
      echo "$(date -u +'%F %T UTC') deleting ${CHECKPOINT_DIR}/${name}"
      rm -rf -- "${CHECKPOINT_DIR:?}/${name}"
    fi
  done
}

while true; do
  cleanup_once
  [[ "${RUN_ONCE}" == "1" ]] && break
  sleep "${POLL_SECONDS}"
done
