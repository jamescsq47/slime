#!/bin/bash
# Keep 25-epoch SFT milestones plus safe rolling checkpoints while training.
set -euo pipefail

CHECKPOINT_DIR=${CHECKPOINT_DIR:-/workspace/Qwen3-8B-browsecomp-sft}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-34}
KEEP_EVERY_EPOCHS=${KEEP_EVERY_EPOCHS:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
POLL_SECONDS=${POLL_SECONDS:-60}
DRY_RUN=${DRY_RUN:-0}
RUN_ONCE=${RUN_ONCE:-0}

# Prevent stale cleaners with older retention rules from running concurrently.
# The lock lives outside the checkpoint directory so checkpoint writes cannot
# replace it. RUN_ONCE checks use a separate lock only when explicitly set.
LOCK_FILE=${LOCK_FILE:-/tmp/browsecomp_sft_checkpoint_cleanup.lock}
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another BrowseComp SFT checkpoint cleaner is already running" >&2
  exit 1
fi

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

  # Record which 25-epoch boundaries already have a permanently retained
  # checkpoint. A save schedule can be offset by resume (for example, a run
  # resumed at iteration 1019 with --save-interval 100 will never create
  # iter_0001699). In that case retain the first checkpoint written after the
  # boundary instead of silently losing the milestone altogether.
  declare -A retained_milestones=()
  local marker milestone_epoch
  for name in "${checkpoints[@]}"; do
    for marker in "${CHECKPOINT_DIR}/${name}"/.keep_epoch_*; do
      [[ -e "${marker}" ]] || continue
      milestone_epoch=${marker##*_}
      retained_milestones["${milestone_epoch}"]=1
    done
  done

  local name iteration completed_steps milestone_steps
  for name in "${checkpoints[@]}"; do
    iteration=$((10#${name#iter_}))
    completed_steps=$((iteration + 1))

    for ((milestone_epoch=KEEP_EVERY_EPOCHS; milestone_epoch<=TOTAL_EPOCHS; milestone_epoch+=KEEP_EVERY_EPOCHS)); do
      milestone_steps=$((STEPS_PER_EPOCH * milestone_epoch))
      if (( completed_steps >= milestone_steps )) && [[ -z "${retained_milestones[${milestone_epoch}]:-}" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "would retain ${CHECKPOINT_DIR}/${name} for epoch ${milestone_epoch}"
        else
          touch "${CHECKPOINT_DIR}/${name}/.keep"
          touch "${CHECKPOINT_DIR}/${name}/.keep_epoch_${milestone_epoch}"
        fi
        retained_milestones["${milestone_epoch}"]=1
      fi
    done

    # A .keep marker provides an explicit escape hatch for manually pinned
    # recovery points and protects them across cleaner restarts/config changes.
    if [[ -e "${CHECKPOINT_DIR}/${name}/.keep" ]]; then
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
