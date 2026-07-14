#!/bin/bash
# Slurm-free eval runner: executes on a compute node (e.g. drained c018) via
# plain ssh + enroot, sequentially evaluating the checkpoints listed below.
# Skips (cell, iter) pairs already in results.csv; appends rows on success.
# Usage (on the node): bash ssh_eval_node.sh
set -x
BASE=/data/home/syang
REPO=${SLIME_REPO:-${BASE}/CM/slime}
LAUNCH=${SLIME_REPO:-${BASE}/CM/slime}/examples/browsecomp/slurm_gb300
LOGD=${SLIME_LOGD:-${BASE}/CM/slime/logs}
RESULTS=${LAUNCH}/eval_results

SPECS=(
  "8B:length_penalty_global_ref:259"
  "8B:length_penalty_trunc:399"
  "8B:length_penalty_trunc:419"
)

# One extracted container reused across evals (node-local /scratch).
enroot list | grep -qx slime-eval || enroot create -n slime-eval ${SLIME_IMAGE:-${BASE}/images/slime-dev.sqsh}

for spec in "${SPECS[@]}"; do
  IFS=: read -r S M I <<< "${spec}"
  grep -q "^qwen3-${S},${M},${I}," ${RESULTS}/results.csv && continue

  CKPT_DIR=${BASE}/ckpts/browsecomp_qwen3-${S}-${M}
  ITER_DIR=$(printf "iter_%07d" ${I})
  [ -d "${CKPT_DIR}/${ITER_DIR}" ] || { echo "missing ${CKPT_DIR}/${ITER_DIR}"; continue; }
  ELOAD=${BASE}/ckpts/eval_tmp/${S}-${M}-${I}
  mkdir -p ${ELOAD}
  ln -sfn ${CKPT_DIR}/${ITER_DIR} ${ELOAD}/${ITER_DIR}
  echo ${I} > ${ELOAD}/latest_checkpointed_iteration.txt

  LOG=${LOGD}/eval_ssh_${S}-${M}-${I}.log
  # RESTRICT_DEV=n: the cluster enroot.conf restricts /dev in containers,
  # which breaks NCCL under manual `enroot start` (pyxis relaxes it itself).
  ENROOT_RESTRICT_DEV=n enroot start --rw --mount /data:/data slime-eval \
    bash ${LAUNCH}/eval_inner.sh ${S} ${M} ${I} ${ELOAD} > ${LOG} 2>&1

  LINE=$(grep -a "eval 0: {" ${LOG} | grep -a "'eval/browsecomp':" | tail -1)
  ACC=$(echo "${LINE}" | grep -oE "'eval/browsecomp': [0-9.]+" | grep -oE "[0-9.]+$")
  RLEN=$(echo "${LINE}" | grep -oE "'eval/browsecomp/response_len/mean': [0-9.]+" | grep -oE "[0-9.]+$")
  TRUNC=$(echo "${LINE}" | grep -oE "'eval/browsecomp-truncated_ratio': [0-9.]+" | grep -oE "[0-9.]+$")
  if [ -n "${ACC}" ]; then
    echo "qwen3-${S},${M},${I},${ACC},${RLEN},${TRUNC},$(date -u +%FT%TZ)" >> ${RESULTS}/results.csv
    echo "EVAL_OK ${S}/${M}@${I} acc=${ACC} len=${RLEN}"
  else
    echo "EVAL_FAILED ${S}/${M}@${I} — see ${LOG}"
  fi
  # Belt-and-suspenders between runs: kill anything the inner trap missed.
  pkill -9 -u $USER -x python3 2>/dev/null; pkill -9 -u $USER -x raylet 2>/dev/null
  sleep 5
done
echo "ALL_SPECS_DONE"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
