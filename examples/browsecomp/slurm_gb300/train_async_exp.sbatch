#!/bin/bash
# BrowseComp length-penalty experiment cell — fully-async, 4 GB300 nodes.
# Submit with:
#   sbatch -J bc-<cell> -w rdx-gb300-r01-c[XXX-YYY] \
#     --export=ALL,MODEL_SIZE=8B,MODE=length_penalty train_async_exp.sbatch
# Last node of the allocation = train node (+ its own search server).
#SBATCH -A customer
#SBATCH -p batch_1
#SBATCH --qos=batch_1_qos
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=800G
#SBATCH -t 30-00:00:00
#SBATCH -o /data/home/syang/CM/slime/logs/exp_%x_%j.log

set -x
MODEL_SIZE=${MODEL_SIZE:?"--export=ALL,MODEL_SIZE=8B|32B required"}
MODE=${MODE:?"--export=ALL,MODE=baseline|length_penalty required"}

BASE=/data/home/syang
REPO=${SLIME_REPO:-${BASE}/CM/slime}
LAUNCH=${SLIME_REPO:-${BASE}/CM/slime}/examples/browsecomp/slurm_gb300
IMG=${SLIME_IMAGE:-${BASE}/images/slime-dev.sqsh}
LOGD=${SLIME_LOGD:-${BASE}/CM/slime/logs}

NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
TRAIN_NODE=${NODES[3]}
ROLLOUT_NODES=(${NODES[0]} ${NODES[1]} ${NODES[2]})
TRAIN_IP=$(getent hosts ${TRAIN_NODE} | awk '{print $1}')
SEARCH_URL="http://${TRAIN_NODE}:8010"
echo "cell=${MODEL_SIZE}/${MODE} train=${TRAIN_NODE} (${TRAIN_IP}) rollout=${ROLLOUT_NODES[*]}"

WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ${BASE}/.netrc)

for n in "${ROLLOUT_NODES[@]}"; do
  srun -N1 -n1 -w ${n} --gres=gpu:4 \
       --container-image=${IMG} \
       --container-mounts=/data:/data \
       --container-workdir=${REPO} \
       bash -c "
set -x
export HOME=${BASE}
export HF_HOME=${BASE}/.cache/huggingface
until ray start --address=${TRAIN_IP}:6379 --num-gpus 4 --disable-usage-stats --block; do
  echo 'ray head not up yet, retrying'; sleep 10
done
" > ${LOGD}/rollout_${SLURM_JOB_ID}_${n}.log 2>&1 &
done

srun -N1 -n1 -w ${TRAIN_NODE} --gres=gpu:4 \
     --container-image=${IMG} \
     --container-mounts=/data:/data \
     --container-workdir=${REPO} \
     bash -c "
set -x
export HOME=${BASE}
export HF_HOME=${BASE}/.cache/huggingface
source ${BASE}/.grader_env

python3 ${REPO}/examples/browsecomp/search_server.py \
  --model ${BASE}/models/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --host 0.0.0.0 --port 8010 > ${LOGD}/search_server_${SLURM_JOB_ID}.log 2>&1 &

until curl -sf http://localhost:8010/health > /dev/null; do
  echo 'waiting for search server'; sleep 10
done
echo 'search server healthy'

export LOCAL_SEARCH_URL=${SEARCH_URL}
export MASTER_ADDR=${TRAIN_IP}
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_KEY=${WANDB_API_KEY}
export MODEL_SIZE=${MODEL_SIZE}
export MODE=${MODE}
export RESUME=${RESUME:-0}
bash ${LAUNCH}/run_browsecomp_async_gb300.sh
"
