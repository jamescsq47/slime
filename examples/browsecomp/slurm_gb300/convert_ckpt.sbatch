#!/bin/bash
# Convert Qwen3-8B HF checkpoint -> Megatron torch_dist (one-time).
#SBATCH -J bc-convert
#SBATCH -A customer
#SBATCH -p batch_1
#SBATCH --qos=batch_1_qos
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -t 04:00:00
#SBATCH -o /data/home/syang/CM/slime/logs/convert_%j.log

set -x
BASE=/data/home/syang
REPO=${SLIME_REPO:-${BASE}/CM/slime}

srun --container-image=${SLIME_IMAGE:-${BASE}/images/slime-dev.sqsh} \
     --container-mounts=/data:/data \
     --container-workdir=${REPO} \
     bash -c "
set -ex
export HOME=${BASE}
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py \
    \${MODEL_ARGS[@]} \
    --hf-checkpoint ${BASE}/models/Qwen3-8B \
    --save ${BASE}/models/Qwen3-8B_torch_dist
"
