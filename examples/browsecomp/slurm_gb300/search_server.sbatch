#!/bin/bash
# BrowseComp-Plus search server: 1 GPU, long-running.
# Writes "http://<node-ip>:8010" to launch/search_server.addr for training jobs.
#SBATCH -J bc-search
#SBATCH -A customer
#SBATCH -p batch_1
#SBATCH --qos=batch_1_qos
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 30-00:00:00
#SBATCH -o /data/home/syang/CM/slime/logs/search_server_%j.log

set -x
BASE=/data/home/syang
REPO=${SLIME_REPO:-${BASE}/CM/slime}
ADDR_FILE=${SLIME_REPO:-${BASE}/CM/slime}/examples/browsecomp/slurm_gb300/search_server.addr
PORT=8010

srun --container-image=${SLIME_IMAGE:-${BASE}/images/slime-dev.sqsh} \
     --container-mounts=/data:/data \
     --container-workdir=${REPO}/examples/browsecomp \
     bash -c "
set -x
export HOME=${BASE}
export HF_HOME=${BASE}/.cache/huggingface
# hostname -I returns the unroutable fabric IP; the short hostname resolves
# to the routable 10.4.x address via cluster DNS.
echo \"http://\$(hostname -s):${PORT}\" > ${ADDR_FILE}
python3 search_server.py \
  --model ${BASE}/models/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --host 0.0.0.0 --port ${PORT}
"
