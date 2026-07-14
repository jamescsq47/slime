#!/bin/bash
# Kill orphaned ray/sglang/miles processes left on a node after scancel
# failed to reap them (enroot containers escape slurm's proctrack here).
#SBATCH -A customer
#SBATCH -p batch_1
#SBATCH --qos=batch_1_qos
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /data/home/syang/CM/miles-search/logs/cleanup_%j.log
set -x
hostname
echo "=== before ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# Kill this user's compute payloads by exact process-name match so this
# script's own bash/slurm processes survive.
for name in python3 python raylet gcs_server sglang; do
  pkill -9 -u $USER -x ${name}
done
sleep 5
echo "=== after ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ps -u $USER -o pid,etime,comm --no-headers | head -20
