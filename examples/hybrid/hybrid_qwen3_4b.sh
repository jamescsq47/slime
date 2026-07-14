#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

pkill -u $(whoami) -f "ray_csq" 2>/dev/null
sleep 1
rm -rf /tmp/ray_csq
mkdir -p /tmp/ray_csq
TEMP_DIR="/tmp/ray_csq"
set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# This host has four isolated NVLink pairs: 0-1, 2-3, 4-5 and 6-7.
# Keep the communication-heavy training tensor-parallel groups inside a pair.
# Ray orders local GPUs by physical ID, so actor ranks 0-3 use GPUs 0-3 and
# rollout engines use GPUs 4-7.
TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
ROLLOUT_GPUS_PER_ENGINE=${ROLLOUT_GPUS_PER_ENGINE:-1}

if (( TRAIN_GPUS + ROLLOUT_GPUS != 8 )); then
    echo "TRAIN_GPUS + ROLLOUT_GPUS must equal the 8 GPUs reserved by this script"
    exit 1
fi
if (( TRAIN_GPUS % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "TRAIN_GPUS must be divisible by TENSOR_MODEL_PARALLEL_SIZE"
    exit 1
fi
if (( ROLLOUT_GPUS % ROLLOUT_GPUS_PER_ENGINE != 0 )); then
    echo "ROLLOUT_GPUS must be divisible by ROLLOUT_GPUS_PER_ENGINE"
    exit 1
fi

GPU_TOPOLOGY=$(nvidia-smi topo -m 2>/dev/null || true)
for GPU_PAIR in "GPU0 GPU1" "GPU2 GPU3"; do
    read -r GPU_A GPU_B <<< "${GPU_PAIR}"
    if ! awk -v a="${GPU_A}" -v b="${GPU_B}" '
        $1 == a {
            column = substr(b, 4) + 2
            if ($column ~ /^NV/) found = 1
        }
        END { exit !found }
    ' <<< "${GPU_TOPOLOGY}"; then
        echo "WARNING: expected NVLink pair ${GPU_A}-${GPU_B} was not detected"
    fi
done

# NVLS is an NVSwitch/NVLink-SHARP collective path. Isolated A100 PCIe pairs
# do not provide it; NCCL still uses ordinary NVLink P2P with NVLS disabled.
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
echo "Parallel layout: train=${TRAIN_GPUS} (TP=${TENSOR_MODEL_PARALLEL_SIZE}, DP=$((TRAIN_GPUS / TENSOR_MODEL_PARALLEL_SIZE))), rollout=${ROLLOUT_GPUS} / ${ROLLOUT_GPUS_PER_ENGINE}-GPU engines"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
echo "=== Running hybrid async benchmark: mode=${MODE} ==="

CKPT_ARGS=(
   --hf-checkpoint /workspace/Qwen3-4B
   #--hf-checkpoint /root/Qwen3-8B-FP8
   --ref-load /workspace/Qwen3-4B_sft_torch_dist
   # --load /root/Qwen3-8B_slime/
   --save /workspace/Qwen3-4B_async_math/
   --save-interval 200
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project hybrid-qwen3-4b
   --wandb-group qwen3-4B-hybrid-math
   --wandb-key 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
)

PROMPT_SET=/workspace/data/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout 800
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192 # 8192&512 
   --rollout-temperature 1

   --global-batch-size 256
   --num-steps-per-rollout 1
   --balance-data
   --rollout-health-check-interval 30
   --rollout-health-check-timeout 30
   # --save-debug-rollout-data /workspace/slime/examples/hybrid/debug/ratio_1_8192/rollout_{rollout_id}.pt
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1 # 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.5
)

CUSTOM_ARGS=(
   --data-source-path custom_data_source.CustomDataSource
   --custom-generate-function-path generate_with_hybrid.generate_unified
   --custom-rm-path generate_with_hybrid.reward_func_unified
   --math-data-path /workspace/data/dapo-math-17k/dapo-math-17k.jsonl
   --qa-data-path /workspace/Search-R1/data/nq_hotpotqa_train/train.parquet
   --math-ratio 1
   # --batch-alternation \
   # --math-batches-per-cycle 100 \
   # --qa-batches-per-cycle 0 \
#    --dynamic-alternation \
#    --lag-version 3 \
    --mask-offpolicy-math 51200
    --mask-offpolicy-qa 51200
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container

# single node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
ray start --head --node-ip-address ${MASTER_ADDR} --temp-dir ${TEMP_DIR} --dashboard-port ${DASHBOARD_PORT} --num-gpus 8 --num-cpus 64 --disable-usage-stats

# Wait for the Ray dashboard and job agent to be fully ready
echo "Waiting for Ray agent to initialize..."
for i in {1..30}; do
  if curl -s http://${MASTER_ADDR}:${DASHBOARD_PORT}/api/jobs/ 1>/dev/null; then
    echo "Ray agent is up!"
    break
  fi
  sleep 1
done

#multi-node
# a10 main
# export MASTER_ADDR="10.0.1.170"
# ray start --head \
#     --node-ip-address=${MASTER_ADDR} \
#     --port=6379 \
#     --num-gpus 8 --num-cpus 64 --disable-usage-stats
# a11 worker 
# export MASTER_ADDR="10.0.1.170"
# export MY_NODE_IP=$(hostname -I | awk '{print $1}')
# ray start --address="${MASTER_ADDR}:6379" \
#           --node-ip-address="${MY_NODE_IP}" \
#           --num-gpus 8 \
#           --num-cpus 64 \
#           --disable-usage-stats


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_ORDER\": \"PCI_BUS_ID\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\"
  }
}"

MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
echo "=== Running hybrid async benchmark: mode=${MODE} ==="

# --- Mode-specific flags ---
MODE_ARGS=()
case "${MODE}" in
    one_step_off)
        ;;
    fully_async)
        MODE_ARGS+=(
            --fully-async-debug-version-tracking
        )
        ;;
    window_partial)
        MODE_ARGS+=(
            --fully-async-debug-version-tracking
            --fully-async-buffer-policy window_evict
            --fully-async-version-window "${FULLY_ASYNC_VERSION_WINDOW}"
            --fully-async-max-completed-samples "${FULLY_ASYNC_MAX_COMPLETED_SAMPLES}"
            --fully-async-eviction-policy "${FULLY_ASYNC_EVICTION_POLICY}"
            --partial-rollout
            # --mask-offpolicy-in-partial-rollout
        )
        ;;
    staleness_partial)
        MODE_ARGS+=(
            --fully-async-debug-version-tracking
            --fully-async-buffer-policy legacy_backpressure
            --staleness-threshold 0.5
            --partial-rollout
            --mask-offpolicy-in-partial-rollout
        )
        ;;
    *)
        echo "Unknown MODE: ${MODE}. Use one of: one_step_off, fully_async, window_partial, staleness_partial"
        exit 1
        ;;
esac

ray job submit --address="http://${MASTER_ADDR}:${DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${TRAIN_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${MODE_ARGS[@]}
