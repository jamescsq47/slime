# BrowseComp train-trajectory SFT pipeline

This directory is intentionally separate from the existing BrowseComp RL and
evaluation implementation. It does not modify or replace their agent, reward,
or launch scripts.

```bash
export LOCAL_SEARCH_URL=http://127.0.0.1:8000
export GRADER_API_KEY=...
export GRADER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export GRADER_MODEL=gemini-3-flash-preview
bash examples/browsecomp/sft/generate.sh

python examples/browsecomp/sft/export_sft.py \
  --input '/workspace/data/browsecomp/qwen3_8b_sft_rollouts/*.pt' \
  --output /workspace/data/browsecomp/browsecomp_qwen3_8b_sft.jsonl

bash examples/browsecomp/sft/train.sh
```

On the target 8xRTX A6000 host, generation defaults to GPU1-7 with seven TP=1
SGLang engines while the vendored search worker owns physical GPU0. Set `GENERATION_GPUS` and
`INFER_TP` to override this partition.

Generation samples eight trajectories for each of the 680 train questions.
Non-exact answers require two positive judge calls. Filtering requires a
correct completed trajectory, train split, search, and an opened evidence page,
and keeps at most four short unique trajectories per question.

The 8-GPU SFT default is TP=2 on 48GB/PCIe hosts and TP=1 on 80GB
NVLink/NVSwitch hosts. CP=1 and PP=1. Set `TP`, `CP`, or `PP` to override the
detected topology. CP is
disabled because FlashAttention 2.8's variable-length CP path hangs on this
Ampere host; fused and unfused Transformer Engine attention do not support CP.
PP=2 was also avoided because dynamic long-sequence microbatches left the
second pipeline stage idle. Full recomputation, distributed optimizer and
dynamic token batching reduce OOM risk.

Training defaults to 100 epochs; set `NUM_EPOCH` to override it. The long-run
learning rate is 1e-6 with cosine decay to 1e-7 and 1% warmup.
If `SAVE_PATH/latest_checkpointed_iteration.txt` exists, `train.sh`
automatically resumes model, optimizer, scheduler, rollout id, and dataset
cursor from that checkpoint. It also fails before starting Ray when no usable
NVIDIA GPU is visible.
The default global batch is 16 to bound synchronization skew from uneven
long trajectories; distributed collectives use a 30-minute timeout.
Dynamic-batch microbatch-count metadata uses the DP Gloo group so the first
lazy NCCL DP collective cannot deadlock before model forward begins.
The NCCL watchdog heartbeat is 3600 seconds so long FlashAttention/recompute
kernels do not trigger PyTorch's shorter watchdog false positive.
SFT cross-entropy log probabilities are computed in 512-token chunks to avoid
materializing a multi-gigabyte FP32 `[sequence, vocab]` temporary.
The launcher uses FlashAttention with a 12,288-token training context. Older
evidence payloads are compacted by `sft_rollout.py` to keep every retained
trajectory within that bound.

To continue with mixed BrowseComp + Retool RL from the final SFT checkpoint,
use the staged launcher in `examples/mixed`:

```bash
STAGE=sft bash examples/mixed/browsecomp_qwen3_8b_sft_then_rl.sh
# After SFT completes (and after starting the multi-node Ray cluster):
STAGE=rl bash examples/mixed/browsecomp_qwen3_8b_sft_then_rl.sh
```
