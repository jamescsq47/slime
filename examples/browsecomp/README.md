# BrowseComp-Plus RL Training

RL training (GRPO) of a deep-research agent on **BrowseComp-Plus**: BrowseComp
questions grounded in a fixed local corpus
([`Tevatron/browsecomp-plus-corpus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus),
~100K docs) with dense retrieval, so no live web access is needed. The
environment, data, and grading are ported from the
[Context-Folding / FoldAgent](https://arxiv.org/abs/2510.11967) open-source
re-implementation; the agent here is a plain linear-context ReAct agent.

The agent has three tools, defined in text format inside the system prompt that
ships with the data:

- `search(query, topk)` — dense retrieval over the corpus
- `open_page(docid | url)` — fetch the full document
- `finish(answer, explanation, confidence)` — submit the final answer

Reward is 0/1 per rollout, scored by the official BrowseComp grader prompt via
an OpenAI-compatible LLM judge (with a lenient exact-match fast path and a
relaxed-EM second-opinion fallback).

## Architecture

```
slime train.py / train_async.py
  └─ browsecomp_agent.generate           (--custom-generate-function-path)
       ├─ SGLang /generate               ── assistant tokens + rollout logprobs
       └─ browsecomp_env.BrowseCompEnv
            │  POST /search, /open       ── FoldAgent search server
            └─ finish → predicted_answer into sample.metadata
  └─ browsecomp_rm.reward_func           (--custom-rm-path)
       └─ BrowseComp grader prompt → LLM judge (GRADER_* env vars)
```

The custom generate function owns the multi-turn loop. Assistant tokens are
trained with loss mask 1; search/open observations are appended to context with
loss mask 0.

## Setup

### 1. Search server (BrowseComp-Plus)

`search_server.py` is vendored here (self-contained; no FoldAgent checkout
needed). It loads the corpus (`Tevatron/browsecomp-plus-corpus`), precomputed
embeddings (`miaolu3/browsecomp-plus`), and `Qwen/Qwen3-Embedding-8B` for query
encoding — all pulled from HuggingFace on first run. Run it on a machine/GPU
separate from training (needs an env with `torch transformers datasets fastapi
uvicorn`):

```bash
cd examples/browsecomp
python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 --port 8010

export LOCAL_SEARCH_URL="http://<search-server-ip>:8010"
```

### 2. LLM judge

Any OpenAI-compatible endpoint:

```bash
export GRADER_API_KEY=...
export GRADER_BASE_URL=...        # optional; default OpenAI
export GRADER_MODEL=...           # e.g. gemini-2.0-flash / gpt-4o-mini
export GRADER_FALLBACK_MODEL=...  # optional; stronger second-opinion judge
```

### 3. Data

The BrowseComp-Plus train/test queries (680 train / 150 test) come from the
FoldAgent release. Download the parquet archive, then convert to slime jsonl:

```bash
cd examples/browsecomp
pip install gdown
python download_data.py --out ./data_raw          # fetches bc_{train,test}.parquet
mkdir -p data
python prepare_data.py --input ./data_raw/bc_train.parquet --output data/bc_train.jsonl
python prepare_data.py --input ./data_raw/bc_test.parquet  --output data/bc_test.jsonl
```

(The parquet is not committed — it's dataset content, not code. `data/` is
git-ignored.)

### 4. Model

```bash
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B

cd /path/to/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist
```

## Run

Single node, colocated (training and inference share 8 GPUs):

```bash
bash examples/browsecomp/run_qwen3_8B.sh
```

Multi-node, fully async and disaggregated (default 1 training node + 3 rollout
nodes; rollout generation runs continuously in a background worker so long
BrowseComp trajectories never block training, TIS corrects the off-policy gap,
and `--max-weight-staleness` bounds how stale a trajectory may be):

```bash
# on each worker node:
ray start --address=<head-ip>:6379 --num-gpus 8
# on the head node:
MASTER_ADDR=<head-ip> NUM_NODES=4 bash examples/browsecomp/run_qwen3_8B_async.sh
```

Defaults mirror the FoldAgent reproduction: 8192-token prompt + 32768-token
session budget, up to 100 ReAct turns with ≤2048 new tokens per turn,
8 rollouts per prompt, temperature 1.0.

Key knobs (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `BROWSECOMP_MAX_TURNS` | 100 | max ReAct turns per rollout |
| `BROWSECOMP_TURN_MAX_NEW_TOKENS` | 2048 | per-turn completion cap |
| `BROWSECOMP_MUST_SEARCH` | 1 | a finish without any prior search scores 0 (and a correct memorized guess is bounced once with a "verify via search" message) |
| `BROWSECOMP_DO_NOT_GIVE_UP` | 0 | bounce "insufficient information" answers once |
| `HF_CHECKPOINT` / `REF_LOAD` / `DATA_DIR` | see script | paths |

## Notes

- **Thinking mode**: the custom generate function renders the Qwen chat template
  on the first turn and appends later user observations directly as masked
  context. With thinking enabled, consider raising
  `BROWSECOMP_TURN_MAX_NEW_TOKENS` so reasoning is not truncated mid-turn.
- **must_search guard**: BrowseComp answers can be memorized; requiring at
  least one search call before `finish` counts prevents the policy from
  collapsing into a closed-book QA model.
- **Judge volume**: each non-EM-matching rollout costs one judge call
  (~batch 32 × 8 rollouts per step upper bound). The relaxed-EM fallback adds
  a stronger-model call only in rare borderline cases.
- **Truncated/failed rollouts**: sessions that hit the token budget, exceed
  max turns, or lose the search backend never submit an answer and receive
  reward 0; samples are still collected. Set `BROWSECOMP_MAX_SEQ_LEN` below the SGLang context length so
  GRPO sees them as negatives.

## Operational lessons (validated on a 4-node Qwen3-4B async run)

These are the non-obvious things that must be right for a multi-node async run
to stay up; each was a real failure we hit and fixed.

- **Context budget must sit below the model's native context.** sglang refuses
  a `--sglang-context-length` larger than the model maximum (40960 for
  Qwen3-4B), so you cannot "add headroom" there. Instead keep sglang at the
  native max and set `BROWSECOMP_MAX_SEQ_LEN` *below* it (the scripts use 36864 = 40960 − 4096).
  The 4096 slack absorbs the final turn's large observation + generation; the
  agent's budget guard stops the loop and marks the sample truncated. Without
  this, mid-rollout requests overflow (`input 42372 > 40960`), sglang returns
  HTTP 400 for every request, and the batch can never form → training stalls.
  A residual ~3% of rollouts still 400 on unusually large observations; the
  agent catches those and ends the rollout gracefully. Capping per-turn
  observation size in `browsecomp_env.py` would remove even those.

- **Disable in-training eval on the async path.** The sync eval at every
  `--eval-interval` step spins up a burst of 150 concurrent agentic sessions
  on top of the checkpoint save + weight update happening on the same step.
  That stack reliably killed a worker with `SYSTEM_ERROR` at every decade-step
  boundary. Set `--eval-interval` very high (we pass `EVAL_INTERVAL=100000`)
  and evaluate checkpoints offline instead — see
  `scripts/run_browsecomp_eval_offline.sh` (slime's `--num-rollout 0`
  eval-only mode). With eval off, training crosses checkpoint boundaries
  cleanly.

- **Physically isolate the search server's GPU from training.** The FoldAgent
  search server re-exports `CUDA_VISIBLE_DEVICES` internally and lands on
  physical GPU 0 regardless of how it's launched, so pin the training container
  off that GPU (e.g. `CUDA_VISIBLE_DEVICES=4,5,6,7` and register fewer GPUs
  with Ray on that node) or put the search server on a node that runs no
  training. Co-locating it on a training GPU causes an OOM on the first
  weight-update step.


- **Weight-update pause mode.** Use `--pause-generation-mode in_place`. The
  default `retract` re-queues in-flight requests and flushes the cache, which
  never drains under the fully-async worker's continuous request stream and
  times out.

- **Non-default Ray ports on shared nodes.** If other users leak Ray clusters
  on the same physical nodes (dashboard agent 52365, GCS 6379), start Ray on
  non-default ports and pass `RAY_ADDRESS` explicitly so `address="auto"`
  doesn't attach to a foreign cluster.

- **Resume.** Checkpoints are saved with `--no-save-optim/--no-save-rng`, so
  resume with `--load <dir> --no-load-optim --no-load-rng` (see `LOAD_DIR` /
  `RESUME_LOAD_DIR` in the run/sbatch scripts). Optimizer momentum restarts
  from zero on resume; weights continue from the checkpointed iteration.

## Attribution & License

This example is derived from the **Context-Folding / FoldAgent** open-source
re-implementation (paper: https://arxiv.org/abs/2510.11967), which is licensed
under the **Apache License 2.0** — the same license as slime.

Ported / vendored from FoldAgent, with modifications:
- `search_server.py` — vendored from FoldAgent `envs/search_server.py`
  (BrowseComp-Plus dense-retrieval backend).
- `browsecomp_env.py` — search/open_page/finish tool env, function-call parser,
  and `em_score`, adapted from FoldAgent `envs/local_search.py`.
- `browsecomp_rm.py` — BrowseComp grader prompt, judge parsing, and relaxed-EM
  fallback, adapted from FoldAgent `envs/local_search.py`.
- `prepare_data.py` / `download_data.py` — consume FoldAgent's BrowseComp-Plus
  parquet split.

The agent (`browsecomp_agent.py`) is an independent linear-context ReAct
implementation for slime's custom-generate path; FoldAgent's context-folding
agent and FoldGRPO are not used here.

Datasets pulled at runtime keep their own terms: `Tevatron/browsecomp-plus-corpus`,
`miaolu3/browsecomp-plus`, and the BrowseComp / BrowseComp-Plus queries.
