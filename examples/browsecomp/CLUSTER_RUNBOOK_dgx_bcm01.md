# BrowseComp runbook — DGX / bcm-01 cluster (internal)

Environment-specific launch notes for the **DGX cluster** (x86_64 `dgx-01..08`,
jump host `bcm-01`, Slurm + pyxis/enroot, 8× H100 per node). This is where the
example was first brought up (the 4B and initial 8B runs).

> **This is NOT the GB300 cluster.** The length-penalty experiment sweep ran on
> a different cluster (`rdx-gb300`, aarch64, 4× GB300 per node) — see
> `slurm_gb300/README.md` and `docs/experiments/browsecomp-length-penalty*.md`
> for that one. The two clusters differ enough that scripts are **not** portable
> between them; see "Differences vs the GB300 cluster" below.

The portable example lives alongside this file (see README.md); this runbook
captures how we actually launched it on DGX, including the non-obvious failures
and their fixes. The launch scripts referenced here live outside the repo (they
hardcode node names, paths, and read secrets), under
`/nobackup/shangy/miles/scripts/`.

## Differences vs the GB300 cluster

| | DGX / bcm-01 (this doc) | GB300 (`slurm_gb300/`) |
|---|---|---|
| Nodes | `dgx-01..08`, x86_64 | `rdx-gb300-r01-c[001-018]`, aarch64 |
| GPUs/node | 8× H100 (80 GB) | 4× GB300 (284 GB) |
| Container image | `miles-latest.sqsh` (x86) | `radixark/miles:dev` imported `enroot import -a aarch64` |
| Internet on compute | none → judge relay on `bcm-01` | (per GB300 setup; see its README) |
| Search server GPU | shares train node GPU 0 (host conda) | one GPU on the train node (see its sbatch) |
| Train topology | 8-GPU nodes; tp4 on 4 GPUs, IP-sorted placement | 4-GPU nodes, TP=4 = whole node |
| Launch | slurm sbatch **or** ssh+enroot workaround (nodes got squatted) | slurm sbatch (`train_async_exp.sbatch`) |
| Scripts | `/nobackup/shangy/miles/scripts/*` (out of repo) | `examples/browsecomp/slurm_gb300/*` (in repo) |

Most "failures we hit" below are cluster-agnostic miles/BrowseComp lessons and
apply to both; a few are DGX-specific (noted inline).

## Topology

- **Search server**: one GPU, host conda env (`FoldAgent`). Loads
  Qwen3-Embedding-8B + corpus embeddings (~16 GB). It always lands on the
  node's physical GPU 0 (it re-exports `CUDA_VISIBLE_DEVICES` internally), so
  keep GPU 0 free for it on whatever node it runs.
- **Judge**: Gemini via OpenAI-compat endpoint. Compute nodes have no internet,
  so a plain-HTTP relay runs on the jump host (`bcm-01`) and forwards to
  `generativelanguage.googleapis.com`. Point `GRADER_BASE_URL` at the relay.
- **Training/rollout**: miles container (`miles-latest.sqsh`), disaggregated
  async — 1 train node + N rollout nodes.

## Two ways to launch

### A. Slurm + pyxis (preferred when nodes are schedulable)

- `submit_browsecomp_8b_async_full.sbatch` (8B) / `submit_browsecomp_async_full.sbatch` (4B):
  allocate nodes, start the search server (host conda) on node 0, then run the
  container training via `srun --container-image`.
- Resume: `sbatch --export=ALL,RESUME_LOAD_DIR=<ckpt>,EVAL_INTERVAL=100000 ...`
- Eval offline: `EVAL_LOAD_DIR=<ckpt> sbatch submit_browsecomp_eval_offline.sbatch`

### B. Pure ssh + enroot (when nodes are squatted outside slurm)

When other users hold nodes via slurm-external processes (so slurm won't grant
them), launch directly:

- `ssh_launch_8b.sh` — writes per-node scripts to shared `/nobackup`, then
  `ssh -f` each node to `enroot start` the container and run the (slurm-free)
  `run_browsecomp_qwen3_8b_ssh.sh`. Node roles set via env
  (`SLURM_PROCID`, `HEAD_NODE`, `NUM_MILES_NODES`, ...).
- `run_browsecomp_qwen3_8b_ssh.sh` — same as the async run script but with the
  global `pkill`/`ray stop` lines removed (enroot shares the host PID namespace,
  so a blanket pkill would kill other users' processes).

## Supporting services

- **Judge relay** (`browsecomp_judge_relay.py`): run on bcm-01,
  `python browsecomp_judge_relay.py --port 18080`. Forwards to Gemini,
  passes the `Authorization` header through untouched.
- **wandb sync** (`sync_wandb_browsecomp_loop.sh`): compute nodes run
  `WANDB_MODE=offline`; this loop on bcm-01 syncs offline runs to the cloud
  every 5 min. Project `ys-2020/miles-browsecomp`.
- **8B checkpoint conversion** (`convert_qwen3_8b.sbatch`): HF → megatron
  torch_dist (once).
- **auto-resume** (`autoresume_8b_once.sh` + cron): submits/relaunches when
  clean nodes appear. Idempotent.

## Failures we hit (and fixes) — read before launching

1. **Ray placement puts training on the wrong node.** miles sorts placement
   bundles by node IP ascending; the actor (training) takes the lowest-IP
   node's GPUs. Make the **train node the lowest IP** and register exactly the
   train GPU count there; rollout goes to the higher-IP node(s). Otherwise
   training and rollout collide on one node → OOM.

2. **Context overflow stalls training.** sglang rejects
   `--sglang-context-length` above the model's native max (40960). Keep sglang
   at native and set `BROWSECOMP_MAX_SEQ_LEN` below it (36864). Without headroom,
   mid-rollout requests exceed 40960, every request 400s, the batch never
   forms, training hangs.

3. **In-training eval kills a worker.** The sync eval (150 concurrent agentic
   sessions) on top of checkpoint+weight-update at each decade step reliably
   crashed a worker with `SYSTEM_ERROR`. Run with `EVAL_INTERVAL=100000` (off)
   and evaluate checkpoints offline instead.

4. **(DGX-specific) 8B OOM on 4 training GPUs with tp2×cp2.** Optimizer states only shard by
   tp=2. Use **tp4×cp1** so params+optimizer shard 4-way.

5. **(DGX-specific) Search server collides with training on GPU 0.** It ignores the launched
   `CUDA_VISIBLE_DEVICES` and uses physical GPU 0. Put training on GPUs 4-7
   (`CUDA_VISIBLE_DEVICES=4,5,6,7`) so it's disjoint, or put search on a node
   with a free GPU 0.

6. **(DGX ssh-launcher) Grader "invalid API key".** The ssh launcher must `source` FoldAgent's
   `.env` to resolve `GRADER_API_KEY=${GRADER_API_KEY:-<key>}`; a naive
   `grep | cut` passes the literal `${...}` expression.

7. **(DGX ssh+enroot) Ray "Session name does not match persisted value" on relaunch.** Leftover
   ray GCS/dashboard state on a node (ports 6380/8266/8277, `/tmp/ray*`) from a
   prior run. Before relaunch, kill `gcs_server|raylet|ray::|plasma|dashboard`
   (by pattern **and** by the PID holding the port), free the ports, and
   `rm -rf /tmp/ray_browsecomp_*`.

8. **Old miles `agentic_tool_call` args "unrecognized".** In slime this
   example uses `--custom-generate-function-path browsecomp_agent.generate`.
   Do not pass `--custom-agent-function-path` or session/TITO flags.

9. **Weight-update flush timeout under fully-async.** Use
   `--pause-generation-mode in_place` (the default `retract`+flush never drains
   under the continuous request stream).

10. **(DGX, squatted cluster) Shared-node hygiene.** Only ever kill your **own** processes
    (`pkill -u shangy ...`); never blanket-kill on nodes other users share.
