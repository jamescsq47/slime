# GB300 Slurm/enroot harness for BrowseComp experiments

> For the **other** cluster (DGX `dgx-01..08` / `bcm-01`, x86_64, 8× H100),
> see `../CLUSTER_RUNBOOK_dgx_bcm01.md`. Scripts are not portable between the
> two — that doc lists the concrete differences.

The exact launch/eval/ops scripts used for the BrowseComp length-penalty
experiment sweep on the `rdx-gb300` cluster (18 nodes × 4× NVIDIA GB300
284GB, aarch64, Slurm + pyxis/enroot). Paths are hardcoded to the original
user's layout (`/data/home/syang/...`) — treat these as a faithful record of
what ran; adjust `BASE`/account/partition for reuse.

Results and analysis: `docs/experiments/browsecomp-length-penalty-results.md`.

## Files

| File | Purpose |
|---|---|
| `run_browsecomp_async_gb300.sh` | Parameterized fully-async trainer (`MODEL_SIZE=8B\|32B`, `MODE=baseline\|length_penalty\|length_penalty_global_ref\|length_penalty_trunc`). 1 train node (TP=4) + 3 rollout nodes (3× sglang TP=4). |
| `train_async_exp.sbatch` | 4-node cell launcher. Last node = train node + its own search server. `--export=ALL,MODEL_SIZE=…,MODE=…[,RESUME=1]`. |
| `search_server.sbatch` | Standalone 1-GPU BrowseComp-Plus search server (writes its URL to a shared addr file). |
| `convert_ckpt.sbatch` / `convert_ckpt_32B.sbatch` | One-time HF → torch_dist conversion in-container. |
| `eval_ckpt.sbatch` | Offline eval of one checkpoint (spins its own search server; parses `eval 0:` metrics into `eval_results/results.csv`). |
| `eval_sweep.sh` | Idempotent sweep: evals every saved ckpt not yet in the CSV, N sequential lanes via `sbatch --wait`. |
| `eval_inner.sh` + `ssh_eval_node.sh` | Slurm-free eval path for drained nodes: plain ssh + `enroot start` (needs `ENROOT_RESTRICT_DEV=n` for NCCL). |
| `health_check.sh` | Periodic health probe: slurm state, fresh log errors (offset-tracked), search-server `/health`, penalty stats, ckpt disk. |
| `node_cleanup.sbatch` | Kills orphaned ray/sglang/python payloads after `scancel` (slurm's proctrack does not reap enroot containers on this cluster). |
| `sync_8b_baseline_wandb.py` | Incremental wandb mirror of a run into another project (used to consolidate all cells under `browsecomp-b300`). |
| `plot_eval_curves.py` | Renders `results.csv` into per-cell offline-eval curves and pushes them to wandb. |

## Operational gotchas (all hit for real)

- `enroot import` on the x86 login node pulls amd64 → `Exec format error` on
  aarch64 compute nodes. Use `enroot import -a aarch64`, temp dir on
  node-local disk.
- `scancel` does NOT kill enroot container payloads here — always run
  `node_cleanup.sbatch` on the freed nodes and verify `nvidia-smi` shows 0 MiB
  before treating them as free.
- The step-0 eval burst (150 concurrent sessions) racing the fully-async
  worker startup can kill the RolloutManager with `SYSTEM_ERROR`; pass
  `--skip-eval-before-train` (default in the run script).
- Eval-only mode (`--num-rollout 0`) needs an explicit `--lr-decay-iters`.
- `hostname -I` returns an unroutable fabric IP; use the short hostname (DNS)
  for anything cross-node.
