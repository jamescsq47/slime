# Native Mooncake c576 retry — 2026-09-10

Requested completion of the missing row in `BROWSECOMP_QWEN3_8B.md`.
Prior failed runs are retained unchanged. This run does not modify the custom
`pd`/`pd_mamba` engines, shared harness, or native `pd_baseline` engine.

## Configuration

- Qwen3-8B; TP=1; P GPUs 0/2/4/6, D GPUs 1/3/5/7.
- Closed-loop c576; fixed BrowseComp n680 source order, cycling; seed 2026.
- 300 s business warmup + 1200 s measurement; engine startup excluded.
- Context 40960; total response budget 36864; 2048 tokens/turn; 100 turns.
- Temperature 0, top-p 1, top-k -1; inference logprobs disabled.
- P/D mem fractions 0.80, except D GPU7 0.60 sharing the search service.
- Native P HiCache 128 GiB/P, D offload Host 56 GiB/D, Mooncake 256 GiB.
- Native NIXL P→D; native Mooncake TCP storage; no custom lifecycle/routing.
- Reproducer: `run.sh`, invoking the existing `scripts/baseline/run_pd_case.sh`.
- Existing process-group cleanup traps own only this run's started services.

## Preflight

- All 8 GPUs were idle before startup; no other agent's services were stopped.
- `pd_baseline` environment check passed; SGLang 0.5.10.post1,
  Mooncake 0.3.12.post1, no custom agentic modules/environment variables.
- Decode offload source SHA256 matches the preceding baseline:
  `7fb5df97e109b23fb14138af67cab1c86a6ec33015bcdca33b73fcc15d532883`.
- Repository-local and external HF embedding caches are byte-identical:
  `45e3a05b49f04227a9b80843f2cc1fa0a37657f036884ecfd3bc271362c9a224`.
- Search-only HTTP preflight: 1/1, 32/32, 576/576 successful responses with
  five documents each. The c576 burst took 8.23 s. Raw response status/timing
  in `search_preflight.json`; reproducible probe in `probe_search.py`.
- This preflight is not proof that the previous search fault is fixed: the
  GPU was not yet sharing D, and only short representative queries were used.
  Actual multi-turn traces and search errors must be checked during the run.

## Acceptance

Report throughput only after a complete valid formal window. Check search
errors/aborts, model-call counts, length distributions, engine errors, and
native cache metrics. Do not relabel aborted one-turn trajectories as a valid
BrowseComp throughput measurement. No numerical-correctness claim follows
solely from a successful DMA or HTTP status.
