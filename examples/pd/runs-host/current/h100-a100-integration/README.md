# H100 implementation integrated on A100

Baseline branches: Slime H100 `2f9ed17`, merged with A100 `82e18c1`;
SGLang pd_node_h100 `20dd76e`, with selected A100 safety fixes.

## Scope

H100 Router, native compute admission, Direct deadlines, TP coordination,
registered memfd windows and batched CUDA copy protocol remain the base.
The Slime merge adds A100 workload/harness/configuration support and retains
H100 PD control code. Deferred allocator frees own their index tensors.
Final safety changes and formal metrics are recorded after verification.

## Verification scope

- Paired Slime PD control/Host/priority/progress/claim suite: 219 passed.
- Workload plugins and SWE harness suite: 47 passed with the existing
  `/homes/siqic/mini-swe-agent-v2/src` optional dependency on PYTHONPATH.
- Router suite after test-session cleanup: 98 passed, no unclosed session.
- Indexed DMA descriptor byte-address equivalence: 4 passed; real CUDA
  P→D round trip and D→P source-release tests: 2 passed.
- SGLang final lifecycle/TP/DMA descriptor/integration safety suite: 321 passed.
  Four Slime test fixtures were updated to carry explicit generation IDs;
  the affected 36 Host/release tests passed without weakening assertions.
- Merged staged sources: 34 Python AST checks and 17 shell syntax checks passed.
- Broader inherited training tests are not all green: the A100 commit lacks
  `examples/mixed/terminal_agent.py`, the pd inference environment lacks
  `wandb`, and 11 three-domain training tests reference APIs absent from that
  branch's `examples/mixed/custom_data_source.py`. These are separate from
  the tested PD workload plugin path; no training-test pass is claimed.

The canonical BrowseComp schedule was absent from both branch Git trees
although launchers referenced it. It is now tracked, byte-identical to the
original A100 file (SHA256 `156cee416e3e3771de680c38f26a70f8347f9d67c4c67c14f0d05d89826838d6`).

## DMA measurements on A100

All raw JSON files, including failed gates, are under `dma/`.

| Probe | D2H | H2D | Outcome |
|---|---:|---:|---|
| Registered 1 GiB extent, wall bandwidth | 13.82 GiB/s | 21.74 GiB/s | Strict gate passed |
| Original batch descriptors, two lanes, NUMA local | 11.28–11.55 GiB/s/lane | 11.19–11.48 GiB/s/lane | Relaxed A100 gate passed |
| Vectorized descriptors, two lanes, NUMA local | 11.34–11.60 GiB/s/lane | 11.20–11.48 GiB/s/lane | Relaxed A100 gate passed |

The paired NUMA-local probes use 20 repetitions and 300 baseline forwards.
The original H100 gate (10 GiB/s/lane, Forward P50 +5%, P95 +10%) is **not
passed** by the batch probe on this A100. The explicitly relaxed gate is
8 GiB/s/lane, P50 +20%, P95 +60%; benchmark defaults were not changed.

With vectorized descriptors, the tiny GEMM's D2H P50/P95 regressions are
-2.33%/+23.91%, and H2D +15.79%/+51.22%. H2D P95 changes from 41.98 to
63.49 microseconds. These are microbenchmark event times, not serving
Forward utilization or application throughput.

CPU-only descriptor construction (1,000 timed repetitions, 36 layers,
4,096 tokens, fragmented 64-token pages) drops from median 879.91 to
126.17 microseconds; P95 drops from 999.10 to 139.12 microseconds. The
paired GPU probe shows essentially unchanged saturated DMA bandwidth.
The optimization removes repeated per-layer Torch slices; it does not
change byte addresses, ownership, fences or transfer ordering.

## Formal configuration

`scripts/new_method/run_h100_integration_a100.sh` records the launch:
Qwen3-8B, BrowseComp canonical source-order n680 cycling, temperature 0,
4P:4D TP=1, c512, warmup 300 seconds and measurement 1,200 seconds.
User changed concurrency from 384 to 512 during service startup; the c384
attempt was stopped before workload execution and is not a performance result.
P GPUs 0/2/4/6, D GPUs 1/3/5/7, search on GPU7.
To compare with A100 r7, D mem fractions remain 0.85/0.85/0.85/0.74;
D→P Host is 128 GiB/P and P→D Host is 32 GiB/P. The H100 implementation
uses memfd, two DMA lanes, 4,096-token chunks and a 640 GiB registration
cache limit. Native SGLang HiCache/Mooncake is disabled.

Formal result: **failed and terminated; no valid throughput result**.
The c512 attempt encountered long runtime Host arena registration stalls,
then D3 on GPU7 failed with CUDA OOM while sharing the card with the search
service. Full chronology and raw monitoring samples are retained in
[the failure report](browsecomp-qwen3-8b-tp1-4p4d-c512-w300-m1200-r1/FAILURE.md).
The intended 300+1200 measurement was not successfully completed.
