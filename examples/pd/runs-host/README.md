# Retained host results

This directory contains reproducible formal baselines, the latest validated
agentic-PD runs, and a small set of ablations needed to explain the design.
Failed, incomplete, superseded, smoke, verification, and diagnostic runs were
removed most recently on 2026-08-23.

## Qwen3-32B, TP=2

- `baseline/formal-qwen3-32b-tp2-browsecomp-colocated-c192-w300-m1200-20260817-r2`
  and `baseline/formal-qwen3-32b-tp2-browsecomp-colocated-c256-w300-m1200-20260820-r1`:
  pure-BrowseComp colocated references using the fixed source-order workload.
- `new-method/formal-qwen3-32b-tp2-browsecomp-2p6d-transfer-list-c192-w300-m1200-20260820-r53`
  and `new-method/formal-qwen3-32b-tp2-browsecomp-4p4d-transfer-list-c192-w300-m1200-20260820-r16`:
  validated c192 topology comparison.
- `new-method/formal-qwen3-32b-tp2-browsecomp-2p6d-promote-c256-w300-m1200-20260823-r1`:
  immediately preceding validated c256 implementation.
- `new-method/formal-qwen3-32b-tp2-browsecomp-2p6d-lifecycle-lock-c256-w300-m1200-20260823-r1`:
  preceding c256 result with one receiver lifecycle lock and TP0-owned Direct
  admission. It completes 724 agents in 1,200 seconds, reaches 1,522.3 Decode
  token/s, and reuses 100% of eligible previous-turn Decode KV.
- `new-method/formal-tp2-background-direct-browsecomp-2p6d-c256-w300-m1200-20260823`:
  current validated BrowseComp TP=2 state.  The physical topology is 2P:6D
  (one logical P group and three logical D groups), with c256 closed-loop load,
  300 seconds of warmup, and 1,200 seconds of measurement.  Background Direct
  progress is decoupled from Prefill forward, the run completes without TP-rank
  divergence or KV loss, and total Decode throughput is 1,433.3 token/s.
- `baseline/formal-qwen3-32b-tp2-terminal-bench-colocated-c256-turn8192-w300-m1200-20260821-r5`
  and `new-method/formal-qwen3-32b-tp2-terminal-2p6d-direct-admission-simplified-c256-w300-m1200-20260822-r5`:
  retained Terminal-Bench colocated/PD pair.

The aligned c256 BrowseComp comparison is in
`QWEN32_TP2_C256_PD_VS_COLLOCATED.md`.

## Qwen3-8B and earlier architecture baselines

- `baseline/current-workload-colocated-4gpu-c256-s2026-w300-m1200`:
  mixed-workload four-GPU colocated reference.
- `baseline/refresh-four-gpu-1p3d-c256-s2026-w300-m1200-20260812` and
  `baseline/refresh-four-gpu-2p2d-c256-s2026-w300-m1200-20260812`:
  no-reverse and stock-Mooncake PD references.
- `baseline/refresh-pure-colocated-4gpu-c256-s2026-w300-m1200-20260812` and
  `baseline/refresh-pure-pd-c256-s2026-w300-m1200-20260812`:
  Retool-only and BrowseComp-only characterization.
- `baseline/mixed-colocated-8gpu-inference-only-s2026-w300-m1200` and
  `baseline/mixed-pd-8gpu-c512-s2026-w300-m1200`:
  aligned c512/c640/c768 colocated and stock-PD references.
- `baseline/formal-browsecomp-source-order-colocated-8gpu-c384-w300-m1200-20260816-r1`,
  `...-c512-...`, and `...-c576-...` plus the matching retained new-method
  runs: pure-BrowseComp concurrency and Direct-reserve studies.
- `baseline/formal-browsecomp5-retool1-colocated-8gpu-c512-w300-m1200-20260815-r1`
  and `new-method/formal-browsecomp5-retool1-4p4d-domainfix-c512-w300-m1200-20260816-r1`:
  fixed BrowseComp:Retool=5:1 comparison.
- `new-method/1p3d-c256-s2026-w300-m1200`,
  `new-method/formal-step2-radix-pin-2p6d-c512-w300-m1200-20260815-r1`, and
  `new-method/formal-retool2-browsecomp1-step2-radix-pin-2p6d-c512-w300-m1200-20260815-r1`:
  validated 1P:3D and 2P:6D results.
- `new-method/formal-no-slow-path-step2-radix-pin-2p6d-c512-w300-m1200-20260815-r1`:
  slow-path ablation.

Each retained formal run keeps its raw requests, two-second engine counters,
service logs, configuration, environment fingerprint, summary JSON, and plots.

Three deleted runs can temporarily remain as tiny directory shells containing
NFS `.nfs*` files while external Mooncake processes still hold their old log
descriptors. They contain no retained experimental data and can be removed
after those processes exit.
