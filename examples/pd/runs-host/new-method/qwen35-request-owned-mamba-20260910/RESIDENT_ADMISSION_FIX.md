# TP1 Mamba resident-admission deadlock: fix and load verification

Updated2026-09-11 02:03UTC. Full500 evaluation still running, not final throughput
or accuracy. User-requested stopped r2 and all failed/smoke artifacts retained.

## Cause and minimal change

The early Direct completion sweep committed78 parent+suffix worksets to real
requests, consuming312/315 Mamba slots. They remained in the metadata queue,
behind16 older requests that could not obtain slots. Each tick re-sorted by
original arrival and scanned only16, so the already-resident requests never
reached Prefill. This was a progress deadlock, not simply an undersized pool.

`sglang-qwen35-integration/python/sglang/srt/managers/scheduler.py` now delivers
committed TP1 Mamba worksets to the existing bootstrap/native Prefill consumer
independently of metadata-only NEW I/O scan/admission limits. Requires gate
complete, workset backed, reserved runtime states, handed lease. Qwen3/native
and TP>1 selection unchanged; no changes to Forward batching, data transfer,
fences, Radix references, capacity ratio, Router, harness or deadlines.

Startup-only additional fix: this isolated launch explicitly uses NCCL ports
23910..23917. R3 failed on a randomly selected TCPStore port35937 before tasks
began. Port options default absent for other users of validation launcher.

## Verification

- Old HEAD queue function fails both78-behind16 regressions; patched8/8 pass.
- Fusion engine agentic lifecycle/TP/fault tests518 pass; native Mamba cache
  unit tests4 pass,522 total. Independent state-machine audit GO.
- Old environment-specific external fixtures8 fail API compatibility; optional
  CPU-only numerical tests3 lack compiled CPU ops in this CUDA build. These are
  documented, not reported as passes or changed to satisfy another checkout.
- Diagnostic smoke6/6 exact raw token references,12 P2D and4 D2P Host Mamba
  hash pairs equal;12 generations all released, four Host paths balanced,
  two Direct attempts fully rolled back, all8 engines empty at idle. Diagnostic
  chunk256/alignment256/digeston; NOT performance measurement.
- Formal r4 uses originalchunk8192/default4096/digestoff, ratio0.5/static0.8,
  2P:6D/TP1/c256, same500 SWE tasks and external harness/sampling. Startup and
  checkpoint-wrapper supervision unchanged. Supervisor PID239352.

## Positive Direct / c256 result so far

Independent log audit at02:02:44UTC, after roughly6min of business execution:

| Metric | P0 | P1 |
|---|---:|---:|
| Direct bound |1907|2066|
| Bound requests entering Prefill |1907|2066|
| Subsequently P2D-released |1893|2062|
| Remaining in pipeline |14|4|
| Oldest remaining |2s|1s|
| bind-to-release mean |2.50s|2.53s|
| bind-to-release P90 |4s|4s|

Times are from second-resolution logs, not high-precision transfer benchmarks.
No bound-but-not-rematched backlog; both P continue Forward. P0 allocator
misses0, P1 initial5 remain unchanged. The targeted deadlock no longer reproduces
in this run; native forwarding/release consumes the positive Direct worksets.

Counter-delta interval **02:01:31–02:02:02UTC**, about30s:
P compute10697 token/s total; D3120 token/s total (~520/card); P Forward
93.9%/96.1%; D Forward~99.7% average. End-of-window D running19/18/21/23/24/22
(mean21.17/card), P Mamba26.35%/24.13%, D Attention usage~24.5% mean.
This early diagnostic window must not be compared to a whole-run baseline as
a final speedup. The previous broken run's661 token/s was at another elapsed
stage; do not present their ratio as a controlled throughput gain.

## Artifacts

- Stopped r2: `/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-request-owned-ratio05-20260911-r2` (`STOPPED.md`,65 durable tasks, raw logs,control-final).
- Smoke: `/tmp/pd-persist/fused-qwen35-resident-admission-smoke-20260911-r1` (`VALIDATION.md`, raw token comparisons, state digests).
- Failed startup r3: same formal prefix ending `20260911-r3`, `STARTUP_FAILED.md`.
- Current r4: `/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c256-request-owned-ratio05-20260911-r4`.
- Reusable launcher: `scripts/new_method/run_qwen35_fused_swe500_2p6d.sh`.

Complete task records fsync to requests.completed.jsonl as tasks finish; full
metrics/summary are written on normal evaluation completion. Formal500 accuracy
and completion-time comparison remain outstanding. No GitHub push in this turn.
