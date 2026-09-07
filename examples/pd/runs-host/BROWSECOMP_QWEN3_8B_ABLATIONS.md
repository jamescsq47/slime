# BrowseComp / Qwen3-8B / Agentic PD Ablations

## Fixed setting

All rows use the same serving workload and differ only in the named ablation.

| Item | Value |
|---|---|
| Model | Qwen3-8B |
| Workload | BrowseComp, fixed source-order n680 schedule, cycling as needed |
| Layout | 4P:4D, TP=1, global Host recovery |
| Concurrency | 512 closed-loop agents |
| Sampling | `temperature=0` |
| Timing | 300 s warmup + 1200 s measurement |
| Request source | `fixed_browsecomp_source_order_n680.json` |
| Native HiCache/Mooncake | disabled |
| Decode admission target | `D_TARGET_KV_FRACTION=1.0` |
| P→D Host decision grace | `P2D_SPILL_DELAY_SECONDS=0.5` |

## Primary results

| Variant | D→P Direct | D→P Host | P→D Host | P/D routing | Decode token/s | Decode token/s/D | Agent/s | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Full method (previous retained reference) | on | on | on | load-aware | 4,436 | 1,109 | 2.219 | retained reference |
| Full method (2026-09-02 rerun) | on | on | on | load-aware | — | — | — | failed: 95 P-ready 600 s timeouts; no valid throughput |
| D→P Direct only | on | off | on | load-aware | 4,662 | 1,165 | 2.204 | complete; 300+1200 s |
| D→P Slow only | off | on | on | load-aware | — | — | — | pending |
| Random P/D routing | on | on | on | random among capacity-feasible workers | — | — | — | pending |
| P→D Direct only | on | on | off | load-aware | — | — | — | pending |

## Resource and workload details

| Variant | Prefill token/s | Actual Prefill/Agent | Decode/Agent | Parent KV reuse | P Forward/card | D Forward/card | P KV | P queue | P inflight | D KV | D running | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full method (previous retained reference) | 30,709 | 13,804 | 1,994 | 99.93% | 81.2% | 98.4% | 89.3% | 18.1 | 18.3 | 86.3% | 41.3 | 0.0 | 0.6 |
| Full method (2026-09-02 rerun) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| D→P Direct only | 38,263 | 17,329 | 2,111 | 90.15% | 96.84% | 96.53% | 64.06% | 64.95 | — | 67.08% | 44.29 | 1.13 | 0.40 |
| D→P Slow only | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Random P/D routing | — | — | — | — | — | — | — | — | — | — | — | — | — |
| P→D Direct only | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Run directories

| Variant | Run directory | Status |
|---|---|---|
| Full method (previous retained reference) | `current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/new-method-agentic-pd` | retained reference |
| Full method (2026-09-02 rerun) | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-work-conserving-20260902-r1/full` | failed; stopped after repeated P-ready timeouts |
| D→P Direct only | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-work-conserving-20260902-r1/d2p-direct-only` | complete; formal window |
| D→P Slow only | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/d2p-slow-only` | pending |
| Random P/D routing | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/random-routing` | pending |
| P→D Direct only | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-direct-only` | pending |

The full-method row is the retained formal result. Ablation rows are marked
complete only after the full 300+1200 second window and path/ownership checks.

## 2026-09-02 rerun notes

- The Full rerun was not assigned a throughput result. During the measurement
  window, 95 requests reached the 600-second P-ready timeout and Router returned
  HTTP 500. At the failure point some P workers had zero running requests but
  341k–375k physical KV tokens resident out of 372k–376k. D→P Host pressure had
  triggered 241 whole-snapshot evictions, while requests associated with lost
  readiness continued waiting. This is a lifecycle/progress failure rather than
  a Host-transfer bandwidth limit.
- D→P Direct-only completed with zero request failures and no P-ready timeout.
  It completed 2,645 agents in the measurement window. The page-aligned parent
  reuse rate fell to 90.15% because Direct misses cannot recover through Host;
  the resulting extra Prefill was about 3,107 tokens per completed agent.
- The remaining ablations were intentionally not started after this run, per
  the user request to pause after D→P Direct-only completed.
