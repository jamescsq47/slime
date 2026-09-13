# Full Router acceptance: 300 s warmup + 1200 s measurement

> **历史结果，需重跑：** 本轮采用旧“Direct失败→Slow”语义和未对齐显存比例，
> 不代表 2026-09-08 起的“快慢路径+Direct失败重算”当前新方法。

Configuration: Qwen3-8B, TP=1, 2P:6D, fixed Retool:BrowseComp 1:1
schedule, closed-loop c512, seed 2026, temperature 0.

## Verdict

The causal-load/draining-reselection change improves aggregate throughput and
average balance, but does **not** pass long-run balance acceptance. The first
300 measured seconds are balanced; irreversible NUMA-local P->D Host ownership
then accumulates and the final 300 seconds split the two domains severely.

## Whole-window result

| Metric | Previous 300+1200 | New 300+1200 |
|---|---:|---:|
| Decode throughput | 8,781.7 token/s | 9,205.0 token/s |
| Decode throughput/GPU | 1,463.6 token/s | 1,534.2 token/s |
| Completed agents/s | 2.167 | 2.336 |
| D Forward/GPU | 99.63% | 99.75% |
| D throughput CV | 31.97% | 12.77% |
| D running CV | 44.04% | 18.73% |
| D KV-utilization CV | 28.30% | 11.81% |
| P compute-throughput CV | 19.11% | 6.55% |
| P queue-length CV | 77.37% | 19.64% |
| P->D Host binds / P->D binds | 5,354 / 5,922 (90.4%) | 4,148 / 7,991 (51.9%) |
| Page-aligned parent KV reuse | 100% | 99.95% |

## Per-engine whole-window means

| Engine | Throughput | Running | KV utilization | Forward |
|---|---:|---:|---:|---:|
| D0 / 27301 | 1,333 token/s | 52.0 | 64.1% | 99.74% |
| D1 / 27302 | 1,345 token/s | 53.0 | 63.6% | 99.81% |
| D2 / 27303 | 1,376 token/s | 53.0 | 63.7% | 99.77% |
| D3 / 27401 | 1,749 token/s | 75.6 | 79.0% | 99.76% |
| D4 / 27402 | 1,816 token/s | 78.5 | 78.6% | 99.81% |
| D5 / 27403 | 1,586 token/s | 53.9 | 83.9% | 99.58% |
| P0 / 27300 | 9,076 prefill token/s | queue 37.9 | 62.6% | 96.84% |
| P1 / 27400 | 7,959 prefill token/s | queue 25.4 | 70.1% | 83.06% |

## Time evolution

Each row is one consecutive 300-second quarter of the measured window.

| Quarter | D0-D2 running | D3-D5 running | P0 Forward | P1 Forward |
|---|---:|---:|---:|---:|
| Q1 | 56.0 | 54.9 | 96.9% | 97.2% |
| Q2 | 75.7 | 65.5 | 95.0% | 95.8% |
| Q3 | 61.7 | 76.4 | 95.6% | 87.8% |
| Q4 | 17.5 | 80.7 | 99.9% | 51.1% |

P->D Host binds by quarter were 0, 1,433, 1,703, and 1,012. Once Host owns a
snapshot, restore and D admission are NUMA-local. A backlog in one domain can
therefore no longer use free capacity in the other domain. Dynamic P routing
reacts to the pressure of new work but cannot migrate already staged Host
snapshots, so the local backlog becomes a long-run positive-feedback loop.

The new causal refresh and draining reselection are still valid: they reduce
premature Host binding and materially improve the whole-window result. They are
not sufficient for final balance because Host ownership itself remains an
irreversible partition after spill.

## D→P 快慢路径比例

正式 1,200 秒窗口按唯一 request-generation snapshot 统计：Direct 9,281，
Slow 453，即 Direct 95.35% / Slow 4.65%。
