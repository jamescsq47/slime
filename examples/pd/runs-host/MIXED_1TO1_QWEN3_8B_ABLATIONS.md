# Retool + BrowseComp 1:1 / Qwen3-8B / Agentic PD Ablations

## Fixed setting

All rows use the same serving workload and differ only in the named ablation.

| Item | Value |
|---|---|
| Model | Qwen3-8B |
| Workload | Retool : BrowseComp = 1 : 1, fixed `seed=2026` schedule |
| Layout | 2P:6D, TP=1, global Host recovery |
| Concurrency | 512 closed-loop agents |
| Sampling | `temperature=0` |
| Timing | 300 s warmup + 1200 s measurement |
| Request source | `fixed_random_s2026_n8192.json` |
| Native HiCache/Mooncake | disabled |

## Primary results

| Variant | D→P Direct | D→P Host | P→D Host | P/D routing | Decode token/s | Decode token/s/D | Agent/s | Status |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Full method (obsolete) | on | on | on | incomplete P pressure / source-local P→D Host | 8,781.7 | 1,463.6 | 2.167 | **invalid; do not compare** |
| D→P Direct only | on | off | on | load-aware | 9,445.5 | 1,574.2 | 2.380 | complete |
| D→P Slow only | off | on | on | load-aware | 683.8 | 114.0 | 0.224 | measured, unhealthy |
| Random P/D routing | on | on | on | random among capacity-feasible workers | 9,702.6 | 1,617.1 | 2.417 | complete |
| P→D Direct only | on | on | off | load-aware | 9,817.0 | 1,636.2 | 2.450 | complete |

## D→P 快慢路径比例

比例按正式 1,200 秒窗口内完成路径的唯一 request-generation snapshot 统计。
短测不回填正式表；待完成的 ablation 在各自 300+1,200 秒运行结束后更新。

| Variant | Direct | Slow | Direct/Slow 比例 |
|---|---:|---:|---:|
| Full method (obsolete) | 8,595 | 458 | invalid old routing semantics |
| D→P Direct only | 10,224 | 0 | 100% / 0% (另有 61 次 Direct 失败后完整重算) |
| D→P Slow only | 0 | 1,158 | 0% / 100% (624 Host→P complete, 533 Host evictions) |
| Random P/D routing | 10,392 | 134 | 98.73% / 1.27% |
| P→D Direct only | 10,622 | 88 | 99.18% / 0.82% |

## Resource and workload details

| Variant | Prefill token/s | Actual Prefill/Agent | Decode/Agent | Parent KV reuse | P Forward/card | D Forward/card | P KV | P queue | P inflight | D KV | D running | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full method (obsolete) | 15,660.0 | 7,215.9 | 4,046.5 | 100% | 84.2% | 99.6% | 73.8% | 30.1 | 28.1 | 66.2% | 57.0 | 0.80 | 0.21 |
| D→P Direct only | 18,946.7 | 7,930.2 | 3,953.4 | 98.17% | 96.9% | 99.9% | 48.2% | 61.6 | 6.67 | 68.8% | 59.4 | 0.16 | 0.20 |
| D→P Slow only | 4,247.4 | 18,908.1 | 3,043.9 | 67.46% | 19.3% | 27.4% | 94.6% | 224.9 | 5.04 | 8.3% | 2.6 | 0.01 | 0.01 |
| Random P/D routing | 18,214.0 | 7,525.0 | 4,008.6 | 100% | 94.5% | 99.8% | 52.1% | 38.6 | 7.10 | 75.6% | 65.7 | 0.15 | 0.25 |
| P→D Direct only | 18,548.4 | 7,554.6 | 3,998.4 | 99.94% | 96.2% | 99.8% | 50.8% | 29.2 | 9.16 | 80.2% | 69.1 | 0.18 | 0.16 |

## Run directories and path accounting

| Variant | Run directory | D→P Direct/Slow | P→D Direct/Slow | Host ownership conservation |
|---|---|---|---|---|
| Full method (obsolete) | `current/qwen3-8b-tp1-mixed1to1-c512-w300-m1200/new-method-agentic-pd` | 94.94% / 5.06% | see retained logs | invalid: P→D Host was not globally late-bound and P pressure omitted downstream delivery |
| D→P Direct only | `current/ablations/mixed1to1-qwen3-8b-2p6d-c512/d2p-direct-only` | 10,224 Direct / 0 Slow / 61 recompute | 13,285 Direct / 1,217 Host | D→P Host disabled；P→D Host complete=release=1,217；无 CAS ownership failure |
| D→P Slow only | `current/ablations/mixed1to1-qwen3-8b-2p6d-c512/d2p-slow-only` | 0 Direct / 1,158 Host D2H | 1,282 P→D releases | D2H complete=release=1,158；624 H2D complete；533 complete-snapshot evictions |
| Random P/D routing | `current/ablations/mixed1to1-qwen3-8b-2p6d-c512/random-routing` | 10,392 Direct / 134 Slow | 13,693 P→D releases；3,035 Host D2H / 3,016 Host H2D | D→P Slow D2H=release=H2D=134；0 Host eviction |
| P→D Direct only | `current/ablations/mixed1to1-qwen3-8b-2p6d-c512/p2d-direct-only` | 10,622 Direct / 88 Slow | 13,658 Direct releases / 0 Host | 87 D→P Host H2D complete；0 Host eviction；P→D Host disabled |

The old 8,781.7-token/s full-method row is retained only as an audit artifact.
It is invalid for paired comparison because that run did not use global P→D
Host recovery and used an incomplete D→P P-pressure model. It will be replaced
by a same-checkpoint rerun after the lifecycle/router fixes pass audit.

## D→P Direct-only阶段性结论

以下旧比较已废弃，不作为结论：

- Decode 吞吐从 8,781.7 提高到 9,445.5 token/s，增加 7.6%；单 D 从
  1,463.6 提高到 1,574.2 token/s。
- Agent 吞吐从 2.167 提高到 2.380 agent/s，增加 9.8%。
- 代价是父 KV 页对齐复用率从 100% 降到 98.17%，实际 Prefill/Agent
  从 7,215.9 增至 7,930.2 tokens，增加 9.9%。
- P Forward/card 从 84.2% 升至 96.9%，P KV 从 73.8% 降至 48.2%；D
  Forward/card 保持 99.9%，D running 从 57.0 升至 59.4。该 workload
  下，省去 D→P Host 的 D2H/H2D 与恢复控制开销，收益大于少量完整重算的成本。
- 正式窗口完成 2,856 agents（Math 1,385 / QA 1,471），无请求失败；平均
  4.58 turns/agent、3,953 Decode tokens/agent。

注意：8,781.7 的 Full method 不仅不是同 checkpoint paired run，而且使用了
错误的 P→D Host/压力路由语义，已经正式作废。上述相对差值全部不再用于结论。

## D→P Slow-only 阶段性结论

- 完整跑过 300 秒预热和 1,200 秒测量，但服务活性不合格：仅完成
  269 agents，Decode 为 683.8 token/s，后半窗口出现 486 次 Router
  HTTP 500，其中约 468 次是 P-ready 等待超过 600 秒。
- 正式窗口内 1,158 个 snapshot 完成 D→Host 并释放 D KV；624 个完成
  Host→P 恢复，533 个在 Host 高水位下被按短 snapshot 优先驱逐。这与
  67.46% 的父 KV 复用率相互印证，不是路径统计遗漏。
- P 平均 KV 为 94.6%、queue 为 224.9，而 D KV 仅 8.3%、running 仅
  2.6；因此损失不是 D 算力不足，而是强制所有反向 KV 经 Host 后，P
  恢复和随后 P→D 交付无法跟上到达速率，最终让 D 长期断供。

## Random-routing 阶段性结论

- 完整跑过 300+1,200 秒，完成 2,900 agents，测量窗口没有 Router
  500、Host 驱逐或 KV ownership 故障。
- Decode 为 9,702.6 token/s，比保留的 load-aware Full reference 高 10.5%；
  Agent/s 高 11.5%。但 Full reference 不是当前 checkpoint 的同轮 paired run，
  因此这是明确的方向信号，不单独作为“随机必然优于 load-aware”的最终证据。
- 逐 D 平均 running 范围为 47.6–73.3，KV 为 71.3%–78.1%；随机路由
  确实增加了负载偏斜，但本轮 D 总平均 running 仍达 65.7，高于 Full
  reference 的 57.0，因而整体 Decode batch 更大。
- D→P 为 10,392 Direct + 134 Slow，所有 Slow 均恢复并释放，父 KV
  page-aligned 复用率为 100%。

## P→D Direct-only 阶段性结论

- 完整跑过 300+1,200 秒，完成 2,940 agents，正式窗口内无 P→D
  Host D2H/H2D，也没有容量或交付超时。
- Decode 为 9,817.0 token/s，D running/KV 为 69.1/80.2%，都高于
  保留的 Full reference。在该 c512 workload 中，D 的容量排空足以让
  P-ready KV 直接交付，P→D Host buffer 没有成为必需的性能路径。
- D→P 仍保留完整快慢路径：10,622 Direct + 88 Slow，87 个 Slow
  在窗口内恢复，0 驱逐。父 KV 复用率 99.94%；唯一明显损失是 1 条
  既定 terminal-repair 语义引起的 46,336 tokens 完整重算，与 P→D Host 开关无关。
- 该结论不表示 P→D Host 可从系统中删除：当 P 更快、D HBM 更紧或
  并发更高时，它仍是避免 Prefill 结果长期占住 P HBM 的容量保险。
