# BrowseComp + Qwen3-8B

## 实验矩阵

除非单独说明，正式结果要求使用固定 source-order BrowseComp workload、
temperature 0、8 张 GPU、300 秒预热和 1,200 秒测量。Decode 为所有 Decode
计算资源的墙钟总吞吐。

> **显存比例校正（2026-09-07）：** colocated 使用普通 GPU `0.80`、承载搜索
> 服务的 GPU 7 使用 `0.60`；下列既有 PD 运行虽然 D 已使用
> `0.80/0.80/0.80/0.60`，但 P 使用了 `0.85`。这些 PD 数字仅保留为历史诊断值，
> 均需以 P `0.80` 重跑后才能用于最终横向比较。Colocated 结果不受影响。

> **新方法语义校正（2026-09-08）：** “当前新方法”现在专指“工具超过1秒走
> Slow；工具1秒内返回则尝试Direct；Direct在1秒内未建立时完整重算”。此前
> “Direct失败→Slow”的快慢路径结果均为旧方案，除下面新增的c512对齐结果外，
> 均需按新语义重跑。

| 方法 | 配置 | 状态 | Decode |
|---|---|---|---:|
| Colocated baseline | 8卡，c384 | 完成 | 4,834 token/s |
| Colocated baseline | 8卡，c512 | 完成 | 4,434 token/s |
| Colocated baseline | 8卡，c576 | 完成 | 4,349 token/s |
| 旧版快慢路径 | 4P:4D，c384 | 需重跑（历史 P=0.85，且不是当前新方法） | 4,700 token/s |
| 旧版快慢路径 | 4P:4D，c512 | 需重跑（历史 P=0.85，且不是当前新方法） | 4,638 token/s |
| 旧版快慢路径 | 4P:4D，c576，Host短snapshot优先驱逐 | 需重跑（历史 P=0.85，且不是当前新方法） | 4,017 token/s |
| **当前新方法（快慢路径+重算）** | **4P:4D，c512** | **完成（对齐显存）** | **4,888 token/s** |
| No-reverse PD | 4P:4D，c384 | 需重跑（历史 P=0.85） | 2,013 token/s |
| No-reverse PD | 4P:4D，c512 | 需重跑（历史 P=0.85） | 2,071 token/s |
| No-reverse PD | 4P:4D，c576 | 需重跑（历史 P=0.85） | 2,179 token/s |
| 原生 Mooncake | 4P:4D，c384 | 需重跑（历史 P=0.85） | 1,076 token/s |
| 原生 Mooncake | 4P:4D，c512 | 需重跑（历史 P=0.85） | 1,352 token/s |
| 原生 Mooncake | 4P:4D，c576 | 需重跑（历史 P=0.85） | 1,460 token/s |

## 已完成结果明细

除加粗的当前新方法 c512 外，以下 PD 明细均是 P `0.85` 的历史测量，待以
P `0.80` 和当前“快慢路径+重算”语义重跑后替换；不要用旧行与 colocated
计算最终收益。

| 方法 | 并发 | Agent/s | Prefill compute | Decode/Agent | 实际 Prefill/Agent | Parent KV复用 |
|---|---:|---:|---:|---:|---:|---:|
| Colocated baseline | 384 | 2.392 | 38,149 token/s | 2,014 tokens | 15,893 tokens | 91.74% |
| Colocated baseline | 512 | 2.181 | 46,523 token/s | 2,031 tokens | 21,308 tokens | 74.11% |
| Colocated baseline | 576 | 2.160 | 47,552 token/s | 2,008 tokens | 21,957 tokens | 74.31% |
| 旧版快慢路径 4P:4D | 384 | 2.352 | 32,669 token/s | 1,993 tokens | 13,856 tokens | 99.98% |
| 旧版快慢路径 4P:4D | 512 | 2.320 | 32,889 token/s | 1,996 tokens | 14,154 tokens | 99.26% |
| 旧版快慢路径 4P:4D + Host驱逐 | 576 | 2.062 | 32,315 token/s | 1,943 tokens | 15,631 tokens | 96.32% |
| **当前新方法（快慢路径+重算）4P:4D** | **512** | **2.275** | **38,232 token/s** | **2,147 tokens** | **16,790 tokens** | **93.44%** |
| No-reverse PD 4P:4D | 384 | 0.989 | 42,857 token/s | 2,030 tokens | 43,224 tokens | 0.00% |
| No-reverse PD 4P:4D | 512 | 1.014 | 42,801 token/s | 2,039 tokens | 42,138 tokens | 0.00% |
| No-reverse PD 4P:4D | 576 | 1.049 | 43,228 token/s | 2,070 tokens | 41,066 tokens | 0.00% |
| 原生 Mooncake 4P:4D | 384 | 0.525 | 21,567 token/s | 2,048 tokens | 41,049 tokens | 21.39% |
| 原生 Mooncake 4P:4D | 512 | 0.627 | 21,849 token/s | 2,152 tokens | 34,784 tokens | 21.02% |
| 原生 Mooncake 4P:4D | 576 | 0.693 | 22,226 token/s | 2,104 tokens | 32,039 tokens | 19.49% |

## D→P 快慢路径比例

旧版三行按正式 1,200 秒窗口内完成路径的唯一 request-generation snapshot
统计；当前新方法 c512 的新分析器计数覆盖预热和正式窗口整轮，绝对次数不能与
旧行直接相比，路径比例仅使用已得到明确结果的 snapshot。TP rank 不重复计数。
其他方法不使用当前新方法的自定义 D→P Direct/Shared-Arena 状态机，因此不适用。

| 方法 | 配置 | Direct | Slow | 重算 | 路径比例 |
|---|---|---:|---:|---:|---|
| 旧版快慢路径 | 4P:4D，c384 | 4,781 | 2,458 | 0 | 66.05% / 33.95% / 0% |
| 旧版快慢路径 | 4P:4D，c512 | 4,753 | 2,340 | 0 | 67.01% / 32.99% / 0% |
| 旧版快慢路径 + Host驱逐 | 4P:4D，c576 | 4,986 | 1,345 | 0 | 78.76% / 21.24% / 0% |
| **当前新方法（快慢路径+重算）** | **4P:4D，c512** | **8,207** | **427** | **637** | **88.5% / 4.6% / 6.9%** |
| Colocated / No-reverse / 原生 Mooncake | — | — | — | — | 不适用 |

## 稳态资源明细

下表均为 1,200 秒正式测量窗口平均值。Forward 按物理 GPU 统计；KV、running、
queue 和 inflight/transfer 按 SGLang 逻辑引擎统计。Colocated 的 P/D 共享同一
KV pool，因此只在 D KV/running/queue 栏记录整体引擎状态。

| 方法 | P Forward/卡 | P KV/引擎 | P queue/引擎 | P inflight/引擎 | D Forward/卡 | D KV/引擎 | D running/引擎 | D queue/引擎 | D transfer/引擎 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Colocated c384 | 51.1% | — | — | — | 48.8% | 67.6% | 46.3 | 0.8 | — |
| Colocated c512 | 60.4% | — | — | — | 39.6% | 84.1% | 59.9 | 3.1 | — |
| Colocated c576 | 62.0% | — | — | — | 37.9% | 84.6% | 66.1 | 4.8 | — |
| 旧版快慢路径 4P:4D c384 | 87.0% | 90.6% | 8.7 | 12.4 | 97.6% | 86.4% | 52.8 | 0.0 | 0.3 |
| 旧版快慢路径 4P:4D c512 | 86.8% | 85.1% | 25.8 | 12.5 | 97.9% | 85.0% | 50.6 | 0.0 | 0.3 |
| 旧版快慢路径 + Host驱逐 c576 | 84.6% | 70.1% | 61.7 | 12.8 | 97.4% | 71.1% | 36.6 | 0.0 | 0.4 |
| **当前新方法（快慢路径+重算）c512** | **97.04%** | **50.60%** | **65.37** | **6.38** | **99.62%** | **70.49%** | **48.44** | **0.0** | **0.20** |
| No-reverse 4P:4D c384 | 99.7% | 7.5% | 18.9 | 0.7 | 99.1% | 92.2% | 7.7 | 0.0 | 21.3 |
| No-reverse 4P:4D c512 | 99.0% | 7.4% | 19.2 | 0.8 | 98.7% | 92.3% | 8.2 | 0.0 | 21.6 |
| No-reverse 4P:4D c576 | 99.4% | 7.3% | 19.1 | 0.7 | 99.1% | 92.2% | 9.0 | 0.0 | 21.6 |
| 原生 Mooncake 4P:4D c384 | 84.7% | 8.0% | 20.1 | 1.1 | 89.1% | 92.6% | 4.7 | 0.0 | 22.5 |
| 原生 Mooncake 4P:4D c512 | 83.8% | 7.7% | 21.3 | 1.1 | 92.2% | 92.4% | 6.4 | 0.0 | 23.9 |
| 原生 Mooncake 4P:4D c576 | 83.4% | 7.5% | 21.0 | 1.1 | 93.3% | 92.2% | 7.1 | 0.0 | 23.6 |

## Slow 路径在 D HBM 中的额外驻留

这里仅统计最终转入 D→P Slow 的 snapshot。从 Decode 结束发布 Direct offer
开始，到该 snapshot 完成 D2H、D 释放父 KV 为止，对 page-aligned token 数做
时间积分。平均占用包括 Direct/工具阈值等待和完整 D2H wall time；不包含最终走
Direct 的 snapshot。Qwen3-8B TP=1 的 KV 为 144 KiB/token。

| 配置 | Slow完成数 | 平均转Slow等待 | 平均D2H wall | Slow额外占用（4D合计） | 平均每D | 峰值（4D合计） | 占D KV池 | 占实际D KV使用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧版快慢路径 c384 | 2,458 | 2.244 s | 0.418 s | 13.39 GiB | 3.35 GiB | 48.23 GiB | 7.66% | 8.85% |
| 旧版快慢路径 c512 | 2,340 | 2.293 s | 0.383 s | 11.93 GiB | 2.98 GiB | 42.42 GiB | 6.82% | 8.01% |

c384 的平均 13.39 GiB 可进一步拆成：Direct/工具阈值等待 10.83 GiB，D2H
wall 阶段 2.56 GiB；c512 分别为 9.77 GiB 和 2.16 GiB。因此这两轮的 Slow
驻留成本约 81% 来自转 Slow 前的等待窗口，而不是 Host DMA 带宽不足。以上均为
旧版快慢路径数据，需按当前“快慢路径+重算”语义重跑后再用于设计结论。

## 原始结果

- Colocated c384: [summary](archive/baseline/formal-browsecomp-source-order-colocated-8gpu-c384-w300-m1200-20260816-r1/offload_analysis_summary.json)
- Colocated c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/baseline-colocated/offload_analysis_summary.json)
- Colocated c576: [summary](archive/baseline/formal-browsecomp-source-order-colocated-8gpu-c576-w300-m1200-20260817-r1/offload_analysis_summary.json)
- 旧版快慢路径 c384: [report](current/h100-a100-integration/browsecomp-qwen3-8b-tp1-4p4d-c384-prewarm-barrier-w300-m1200-r2/RESULTS.md)
- 旧版快慢路径 c512: [report](current/h100-a100-integration/browsecomp-qwen3-8b-tp1-4p4d-c512-prewarm-barrier-w300-m1200-r2/RESULTS.md)
- 旧版快慢路径 c576（Host驱逐策略）: [summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/new-method-agentic-pd-host-evict-shortest-low75-r1/offload_analysis_summary.json)
- 旧版快慢路径 c576（驱逐前失败现场）: [summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/new-method-agentic-pd/offload_analysis_summary.json)
- 当前新方法 c512: [summary](current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260908-r4/fast-direct-fail-recompute-1s/offload_analysis_summary.json)
- No-reverse c384: [summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/no-reverse-pd-4p4d/offload_analysis_summary.json)
- No-reverse c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/no-reverse-pd-4p4d/offload_analysis_summary.json)
- No-reverse c576: [summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/no-reverse-pd-4p4d/offload_analysis_summary.json)
- 原生 Mooncake c384（首次预热失败现场）: [run](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/native-mooncake-pd-4p4d-failed-storage-r1/)
- 原生 Mooncake c384（正式重跑）: [summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/native-mooncake-pd-4p4d/offload_analysis_summary.json)
- 原生 Mooncake c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/native-mooncake-pd-4p4d/offload_analysis_summary.json)
- 原生 Mooncake c576: [summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native-mooncake-pd-4p4d/offload_analysis_summary.json)

## 原生 Mooncake c384 预热失败

- source-order n680、temperature 0、4P:4D、c384，与同组正式实验保持一致。
- 预热约 185 秒时停止；在进入 1,200 秒测量窗口前已经产生 308 个 Decode HTTP 500。
- 首个故障信号来自 P 侧原生 Mooncake：`BatchGet LEASE_EXPIRED`、
  `Batch finalization RPC_TIMEOUT`，随后 P0 超过 20 秒无法推进 detokenizer heartbeat 并退出。
- P 退出使 NIXL 连接断开，四个 D 随后分别记录 182、111、178、55 次
  `Decode transfer failed`；Router 最终报告所有 Decode circuit unavailable。
- 故障发生时 Mooncake store 约 153/256 GiB（59.8%），且未触发驱逐，因此不是
  共享存储容量耗尽，而是原生恢复/存储控制路径阻塞 P 后引发的级联失败。
- 本轮不产生可比较吞吐结果。

## 原生 Mooncake c512 正式结果补充

- 完整通过 300 秒预热和 1,200 秒测量，测量窗口内 P storage error、D transfer
  error 和 HTTP 500 均为 0。
- Mooncake 平均/峰值使用率为 81.29%/85.00%；测量窗口内成功驱逐 46 次、
  约 1,005.35 GB。
- D 平均每引擎还有 97.54 个 prealloc 请求、23.93 个 transfer 请求，但只有
  6.41 个 running 请求；因此 92.39% 的 D KV 占用没有转化成高 Decode 并行度。

## 原生 Mooncake c384 正式重跑

- 重跑完整通过 301 秒预热和 1,200 秒测量，完成 630 个 Agent；全窗口
  HTTP 500、Mooncake RPC timeout 和 transfer failure 均为 0。首次预热失败
  因而不是 c384 必现故障。
- Mooncake 平均/峰值使用率为 80.08%/85.00%，成功驱逐 46 次、约
  1,005.38 GB，无 allocation failure。
- D 平均每引擎有 68.93 个 prealloc、22.46 个 transfer，但只有 4.68 个
  running；92.64% 的 KV 使用率没有形成有效 Decode batch。
- Parent KV 复用率 21.39%，父 KV 丢失导致额外 Prefill 20,503 tokens/Agent。

## 原生 Mooncake c576 正式结果补充

- 完整通过 301 秒预热和 1,200 秒测量，测量窗口完成 832 个 Agent；P/D 日志中
  HTTP 500、Mooncake RPC timeout 和 transfer failure 均为 0。
- Mooncake 平均/峰值使用率为 80.40%/84.90%；成功驱逐 46 次、约
  1,009.47 GB，无 allocation failure。
- D 平均每引擎有 113.32 个 prealloc、23.57 个 transfer，但只有 7.07 个
  running；92.22% 的 D KV 主要被等待状态占据。并发从 512 增至 576 后，
  Decode 仅从 1,352 增至 1,460 token/s，仍远低于 No-reverse 和当前新方法。
- 完成集合的 page-aligned Parent KV 复用率仅 19.49%，由父 KV 丢失明确造成的
  额外 Prefill 为 13,954 tokens/Agent。

## c576 Shared Arena 驱逐结果

- 90% 水位触发，按 snapshot token 数从短到长驱逐；长度相同时优先等待更久者，目标回落至 75%。
- 共驱逐 355 个 request-generation snapshot、3,473,856 tokens（约 477.1 GiB）。
- 驱逐 snapshot 长度：中位数 9,856 tokens，P90 16,640 tokens，范围 960–20,544 tokens。
- Router 500、异步控制错误和 P-ready 600 秒超时均为 0；旧 c576 的容量闭环未复现。
- 驱逐后完成集合的父 KV page-aligned 复用率为 96.32%，额外重复 Prefill 为 1,356 tokens/Agent。
- 相比驱逐前失败现场，Decode 从 907 提升到 4,017 token/s；它比旧版快慢路径
  c512 低 9.45%，比当前“快慢路径+重算”c512 低约 17.82%。该结果说明主动驱逐
  解决了当时的活性问题，但仍属于需要重跑的旧方案。

## 完成集合数据特性

`实际 Prefill/Agent` 是 GPU 真正执行的 Prefill tokens；`Parent KV 未复用率`
等于 `1 - Parent KV 复用率`，表示上一轮父 KV 中需要重复计算的 token 比例，
不能直接解释为实际 Prefill 的同等比例。只有分析器明确记录绝对重算量时，才填入
`明确额外 Prefill/Agent`，避免用不同完成集合进行不可靠的相减。

| 方法 | 并发 | Decode/Agent | 实际 Prefill/Agent | Parent KV 未复用率 | 明确额外 Prefill/Agent |
|---|---:|---:|---:|---:|---:|
| Colocated baseline | 384 | 2,014 tokens | 15,893 tokens | 8.26% | 未单独记录 |
| Colocated baseline | 512 | 2,031 tokens | 21,308 tokens | 25.89% | 未单独记录 |
| Colocated baseline | 576 | 2,008 tokens | 21,957 tokens | 25.69% | 未单独记录 |
| 旧版快慢路径 4P:4D | 384 | 1,993 tokens | 13,856 tokens | 0.02% | 7.3 tokens |
| 旧版快慢路径 4P:4D | 512 | 1,996 tokens | 14,154 tokens | 0.74% | 未单独记录 |
| 旧版快慢路径 4P:4D + Host驱逐 | 576 | 1,943 tokens | 15,631 tokens | 3.68% | 1,356 tokens |
| **当前新方法（快慢路径+重算）4P:4D** | **512** | **2,147 tokens** | **16,790 tokens** | **6.56%** | **2,146 tokens（整轮637次显式Direct失败重算）** |
| No-reverse PD 4P:4D | 384 | 2,030 tokens | 43,224 tokens | 100.00% | 未单独记录 |
| No-reverse PD 4P:4D | 512 | 2,039 tokens | 42,138 tokens | 100.00% | 未单独记录 |
| No-reverse PD 4P:4D | 576 | 2,070 tokens | 41,066 tokens | 100.00% | 未单独记录 |
| 原生 Mooncake 4P:4D | 384 | 2,048 tokens | 41,049 tokens | 78.61% | 20,503 tokens |
| 原生 Mooncake 4P:4D | 512 | 2,152 tokens | 34,784 tokens | 78.98% | 15,647 tokens |
| 原生 Mooncake 4P:4D | 576 | 2,104 tokens | 32,039 tokens | 80.51% | 13,954 tokens |

重构前的 4,660 token/s 和 4,847 token/s 新方法结果只保留在 archive 中用于历史
回归，不填入当前实验矩阵。
