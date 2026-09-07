# BrowseComp + Qwen3-32B TP=2

## 实验矩阵

正式结果使用固定 source-order BrowseComp workload、temperature 0、8 张 GPU、
TP=2、300 秒预热和 1,200 秒测量。`2P:6D` 表示一个 TP=2 Prefill 组和三个
TP=2 Decode 组；`4P:4D` 表示两个 TP=2 Prefill 组和两个 TP=2 Decode 组。

| 方法 | 配置 | 状态 | Decode |
|---|---|---|---:|
| Colocated baseline | c256 | 完成 | 1,366 token/s |
| Colocated baseline | c320 | 完成 | 1,456 token/s |
| 原生 Mooncake | 2P:6D，c256 | 完成 | 862 token/s |
| 当前新方法 | 2P:6D，c256 | 完成 | 1,447 token/s |
| No-reverse PD | 2P:6D，c256 | 完成 | 816 token/s |
| No-reverse PD | 4P:4D，c256 | 完成 | 1,020 token/s |
| 原生 Mooncake | 4P:4D，c256 | 完成 | 1,023 token/s |

## 已完成结果明细

| 方法 | 并发 | Agent/s | Prefill compute | Decode/Agent | 实际 Prefill/Agent | Parent KV复用 |
|---|---:|---:|---:|---:|---:|---:|
| Colocated baseline | 256 | 0.518 | 9,607 token/s | 2,626 tokens | 18,472 tokens | 9.39% |
| Colocated baseline | 320 | 0.542 | 9,580 token/s | 2,675 tokens | 17,606 tokens | 10.39% |
| 原生 Mooncake 2P:6D | 256 | 0.330 | 4,805 token/s | 2,610 tokens | 14,555 tokens | 19.65% |
| 当前新方法 2P:6D | 256 | 0.557 | 5,135 token/s | 2,591 tokens | 9,196 tokens | 100.00% |
| No-reverse PD 2P:6D | 256 | 0.319 | 5,551 token/s | 2,553 tokens | 17,362 tokens | 0.00% |
| No-reverse PD 4P:4D | 256 | 0.408 | 6,798 token/s | 2,490 tokens | 16,595 tokens | 0.00% |
| 原生 Mooncake 4P:4D | 256 | 0.401 | 5,420 token/s | 2,544 tokens | 13,480 tokens | 27.66% |

## D→P 快慢路径比例

按正式 1,200 秒窗口内完成路径的唯一 request-generation snapshot 统计，两个
TP rank 合并为一个逻辑 snapshot。

| 方法 | 配置 | Direct | Slow | Direct/Slow 比例 |
|---|---|---:|---:|---:|
| 当前新方法 | 2P:6D，c256 | 631 | 308 | 67.20% / 32.80% |
| Colocated / No-reverse / 原生 Mooncake | — | — | — | 不适用 |

## 稳态资源明细

下表均为 1,200 秒正式测量窗口平均值。Forward 按物理 GPU 统计；KV、running、
queue 和 inflight/transfer 按逻辑 TP 引擎统计。一个 TP=2 引擎的两个 rank 执行
相同请求并保持相同 KV 利用率，所以 running/引擎也等于每个 rank 观察到的请求数。
Colocated 的 P/D 共享同一 KV pool，只在 D KV/running/queue 栏记录整体状态。

| 方法 | P Forward/卡 | P KV/引擎 | P queue/引擎 | P inflight/引擎 | D Forward/卡 | D KV/引擎 | D running/引擎 | D queue/引擎 | D transfer/引擎 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Colocated c256 | 45.7% | — | — | — | 54.3% | 92.9% | 28.0 | 35.4 | — |
| Colocated c320 | 45.3% | — | — | — | 54.6% | 93.1% | 29.7 | 49.8 | — |
| 原生 Mooncake 2P:6D c256 | 99.9% | 10.5% | 50.1 | 1.6 | 99.8% | 90.8% | 9.0 | 0.0 | 18.2 |
| 当前新方法 2P:6D c256 | 97.7% | 66.7% | 5.6 | 6.6 | 99.8% | 77.9% | 19.3 | 0.0 | 0.1 |
| No-reverse 2P:6D c256 | 100.1% | 10.0% | 50.8 | 1.4 | 97.8% | 90.8% | 8.2 | 0.0 | 18.1 |
| No-reverse 4P:4D c256 | 60.9% | 6.7% | 2.1 | 1.4 | 99.9% | 90.1% | 19.7 | 0.0 | 4.0 |
| 原生 Mooncake 4P:4D c256 | 54.6% | 6.4% | 2.1 | 1.4 | 99.2% | 90.2% | 20.2 | 0.0 | 4.2 |

## 原始结果

- Colocated c256: [summary](current/qwen3-32b-tp2-browsecomp-c256-w300-m1200/baseline-colocated/offload_analysis_summary.json)
- Colocated c320: [summary](archive/baseline/formal-qwen3-32b-tp2-browsecomp-colocated-c320-w300-m1200-20260824-r1/offload_analysis_summary.json)
- 原生 Mooncake 2P:6D c256: [summary](archive/baseline/formal-qwen3-32b-tp2-browsecomp-native-mooncake-2p6d-c256-w300-m1200-20260824-r1/offload_analysis_summary.json)
- 当前新方法 2P:6D c256: [summary](current/qwen3-32b-tp2-browsecomp-c256-w300-m1200/new-method-agentic-pd/offload_analysis_summary.json)
- No-reverse 2P:6D c256: [summary](current/qwen3-32b-tp2-browsecomp-c256-w300-m1200/no-reverse-pd-2p6d/offload_analysis_summary.json)
- No-reverse 4P:4D c256: [summary](current/qwen3-32b-tp2-browsecomp-c256-w300-m1200/no-reverse-pd-4p4d/offload_analysis_summary.json)
- 原生 Mooncake 4P:4D c256: [summary](current/qwen3-32b-tp2-browsecomp-c256-w300-m1200/native-mooncake-pd-4p4d/offload_analysis_summary.json)

重构前的 Agentic-PD c256/c320 结果不作为当前方法数据回填。

## 完成集合数据特性

`Parent KV 未复用率 = 1 - Parent KV 复用率`，表示上一轮父 KV 中需要重算的
token 比例；它不是实际 Prefill tokens 的简单百分比。绝对重复 Prefill 未被旧实验
单独记录时保持为空，不跨不同完成集合做相减估算。

| 方法 | 配置 | Decode/Agent | 实际 Prefill/Agent | Parent KV 未复用率 | 明确额外 Prefill/Agent |
|---|---|---:|---:|---:|---:|
| Colocated baseline | c256 | 2,626 tokens | 18,472 tokens | 90.61% | 未单独记录 |
| Colocated baseline | c320 | 2,675 tokens | 17,606 tokens | 89.61% | 未单独记录 |
| 原生 Mooncake | 2P:6D，c256 | 2,610 tokens | 14,555 tokens | 80.35% | 未单独记录 |
| 当前新方法 | 2P:6D，c256 | 2,591 tokens | 9,196 tokens | 0.00% | 约 0 tokens |
| No-reverse PD | 2P:6D，c256 | 2,553 tokens | 17,362 tokens | 100.00% | 6,796 tokens |
| No-reverse PD | 4P:4D，c256 | 2,490 tokens | 16,595 tokens | 100.00% | 6,627 tokens |
| 原生 Mooncake | 4P:4D，c256 | 2,544 tokens | 13,480 tokens | 72.34% | 4,658 tokens |
