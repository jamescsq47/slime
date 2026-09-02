# Retool + BrowseComp 1:1 + Qwen3-8B

## 实验矩阵

正式结果使用固定 1:1 workload 顺序、temperature 0、8 张 GPU、300 秒预热和
1,200 秒测量。Decode 为所有 Decode 计算资源的墙钟总吞吐。

| 方法 | 配置 | 状态 | Decode |
|---|---|---|---:|
| Colocated baseline | 8卡，c512 | 完成 | 8,971 token/s |
| Colocated baseline | 8卡，c640 | 完成 | 8,380 token/s |
| Colocated baseline | 8卡，c768 | 完成 | 7,141 token/s |
| 原生 Mooncake | 2P:6D，c512 | 完成 | 3,509 token/s |
| 原生 Mooncake | 4P:4D，c512 | 完成 | 4,914 token/s |
| No-reverse PD | 2P:6D，c512 | 完成 | 3,213 token/s |
| No-reverse PD | 4P:4D，c512 | 完成 | 4,942 token/s |
| 当前新方法 | 2P:6D，c512，全局Host恢复，temperature=1 | 历史结果，不纳入最终对比 | 9,006 token/s |
| 当前新方法 | 2P:6D，c512，全局Host恢复、P→D可行请求先行，temperature=0 | 完成 | 9,800 token/s |
| 当前新方法 | 2P:6D，c640 | 完成 | 9,819 token/s |
| 当前新方法 | 2P:6D，c768 | 完成 | 9,146 token/s |

## 已完成结果明细

| 方法 | 并发 | Agent/s | Prefill compute | Decode/Agent | 实际 Prefill/Agent | Parent KV复用 |
|---|---:|---:|---:|---:|---:|---:|
| Colocated baseline | 512 | 2.264 | 23,983 token/s | 3,958 tokens | 10,581 tokens | 未统一记录 |
| Colocated baseline | 640 | 2.106 | 31,455 token/s | 3,968 tokens | 14,896 tokens | 未统一记录 |
| Colocated baseline | 768 | 1.800 | 42,854 token/s | 3,959 tokens | 23,755 tokens | 未统一记录 |
| 原生 Mooncake 2P:6D | 512 | 0.841 | 17,503 token/s | 4,159 tokens | 20,747 tokens | 未统一记录 |
| 原生 Mooncake 4P:4D | 512 | 1.163 | 26,619 token/s | 4,211 tokens | 22,813 tokens | 未统一记录 |
| No-reverse PD 2P:6D | 512 | 0.748 | 23,010 token/s | 4,282 tokens | 30,661 tokens | 未统一记录 |
| No-reverse PD 4P:4D | 512 | 1.217 | 42,230 token/s | 4,047 tokens | 34,585 tokens | 未统一记录 |
| 当前新方法 2P:6D（temperature=1，历史） | 512 | 2.351 | 18,924 token/s | 3,823 tokens | 8,034 tokens | 100.00% |
| 当前新方法 2P:6D（temperature=0，P→D可行请求先行） | 512 | 2.428 | 18,215 token/s | 4,027 tokens | 7,486 tokens | 100.00% |
| 当前新方法 2P:6D（temperature=0） | 640 | 2.442 | 18,110 token/s | 4,017 tokens | 7,410 tokens | 100.00% |
| 当前新方法 2P:6D（temperature=0） | 768 | 2.345 | 18,169 token/s | 3,896 tokens | 7,740 tokens | 98.62% |

## D→P 快慢路径比例

按正式 1,200 秒窗口内完成路径的唯一 request-generation snapshot 统计。

| 方法 | 配置 | Direct | Slow | Direct/Slow 比例 |
|---|---|---:|---:|---:|
| 当前新方法（temperature=1，历史） | 2P:6D，c512 | 8,814 | 355 | 96.13% / 3.87% |
| 当前新方法（temperature=0，P→D可行请求先行） | 2P:6D，c512 | 10,715 | 84 | 99.22% / 0.78% |
| 当前新方法（temperature=0） | 2P:6D，c640 | 9,564 | 641 | 93.72% / 6.28% |
| 当前新方法（temperature=0） | 2P:6D，c768 | 9,384 | 799 | 92.15% / 7.85% |
| Colocated / No-reverse / 原生 Mooncake | — | — | — | 不适用 |

## 稳态资源明细

下表均为 1,200 秒正式测量窗口平均值。Forward 按物理 GPU 统计；KV、running、
queue 和 inflight/transfer 按 SGLang 逻辑引擎统计。Colocated 的 P/D 共享同一
KV pool，因此只在 D KV/running/queue 栏记录整体引擎状态。

| 方法 | P Forward/卡 | P KV/引擎 | P queue/引擎 | P inflight/引擎 | D Forward/卡 | D KV/引擎 | D running/引擎 | D queue/引擎 | D transfer/引擎 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Colocated c512 | 30.7% | — | — | — | 69.3% | 75.3% | 60.8 | 2.3 | — |
| Colocated c640 | 37.8% | — | — | — | 62.1% | 85.9% | 71.0 | 7.9 | — |
| Colocated c768 | 48.5% | — | — | — | 51.4% | 88.6% | 76.0 | 18.9 | — |
| 原生 Mooncake 2P:6D c512 | 99.6% | 8.5% | 100.5 | 1.6 | 98.4% | 90.4% | 12.6 | 0.0 | 35.2 |
| 原生 Mooncake 4P:4D c512 | 70.4% | 6.6% | 5.0 | 1.5 | 99.6% | 88.5% | 49.9 | 0.0 | 8.0 |
| No-reverse 2P:6D c512 | 100.0% | 7.0% | 103.7 | 1.1 | 98.0% | 91.1% | 8.4 | 0.0 | 35.7 |
| No-reverse 4P:4D c512 | 91.7% | 6.5% | 9.7 | 1.1 | 99.8% | 88.5% | 41.2 | 0.0 | 12.8 |
| 当前新方法 2P:6D c512（temperature=1，历史） | 97.5% | 68.1% | 66.2 | 6.5 | 99.8% | 68.1% | 56.5 | 0.0 | 0.2 |
| 当前新方法 2P:6D c512（temperature=0，P→D可行请求先行） | 94.2% | 56.3% | 24.0 | 10.6 | 99.7% | 81.0% | 69.7 | 0.0 | 0.22 |
| 当前新方法 2P:6D c640（temperature=0） | 93.4% | 74.9% | 57.8 | 16.8 | 99.7% | 78.4% | 67.1 | 0.0 | 0.3 |
| 当前新方法 2P:6D c768（temperature=0） | 94.1% | 77.9% | 110.9 | 16.4 | 99.7% | 75.5% | 60.5 | 0.0 | 0.3 |

## 原始结果

- Colocated c512: [summary](current/qwen3-8b-tp1-mixed1to1-c512-w300-m1200/baseline-colocated/offload_analysis_summary.json)
- Colocated c640: [summary](archive/baseline/mixed-colocated-8gpu-inference-only-s2026-w300-m1200/c640/offload_analysis_summary.json)
- Colocated c768: [summary](archive/baseline/mixed-colocated-8gpu-inference-only-s2026-w300-m1200/c768/offload_analysis_summary.json)
- 原生 Mooncake 2P:6D c512: [summary](archive/baseline/mixed-pd-8gpu-c512-s2026-w300-m1200/pd-native-mooncake-2p6d/offload_analysis_summary.json)
- 原生 Mooncake 4P:4D c512: [summary](archive/baseline/mixed-pd-8gpu-c512-s2026-w300-m1200/pd-native-mooncake-4p4d/offload_analysis_summary.json)
- No-reverse 2P:6D c512: [summary](archive/baseline/mixed-pd-8gpu-c512-s2026-w300-m1200/pd-no-reverse-2p6d/offload_analysis_summary.json)
- No-reverse 4P:4D c512: [summary](archive/baseline/mixed-pd-8gpu-c512-s2026-w300-m1200/pd-no-reverse-4p4d/offload_analysis_summary.json)
- 当前新方法 2P:6D c512（temperature=1，历史）: [summary](current/qwen3-8b-tp1-mixed1to1-c512-global-host-restore-w300-m1200/offload_analysis_summary.json)
- 当前新方法 2P:6D c512（temperature=0，P→D可行请求先行）: [summary](current/ablations/mixed1to1-qwen3-8b-2p6d-c512/target1-spill0p5-nonstrict/full/offload_analysis_summary.json)
- 当前新方法 2P:6D c512（temperature=0，旧严格FIFO参考）: [summary](current/ablations/mixed1to1-qwen3-8b-2p6d-c512/lifecycle-router-fix/full/offload_analysis_summary.json)
- 当前新方法 2P:6D c640（temperature=0）: [summary](current/qwen3-8b-tp1-mixed1to1-c640-w300-m1200/new-method-agentic-pd/offload_analysis_summary.json)
- 当前新方法 2P:6D c768（temperature=0）: [summary](current/qwen3-8b-tp1-mixed1to1-c768-w300-m1200/new-method-agentic-pd/offload_analysis_summary.json)

`c512` 的 `temperature=1` 旧结果只保留为历史诊断；最终横向对比使用
`temperature=0` 的正式结果。

## 完成集合数据特性

本表的数据组成均为固定 Retool:BrowseComp = 1:1。旧实验没有统一保存父 KV 的
token 加权复用统计时标为“未记录”；No-reverse 的上一轮父 KV 不会被反向传回，
因此其父 KV 未复用率按设计为 100%，但绝对重复 tokens 仍需逐请求事件才能严谨计算。

| 方法 | 配置 | Decode/Agent | 实际 Prefill/Agent | Parent KV 未复用率 | 明确额外 Prefill/Agent |
|---|---|---:|---:|---:|---:|
| Colocated baseline | 8卡，c512 | 3,958 tokens | 10,581 tokens | 未记录 | 未单独记录 |
| Colocated baseline | 8卡，c640 | 3,968 tokens | 14,896 tokens | 未记录 | 未单独记录 |
| Colocated baseline | 8卡，c768 | 3,959 tokens | 23,755 tokens | 未记录 | 未单独记录 |
| 原生 Mooncake | 2P:6D，c512 | 4,159 tokens | 20,747 tokens | 未记录 | 未单独记录 |
| 原生 Mooncake | 4P:4D，c512 | 4,211 tokens | 22,813 tokens | 未记录 | 未单独记录 |
| No-reverse PD | 2P:6D，c512 | 4,282 tokens | 30,661 tokens | 100.00%（设计值） | 未单独记录 |
| No-reverse PD | 4P:4D，c512 | 4,047 tokens | 34,585 tokens | 100.00%（设计值） | 未单独记录 |
| 当前新方法（temperature=1，历史） | 2P:6D，c512 | 3,823 tokens | 8,034 tokens | 0.00% | 约 0 tokens |
| 当前新方法（temperature=0，P→D可行请求先行） | 2P:6D，c512 | 4,027 tokens | 7,486 tokens | 0.00% | 0 tokens |
| 当前新方法 | 2P:6D，c640 | 4,017 tokens | 7,410 tokens | 0.00% | 0 tokens |
| 当前新方法 | 2P:6D，c768 | 3,896 tokens | 7,740 tokens | 1.38% | 341 tokens |

## 并发扫描结论

- 当前非严格 FIFO 的 `c512` Decode 为 9,800 token/s，与历史 `c640` 的
  9,819 token/s 基本相同，同时保持 page-aligned Parent KV 100% 复用。
- `c768` 的 P queue 增至 110.9/卡，Shared Host Arena 触发 shortest-first
  request-generation 驱逐，Parent KV 复用因此降至 98.62%，Decode 回落到
  9,146 token/s。这是容量压力下的预期降级，不是传输死锁或 snapshot
  状态丢失。
