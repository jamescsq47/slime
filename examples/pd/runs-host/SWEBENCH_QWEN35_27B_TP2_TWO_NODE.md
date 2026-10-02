# SWE-bench Verified · Qwen3.5-27B TP=2 · 双节点对比

本文只对比同一模型和 workload 下的两种双节点方法：

1. Colocated baseline；
2. 当前新方法 Agentic-PD 分离。

不纳入 Qwen3.8，也不把两次 Colocated 运行当作两种方法。两列正式结果分别是：

- Colocated：`qwen35-27b-tp2-colocated-2node-c128-r6`；
- Agentic-PD：`qwen35-27b-tp2-global-pd-c128-r2`。

## 1. 配置对齐

| 项目 | Colocated baseline | 当前 Agentic-PD |
|---|---|---|
| 模型 | Qwen3.5-27B | Qwen3.5-27B |
| 节点 / GPU | a10 + a11；共 16 张 GPU | a10 + a11；共 16 张 GPU |
| 实例布局 | 每节点 4 个 TP=2 副本，共 8 个副本 | 4 个 P TP=2 组 + 4 个 D TP=2 组 |
| 数据集 | SWE-bench Verified；固定顺序 500 题 | 相同 |
| 全局并发 | 128 | 128 |
| TP / 显存 | TP=2；`mem_fraction_static=0.80` | 相同 |
| Mamba | `mamba_full_memory_ratio=0.90`，track interval 64 | 相同 |
| KV page / 上下文 | page size 64；最大 131,072 tokens | 相同 |
| Prefill | chunked/max prefill 均为 8,192 tokens | 相同 |
| Agent 限制 | 单轮 8,192；累计输出 81,920；最多 64 轮 | 相同 |
| 采样 | temperature=0.6，top_p=0.95，top_k=20，seed=2026 | 相同 |
| Harness | `swe_bench_openenv` + `openai_tools` | 相同 |

## 2. 全量结果

| 指标 | Colocated baseline | 当前 Agentic-PD |
|---|---:|---:|
| 完成 / 失败 / 截断 | **500 / 0 / 0** | **500 / 0 / 0** |
| SWE-bench 通过 | **312 / 500 = 62.4%** | **325 / 500 = 65.0%** |
| Verifier 完成 / 超时 | 499 / 1 | 499 / 1 |
| Verifier infrastructure error | **0** | **0** |
| T250 / T450 / T500 | 1,298.02 / 2,247.43 / 6,629.93 s | 2,665.34 / 4,202.69 / 6,766.54 s |
| 全量完成时间 | **6,629.93 s** | **6,766.53 s** |
| 收尾速率 | 0.0754 agent/s | 0.0739 agent/s |
| Agent 耗时，平均 / P50 / P90 | 570.89 / 520.15 / 914.72 s | 1,066.14 / 1,091.35 / 1,549.40 s |
| 全程实际 Prefill 吞吐 | 2,923.70 token/s | 2,612.19 token/s |
| 全程 Decode 吞吐 | 764.63 token/s | 737.01 token/s |
| Launcher 退出码 | **0** | **0** |

两组 T500 都包含最后一个 verifier 达到 2,400 秒上限的长尾；均无基础设施错误。因此方法吞吐差异应优先看 T250、T450 和固定中段窗口，不能只看被相同 verifier 长尾主导的 T500。

## 3. 中段性能与资源

统一使用 300–1,500 秒窗口。Prefill/Decode Forward 来自同一套 device timer，因此两者之和不超过 100%。

| 指标 | Colocated baseline | 当前 Agentic-PD |
|---|---:|---:|
| 窗口内完成题数 | 271 | 118 |
| 实际 Prefill 计算量 | 约 9,776,512 tokens | 4,441,280 tokens |
| 实际 Prefill 吞吐 | **8,162.70 token/s** | **3,708.14 token/s** |
| Decode 计算量 | 约 2,778,837 tokens | 1,281,388 tokens |
| Decode 吞吐 | **2,320.13 token/s** | **1,069.86 token/s** |
| Prefill Forward / 物理 GPU | **28.55%** | **37.25%（P GPU）** |
| Decode Forward / 物理 GPU | **70.43%** | **99.36%（D GPU）** |
| 同卡 Forward 总占比 | **98.99%** | P/D 分卡，应分别统计 |
| Running / TP 组 | 13.16 | P 0.00 / D 8.01 |
| Waiting / TP 组 | 0.14 | P 约 0.001 / D 0.00（仅原生 scheduler 队列） |
| Attention KV active | 44.83% | P 38.27% / D 48.27% |
| Mamba active | 22.68% | P 20.93% / D 17.00% |
| Prefix token 命中率 | 约 96.30% | 约 96.58% |

Colocated 中段的正确 Forward 口径为：**28.55% + 70.43% = 98.99%**。PD 的 TP=2 device timer 在 endpoint 上聚合两个 rank，表中已除以 2，得到每张物理 GPU 的占比；P/D 位于不同 GPU，不能相加。

PD 的 D Forward 已接近满载，但 Running 只有 baseline 的约 61%，Decode 吞吐也只有约 46%。这说明本轮主要损失是 **D batch 太小、P/恢复流水线供给不足**，不是 Decode scheduler 被控制面暂停。P 侧同时只有 37.25% Forward，且单位 active Forward 的 token 产出也低于 colocated，说明恢复后 Prefill 的批量化仍不足。

## 4. 数据特征

| 指标 | Colocated baseline | 当前 Agentic-PD |
|---|---:|---:|
| 模型调用轮数 / 题，平均 / P50 / P90 | 52.17 / 63.5 / 64 | 51.14 / 60 / 64 |
| Decode 输出 / 题 | 10,140.21 tokens | 9,954.93 tokens |
| Decode 输出合计 | 5,070,107 tokens | 4,977,466 tokens |
| 各轮完整 Prompt 累计 / 题 | 931,995.61 tokens | 928,573.52 tokens |
| GPU 实际 Prefill / 题 | 38,767.87 tokens | 35,295.36 tokens |
| GPU 实际 Prefill 合计 | 19,383,936 tokens | 17,647,680 tokens |
| 相对完整 Prompt 累计量的计算减少率 | 95.84% | 96.20% |
| 工具 observation / 题 | 18,774.04 tokens | 19,042.89 tokens |
| Shell 调用 / 题 | 51.66 | 50.60 |
| 工具时间 / 题，平均 / P50 / P90 | 68.17 / 46.21 / 102.44 s | 66.15 / 41.51 / 87.73 s |
| Verifier 时间 / 题，平均 / P50 / P90 | 18.97 / 9.63 / 20.75 s | 18.99 / 9.61 / 24.29 s |

两轮轨迹长度、Decode 量、累计 Prompt 和工具时间接近；PD 甚至少计算约 9.0% 的实际 Prefill tokens。因此性能下降不能归因于 PD 数据更长或正确性语义不一致。

## 5. PD 路径健康度

- 500/500 正常收尾，0 请求失败、0 infrastructure error，运行日志没有 Traceback、OOM、TP tombstone 上限或生命周期断言。
- D→P Direct admission deadline 到期 5,502 次；这些请求随后进入 Slow，而不是丢失 KV。
- D→P Host 峰值约 427.38 GiB / 512 GiB；触发 386 个完整 request-generation snapshot 驱逐，按既定语义转为重算。
- P→D pending 可周期性清零；P→D Host 与 Direct 都持续完成，没有永久 P-ready 死锁。
- 测量结束后 a10/a11 均无 GPU compute process，未遗留占卡孤儿进程。

Host 驱逐和大量 Direct deadline fallback 会降低父 KV 复用收益，并把更多工作重新送回 P；但最直接的中段瓶颈仍是 P 产出和恢复交接没有持续把 D running 填高。

## 6. 原始数据

| 内容 | Colocated baseline | 当前 Agentic-PD |
|---|---|---|
| 正式结果目录 | `runs/dualpd/qwen35-27b-tp2-colocated-2node-c128-r6` | `runs/dualpd/qwen35-27b-tp2-global-pd-c128-r2` |
| 总结 | `workload/summary.json` | `workload/summary.json` |
| 轨迹分析 | `workload/swe_bench_profile_summary.{json,md}` | `workload/swe_bench_profile_summary.{json,md}` |
| 请求轨迹 | `workload/requests.completed.jsonl` | `workload/requests.completed.jsonl` |
| 引擎采样 | `workload/engine_metrics.jsonl` | `workload/engine_metrics.jsonl` |
| 吞吐序列 | `workload/engine_throughput_2s.jsonl` | `workload/engine_throughput_2s.jsonl` |

历史 `qwen35-27b-tp2-colocated-2node-c128-r5` 是提前停止的 Colocated 阶段结果，不列为第二种方法。
