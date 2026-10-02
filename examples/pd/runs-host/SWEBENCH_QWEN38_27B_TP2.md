# SWE-bench Verified · Qwen3.8-27B TP=2 · 双节点对比

本文对比两组双节点、c128、SWE-bench Verified 500题正式实验：

- **Colocated baseline**：`qwen38-27b-tp2-colocated-2node-c128-r2`；
- **当前 Agentic-PD**：`qwen38-27b-tp2-global-c128-evictfix-full-20260930`。

Colocated `r1` 为误运行，不计入结果。全量时间均包含模型调用、工具执行、verifier和最终drain，不含模型加载。

## 1. 配置对比

### 1.1 对齐配置

| 项目 | 两组共同配置 |
|---|---|
| 模型 | Qwen3.8-27B |
| 节点 / GPU | a10 + a11；共16张GPU |
| 数据集 | SWE-bench Verified；固定顺序500题，各执行一次 |
| 全局并发 | 128；一题收尾后补入下一题 |
| TP / 显存 | TP=2；`mem_fraction_static=0.80` |
| Mamba | `mamba_full_memory_ratio=0.90`，track interval 64 |
| KV page / 上下文 | page size 64；最大上下文131,072 tokens |
| Prefill | chunked prefill 8,192；每批最多8,192 tokens |
| Agent限制 | 单轮最多8,192 tokens；累计输出最多81,920 tokens；最多64轮 |
| 采样 | temperature=0.6，top_p=0.95，top_k=20，seed=2026，thinking=true |
| Harness | `swe_bench_openenv`；Chat Completions + `openai_tools` |
| Parser | reasoning=`glm45`；tool call=`qwen3_coder` |

### 1.2 方法差异

| 项目 | Colocated baseline | 当前 Agentic-PD |
|---|---|---|
| 实例布局 | 每节点4个TP=2副本，共8个副本 | a10上4个P TP=2组；a11上4个D TP=2组 |
| Prefill / Decode | 每个副本同时承担P和D | P/D物理分离 |
| KV路径 | 本地Radix/Mamba缓存 | D→P和P→D均支持Direct与Host慢路径 |
| Router | 8个副本cache-aware router | P/D全局late binding，按容量和负载选择目标TP组 |
| TP一致性 | 原生SGLang TP | rank0唯一决策，组内rank完成后统一提交 |
| Direct阈值 | 不适用 | 工具fast-arrival 1 s；Direct admission 1 s |
| Decode增长预留 | 原生动态增长 | 8,192 tokens/request |
| Host Arena | 不适用 | D→P 64 GiB/rank，共512 GiB；P→D 16 GiB/rank，共128 GiB |
| 控制面 | 原生调度 | TCP/内存控制，运行期不轮询NFS |
| 环境 | `pd_mamba_baseline` | `pd_multi_node_v3` |

## 2. 全量结果对比

| 指标 | Colocated baseline | 当前 Agentic-PD | PD相对baseline |
|---|---:|---:|---:|
| 收尾 / 失败 / 截断 | 500 / 0 / 0 | 500 / 0 / 0 | 持平 |
| SWE-bench通过 | 269 / 500 = **53.8%** | 265 / 500 = **53.0%** | -0.8 pct |
| Verifier完成 / 超时 | 499 / 1 | 499 / 1 | 持平 |
| Verifier infrastructure error | 0 | 0 | 持平 |
| 全量完成时间 | **5,144.26 s** | **6,499.20 s** | +26.34% |
| T250 / T450 / T500 | 1,982.58 / 3,630.61 / 5,144.26 s | 2,531.69 / 4,973.31 / 6,499.20 s | — |
| 收尾速率 | 0.0972 agent/s | 0.0769 agent/s | -20.85% |
| Agent耗时，平均 / P50 / P90 | 917.58 / 754.42 / 1,793.45 s | 1,244.01 / 910.41 / 2,592.39 s | — |
| Prefill实际计算吞吐 | 2,429.23 token/s | 2,573.56 token/s | +5.94% |
| Decode吞吐，全程总计 | **2,007.65 token/s** | **1,590.52 token/s** | -20.78% |

## 3. 稳态性能与资源对比

Colocated使用300–1,500秒窗口；Agentic-PD使用自动识别的稳态窗口，约1,249–5,229秒。两者都排除启动和末段drain，但时间窗口并不完全相同。

| 指标 | Colocated baseline | 当前 Agentic-PD |
|---|---:|---:|
| Prefill实际计算吞吐 | 3,936.05 token/s | 2,712.04 token/s |
| Decode吞吐，总计 | **3,018.05 token/s** | **1,660.16 token/s** |
| Decode吞吐 / D TP组 | 377.26 token/s | 415.04 token/s |
| Decode吞吐 / D物理GPU摊销 | 188.63 token/s | 207.52 token/s |
| D Running，总计 | 约118.88 | 62.22 |
| D Running / TP组 | 14.86 | 15.56 |
| D Running峰值 | 未单独记录 | 84 |
| Prefill Forward / 物理GPU | 13.29%（共置Prefill阶段） | 26.56%（P GPU） |
| Decode Forward / 物理GPU | 85.86%（共置Decode阶段） | **99.74%**（D GPU） |
| 同卡Forward总占比 | **99.15%** | 不适用；P/D位于不同GPU |
| Attention KV活跃占比 | 52.04% | D：73.80%；P：16.65% |
| Attention KV可驱逐占比 | 10.80% | 按request-generation转移，不同口径 |
| D→P Host驱逐 | 不适用 | 237个logical snapshot |
| 硬错误 / Launcher退出码 | 0 / 0 | 0 / 0 |

Colocated共8个副本同时参与Decode，PD只有4个D组，因此不能只比较总Decode吞吐。按D TP组计，PD从377.26提高到415.04 token/s，约提高10.0%；但这远未达到“D只做Decode后单组吞吐接近2倍”的理想预期，所以PD总吞吐仍较低。

Agentic-PD的237次Host驱逐均没有再触发请求生命周期错误；该运行已跨过修复前第79次驱逐时的必现故障点。

## 4. 数据与轨迹特征对比

| 指标 | Colocated baseline | 当前 Agentic-PD | PD相对baseline |
|---|---:|---:|---:|
| 模型调用轮数 / 题，平均 / P50 / P90 | 31.80 / 30 / 64 | 32.75 / 31 / 64 | 轨迹接近 |
| Decode输出 / 题 | 20,657.73 | 20,616.23 | -0.20% |
| Decode输出合计 | 10,328,865 | 10,308,116 | -0.20% |
| 各轮完整Prompt累计 / 题 | 679,354.85 | 699,840.06 | +3.02% |
| GPU实际Prefill计算 / 题 | 24,995.20 | 33,448.78 | +33.82% |
| GPU实际Prefill计算合计 | 12,497,600 | 16,724,392 | +33.82% |
| 相对完整Prompt累计量的计算减少率 | 96.32% | 95.22% | -1.10 pct |
| 工具observation / 题 | 11,166.20 | 11,319.96 | +1.38% |
| Shell调用 / 题 | 30.90 | 31.86 | +3.11% |
| 工具时间 / 题，平均 / P50 / P90 | 49.64 / 28.85 / 102.39 s | 65.79 / 29.51 / 139.72 s | — |
| Verifier时间 / 题，平均 / P50 / P90 | 18.83 / 9.77 / 20.42 s | 18.80 / 9.67 / 22.30 s | 接近 |

两轮正确率、平均Decode长度、轮数和累计Prompt规模接近，说明数据与Agent语义基本可比。PD实际Prefill量多33.82%，是当前与baseline差距中需要继续分解的重要部分。

| 终止原因 | Colocated baseline | 当前 Agentic-PD |
|---|---:|---:|
| `task_complete` | 230 | 227 |
| `max_tokens_per_turn` | 172 | 168 |
| `max_turns` | 49 | 53 |
| `tool_format_error` | 24 | 24 |
| `no_command` | 17 | 22 |
| `final_answer` | 7 | 5 |
| `repeated_command_outcome` | 1 | 1 |

## 5. 结论

- **正确性可比**：通过率53.8%对53.0%，轨迹长度和Decode量几乎一致。
- **PD逻辑链已长时间跑通**：500/500收尾、0失败、0硬错误，并安全经过237次Host驱逐。
- **D计算本身已打满**：D Forward约99.74%，单D TP组Decode吞吐高于baseline约10%。
- **端到端性能仍未达到baseline**：全量完成时间增加26.34%，Decode总吞吐降传20.78%。
- **下一个重点**：继续分解PD为何比baseline多计算33.82%的Prefill tokens，以及为何D-only单组收益只有约10%。

## 6. 可复现入口与原始数据

| 项目 | Colocated baseline | 当前 Agentic-PD |
|---|---|---|
| 启动脚本 | `tools/dualpd/run_qwen38_27b_two_node_colocated_c128.sh` | `tools/dualpd/run_qwen38_27b_two_node_pd_c128.sh` |
| 结果目录 | `runs/dualpd/qwen38-27b-tp2-colocated-2node-c128-r2` | `runs/dualpd/qwen38-27b-tp2-global-c128-evictfix-full-20260930` |
| 总结 | `workload/summary.json` | `workload/summary.json` |
| 轨迹分析 | `workload/swe_bench_profile_summary.{json,md}` | `workload/swe_bench_profile_summary.{json,md}` |
| 请求轨迹 | `workload/requests.completed.jsonl` | `workload/requests.completed.jsonl` |
| 引擎采样 | `workload/engine_metrics.jsonl` | `workload/engine_metrics.jsonl` |
| 吞吐序列 | `workload/engine_throughput_2s.jsonl` | `workload/engine_throughput_2s.jsonl` |
| Launcher退出码 | 0 | 0 |

`qwen38-27b-tp2-colocated-2node-c128-r1` 含 `MISRUN.md`，不作为可比较结果。
