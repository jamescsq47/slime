# Qwen3.5-9B SWE500：H2D 槽位解耦 R9 → R11

2026-09-11。结论：**恢复槽无效占用和恢复前排队减少，但本轮没有吞吐收益，不能把该修改作为性能验收通过。**
这是完整500题测评，各任务执行一次，不是持续补充任务的 closed-loop 稳态实验。

## 改动和对齐条件

R9已先上传：SGLang `pd_mamba` commit `9262db4fa3`，slime `main` commit `61ca2d8689`。
R11在该引擎基线上开启 `SGLANG_AGENTIC_KV_P_H2D_DECOUPLED=true`：

- 保留每P **4个物理H2D槽**；完整 Attention＋Mamba copy fence 完成、无在途CPU/DMA引用后立即归还物理槽，不再等到scheduler完成Radix绑定。
- 完整 parent＋suffix KV/Mamba workset lease、Host所有权继续保留到原有交接点；释放传输槽不等于释放KV或Host。
- 在真实PD调度入口的broker service之后，及时推进已选中的恢复请求；不把allocator/Radix写入移到后台线程，也不改变请求类别优先级。
- 已选中/驻留恢复的额度最多 **8个/P**（物理槽数的2倍）。这是显式增加的有界staging额度，不是严格同额度的单因素计时消融；没有额外预分配HBM池。
- 默认关闭；仅TP=1、request-owned Mamba模式生效。dense/Qwen3、TP>1保留原逻辑。

其余对齐：Qwen3.5-9B、TP=1、2P:6D、c500、同一500题和顺序、同一外部OpenEnv/Miles PR51 fenced-shell harness；每轮8192 tokens、最多64轮、context131072；temperature0.6、top-p0.95、top-k20、min-p0、thinking开启。8卡静态显存均0.80，Mamba:Attention比例0.5，page/稳定checkpoint间隔64。原生HiCache/Mooncake关闭，自定义Direct＋Shared Host回传；关闭拥堵重算及固定失败重算，Q32/32只记录、不控制出口。Direct工具阈值和建链deadline均1秒。Docker每题2CPU/4GiB，shell600秒、verifier2400秒，保持不变。

R10曾在早期运行发现真正PD入口未推进已预选槽的问题，已停止并保留记录，**不纳入性能对比**。修复后增加真实PD入口与满槽回归测试；603个生命周期/故障/TP/Mamba测试通过，独立审计GO后才启动R11。没有修改安装的pd环境或另一agent维护的h100源码。

## 完整500题结果

| 指标 | R9：4槽/P | R11：4槽/P，物理槽解耦 |
|---|---:|---:|
| 正常结束 / 环境失败 | 498 / 2 | 496 / 4 |
| 正确题数 / 正确率 | 160 / 32.0% | 148 / 29.6% |
| 全部500题收尾 | 60分35秒 | 58分58秒 |
| T450，仅status=completed | 31分24秒 | 31分20秒 |
| 单题耗时平均 / P50 / P90 | 1293.87 / 1362.26 / 1879.20秒 | 1277.28 / 1357.36 / 1872.14秒 |
| 平均轮数 | 44.342 | 43.212 |
| 平均Decode / 题 | 13,451.97 | 13,567.04 |
| 平均实际Prefill / 题 | 21,643.66 | 21,065.54 |
| 平均理想必要增量 / 题，不计page边界 | 20,281.21 | 19,742.78 |
| 平均全命中Prefill / 题，含page64边界 | 21,643.66 | 21,065.54 |
| 超出全命中值的历史重算 / 题 | 0 | 0 |
| Prompt token加权命中率 | 96.16% | 96.07% |
| 全程精确未命中P token/s，总计 | 2,976.92 | 2,977.01 |
| 全程模型输出D token/s，总计 | 1,850.21 | 1,917.31 |
| 全程P / D Forward，每卡平均 | 33.58% / 68.40% | 32.95% / 76.68% |
| 全程D running / 卡 | 13.87 | 14.52 |
| 全程P / D Attention活跃池占比 | 9.05% / 24.05% | 12.71% / 25.65% |
| 全程P / D Mamba活跃池占比 | 11.03% / 15.07% | 14.76% / 15.71% |

R11业务区间：2026-09-11 **21:07:23.921–22:06:21.964 UTC**，3538.043秒。
准确总数：21,606次模型调用；Decode6,783,518；实际Prefill10,532,772；不计边界的理想增量9,871,391；page边界661,381 tokens。逐题500行actual−ideal64全部为0。
理论必要增量仍按harness删除旧reasoning后的稳定Prompt前缀计算，包含重新序列化的可见回复、工具结果和template；不把所有轮完整Prompt累加当作必要Prefill。

R11正常终止分布：max_turns162、task_complete109、max_tokens_per_turn102、repeated_command_outcome94、no_command26、command_timeout3；另4题environment_error（2 TimeoutError、2 RuntimeError）。失败题为django-12143、15277、12304、16642；与R9失败题不同。模型/工具轨迹未固定，且仅单次对比，不能把正确率差异证明为缓存损坏或调度因果。

## 同一中段300–1500秒

| 指标 | R9 | R11 | 变化 |
|---|---:|---:|---:|
| P compute token/s，总计 | 6,173.23 | 5,893.05 | −4.54% |
| D token/s，总计 | 4,059.16 | 3,994.63 | −1.59% |
| D token/s/卡 | 676.53 | 665.77 | −1.59% |
| P Forward/卡 | 71.92% | 67.74% | −4.18百分点 |
| D Forward/卡 | 97.23% | 95.02% | −2.21百分点 |
| D running/卡 | 30.85 | 30.89 | 基本相同 |
| P Attention活跃池占比 | 18.68% | 24.78% | +6.10百分点 |
| D Attention活跃池占比 | 51.02% | 50.55% | −0.47百分点 |
| P Mamba活跃池占比 | 19.08% | 25.30% | +6.22百分点 |
| D Mamba活跃池占比 | 32.46% | 32.64% | +0.18百分点 |
| P原生计算等待队列/卡 | 0.067 | 0.072 | 仍很小 |
| P prefill-inflight队列/卡 | 5.88 | 8.59 | 增加 |
| D prealloc / transfer队列/卡 | 0.105 / 0.108 | 0.116 / 0.139 | 略增加 |
| 活跃agent，系统总平均 | 371.69 | 362.50 | 样本集合/阶段并非固定 |

GPU counter按各endpoint分别插值差分；gauge按时间积分，不能简单相加缺失scrape的聚合计数器。中段最大scrape间隙R9为25.38秒、R11为22.86秒，跨间隙采用线性估计。全程最后一小段超出最后scrape采用端点保持；精确墙钟token吞吐来自原始请求总量，不依赖该估计。池占比不是整卡HBM占比。

## 槽位与恢复等待：机制有效，但不是吞吐充分条件

以下排队量为两个P合计、唯一snapshot的生命周期时间积分。worker日志有1秒分辨率；不可将小于1秒的阶段解释为精确瞬时并发。

| 指标 | R9 | R11 |
|---|---:|---:|
| Host durable＋下一轮已到，等待恢复I/O | 91.24个 | 62.58个（−31.4%） |
| 已materialize、尚未进入I/O | 3.79个 | 1.39个 |
| 恢复I/O等效并行总量，按wall时长 | 1.53个 | 1.62个 |
| Host已durable、下一轮尚未到达 | 53.11个 | 61.68个 |
| 真实shell同时执行数量的估计范围 | 16.28–16.38个 | 16.63–16.85个 |
| 每P被占物理槽，周期采样均值 | 3.1875 / 4 | 1.075 / 4 |
| 物理槽全满，周期样本比例 | 61.25% | 18.75% |
| 中段H2D完成snapshot数 | 11,917 | 12,351 |
| 每snapshot平均大小 | 0.4813GiB | 0.4675GiB |
| H2D完成字节量 / 1200秒，系统总计 | 4.780GiB/s | 4.812GiB/s |
| H2D GPU event均值 | 46.45ms | 51.04ms |
| H2D I/O wall均值 / P90 | 153.79 / 248.36ms | 157.87 / 239.81ms |

槽占用统计每P约30秒采样、各40个样本，不是DMA忙碌占比。R11恢复前排队量的时间戳量化范围约58.03–67.62个。Host等待下一轮包含应用/HTTP间隙，不能都称为正在执行工具。

R11新埋点（按handoff落在中段的12,356个snapshot）：

| 阶段 | 平均 | P50 | P90 |
|---|---:|---:|---:|
| 被选中→workset grant | 24.98ms | 23.57ms | 36.27ms |
| I/O开始→复合Attention＋Mamba fence | 156.88ms | 124.60ms | 238.79ms |
| fence→scheduler交接 | 336.88ms | 232.78ms | 737.34ms |

最后一行在R11不再占物理槽，但仍占合法workset/Host所有权；其最大值16.84秒，是控制/交接阶段长尾，不是16秒PCIe DMA。该埋点按handoff取样，H2D表按copy完成取样，边界处5个snapshot的差别正常。**仍没有完整CPU tracing，因此不能把P剩余32.26%的时间全部归因于这个阶段。**

解释：局部恢复前排队确实下降，物理槽空出来了，但总H2D字节吞吐几乎不变，P inflight和P活跃KV/Mamba驻留反而增加。D running基本未变。因此“只要把槽提前释放，P就能接近100%并显著喂饱D”没有被本轮支持。下一步若继续优化，应测量完整已恢复→绑定→原生bootstrap/ready队列→Forward→P2D交付链，而不是继续凭队列长度扩大槽位。上述是待验证方向，不是已证实的唯一瓶颈；本轮不再改逻辑追逐指标。

## 工具时间和生命周期守恒

| 工具时间，全程shell调用 | R9 | R11 |
|---|---:|---:|
| 调用次数 | 21,934 | 21,365 |
| 平均 / P50 / P90 | 1.328 / 0.818 / 1.809秒 | 1.287 / 0.853 / 1.929秒 |
| P95 / P99 | 2.310 / 5.172秒 | 2.383 / 4.881秒 |
| 最大 | 600.29秒 | 604.39秒 |

R11 shell区间分布：≤0.5秒953；0.5–1秒11,795；1–2秒6,716；2–5秒1,702；5–10秒117；10–60秒68；60–300秒11；>300秒3。包含工具超时处理开销，因此观测wall可能略超过600秒限制。

R11唯一snapshot计数：

```
6,196 Direct + 15,039 Host durable + 269 app-final + 102 length = 21,606调用
15,039 Host durable = 15,039 D source释放
14,910 Host恢复 + 129 已结束Host清理 = 15,039
6,196 Direct + 14,910 Host恢复 = 21,106 = 21,606 - 500首轮
P→D Host: 48 queued = 48 durable = 48 P释放 = 48 D恢复 = 48 Host释放
```

Direct/Host出口比例29.18%/70.82%，无显式重算、无Host驱逐、无剩余未解释Host snapshot。500题逐题缓存命中核对通过，不代表模型答案必然正确。结束时所有本轮GPU进程、容器和watcher已退出；GPU7他人PID1868643未触碰。CUDA退出曾短暂延迟，经过既有有界清理过程后显存已释放，未遗留本轮CUDA进程。

## 文件位置与复现

- 总表：[SWEBENCH_QWEN35_9B_TP1.md](SWEBENCH_QWEN35_9B_TP1.md)。
- 轻量汇总和500行token明细：[qwen35-h2d-decoupling-r11-20260911](new-method/qwen35-h2d-decoupling-r11-20260911/)。原始logs/traces不上传GitHub。
- R11完整记录：`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-decoupled-q32-32-20260911-r11`。
- R9完整记录：`/tmp/pd-persist/fused-qwen35-9b-tp1-swe500-2p6d-c500-h2d4-q32-32-20260911-r9`。
- 失败R10保留同前缀的`...-r10`目录和`FAILED_PREFLIGHT_PERFORMANCE.md`，不得与R11混合。
- 本地启动器：`scripts/new_method/run_qwen35_fused_swe500_2p6d.sh`；本轮运行前的启动器、harness、workload在R11的`source-snapshot/`；引擎base为`9262db4fa3`加`engine.patch`。
- 离线统计：`scripts/tools/summarize_swe_pd_comparison.py`、`summarize_swe_incremental_prefill.py`、`summarize_swe_h2d_window.py`、`summarize_swe_p_stage_window.py`。

全程墙钟D吞吐约+3.63%，但T450只差3.9秒（−0.21%）、中段D吞吐−1.59%、正确率少12题。**不能据此宣称整体性能改善，也不能用这一单次有限数据实验断言该机制普遍更差。** 默认不开启，保留为可复现实验开关。
