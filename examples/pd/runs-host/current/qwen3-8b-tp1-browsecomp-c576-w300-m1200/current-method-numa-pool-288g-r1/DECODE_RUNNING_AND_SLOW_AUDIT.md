# Decode running与Slow积压复核

本次只读分析，未修改生产代码、未启动GPU实验。旧组为同c576的
`current-method-slow-congestion-1s-20260909-r1`，不是c512的5148组。

## 1. running不是transfer

`scheduler_metrics_mixin.py:681`的指标来自`batch.reqs`，Transfer和Prealloc分别统计。
已结束请求的offload由独立manager持有；它的KV可能尚未释放，但不是持续参与Decode的running。
Gauge是周期性发布的最近batch状态，而不是每纳秒精确跟踪。

| 正式窗口每D平均 | 旧 | 新 |
|---|---:|---:|
| 实际Decode batch running | 49.67 | 51.19 |
| P→D prealloc queue | 1.57 | 1.85 |
| P→D transfer queue | 0.207 | 0.343 |
| 普通D waiting queue | 0 | 0 |
| 已结束Decode但D源KV尚未释放（日志时间重建） | 约1.94 | 约1.96 |

最后一行以`request_seen`（完成Decode时）到同req的`d_generation_release`积分，
只计具有Direct offer的parent snapshot，跨测量边界裁剪，秒级日志有量化误差。
对应page-aligned parent逻辑tokens平均每D约24894→27663；不是精确去重物理页数。
它与running分开，不应从49.67/51.19里扣除。

## 2. 平均并发上升掩盖了分布变化

以下合并4张D的正式窗口采样，每卡每时刻一个样本。

| running分布 | 旧 | 新 |
|---|---:|---:|
| 均值 | 49.67 | 51.19 |
| 中位数 | 49 | 43 |
| P90 | 82 | 96 |
| P95 | 95 | 113 |
| 小于32的样本占比 | 28.57% | 33.15% |
| 大于等于96的样本占比 | 4.61% | 10.31% |

新组更不平稳：低并发更多，同时高并发突发把均值抬高。
4个D的Decode日志均每40步输出一次。正式窗口3101→2796条，结合GPU Forward
累计时间及实际生成token counter，可以估算：

| 每个Decode Forward step | 旧 | 新 |
|---|---:|---:|
| 平均产出tokens | 46.65 | 49.97 |
| 平均GPU Forward时间 | 38.35 ms | 42.68 ms |

边界可能有每卡不足40步的误差；这不是逐kernel profiler。该估计与打印batch均值
46.66/49.92一致。产出每步增加约7%，但耗时增加约11%，故每秒吞吐下降。

按相邻2秒采样的running分桶，再累计token/Forward秒：

| running区间 | 旧活跃token/s/卡 | 新活跃token/s/卡 | 旧不可驱逐KV tokens均值 | 新均值 |
|---|---:|---:|---:|---:|
| 16–31 | 699 | 579 | 214.5k | 249.4k |
| 32–47 | 1008 | 901 | 229.6k | 271.7k |
| 48–63 | 1361 | 1258 | 255.9k | 267.1k |
| 64–79 | 1592 | 1582 | 273.9k | 284.6k |

同样running区间新组也更慢，且低并发区间KV占用更多。该KV含等待释放的非running
父KV，不能将其除以running冒充精确活跃上下文长度。需要逐batch序列长度和DMA重叠
数据，才能进一步分离长上下文与搬运争用；不能声称已经证明全部差额来自其中一个。

## 3. D→P Fast / Slow / Recompute比例没有突变

正式窗口按各出口的唯一snapshot首次日志计数，比例是出口事件构成，不是完成Agent比例。
秒级边界与之前按不同事件/最后记录统计的表会有几个snapshot的差异。

| 出口 | 旧次数 / 比例 | 新次数 / 比例 |
|---|---|---|
| Direct发送完成 | 5639 / 80.38% | 5561 / 79.97% |
| Slow fallback | 1070 / 15.25% | 1075 / 15.46% |
| 显式反馈重算 | 306 / 4.36% | 318 / 4.57% |

Slow fallback中tool_confirmed=False旧1、新103；该字段表示应用ACK/arrival尚未被
确认，不等于纯工具执行耗时超过1秒。完成Agent的累计工具时间均值仅0.264→0.303秒。
因此不能把这103次直接解释为真实慢工具，需单独关联应用ACK/控制时序才能细分。

D完成Decode→Slow durable平均1.58→1.48秒（含1秒deadline），P90均2秒，新最大4秒；
Slow offer→D2H启动均约0.03秒，D2H完成→D释放在同一日志秒内。
不支持“D端D2H排队大幅恶化”这一解释。明显增加的是P→D经Host路径，见
[带宽报告](HOST_PERFORMANCE_ANALYSIS.md)：次数+71%、字节量+132%。

## 4. 新发现：预算结束的Host副本未清理

逐snapshot关联最终Host ledger、D2H/H2D日志、Agent metadata.agentic_request_id、
status和stop_reason后，发现一个与“有用Slow恢复堵塞”不同的问题：

| 测量结束前已经结束的Agent，却未清理最后一轮Host | 旧 | 新 |
|---|---:|---:|
| Agent数量 | 5 | 30 |
| 结束原因 | 全部truncated/budget | 全部truncated/budget |
| 残留KV大小 | 24.26 GiB | 139.46 GiB |
| 正式窗口过期副本的估算平均占用 | 6.31 GiB | 56.71 GiB |

新组30个均未提交该snapshot的下一轮arrival，且状态仍HOST_READY。
例：`56d01079645b40b999cfdc0d70f8c43d:4`，32128 tokens、4.41 GiB，
23:52:18已Host durable且D释放；Agent因budget结束，无generation5，但Host一直保留。
它不是应该继续H2D的请求，而是应取消并释放的终止generation。

代码存在对应生命周期缺口：BrowseComp预算结束时发送final确认；D的
`_agentic_complete_final_candidate`主要处理未开始staging的Direct candidate，
一旦staging或sent就不在该处完成终止。Host manager没有直接消费该final确认的路径，
主要根据CONSUMED/FAILED等Host状态释放。D源已释放、candidate已退休后，迟到的
应用final缺少向Host所有者传播终止的闭环。ACK实际发布/消费的每次时间并未完整落盘，
不能把所有个例的竞态时间细分到毫秒，但终止Agent仍占Host这个事实已确认。

这不是本次才第一次出现：旧组也有5个；本轮明显放大。不能宣称当前代码没有问题。
但本轮两个NUMA的D→P峰值仍低于各自96 GiB保底，尚未因这些过期副本压缩P→D的
192 GiB上限，故不能把此bug当作本轮全部吞吐下降的已证实原因。

## 5. 纠正“123个未恢复”的含义

此前整轮结束ledger中123个HOST_READY：

- 30个在正式结束前写入，均为上述已预算结束Agent，139.46 GiB。
- 93个在停止业务后、服务退出阶段才写入，140.84 GiB；不可算成正式窗口的恢复积压。

跟踪**在正式窗口完成D→P D2H的同一批snapshot**：

| 组 | 写入 | 窗口结束前恢复 | 窗口后恢复 | 无下一轮且未清理 |
|---|---:|---:|---:|---:|
| 旧 | 1069 | 1064 | 0 | 5 |
| 新 | 1075 | 1045 | 1 | 29 |

新组另1个终止残留在预热末期写入，所以终止残留总数为30。
这批真正需要下一轮的Slow最终都有H2D完成证据，未发现其永久停止恢复；
但恢复延迟平均约9秒、个别超过100秒依然存在，不能说都“及时”。
之前把D→P Host分段增长全部解释成活跃恢复积压是不准确的；新组平均Host77.44 GiB中，
约56.71 GiB已是终止后残留，扣除后约20.73 GiB才是其他在途/等待/尚未结束的数据。

## 建议

优先修复应用terminal/budget/cancel到当前Host owner的幂等终止传播，遵守现有DMA
fence和TP组级释放，不改快慢路径阈值。之后仍需用同等running和上下文区间验证
Host→D复制重叠的影响；不能仅凭平均running或Forward99%判定无性能损失。
本轮只是分析，没有执行该修复。
