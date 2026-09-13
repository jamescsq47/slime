# BrowseComp + Qwen3-8B

固定 source-order n680 循环，temperature=0，8卡、TP=1，300秒业务预热 + 1200秒正式测量。
显存按物理GPU对齐：普通卡0.80，GPU7（搜索服务）0.60；PD为4P:4D，P=0/2/4/6，D=1/3/5/7。


## 当前完整新方案

2026-09-09起采用 **快慢路径＋全局Slow恢复拥堵反馈重算**，对应c512实测 **5,148.4 token/s**。
c384/c576均已用当前完整方法重新完成。

- 工具超过1秒未返回：Slow；工具1秒内返回：尝试Direct，建链deadline为1秒。
- 快工具Direct失败：恢复队列不拥堵时转Slow；拥堵时显式完整重算。
- 全局Q只计工具已返回、Host durable、尚未被H2D worker接手的唯一parent generation；跨P不重复，等待工具不计入。
- 每秒采样，连续两次Q≥32进入拥堵，Q≤8退出；信号超过3秒陈旧或无效时保守Slow。
- 只决定新发生的Direct失败出口，不取消已在Host中的请求，不抢占在途KV；DMA fence、workset和TP规则不变。
- Q是等待worker接手的代理指标，不是实际DMA等待队列；32/8是已测参数，不宣称最优。

新方案复现须设置 `SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=true`、`SLOW_CONGESTION_HIGH=32`、
`SLOW_CONGESTION_LOW=8`（后两项完整前缀同为`SGLANG_AGENTIC_KV_`）。
原生HiCache/Mooncake关闭；D→P Host=128 GiB/P、P→D Host=32 GiB/P、H2D=2 lanes/P，
D接收目标1.0，P→D grace=0.5秒；全部Host预注册完成后才开始业务预热。
No-reverse、Colocated及原生Mooncake基线保留已经完成的配置对齐结果。

## 实验矩阵

2026-09-10额外c576消融：已回退统一NUMA池到SGLang `921fbd46ab6d`，
只关闭P→D Host、保留late binding和D→P全部路径，正式300+1200秒为
**4965.7 token/s**，较旧独立Host完整方法4822.5提高2.97%。本轮2874 Agent完成、0失败，
D Forward 99.43%、P Forward 93.87%、D running/卡47.54、D KV/卡69.0%。
不据此替换完整方案默认设置；单轮差异及3个预算终止Host副本残留见
[本轮完整报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-p2d-direct-only-20260910-r1/RESULTS.md)。

| 方法 | 配置 | 状态 | Decode |
|---|---|---|---:|
| Colocated baseline | 8卡，c384 | 完成 | 4,834 token/s |
| Colocated baseline | 8卡，c512 | 完成 | 4,434 token/s |
| Colocated baseline | 8卡，c576 | 完成 | 4,349 token/s |
| 当前新方法（Slow拥堵反馈重算） | 4P:4D，c384 | 完成 | 4,737.0 token/s |
| 当前新方法（Slow拥堵反馈重算） | 4P:4D，c512 | 完成 | 5,148.4 token/s |
| 当前新方法（Slow拥堵反馈重算） | 4P:4D，c576 | 完成 | 4,822.5 token/s |
| No-reverse PD | 4P:4D，c384 | 完成 | 1,982 token/s |
| No-reverse PD | 4P:4D，c512 | 完成 | 1,984 token/s |
| No-reverse PD | 4P:4D，c576 | 完成 | 2,153 token/s |
| 原生 Mooncake | 4P:4D，c384 | 完成 | 1,193 token/s |
| 原生 Mooncake | 4P:4D，c512 | 完成 | 1,320 token/s |
| 原生 Mooncake | 4P:4D，c576 | 完成（显式本地NUMA，见下） | 2,041 token/s |

2026-09-10补齐原生Mooncake c576：使用未修改的`pd_baseline`引擎，300.54秒业务预热＋1200.00秒正式测量。
本轮在独立启动脚本中显式按物理GPU绑定本地NUMA，并把scheduler watchdog从300秒延长到1200秒；
旧c384/c512启动日志为全部TP1 worker自动绑定NUMA0。因此缓存容量、模型和数据参数对齐，
但CPU/NUMA放置并不完全一致，不能把c576相对旧行的吞吐差异仅归因为并发变化。
未修改共享启动脚本、harness、`pd_baseline`源码或另一个agent的自定义引擎。

## 正式窗口吞吐与数据特性

| 方法 | 并发 | Agent/s | Prefill token/s | Decode/Agent tokens | 实际Prefill/Agent tokens | Parent KV复用 |
|---|---:|---:|---:|---:|---:|---:|
| Colocated baseline | 384 | 2.392 | 38,149 | 2,014 | 15,893 | 91.74% |
| Colocated baseline | 512 | 2.181 | 46,523 | 2,031 | 21,308 | 74.11% |
| Colocated baseline | 576 | 2.160 | 47,552 | 2,008 | 21,957 | 74.31% |
| 当前新方法（Slow拥堵反馈重算） | 384 | 2.333 | 34,486 | 2,029 | 14,769 | 97.48% |
| 当前新方法（Slow拥堵反馈重算） | 512 | 2.387 | 35,195 | 2,153 | 14,719 | 96.30% |
| 当前新方法（Slow拥堵反馈重算） | 576 | 2.334 | 35,472 | 2,062 | 15,165 | 95.86% |
| No-reverse PD | 384 | 0.978 | 42,976 | 2,022 | 43,846 | 0.00% |
| No-reverse PD | 512 | 0.966 | 42,803 | 2,051 | 44,251 | 0.00% |
| No-reverse PD | 576 | 1.019 | 42,680 | 2,106 | 41,753 | 0.00% |
| 原生 Mooncake | 384 | 0.582 | 21,509 | 2,043 | 36,823 | 23.26% |
| 原生 Mooncake | 512 | 0.619 | 21,409 | 2,130 | 34,550 | 19.54% |
| 原生 Mooncake（本地NUMA） | 576 | 0.993 | 37,850 | 2,052 | 38,054 | 15.13% |

Parent复用按完成轨迹的完整page-aligned父前缀统计；新方法显式重算会降低该比例。Prefill/Agent与Decode/Agent为窗口总计算量除以完成Agent数。

## 稳态资源

资源为正式窗口各引擎采样均值；Forward为每物理卡平均。Colocated的KV池由P/D共享。

| 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Colocated c384 | 51.1% | — | — | — | 48.8% | 67.6% | 46.3 | 0.8 | — | — |
| Colocated c512 | 60.4% | — | — | — | 39.6% | 84.1% | 59.9 | 3.1 | — | — |
| Colocated c576 | 62.0% | — | — | — | 37.9% | 84.6% | 66.1 | 4.8 | — | — |
| 新方法 c384 | 90.04% | 67.08% | 25.56 | 8.70 | 98.46% | 79.32% | 50.33 | 0.00 | 1.43 | 0.24 |
| 新方法 c512 | 88.99% | 55.01% | 55.76 | 9.37 | 99.07% | 74.91% | 52.48 | 0.00 | 1.65 | 0.20 |
| 新方法 c576 | 90.57% | 57.27% | 75.10 | 8.41 | 99.11% | 76.87% | 49.67 | 0.00 | 1.57 | 0.21 |
| No-reverse PD c384 | 99.41% | 7.96% | 17.62 | 0.72 | 98.28% | 92.27% | 7.61 | 0.00 | 68.27 | 20.02 |
| No-reverse PD c512 | 99.20% | 7.92% | 16.84 | 0.72 | 98.59% | 92.01% | 8.63 | 0.00 | 100.05 | 19.25 |
| No-reverse PD c576 | 98.42% | 7.84% | 17.37 | 0.74 | 99.15% | 91.99% | 8.72 | 0.00 | 115.40 | 19.80 |
| 原生 Mooncake c384 | 84.22% | 8.78% | 18.37 | 1.16 | 92.23% | 92.31% | 5.16 | 0.00 | 69.86 | 20.93 |
| 原生 Mooncake c512 | 82.85% | 8.33% | 19.75 | 1.12 | 92.67% | 92.18% | 5.83 | 0.00 | 100.08 | 22.31 |
| 原生 Mooncake c576（本地NUMA） | 97.26% | 9.15% | 17.52 | 1.02 | 99.03% | 91.94% | 8.21 | 0.00 | 115.33 | 20.33 |

原生Mooncake c576正式窗口结束1192个Agent：486正常完成、706按原配置截断（422单轮长度、284预算），
无`aborted`、无记录的请求/搜索后端错误；共3834次模型调用，平均3.216轮。
396条一轮轨迹中381条为单轮长度终止、15条正常结束，不是此前搜索失败导致的全部一轮退出。
Agent/s沿用本表“无请求错误的结束Agent（含截断）/正式时间”口径，不等于正确率或完整解题率。
Mooncake正式窗口平均占用80.38%、峰值84.90%；全程累计成功驱逐46次、分配失败0。

## 新方法路径计数

统一为正式1200秒，按唯一snapshot去重；不是含预热的整轮计数，也不是Agent比例。

| 并发 | Direct完成 | Slow | 显式重算 | Direct / Slow / 重算 | D→P Host写入 / 恢复 | P→D Host写入 / 恢复 |
|---|---:|---:|---:|---|---|---|
| c384 | 5,582 | 1,503 | 136 | 77.30% / 20.81% / 1.88% | 1,503 / 1,500 | 4,349 / 4,364 |
| c512 | 5,785 | 1,126 | 244 | 80.85% / 15.74% / 3.41% | 1,126 / 1,095 | 3,090 / 3,013 |
| c576 | 5,639 | 1,070 | 306 | 80.39% / 15.25% / 4.36% | 1,070 / 1,064 | 3,845 / 3,808 |

正式窗口写入的1126个D→P snapshot中，1095个窗口内恢复、29个窗口后恢复，2个Agent因budget结束无需下一轮。
1126个均有D HBM释放记录。1次D→P Host释放、4次P→D源P释放跨窗口边界，已核对到释放。
预算结束后的Host清理及时性仍需核对，不宣称所有生命周期边界已经验收完毕。
本轮CPU回归446项通过、独立审核GO、正式请求失败0；仅一次正式结果，不代表多seed统计结论。

## 统一NUMA Host池试验（不替换当前完整方法）

2026-09-10：c576，其他计算/阈值配置不变，Host改为每NUMA288 GiB，两方向各96 GiB
保底并共享96 GiB。完成300+1200秒；存在1次终止/续轮不一致的Router超时，以下仅为诊断结果。

| 指标 | 原独立Arena c576 | 统一NUMA池 c576 |
|---|---:|---:|
| Decode token/s | 4822.5 | 4656.9 |
| P Forward/卡 | 90.57% | 91.22% |
| D Forward/卡 | 99.11% | 99.45% |
| P KV/卡 | 57.27% | 64.34% |
| D KV/卡 | 76.87% | 82.52% |
| D running/卡 | 49.67 | 51.19 |

物理池能够填充和排空，P→D确实使用了弹性容量；但吞吐下降3.44%，暂不作为性能升级。
详见[配置、Host占用、路径计数及异常记录](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-numa-pool-288g-r1/RESULTS.md)。
本轮实验服务均已停止。

## 原始结果

- 新方法 c512：[summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1/offload_analysis_summary.json)、[原始实验报告](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1/RESULT.md)。
- 新方法 c384：[summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/current-method-slow-congestion-1s-20260909-r1/offload_analysis_summary.json)。正式窗口完成2800个Agent、请求失败0；D→P和P→D写入/恢复计数受窗口边界影响，不能直接把差额解释为丢失。
- 新方法 c576：[summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-slow-congestion-1s-20260909-r1/offload_analysis_summary.json)。正式窗口完成2801个Agent、请求失败0；同样按窗口内事件独立计数，边界差额不是KV丢失量。
- Colocated c384: [summary](archive/baseline/formal-browsecomp-source-order-colocated-8gpu-c384-w300-m1200-20260816-r1/offload_analysis_summary.json)
- Colocated c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/baseline-colocated/offload_analysis_summary.json)
- Colocated c576: [summary](archive/baseline/formal-browsecomp-source-order-colocated-8gpu-c576-w300-m1200-20260817-r1/offload_analysis_summary.json)
- No-reverse PD c384: [summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/no_reverse-aligned-20260908-r1/offload_analysis_summary.json)
- No-reverse PD c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/no_reverse-aligned-20260908-r1/offload_analysis_summary.json)
- No-reverse PD c576: [summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/no_reverse-aligned-20260908-r1/offload_analysis_summary.json)
- 原生 Mooncake c384: [summary](current/qwen3-8b-tp1-browsecomp-c384-w300-m1200/native_mooncake-aligned-20260908-r1/offload_analysis_summary.json)
- 原生 Mooncake c512: [summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/native_mooncake-aligned-20260908-r1/offload_analysis_summary.json)
- 原生 Mooncake c576（2026-09-10有效重试）：[summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r5/offload_analysis_summary.json)、[数据有效性及资源](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r5/data_validity_and_resources.json)、[实验报告及复现配置](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r5/RESULTS.md)。
- 原生 Mooncake c576历史失败记录保留：第一次业务预热约124秒NIXL断连，见[故障报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260909-r1/FAILURE.md)；第二次完整运行但7038条轨迹搜索失败，见[数据正确性报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260909-r3/FAILURE.md)；本次r4在业务前因NIXL初始化超过300秒watchdog退出，见[启动故障报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/native_mooncake-aligned-20260910-r4/FAILURE.md)。均不纳入正式吞吐。

消融见 [BrowseComp Qwen3-8B消融表](BROWSECOMP_QWEN3_8B_ABLATIONS.md)。
