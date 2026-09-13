# BrowseComp + Qwen3-8B：完整方案与消融

2026-09-09更新。第一行固定为 **全局Slow恢复拥堵反馈重算** 完整方案（5148组）。
此前混合失败出口的旧结果已从此表移除；“Direct失败一律重算”使用新的4668组。
其余保留结果作为消融或基线参考，不把旧数据重新标成同一代码版本测试。

## 对齐配置与方法差异

共同配置：Qwen3-8B，BrowseComp固定source-order n680循环，TP=1、4P:4D、c512，
temperature=0/top_p=1/top_k=-1，300秒业务预热＋1200秒正式测量。
P四卡0.80，D四卡0.80/0.80/0.80/0.60，搜索服务在GPU7。
自定义方法原生HiCache/Mooncake关闭；启用Host的方向沿用D→P 128 GiB/P、P→D 32 GiB/P，
D接收目标1.0，P→D grace=0.5秒，预注册完成后才开始业务预热。

| 方案 | 具体语义 |
|---|---|
| 完整方案 | 工具>1秒走Slow；快工具Direct失败后，恢复队列拥堵则重算，否则Slow |
| 快慢路径 | 去掉拥堵反馈，快工具Direct失败也总是Slow；保留容量保护/显式驱逐等原有正确性出口 |
| 仅慢路径 | 不尝试D→P Direct，所有可复用parent均走Host |
| Direct失败一律重算 | 工具>1秒仍Slow；快工具Direct失败，无论是否曾claim，安全取消后均重算，不看拥堵 |
| 仅快路径＋失败重算 | 工具不设实际可触发的快慢阈值；返回后尝试Direct，建链超过1秒重算，关闭D→P Slow |
| 纯重算（当前方法单变量消融） | 保留当前方法的P→D late binding、D Router和P→D Host；仅关闭D→P Direct/Slow，所有后续轮完整Prefill |
| P→D预绑定＋仅Direct（失败） | 关闭P→D late binding和P→D Host；预热阶段发生D transfer与P workset环形等待，不计吞吐 |
| 关闭P→D Host | 保留P→D late binding和D→P全部路径，只关闭Prefill完成KV的Shared Host staging |
| 原生No-reverse PD参考 | 同样不回传D→P KV，但使用原生PD预留/transfer控制路径，不是当前方法的单变量消融 |

完整方案Q：工具已返回、Host durable、尚未被H2D worker接手的唯一parent generation。
跨P不重复、等待工具不计入；每秒采样，连续两次Q≥32启用重算，Q≤8退出。
Q不包括worker接手后的CPU准备/DMA；信号缺失或超过3秒陈旧则保守Slow。
已claim的Direct失败也必须遵循DMA fence，不能因为拥堵提前释放源KV。

**可比性限制：**纯重算已使用当前自定义引擎单开关正式重跑；原生No-reverse PD只作为控制路径参考，
不能用来单独归因反向KV收益。其余结果是相同工作负载/显存设置下的历史正式运行，
并非全部同一commit同步重跑。表中性能差异不能全部机械归因于单一开关。

## 正式吞吐

Decode为四张D的总墙钟吞吐，单张D为总量除以4。

| 方案 | Decode token/s | 单张D token/s | Agent/s | 相对完整方案 |
|---|---:|---:|---:|---:|
| 完整方案：快慢路径＋拥堵反馈重算 | 5,148.4 | 1,287.1 | 2.387 | 基准 |
| 消融：快慢路径 | 4,550.7 | 1,137.7 | 2.247 | -11.61% |
| 消融：仅慢路径 | 4,398.1 | 1,099.5 | 2.161 | -14.57% |
| 消融：Direct失败一律重算 | 4,668.1 | 1,167.0 | 2.293 | -9.33% |
| 消融：仅快路径＋失败重算 | 4,584.4 | 1,146.1 | 2.217 | -10.95% |
| 消融：纯重算（仅关闭D→P） | 1,938.8 | 484.7 | 0.957 | -62.34% |
| 消融：P→D预绑定＋仅Direct | 失败（预热停滞） | — | — | 不计 |
| 消融：关闭P→D Host | 4,867.7 | 1,216.9 | 2.388 | -5.45% |
| 控制路径参考：原生No-reverse PD | 1,984.4 | 496.1 | 0.966 | -61.46% |

## P→D Host消融（单独并发表）

细粒度复核：[c512/c576/统一Host池的30秒分析](analysis/p2d-host-c512-c576/ANALYSIS.md)，
[时间序列图](analysis/p2d-host-c512-c576/time_slices_30s.png)。

以下按c512、c576分别比较。各组均为4P:4D、source-order n680、temperature=0、
P显存比例均0.80、D为0.80/0.80/0.80/0.60、300秒业务预热＋1200秒正式测量；消融只关闭
P→D Shared Host Arena，保留late binding以及D→P Direct/Slow/拥堵反馈重算。

### c512

| c512方案 | Decode token/s | 单张D token/s | Agent/s | P Forward | D Forward | P KV | D KV | D running |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整方案 | **5,148.4** | **1,287.1** | 2.387 | 88.99% | 99.07% | 55.01% | 74.91% | 52.48 |
| 消融：关闭P→D Host | 4,867.7 | 1,216.9 | 2.388 | 93.08% | 99.29% | 59.9% | 68.8% | 46.78 |

关闭P→D Host后，本轮Decode吞吐下降5.45%，Agent/s基本持平；窗口Decode/完成Agent
由2153降至2034 tokens，完成集合差异需一并考虑。路径、分段和Host统计见
[c512完整报告](current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-host-disabled-20260910-r1/RESULTS.md)。

### c576

| c576方案 | Decode token/s | 单张D token/s | Agent/s | P Forward | D Forward | P KV | D KV | D running |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整方案（旧独立Host） | 4,822.5 | 1,205.6 | 2.334 | 90.57% | 99.11% | 57.2% | 76.8% | 49.61 |
| 消融：关闭P→D Host | **4,965.7** | **1,241.4** | **2.395** | 93.87% | 99.43% | 54.6% | 69.0% | 47.54 |

本次单轮提高2.97%，但完成集合及轨迹存在自然差异，不能据此直接认定所有负载下
关闭P→D Host都更优。路径、分段和生命周期核对见
[c576完整报告](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-p2d-direct-only-20260910-r1/RESULTS.md)。

## 数据与计算量

| 方案 | Prefill token/s | 实际Prefill/完成Agent | Decode/完成Agent | 父KV token复用率 |
|---|---:|---:|---:|---:|
| 完整方案：快慢路径＋拥堵反馈重算 | 35,195 | 14,719 | 2,153 | 96.30% |
| 消融：快慢路径 | 31,537 | 14,000 | 2,020 | 99.30% |
| 消融：仅慢路径 | 31,708 | 14,637 | 2,030 | 97.55% |
| 消融：Direct失败一律重算 | 38,707 | 16,845 | 2,032 | 91.12% |
| 消融：仅快路径＋失败重算 | 38,429 | 17,299 | 2,064 | 91.17% |
| 消融：纯重算（仅关闭D→P） | 43,009 | 44,890 | 2,024 | 0.00%（D→P） |
| 消融：关闭P→D Host | 36,332 | 15,185 | 2,034 | 96.44% |
| 控制路径参考：原生No-reverse PD | 42,803 | 44,251 | 2,051 | 0.00% |

Prefill/完成Agent、Decode/完成Agent为窗口引擎计算量除以完成数；
父KV复用为完成轨迹的page-aligned parent-prefix token比例，显式重算也计入未复用量，不等于KV无故丢失。

## KV、队列与Forward

正式窗口每物理卡/每引擎平均值。

| 方案 | P Forward | P KV | P queue | P inflight | D Forward | D KV | D running | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整方案：快慢路径＋拥堵反馈重算 | 88.99% | 55.01% | 55.76 | 9.37 | 99.07% | 74.91% | 52.48 | 1.65 | 0.20 |
| 消融：快慢路径 | 83.08% | 96.78% | 13.22 | 13.03 | 96.91% | 86.41% | 53.77 | 1.77 | 0.30 |
| 消融：仅慢路径 | 84.42% | 83.40% | 3.04 | 11.30 | 94.85% | 85.80% | 58.44 | 1.85 | 0.32 |
| 消融：Direct失败一律重算 | 98.10% | 45.68% | 72.41 | 5.90 | 99.64% | 65.02% | 43.14 | 1.03 | 0.18 |
| 消融：仅快路径＋失败重算 | 98.10% | 50.12% | 69.42 | 6.23 | 99.72% | 70.18% | 44.79 | 1.12 | 0.20 |
| 消融：纯重算（仅关闭D→P） | 99.99% | 8.62% | 118.83 | 1.00 | 98.94% | 10.73% | 6.96 | 0.03 | 0.03 |
| 消融：关闭P→D Host | 93.08% | 59.9% | 67.35 | 9.44 | 99.29% | 68.8% | 46.78 | 1.28 | 0.13 |
| 控制路径参考：原生No-reverse PD | 99.20% | 7.92% | 16.84 | 0.72 | 98.59% | 92.01% | 8.63 | 100.05 | 19.25 |

## 正式窗口路径计数

下表统一为1200秒，不再混用含预热的整轮计数。路径按snapshot去重；
重复fallback尝试不重复计数，Slow选择与Host完成事件的少量差额按原始边界分别保留。
比例分母仅为三类已记录路径结果，不是Agent比例。
纯重算不创建D→P snapshot事件；“4090次终态释放”是正式窗口内D完成generation的计数，
不能把0个Direct/Slow事件写成0%重算。P→D仍使用当前方法，本轮D始终可接收，未触发P→D Host。

| 方案 | Direct完成 | Slow | 显式重算 | Direct / Slow / 重算 | D→P Host写入 / 恢复 | P→D Host写入 / 恢复 |
|---|---:|---:|---:|---|---|---|
| 完整方案：快慢路径＋拥堵反馈重算 | 5,785 | 1,126 | 244 | 80.85% / 15.74% / 3.41% | 1,126 / 1,095 | 3,090 / 3,013 |
| 消融：快慢路径 | 2,839 | 4,068 | 0 | 41.10% / 58.90% / 0.00% | 4,068 / 4,020 | 5,901 / 5,895 |
| 消融：仅慢路径 | 0 | 7,321 | 0 | 0.00% / 100.00% / 0.00% | 7,318 / 6,463 | 5,647 / 5,656 |
| 消融：Direct失败一律重算 | 6,413 | 80 | 518 | 91.47% / 1.14% / 7.39% | 80 / 66 | 2,375 / 2,375 |
| 消融：仅快路径＋失败重算 | 6,417 | 0 | 499 | 92.78% / 0.00% / 7.22% | 0 / 0 | 3,482 / 3,482 |
| 消融：纯重算（仅关闭D→P） | 0 | 0 | 全部后续轮完整Prefill | 不适用 | 0 / 0 | 0 / 0 |
| 消融：关闭P→D Host | 5,815 | 1,450 | 324 | 76.62% / 19.11% / 4.27% | 1,450 / 1,447 | 0 / 0 |
| 控制路径参考：原生No-reverse PD | 不适用 | 不适用 | 全部后续轮不回传重算 | 不适用 | 不适用 | 不适用 |

## 完整方案Slow详细核对

- 正式窗口D→Host写入1126个、1991.7 GiB，1126个均有D HBM释放记录。
- 同批snapshot中1095个窗口内恢复到P；29个在边界后恢复；2个Agent因budget结束不再需要下一轮。
- H2D窗口累计1947.2 GiB。累计GiB不是同时Host占用量。
- Host durable到H2D开始平均约4.76秒，P90约10.45秒；该值为已恢复集合按秒级日志匹配，
  不含尚未恢复项，不能当作D→Host排队耗时。
- D2H任务墙钟平均288毫秒、H2D任务墙钟平均251毫秒；前者不包含启动D2H前全部排队。
- P→D Host写入3090、恢复3013；4次P释放及1次D→P Host释放跨窗口边界，已找到释放记录。
- Q样本平均4.8、峰值42，拥堵模式占采样4.4%；不是精确的时间加权占比。
- 正式请求失败0，CPU回归446项通过、独立审核GO。预算终止后Host清理及时性仍需核对，
  不把“无恢复需求”自动解释为Host已经释放。

完整方案在当前保留结果中吞吐最高，同时相较固定失败重算提高了父KV复用。
这是一轮配置匹配比较，尚不是多seed显著性结论；c384/c576需按完整方案重测。

## 原始结果

- 完整方案：快慢路径＋拥堵反馈重算：[summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1/offload_analysis_summary.json)，目录：`current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1`。
- 消融：快慢路径：[summary](current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260907-r3/full-1s/offload_analysis_summary.json)，目录：`current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260907-r3/full-1s`。
- 消融：仅慢路径：[summary](current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260907-r3/d2p-slow-only-1s/offload_analysis_summary.json)，目录：`current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260907-r3/d2p-slow-only-1s`。
- 消融：Direct失败一律重算：[summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-all-direct-fail-recompute-1s-20260909-r2/offload_analysis_summary.json)，目录：`current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-all-direct-fail-recompute-1s-20260909-r2`。
- 消融：仅快路径＋失败重算：[summary](current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260908-r9/direct-only-recompute-1s/offload_analysis_summary.json)，目录：`current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260908-r9/direct-only-recompute-1s`。
- 消融：纯重算（仅关闭D→P）：[summary](current/ablations/browsecomp-qwen3-8b-4p4d-c512/pure-recompute-20260909-r1/offload_analysis_summary.json)，目录：`current/ablations/browsecomp-qwen3-8b-4p4d-c512/pure-recompute-20260909-r1`。
- 消融：P→D预绑定＋仅Direct：[失败记录](current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-prebind-direct-only-20260909-r1/FAILURE.md)，目录：`current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-prebind-direct-only-20260909-r1`。
- 消融：关闭P→D Host：[完整报告](current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-host-disabled-20260910-r1/RESULTS.md)，[summary](current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-host-disabled-20260910-r1/offload_analysis_summary.json)，目录：`current/ablations/browsecomp-qwen3-8b-4p4d-c512/p2d-host-disabled-20260910-r1`。
- 控制路径参考：原生No-reverse PD：[summary](current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/no_reverse-aligned-20260908-r1/offload_analysis_summary.json)，目录：`current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/no_reverse-aligned-20260908-r1`。
- c576关闭P→D Host：[summary](current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-p2d-direct-only-20260910-r1/offload_analysis_summary.json)，目录：`current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-p2d-direct-only-20260910-r1`。

主矩阵：[BrowseComp + Qwen3-8B](BROWSECOMP_QWEN3_8B.md)。
