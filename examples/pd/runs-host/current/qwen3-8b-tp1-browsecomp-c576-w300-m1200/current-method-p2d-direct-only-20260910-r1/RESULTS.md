# c576：回退统一池，关闭P→D Host消融

2026-09-10。SGLang恢复到`921fbd46ab6d`；本次只关闭P→D Host staging，
保留P→D late binding、D→P Direct/Slow和全局Slow拥堵反馈重算。
未改完整方法默认开关。运行入口：`scripts/new_method/run_browsecomp_c576_p2d_direct_only.sh`。

## 结论

完整300+1200秒完成：**Decode 4965.7 token/s**，相比旧独立Host完整方法
4822.5提升**2.97%**，相比已撤回统一池4656.9提升**6.63%**。
这是单轮结果，不代表统计显著或所有负载下都应关闭Host。

关闭P→D Host后，平均D running和KV反而更低，但Decode活跃吞吐略高；
这再次说明平均running并不单独决定吞吐，轨迹、上下文和阶段分布也会变化。
P仍出现过HBM接近满和Forward回落，因此不能说容量反压消失。

## 对齐设置

- Qwen3-8B，8卡，TP=1，4P:4D，c576闭环补充；BrowseComp source-order n680循环。
- temperature=0；P=0/2/4/6、D=1/3/5/7，搜索在GPU7。
- P显存比例全部0.80；D=0.80/0.80/0.80/0.60。
- 工具阈值1秒，Direct建立deadline1秒，拥堵Q高/低水位32/8。
- D→P Host仍128 GiB/P；原生HiCache/Mooncake关闭。
- 8/8 Host预注册完成后业务预热301.87秒，再测量1200.001秒。
- 本轮注册cache上限640 GiB，旧独立Host组1280 GiB；均足够覆盖所有映射且预注册成功，不发生因该上限导致的窗口换入换出。

## 整体指标

吞吐来自正式窗口GPU token counter增量/墙钟时间。Forward按每卡平均。
KV/running/queue为同一窗口的时间加权平均，故与早先直接样本平均可能有小数差异。

| 指标 | 旧独立Host完整方法 | 已撤回统一NUMA池 | 本轮关闭P→D Host |
|---|---:|---:|---:|
| Decode总吞吐 token/s | 4822.5 | 4656.9 | 4965.7 |
| Decode单卡墙钟 token/s | 1205.6 | 1164.2 | 1241.4 |
| Decode单卡活跃 token/s | 1216.5 | 1170.6 | 1248.5 |
| D Forward/卡 | 99.11% | 99.45% | 99.43% |
| P Forward/卡 | 90.57% | 91.22% | 93.87% |
| Prefill实际计算 token/s | 35472.2 | 35404.4 | 36827.9 |
| D running/卡 | 49.61 | 51.21 | 47.54 |
| D KV/卡 | 76.8% | 82.5% | 69.0% |
| D transfer/卡 | 0.204 | 0.343 | 0.144 |
| D prealloc/卡 | 1.568 | 1.851 | 1.158 |
| P KV/卡 | 57.2% | 64.3% | 54.6% |
| P queue/卡 | 75.23 | 53.48 | 84.05 |
| P inflight/卡 | 8.39 | 13.49 | 8.57 |
| Agent/s | 2.334 | 2.322 | 2.395 |
| 完成Agent数 | 2801 | 2786 | 2874 |

统一池组包含一次Router超时，仅作诊断对照。本次窗口前后边界的failure均为0，
模型/Router日志未发现测量期间的500、OOM或断言错误。
running来自实际Decode batch，不包含上述单独列出的transfer/prealloc。

## 本轮300秒分段

| 正式时间 | Decode总token/s | D running/卡 | D KV/卡 | D Forward/卡 | P KV/卡 | P Forward/卡 | P inflight/卡 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–300秒 | 5743.3 | 56.64 | 71.3% | 99.67% | 45.1% | 99.03% | 6.04 |
| 300–600秒 | 4327.5 | 40.92 | 67.2% | 99.19% | 61.7% | 89.20% | 10.34 |
| 600–900秒 | 5330.2 | 55.94 | 82.8% | 99.28% | 74.3% | 88.20% | 14.91 |
| 900–1200秒 | 4455.6 | 36.49 | 54.5% | 99.60% | 37.0% | 99.11% | 2.88 |

各分段counter采用段内首末采样，边界约2秒误差；不能用四个舍入后的分段均值精确复算总counter。

## D→P路径与Host恢复

正式窗口按唯一snapshot首次出口事件计数；分母7467，非完成Agent比例。
秒级日志边界会与按末次记录/其他事件统计略有不同。

| 路径 | 次数 | 比例 |
|---|---:|---:|
| Direct发送完成 | 6006 | 80.43% |
| Slow fallback | 1109 | 14.85% |
| 显式拥堵反馈重算 | 352 | 4.71% |

- P→D Host D2H启动：**0次**，运行进程开关确认为false，late binding保持开启。
- D→P Host D2H启动1109次；窗口内完成1108次、写入1916.98 GiB。
- 该完成集合1105个在窗口内Host→P恢复，1903.46 GiB；余下3个为预算结束的最终snapshot。
- 以D2H完成到H2D完成的日志区间重建，D→P durable Host平均约13.53 GiB，峰值72.80 GiB，结束13.52 GiB；总容量512 GiB。此为逻辑durable字节估算，不含写入中extent、碎片和释放ACK短延迟。
- D→Host：总bytes/累计GPU DMA秒=10.40 GiB/s；平均GPU 0.166秒、完整工作段0.212秒。
- Host→P：总bytes/累计GPU DMA秒=11.93 GiB/s；平均GPU 0.144秒、完整工作段0.214秒。
  这些是活跃传输速率，不是链路上限，也不是除以1200秒的业务平均带宽。

## KV正确性及已知终止清理问题

完成轨迹page-aligned父KV token总复用率96.30%，该分母包含显式选择重算的请求。
7406个后续调用中，6993个完整命中；413个不足完整父前缀，逐个关联后：

- 412个对应显式Direct失败重算，额外3304384 tokens；
- 1个对应既定终止格式纠错，额外6912 tokens。

其余没有发现无法解释的父KV缺失。412是完成Agent整条轨迹的集合，含预热中父generation；
352是正式窗口发生的出口事件数，二者不是同一统计集合。

3个已预算结束Agent的最终snapshot仍留在Host，合计13.52 GiB，属于上一版本已有的终止清理缺口：

| snapshot | Host GiB |
|---|---:|
| d92c734092124133a33421db9227a794:12 | 4.544 |
| 20e8b2c65b13496e814dfc86a3e88a30:5 | 3.999 |
| 726078f591eb40cda82fa538f4491fdd:3 | 4.975 |

它们不是应继续Prefill而恢复失败的请求，而是应清理的终止副本。
停机阶段另有101个snapshot写Host、212.09 GiB，创建时间在测量结束后，不能计作稳态末尾积压。
现场ledger事件归档在`end-host-events/`。本次按单变量实验要求没有顺带修复这个旧缺口。

## 数据特性

| 完成Agent均值 | 旧独立Host | 本轮 |
|---|---:|---:|
| 模型调用轮次 | 3.466 | 3.577 |
| 累计Prompt tokens | 43511 | 44974 |
| 累计模型Decode tokens | 2020 | 1991 |
| harness总response tokens（含工具等，不等于Decode） | 16164 | 16468 |
| 累计tool时间/Agent | 0.264秒 | 0.264秒 |

源数据顺序和生成参数对齐，但完成集合与实际生成轨迹并非逐条完全一致，单轮约3%的差异需要谨慎解读。

## 验证与清理

- 回滚后547项回归通过；独立审核GO后启动。
- 实验全程未再次修改生产代码；SGLang工作区保持HEAD干净。
- 业务停止后等待CUDA Host注销完成，所有本轮P/D/搜索进程退出。
- 最终GPU0–6显存0 MiB，GPU7恢复到启动前既有599 MiB；没有启动下一组实验。
- 结果：`offload_analysis_summary.json`、`closed_loop_boundaries.json`、`engine_metrics.jsonl`、`requests.jsonl`、`pd_throughput.png`、`steady_offload_analysis.png`。
