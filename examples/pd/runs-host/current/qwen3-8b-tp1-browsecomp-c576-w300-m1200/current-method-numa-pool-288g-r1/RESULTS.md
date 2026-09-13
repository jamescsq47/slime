# c576：统一NUMA Host池正式测量

## 结论

完成约302秒业务预热＋1200秒正式测量。统一池正常运行，两个方向保持独立队列，
P→D占用能够反复回落；但Decode吞吐从旧c576的4822.5降至4656.9 token/s（−3.44%）。
不能把扩大和共享Host容量认定为吞吐收益，也不替换原完整方法结果。

本轮还有1次D判为terminal、Agent却要求下一轮的Router HTTP500，详见
[问题记录](OBSERVED_ISSUE.md)。该parent未进入Host分配流程，不能归因于Host extent
丢失；也不能因此宣称全部正确性边界已通过。按用户“只改Host”的范围未修改终止逻辑。

## 配置与代码

- Qwen3-8B，BrowseComp source-order n680循环，temperature=0，TP=1，4P:4D，c576。
- 四张P均mem_fraction_static=0.80；D为0.80/0.80/0.80/0.60，搜索服务GPU7。
- tool/Direct deadline均1秒，Q32/8全局Slow拥堵反馈重算，D目标1.0、P→D grace0.5秒不变。
- 每NUMA一个288 GiB memfd DRAM池；D→P/P→D各保底96 GiB，共享96 GiB，整机576 GiB。
- 本地NUMA优先写入，空间不足远端兜底；Host恢复全局可行容量过滤和原子预留，近似负载优先本地。
- 原registered-window/batch DMA、HBM workset、TP fence、计算调度和快慢路径判据未改。
- 实际源码：`/homes/siqic/sglang-h100-integration/python/sglang`。
- 修改前备份`921fbd46ab6d3baf68be66f097661a918dec6707`已推送并核实远端pd、pd_node_a。
  新Host改动仍在本地，未覆盖该备份。入口：`run_browsecomp_numa_pool_c576.sh`，后端为显式开关。
- 门禁：482项回归通过，额外小池真实CUDA往返1项通过，独立状态机审计GO。
  这不是TP>1正式性能验收。

## 正式窗口对照

旧组为`current-method-slow-congestion-1s-20260909-r1`；指标均对应1200秒测量窗口。
Forward为每张角色GPU平均，KV/running/queue为引擎采样均值。

| 指标 | 旧c576，独立Arena | 新c576，统一NUMA池 |
|---|---:|---:|
| Decode总吞吐 token/s | 4822.5 | 4656.9 |
| Decode单卡 token/s | 1205.6 | 1164.2 |
| Decode活跃时单卡 token/s | 1216.5 | 1170.6 |
| Prefill总吞吐 token/s | 35472.2 | 35404.4 |
| Agent/s | 2.334 | 2.322 |
| 完成Agent | 2801 | 2786 |
| P Forward/卡 | 90.57% | 91.22% |
| D Forward/卡 | 99.11% | 99.45% |
| P KV利用率/卡 | 57.27% | 64.34% |
| D KV利用率/卡 | 76.87% | 82.52% |
| P queue/卡 | 75.10 | 53.48 |
| P inflight/卡 | 8.41 | 13.48 |
| D running/卡 | 49.67 | 51.19 |
| D transfer/卡 | 0.207 | 0.343 |
| 完成轨迹平均轮次 | 3.466 | 3.651 |
| 完成轨迹累计Prompt/Agent tokens | 43511 | 47754 |
| 完成轨迹父page-aligned KV复用率 | 95.86% | 94.54% |

D并行数、KV利用率和Forward略升，但活跃时Decode速率下降，最终墙钟吞吐下降。
同一输入顺序不保证完成集合及生成轨迹相同，本轮轮次和累计Prompt也变化；单轮结果
不能把全部差额精确归因于Host实现或某一个硬件争用因素。

## 物理Host占用

来自broker每2秒日志，按正式窗口时间加权，单位GiB。两个NUMA各288 GiB；
单方向最多192 GiB以保留另一方向96 GiB。不能把每P逻辑视图相加作为物理容量。

| NUMA | D→P平均 / 峰值 / 窗口末 | P→D平均 / 峰值 / 窗口末 | 总平均 / 峰值 |
|---|---|---|---|
| 0 | 37.23 / 91.73 / 80.30 | 100.86 / 191.54 / 135.94 | 138.08 / 261.98 |
| 1 | 40.22 / 81.95 / 62.64 | 101.14 / 191.93 / 115.69 | 141.35 / 259.54 |

两池合计平均279.44/576 GiB。两个池各自峰值未必同时发生，不能相加当作全局峰值。
P→D在两池都曾降到0，随后再次接近192 GiB方向限额，并非一直单调增长无法排空；
但弹性额度耗尽时仍会反压。旧组为每P独立32 GiB的P→D空间，本轮确实用到了额外容量。

![物理Host占用](numa_host_usage.png)

## 路径与未完成项

下表是整轮（含预热和退出阶段）的唯一snapshot计数，不与旧文档的纯1200秒比例混用。

| 路径 | 计数 |
|---|---:|
| D→P Direct完成 | 6974 |
| D→P显式反馈重算 | 425 |
| D→P Host offered | 1538 |
| D→P Host D2H完成 | 1521 |
| D→P Host H2D完成 | 1398 |
| P→D Host queued | 8400 |
| P→D Host D2H完成 | 8400 |
| P→D Host H2D完成 | 8320 |

退出后保留的ledger元数据逐ID核对：D→P已写入但未恢复的123个全部为HOST_READY；
另17个未durable offer为10 offered、6 host_writing、1 host_reserved。
P→D已写入但未恢复的80个为78 HOST_READY、2 H2D_LOADING，另8320个为CONSUMED。
后续逐Agent复核更正：D→P的123个中，30个是正式结束前已budget结束的Agent，
139.46 GiB，本应删除而不应等待恢复；另93个在业务停止后服务退出阶段写入。
因此不能把123个全部称为正常未完成工作；存在终止向Host所有者传播的生命周期缺口。
详见[后续审计](DECODE_RUNNING_AND_SLOW_AUDIT.md)。
P→D queued=durable已核对；本表不以聚合计数冒充每个generation的全部源释放审计。

## 停止状态

模型退出时大容量Host注册清理超过launcher等待期限，launcher按安全规则保留broker。
继续监测，确认8个模型进程组均无活跃CUDA线程后，人工关闭两份broker。
最终GPU0–6为0 MiB，GPU7仅保留运行前已有599 MiB；本轮模型与Host池均已停止。
退出慢是仍需记录的工程成本，不能宣称完全自动快速回收。

原始数据：[分析摘要](offload_analysis_summary.json)、[Host摘要](numa_host_summary.json)、
[配置](RUN_SETTINGS.md)、`logs/host-pool-{0,1}.log`、`logs/router.log`。
