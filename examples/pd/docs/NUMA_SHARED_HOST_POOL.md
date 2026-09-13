# NUMA统一Host池：c576验证

修改前SGLang备份：`921fbd46ab6d3baf68be66f097661a918dec6707`，已同步远端
`pd`和`pd_node_a`。实际运行源码为`/homes/siqic/sglang-h100-integration/python`，
不是pd环境默认editable指向的`sglang-agentic`。

## 修改边界

每NUMA一份288 GiB memfd：D→P/P→D各96 GiB保底，另96 GiB弹性共享。
NUMA-local broker管理extent；原有ledger仍管理snapshot状态和TP fence。
写入源GPU本地优先，失败远端；恢复完整容量过滤、原子预留，近似负载时同NUMA优先。
注册与batch DMA沿用现有实现，启动预热全部池后再300秒业务预热+1200秒测量。
Direct1秒、tool1秒、Q32/8拥堵重算、计算调度及HBM配置不变。

成功、超时、取消、容量不足、退出均不改变原ownership决策；broker只响应已经
获准的extent分配/释放。只有完成fence并关闭snapshot后才回收。ACK丢失幂等重试。
P→D不驱逐；容量不够仍RETAIN_P。D→P保持短snapshot优先、未claim才可驱逐。
池仅在全部CUDA生产者和消费者退出后销毁。

## 八项验收门禁

| 不变量 | 本次实现与检查 |
|---|---|
| 唯一所有者 | 原ledger CAS保留；物理extent使用唯一lease ID |
| P→D Direct释放 | 未修改 |
| P→D Host释放 | 原durable fence后release，broker ACK丢失可重试 |
| D→P Host释放 | 原durable fence和源KV释放未修改 |
| 进度解耦 | 两方向/Direct/Slow队列保持；broker无CUDA，无Forward工作 |
| TP一致 | 原全shard prepare/commit/fence保留，每grant记录实际NUMA |
| 父KV正确性 | 不改传输长度/布局；显式重算与驱逐仍单独记录 |
| 审核 | audit_numa_pool独立审核GO；要求Host/TP故障测试通过后启动 |

审核修复了普通host_ready恢复入口漏过滤以及release ACK丢失重试责任。
Host/TP/router/容量/取消回归已通过：482 passed（含现有CUDA往返）。旧LazySnapshot
字段读取回归已改为读取元数据字典，避免触发尚未materialize的代理访问。
已检查并发保底、远端兜底、ABA/重试、TP部分准备回滚、关闭失败、无调用方重试、
locality/原子shadow、压力采样交接。额外小池真实CUDA往返测试1 passed。

## c576正式结果（2026-09-10）

完成300秒业务预热＋1200秒测量：Decode4656.9 token/s，旧c576为4822.5（−3.44%）；
P Forward91.22%、D Forward99.45%，D running51.19/卡、KV82.52%。
两份288 GiB物理池均有填充/回落，不是持续增长卡死；更大Host容量没有带来吞吐提升。
本轮1次terminal/续轮不一致导致Router超时，不属于Host分配路径；未越界修改终止逻辑。
因此记录为诊断试验，不替换原完整方法，不宣称零错误或TP>1性能已验收。

P→D整轮8400 queued=8400 durable，8320恢复、80项退出时未完成；
D→P1521 durable、1398恢复、123项退出时HOST_READY。完整状态与窗口口径见
[正式报告](../runs-host/current/qwen3-8b-tp1-browsecomp-c576-w300-m1200/current-method-numa-pool-288g-r1/RESULTS.md)。
实验模型与Host池已全部停止；CUDA退出清理较慢，确认模型线程结束后人工停止安全保留的broker。

运行入口：`scripts/new_method/run_browsecomp_numa_pool_c576.sh`。
物理池使用量看`logs/host-pool-{0,1}.log`，不能相加每P的逻辑capacity。
