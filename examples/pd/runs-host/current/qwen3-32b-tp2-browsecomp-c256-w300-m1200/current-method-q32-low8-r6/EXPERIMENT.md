# TP组回滚释放修复后：Q32/8正式复测

状态：已完成300.996秒业务预热+1200.001秒正式测量；671条Agent完成，0失败。

## 正式结果

| 指标 | Q8/2 r4 | Q32/8 r6 |
|---|---:|---:|
| 总Decode token/s | 1361.86 | 1405.70 |
| 总Prefill compute token/s | 5246.57 | 5208.40 |
| P Forward/物理卡 | 98.96% | 98.27% |
| D Forward/物理卡 | 99.89% | 99.88% |
| P KV利用率 | 36.88% | 39.25% |
| D KV利用率/逻辑组平均 | 60.41% | 60.32% |
| D running/逻辑组 | 15.76 | 16.47 |
| Direct组级完成 | 507 | 552 |
| Slow写入完成 | 269 | 323 |
| Slow恢复完成 | 262 | 331 |
| 拥堵重算 | 81 | 11 |

吞吐提高3.22%，仍比历史1446.95低2.85%，不声称最优。
正式窗口Direct/Slow/recompute按唯一snapshot最终路径事件统计：552/323/11，
即62.30%/36.46%/1.24%；不同阶段跨窗口，写入与恢复数不必相等。
Slow D2H complete与D source release均323；TP后台Direct回滚完成102次。
未发现运行期Scheduler异常、RuntimeError或500；此前release_pending计划移除故障未复现。
完成轨迹父page-aligned KV复用96.944%；未复用189376 tokens全部匹配明确重算日志，
无未归因父KV缺失。每完成Agent平均模型调用2.408，Decode2588.85 tokens，
实际Prefill7956.20 tokens（完成轨迹口径，不是窗口tokens/完成数）。
测量结束后进程已退出，未启动下一轮。

复用r5全部设置：Qwen3-32B、BrowseComp source-order n680、temperature=0、
TP2、2P:6D、c256，工具/Direct均1s，Q high=32 low=8，
P=[0,4] mem=.80；D=[1,5]/[2,6]/[3,7] mem=.80/.80/.60；搜索GPU7。
Host注册全部完成后300秒业务预热+1200秒测量；原生HiCache/Mooncake关闭。

唯一代码变化：TP plan中retire_ready lease的重复request_release仍进入
既有组级退休协议，不再直接变releasing；等待全rank物理fence完成后统一commit。
TP1无plan路径不变，不修改传输队列、计算或路由策略。

新增TP2/4交错fence与重复release回归：修复前失败，修复后通过；
相关安全/TP/生命周期/Router共492项通过，diff check通过。
八项不变量：所有权/fence守恒；P→D快慢及D→P Host释放语义不变；
I/O与Forward解耦不变；TP释放严格组级；显式重算单列；独立GO后运行。
