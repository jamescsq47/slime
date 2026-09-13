# c512：关闭P→D Host消融结果

本轮只关闭P→D Shared Host staging；P→D late binding、D→P Direct/Slow、全局Slow恢复拥堵反馈重算均保留。
运行源码为独立干净worktree中的SGLang `921fbd46ab6d3baf68be66f097661a918dec6707`。

## 正式窗口

| 指标 | 结果 |
|---|---:|
| 业务预热 / 正式测量 | 300.47 / 1200.00秒 |
| 完成 / 失败 | 2866 / 0 |
| Agent吞吐 | 2.388 Agent/s |
| Decode总吞吐 | 4867.7 token/s |
| Decode单卡吞吐 | 1216.9 token/s |
| Decode活跃时单卡吞吐 | 1225.6 token/s |
| Prefill吞吐 | 36332 token/s |
| 实际Prefill/完成Agent | 15184.7 tokens |
| Decode/完成Agent | 2034.4 tokens |
| 父KV token复用率 | 96.44% |

## 资源状态（正式窗口时间加权）

| 指标 | 每卡平均 |
|---|---:|
| P Forward | 93.08% |
| P KV | 59.9% |
| P queue / inflight | 67.35 / 9.44 |
| D Forward | 99.29% |
| D KV | 68.8% |
| D running | 46.78 |
| D prealloc / transfer | 1.28 / 0.13 |

## 路径和Host

按正式窗口首次终态事件对request-generation去重：

| 路径 | 次数 | 比例 |
|---|---:|---:|
| D→P Direct完成 | 5815 | 76.62% |
| D→P Slow | 1450 | 19.11% |
| 显式重算 | 324 | 4.27% |

- P→D Host D2H启动为0，确认该方向的Host staging确实关闭。
- D→P Host写入1450个，正式窗口内完成H2D恢复1447个；这1447个已启动恢复的snapshot全部完成。
- D→P写入2629.8 GiB，活跃GPU D2H带宽10.74 GiB/s；恢复2615.2 GiB，活跃GPU H2D带宽12.61 GiB/s。
- 由Host写完到P恢复完成估算的持久占用平均15.9 GiB、峰值83.7 GiB、窗口结束时14.6 GiB；总容量为512 GiB。
- 窗口结束后出现的98个、154.5 GiB Host snapshot属于停止请求和清理阶段，不计作稳态积压。

## 分段

| 正式窗口 | P Forward | P KV | P queue | D token/s | D Forward | D running | D KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–300秒 | 92.19% | 55.5% | 53.15 | 6006.2 | 99.57% | 61.49 | 77.2% |
| 300–600秒 | 93.49% | 60.0% | 74.37 | 4438.1 | 99.23% | 39.75 | 60.9% |
| 600–900秒 | 89.55% | 65.7% | 60.05 | 5226.8 | 99.41% | 51.12 | 72.4% |
| 900–1200秒 | 97.29% | 57.5% | 82.35 | 3790.8 | 98.98% | 34.50 | 64.1% |

## 与c512完整方案的单轮比较

| 指标 | 完整方案 | 关闭P→D Host | 变化 |
|---|---:|---:|---:|
| Decode token/s | 5148.4 | 4867.7 | -5.45% |
| Agent/s | 2.387 | 2.388 | +0.06% |
| P Forward | 88.99% | 93.08% | +4.09个百分点 |
| D running | 52.48 | 46.78 | -5.70 |
| D KV | 74.91% | 68.8% | -6.1个百分点 |
| Decode/完成Agent | 2153 | 2034 | -5.5% |

关闭P→D Host后，本轮P计算更忙，但D平均batch和KV占用下降，Decode token吞吐降低。
Agent/s几乎相同，但窗口Decode/完成数不能当作完成轨迹的真实平均输出长度：
该值含未完成Agent在窗口内贡献的tokens。实际完成轨迹的平均模型输出为1992 tokens，
完整方案为2005 tokens，仅降低约0.65%；不能据此直接解释5.45%的吞吐下降。
更细的batch、周期、搬运和轨迹分析见[30秒分析](../../../../analysis/p2d-host-c512-c576/ANALYSIS.md)。
这是单seed历史对照，并非同一时刻同步A/B，结论应以Decode吞吐和资源状态为主。
