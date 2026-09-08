# BrowseComp + Qwen3-8B：D→P 路径消融实验

## 对齐设置

四组实验采用完全相同的服务负载，仅改变表中注明的 D→P 路径策略。

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-8B |
| 数据 | BrowseComp，固定 source-order `n680`，完成后按同一顺序循环 |
| PD 配置 | 4P:4D，TP=1，全局 Host 恢复 |
| 并发 | 512 个 closed-loop Agent |
| 采样 | `temperature=0`、`top_p=1`、`top_k=-1` |
| 时间 | 300 秒预热 + 1,200 秒正式测量 |
| 原生 HiCache/Mooncake | 关闭 |
| P 显存比例 | 四张 P 均为 `mem_fraction_static=0.80` |
| D 显存比例 | `0.80/0.80/0.80/0.60`；GPU 7 同时运行搜索服务 |
| D 接收目标 | `D_TARGET_KV_FRACTION=1.0` |
| P→D Host 判定等待 | 0.5 秒 |
| P→D Host 慢路径 | 四组均开启 |

## 四种消融语义

1. **快慢路径**：工具在 1 秒内返回则尝试 Direct；Direct 在 1 秒内未建立，
   或工具超过 1 秒才返回，均进入 Slow。这是旧版完整方法。
2. **慢路径**：D→P Direct 完全关闭，所有可复用 parent snapshot 均进入
   Shared Host Arena。
3. **快慢路径 + 重算（当前新方法）**：工具超过 1 秒才返回时进入 Slow；工具
   在 1 秒内返回则尝试 Direct，Direct 在 1 秒内未建立时显式完整重算，不再
   转入 Slow。
4. **快路径 + 重算**：不设置实际可触发的工具快慢阈值；工具返回后均尝试
   Direct，Direct 在 1 秒内未建立时完整重算，D→P Slow 关闭。

## 正式结果

| 方案 | Decode token/s | 单张 D | Agent/s | 相对“快慢路径” | 状态 |
|---|---:|---:|---:|---:|---|
| 快慢路径 | 4,550.7 | 1,137.7 | 2.247 | 基准 | 完成 |
| 慢路径 | 4,398.1 | 1,099.5 | 2.161 | -3.35% | 完成 |
| **快慢路径 + 重算（当前新方法）** | **4,888.2** | **1,222.1** | **2.275** | **+7.42%** | **完成** |
| 快路径 + 重算 | 4,584.4 | 1,146.1 | 2.217 | +0.74% | 完成 |

## 资源与数据特征

所有利用率和队列指标均为 1,200 秒正式窗口的平均值。Forward 按每张物理 GPU
统计；KV、queue、running、prealloc 和 transfer 也换算为每张卡平均值。

| 方案 | Prefill token/s | 实际 Prefill/Agent | Decode/Agent | 父KV复用率 | P Forward | D Forward | P KV | P queue | P inflight | D KV | D running | D prealloc | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 快慢路径 | 31,537 | 14,000 | 2,020 | **99.30%** | 83.08% | 96.91% | 96.78% | 13.22 | 13.03 | 86.41% | 53.77 | 1.77 | 0.30 |
| 慢路径 | 31,708 | 14,637 | 2,030 | 97.55% | 84.42% | 94.85% | 83.40% | 3.04 | 11.30 | 85.80% | 58.44 | 1.85 | 0.32 |
| **快慢路径 + 重算（当前新方法）** | **38,232** | 16,790 | 2,147 | 93.44% | 97.04% | 99.62% | 50.60% | 65.37 | 6.38 | 70.49% | 48.44 | 1.30 | 0.20 |
| 快路径 + 重算 | 38,429 | 17,299 | 2,064 | 91.17% | 98.10% | 99.72% | 50.12% | 69.42 | 6.23 | 70.18% | 44.79 | 1.12 | 0.20 |

“父KV复用率”是 page-aligned parent-prefix token 的复用率。重算方案用更多
P 计算换取更少的反向 Host 传输，因此该指标下降是显式策略结果，不代表 KV
无故丢失。

## 路径统计

下表计数覆盖整轮运行，包括 300 秒预热及正式窗口边界附近的尾部事件，不应直接
除以 1,200 秒吞吐计数。比例仅以已经得到明确路径结果的 snapshot 为分母。

| 方案 | D→P Direct完成 | D→P Slow/fallback | Direct失败重算 | D→P路径比例 | P→D Host D2H/H2D |
|---|---:|---:|---:|---|---:|
| 快慢路径 | 3,829 | 5,145 | 0 | 42.7% Direct / 57.3% Slow | 7,253 / 7,225 |
| 慢路径 | 0 | 9,435 | 0 | 100% Slow | 6,991 / 6,950 |
| **快慢路径 + 重算（当前新方法）** | **8,207** | 427 | 637 | 88.5% Direct / 4.6% Slow / 6.9%重算 | 4,690 / 4,650 |
| 快路径 + 重算 | 7,998 | 0 | 740 | 91.5% Direct / 8.5%重算 | 4,601 / 4,488 |

## 结论

- “慢路径”比“快慢路径”低 3.35%。所有 parent KV 都支付 D2H、Host
  ownership/recovery 和 H2D 开销后，D Forward 也是四组最低。
- “快慢路径 + 重算”吞吐最高。这个 workload 下 P 有计算余量；将少量失败
  Direct 改为重算，显著减少 D→P Host 压力，使 D Forward 达到 99.62%。
- 该收益存在明确代价：实际 Prefill 从 14.0k 增至 16.8k tokens/Agent，父KV
  复用率从 99.30% 降至 93.44%。因此必须同时报告吞吐、重算次数和额外Prefill。
- “快路径 + 重算”取消了慢工具的 Host 保护，重算更多，但吞吐只比“快慢路径”
  高 0.74%。Slow 仍应保留给慢工具，不能被完全删除。
- 根据这组对齐消融，后续文档中的“当前新方法”专指第3种“快慢路径 + 重算”。
  旧“Direct失败→Slow”的实验均属于历史方案，需要按新语义重跑。

## 结果目录

| 方案 | 目录 |
|---|---|
| 快慢路径 | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260907-r3/full-1s` |
| 慢路径 | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260907-r3/d2p-slow-only-1s` |
| 快慢路径 + 重算 | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260908-r4/fast-direct-fail-recompute-1s` |
| 快路径 + 重算 | `current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260908-r9/direct-only-recompute-1s` |

## 正确性说明

“快路径 + 重算”初次测试暴露了一个生命周期边界：P 可能在 NIXL 尚未提交前
退回 Direct claim，使 D 停留在 `DIRECT_READY` 并等待已经被消费的 marker。
修复后，该状态会在 D 释放 KV 前原子、持久地转为显式重算。最终正式窗口完成
2,660 个 Agent，请求失败为 0，没有 Router 500 或 600 秒 P-ready 超时；
相关 SGLang Agentic-PD 专项测试为 `326 passed`。
