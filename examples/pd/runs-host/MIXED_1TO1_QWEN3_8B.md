# Retool + BrowseComp 1:1 + Qwen3-8B

<!-- mixed-current-start -->
## 当前对齐参数重跑（自动更新）

temperature=0；固定n8192顺序；300+1200秒；普通卡0.80、搜索GPU7为0.60。
按表格顺序串行；失败立即暂停。新方法使用当前完整方案，工具/Direct均1s，
Q采用默认公式（2P×2 H2D lanes：high=16、low=4），不沿用TP2专用32/8调参。

| 方法 | 配置 | 状态 | Decode token/s |
|---|---|---|---:|
| 原生 Mooncake | 2P:6D c384 | 待运行 | — |
| 原生 Mooncake | 2P:6D c512 | 完成 | 3,007.8 |
| 原生 Mooncake | 2P:6D c640 | 待运行 | — |
| 原生 Mooncake | 4P:4D c512 | 完成 | 4,934.0 |
| No-reverse PD | 2P:6D c384 | 待运行 | — |
| No-reverse PD | 2P:6D c512 | 完成 | 3,251.8 |
| No-reverse PD | 2P:6D c640 | 待运行 | — |
| No-reverse PD | 4P:4D c512 | 完成 | 5,219.2 |
| 当前新方法 | 2P:6D c384 | 待运行 | — |
| 当前新方法 | 2P:6D c512 | 完成 | 9,501.8 |
| 当前新方法 | 2P:6D c640 | 完成 | 9,069.3 |

| 方法/配置 | Agent/s | P token/s | P Forward | P KV | D Forward | D KV | D running | D transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [原生 Mooncake 2P c512](current/mixed-aligned-default-t0-20260910/native_mooncake-2p6d-c512/result.json) | 0.696 | 16,401.2 | 99.0% | 9.0% | 98.4% | 90.9% | 9.9 | 33.38 |
| [原生 Mooncake 4P c512](current/mixed-aligned-default-t0-20260910/native_mooncake-4p4d-c512/result.json) | 1.193 | 28,290.5 | 73.5% | 7.3% | 99.7% | 88.3% | 46.9 | 8.95 |
| [No-reverse PD 2P c512](current/mixed-aligned-default-t0-20260910/no_reverse-2p6d-c512/result.json) | 0.764 | 23,051.9 | 100.0% | 7.7% | 99.0% | 91.0% | 8.5 | 33.84 |
| [No-reverse PD 4P c512](current/mixed-aligned-default-t0-20260910/no_reverse-4p4d-c512/result.json) | 1.289 | 41,793.8 | 90.2% | 6.9% | 99.8% | 88.6% | 42.2 | 12.77 |
| [当前新方法 2P c512](current/mixed-aligned-default-t0-20260910/full-2p6d-c512/result.json) | 2.344 | 17,937.1 | 92.5% | 58.8% | 99.6% | 80.5% | 65.8 | 0.17 |
| [当前新方法 2P c640](current/mixed-aligned-default-t0-20260910/full-2p6d-c640/result.json) | 2.175 | 18,812.0 | 96.1% | 63.4% | 99.6% | 71.3% | 57.8 | 0.13 |

<!-- mixed-current-end -->


统一配置：temperature=0；固定 workload（n8192）；300 秒预热 + 1200 秒正式测量；
普通 GPU `mem_fraction_static=0.80`，搜索 GPU7 为 `0.60`。

## 最新有效结果

| 方法 | 配置 | 状态 | Decode 总吞吐 |
|---|---|---|---:|
| 原生 Mooncake | 2P:6D，c384 | 待运行 | — |
| 原生 Mooncake | 2P:6D，c512 | 已完成 | 3,007.8 token/s |
| 原生 Mooncake | 2P:6D，c640 | 待运行 | — |
| 原生 Mooncake | 4P:4D，c512 | 已完成 | 4,934.0 token/s |
| No-reverse PD | 2P:6D，c384 | 待运行 | — |
| No-reverse PD | 2P:6D，c512 | 已完成 | 3,251.8 token/s |
| No-reverse PD | 2P:6D，c640 | 待运行 | — |
| No-reverse PD | 4P:4D，c512 | 已完成 | 5,219.2 token/s |
| 当前新方法 | 2P:6D，c384 | 待运行 | — |
| 当前新方法 | 2P:6D，c512 | 已完成 | 9,501.8 token/s |
| 当前新方法 | 2P:6D，c640 | 已完成 | 9,069.3 token/s |

## 正式窗口吞吐与数据特性

| 方法/配置 | 完成样本 | Agent/s | Prefill token/s | 平均 Decode tokens/Agent | 平均实际 Prompt tokens/Agent | 平均累计 cached tokens/Agent | 平均轮次 | 平均工具时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原生 Mooncake 2P:6D c512 | 835 | 0.696 | 16,401 | 4,364 | 19,483 | 3,220 | 2.39 | 0.155 s |
| 原生 Mooncake 4P:4D c512 | 1,432 | 1.193 | 28,291 | 3,979 | 27,936 | 7,374 | 3.43 | 0.438 s |
| No-reverse PD 2P:6D c512 | 917 | 0.764 | 23,052 | 4,010 | 23,275 | 0 | 2.63 | 0.305 s |
| No-reverse PD 4P:4D c512 | 1,547 | 1.289 | 41,794 | 3,942 | 28,369 | 0 | 3.64 | 0.244 s |
| 当前新方法 2P:6D c512 | 2,813 | 2.344 | 17,937 | 3,789 | 32,644 | 27,165 | 4.41 | 0.844 s |
| 当前新方法 2P:6D c640 | 2,610 | 2.175 | 18,812 | 3,740 | 32,330 | 26,316 | 4.41 | 0.378 s |

说明：平均累计 Prompt 是各轮模型调用 prompt 的总和；cached tokens 是各轮报告的
prefix-cache 命中量之和，不能直接等同于完整父 snapshot 的 page-aligned 复用率。

## 稳态资源与并发

| 方法/配置 | P Forward/卡 | P KV/引擎 | P queue | D Forward/卡 | D KV/引擎 | D running/卡 | D queue | D transfer/卡 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原生 Mooncake 2P:6D c512 | 99.0% | 9.0% | 95.7 | 98.4% | 90.9% | 9.9 | 0 | 33.38 |
| 原生 Mooncake 4P:4D c512 | 73.5% | 7.3% | 5.9 | 99.7% | 88.3% | 46.9 | 0 | 8.95 |
| No-reverse PD 2P:6D c512 | 100.0% | 7.7% | 97.9 | 99.0% | 91.0% | 8.5 | 0 | 33.84 |
| No-reverse PD 4P:4D c512 | 90.2% | 6.9% | 9.5 | 99.85% | 88.6% | 42.2 | 0 | 12.77 |
| 当前新方法 2P:6D c512 | 92.5% | 58.8% | 31.5 | 99.6% | 80.5% | 65.8 | 0 | 0.17 |
| 当前新方法 2P:6D c640 | 96.1% | 63.4% | 121.0 | 99.62% | 71.3% | 57.8 | 0 | 0.13 |

其中 KV 为正式窗口采样均值；D running/卡 是每张 D 卡的平均活跃请求数，D transfer/卡
是 SGLang decode transfer queue 的平均长度，不是带宽。

## 路径与 Host 状态

| 方法/配置 | D→P Direct/Slow/重算 | P→D Host | D→P Host | Host/Mooncake 状态 |
|---|---|---|---|---|
| 原生 Mooncake 2P:6D c512 | 原生路径（不适用） | 原生 HiCache/Mooncake | 原生 HiCache/Mooncake | 原生存储开启 |
| 原生 Mooncake 4P:4D c512 | 原生路径（不适用） | 原生 HiCache/Mooncake | 原生 HiCache/Mooncake | 原生存储开启 |
| No-reverse PD 2P:6D c512 | 不创建 D→P 回传 | P→D 原生路径 | 不适用 | 无反向复用 |
| No-reverse PD 4P:4D c512 | 不创建 D→P 回传 | P→D 原生路径 | 不适用 | 无反向复用 |
| 当前新方法 2P:6D c512 | Direct 9,278 / Slow 559 / 重算 9 | Direct 12,937 / Host 4,822 | Host 写入 559 / 恢复 558 | 不依赖原生 HiCache/Mooncake |
| 当前新方法 2P:6D c640 | Direct 8,319 / Slow 967 / 重算 229 | Direct 12,313 / Host 2,327 | Host 写入 967 / 窗口内恢复 959 | 不依赖原生 HiCache/Mooncake |

## 当前新方法路径统计

### D→P 反向 KV

| 配置 | Direct 成功 | Slow 写入 Host | 快路径失败重算 | Direct / Slow / 重算 | Slow Host→P 恢复 | Host 写入量 |
|---|---:|---:|---:|---:|---:|---:|
| 2P:6D c512 | 9,278 | 559 | 9 | 94.23% / 5.68% / 0.09% | 558 / 559 | 646.73 GiB |
| 2P:6D c640 | 8,319 | 967 | 229 | 87.43% / 10.16% / 2.41% | 959 / 967 | 1,149.30 GiB |

| Direct 平均传输时间 | Slow D→Host 平均时间 | Slow Host→P 平均时间 | Slow 平均 snapshot 大小 |
|---:|---:|---:|---:|
| 120.46 ms | 134.99 ms | 87.28 ms | 1,184.70 MiB |
| 113.26 ms | 132.91 ms | 96.36 ms | 1,217.05 MiB |

559 个 Slow snapshot 中，558 个在正式窗口内完成 Host→P 恢复。剩余 1 个
`1af34774149740539d45ca74a52295d6:3` 在 02:55:32 完成 Host 写入，但对应 QA 已于
02:55:30 以 `truncated` 结束，没有下一轮 Prefill；因此这不是恢复丢失。

### P→D 正向 KV

| 配置 | Direct 成功 | 写入 P→D Host | Host→D 恢复 | Direct / Host | Host 写入量 |
|---|---:|---:|---:|---:|---:|
| 2P:6D c512 | 12,937 | 4,822 | 4,817 | 72.85% / 27.15% | 4,312.64 GiB |
| 2P:6D c640 | 12,313 | 2,327 | 2,280 | 84.11% / 15.89% | 1,959.21 GiB |

| P→Host 平均时间 | Host→D 平均时间 | Host 平均 snapshot 大小 |
|---:|---:|---:|
| 76.29 ms | 83.06 ms | 915.83 MiB |
| 76.28 ms | 83.20 ms | 862.15 MiB |

P→D 的 4,822 次 Host 写入中，4,817 次在同一正式窗口内恢复；5 次的差值是
窗口末尾尚未完成的在途请求。闭环在测量结束时仍保持 512 个活跃 Agent，
因此“写入于窗口内、恢复于窗口外”是正常的边界现象。

`c640` 的 D→P 967 次 Host 写入中有 959 次在同一正式窗口恢复；P→D 的
2,327 次 Host 写入中有 2,280 次在窗口内恢复。闭环结束时仍保持640个活跃Agent，
这些差值同样包括测量窗口末尾的在途 snapshot，不能直接解释为恢复丢失。

以上计数来自正式 1200 秒窗口内 P/D 原始日志，按
`snapshot` / `extra_key` 去重；`engine_metrics.jsonl` 本身只保存吞吐、Forward、
KV、running、queue 和 transfer 采样，未把这些路径事件二次汇总。

当前新方法的 ownership、完成边界和原始逐请求记录见：
[实验目录](current/mixed-aligned-default-t0-20260910/full-2p6d-c512/)。

未完成的实验保留在主表中，结果为空；旧版历史实验和不对齐配置已移除。
