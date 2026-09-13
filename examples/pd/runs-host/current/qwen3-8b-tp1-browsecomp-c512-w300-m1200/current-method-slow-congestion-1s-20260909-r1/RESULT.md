# c512：全局Slow恢复拥堵反馈

Q只计工具已返回、Host durable、等待H2D worker接手的唯一parent generation。
Q连续两次1秒采样>=32后，快工具Direct失败转重算；Q<=8后恢复Slow。
慢工具始终Slow；信号超过3秒陈旧则Slow。DMA/TP/释放fence不变。
Qwen3-8B / BrowseComp source-order n680 / TP1 4P:4D / c512 / t0。
全部Host预注册完成后300秒预热+1200秒正式测量。

| 指标 | 本轮 | 4888参考 |
|---|---:|---:|
| Decode token/s | 5148.410 | 4888.225 |
| P Forward/卡 | 88.99% | 97.04% |
| D Forward/卡 | 99.07% | 99.62% |

正式窗口路径/资源见comparison.json；拥堵原始采样见slow_congestion_samples.json。
窗口边缘的未完成项须逐snapshot审核，不能把写入/恢复差额直接称为丢失。
