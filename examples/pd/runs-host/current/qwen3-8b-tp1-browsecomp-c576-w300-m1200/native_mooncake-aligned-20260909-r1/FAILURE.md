# 原生 Mooncake c576：第一次运行失败

- 日期：2026-09-09
- 配置：Qwen3-8B，BrowseComp source-order n680，TP=1，4P:4D，c576，temperature=0。
- 目标窗口：300 秒业务预热 + 1200 秒正式测量。
- 结果：业务预热阶段发生原生 NIXL 连接级故障，主动终止；不计吞吐结果。

## 故障证据

- 业务于20:06:28 UTC开始；约20:08:32，D0的NIXL agent连接断开。
- 当时Mooncake仅使用167.93/256 GiB（65.6%），不是Mooncake容量耗尽。
- 四个P scheduler正在向D0发送或轮询传输，均收到
  `NIXL_ERR_REMOTE_DISCONNECT`，随后把该异常作为致命错误退出。
- Router之后持续返回`No available prefill workers`；运行已不可恢复。
- 故障前D0/D1的KV预留与transfer已经明显积压；D2/D3仍在Decode，说明它不是
  全机GPU或搜索服务统一退出。

## 处理

保留原始日志，并以完全相同的代码、数据顺序和配置进行一次独立重试。只有重试
完整通过300秒预热和1200秒测量后才填写正式吞吐；若重试再次出现同类故障，
则将原生Mooncake c576记为不可稳定运行，而不是报告临时窗口吞吐。

