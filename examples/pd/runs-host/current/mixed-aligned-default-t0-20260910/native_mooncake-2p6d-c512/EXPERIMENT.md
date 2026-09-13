# Mixed 1:1：原生 Mooncake 2P:6D c512

正式测量通过独立复核。Qwen3-8B，TP=1，temperature=0，固定 n8192 顺序。
普通 GPU mem_fraction_static=0.80，搜索 GPU7=0.60；使用 pd_baseline 原生环境。

| 指标 | 正式窗口 |
|---|---:|
| 业务预热 | 301.713 s |
| 测量时长 | 1200.010 s |
| 完成 Agent / 失败 | 835 / 0 |
| Agent/s | 0.6958 |
| P 合计实际 Prefill | 16,401.2 token/s |
| D 合计 Decode | 3,007.8 token/s |
| P 平均 Forward | 98.96% |
| D 平均 Forward | 98.39% |
| P 平均 KV 利用率 | 8.99% |
| D 平均 KV 利用率 | 90.91% |
| D 平均 running/卡 | 9.86 |
| D 平均 transfer/卡 | 33.38 |
| P 平均 queue/卡 | 95.69 |

两个角色各 520 个正式窗口指标采样，始终有完整 2P/6D endpoint 数据，
累计 counter 差值覆盖约 1198.19 秒；吞吐不使用最后瞬时读数。

## 故障与收尾说明

本轮未复现上一次 NIXL loadRemoteMD/ucp_ep_rkey_unpack 段错误。
仅干净重启并增加 UCX info 日志，没有修改原生 SGLang、NIXL、KV 路径或推理参数；
因此不能声称已修复底层偶发竞态。

测量于 21:18:00 UTC 结束，launcher 输出 case complete 后进入服务清理。
21:18:19 detokenizer 的正常 SIGTERM（-15）被监测器误判为运行故障；
21:18:43/50 两个 P 在服务退出期间另有 NIXL_ERR_REMOTE_DISCONNECT。
这些均在测量结束后，不影响上述窗口，但暴露了监测器收尾边界问题。
已仅清理本轮明确归属的进程组，确认GPU上只剩用户原有Jupyter，无本轮残留。

控制器现按明确完成标记及测量结束时间区分运行错误与收尾事件；不忽略窗口内错误，
缺少时间戳时仍保守报错，且不再次中断已经运行的 EXIT cleanup。
