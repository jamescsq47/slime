# BrowseComp c512：当前方法纯重算消融

Qwen3-8B，TP=1、4P:4D、source-order n680、temperature=0；300秒业务预热，
1200秒正式测量。显存与完整方法一致：P为0.80×4，D为0.80/0.80/0.80/0.60。

本轮保留当前方法的P→D late binding、D Router与P→D Host staging，只设置
`SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=true`：请求不声明parent generation，D完成后
不创建反向Direct/Host状态，所有后续轮完整Prefill。

| 指标 | 纯重算 | 完整方法5148组 |
|---|---:|---:|
| Decode token/s（4D） | 1,938.839 | 5,148.410 |
| Decode token/s/卡 | 484.710 | 1,287.102 |
| Agent/s | 0.957 | 2.387 |
| Prefill token/s（4P） | 43,009 | 35,195 |
| 实际Prefill/完成Agent | 44,890 tokens | 14,719 tokens |
| Decode/完成Agent | 2,024 tokens | 2,153 tokens |
| P Forward/卡 | 99.99% | 88.99% |
| D Forward/卡 | 98.94% | 99.07% |
| P KV/卡 | 8.62% | 55.01% |
| D KV/卡 | 10.73% | 74.91% |
| P queue/卡 | 118.83 | 55.76 |
| D running/卡 | 6.96 | 52.48 |

正式窗口有4090个D generation完成并走终态释放，4090次均记录
`reverse_reuse_disabled`和D KV释放；D→P Direct offer、D→P Host写入/恢复均为0。
P→D路径保持启用；本轮D供给不足而始终有容量，所以P→D Host写入/恢复均为0。
全程请求失败0，无OOM、NIXL、Router 500或scheduler异常。

完成轨迹的D→P复用为0。通用分析器仍报告5.45%的parent-prefix cached比例，来源是
P本地Radix对公共/跨请求前缀的命中，不是D→P反向KV复用。
