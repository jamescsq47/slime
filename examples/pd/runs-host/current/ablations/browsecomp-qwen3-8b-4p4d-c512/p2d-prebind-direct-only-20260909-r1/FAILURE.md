# P→D 无 Host、关闭 late binding 消融：失败

- 日期：2026-09-09
- 配置：Qwen3-8B，BrowseComp source-order n680，TP=1，4P:4D，c512，temperature=0。
- 目标窗口：300 秒业务预热 + 1200 秒正式测量。
- 结果：业务预热阶段出现系统性停滞，主动终止；不计吞吐结果。

## 观测

- 四张 D 同时降到 `running=0`、GPU utilization=0；每张 D 的 P→D transfer
  达到配置上限 8，prealloc queue 约 25–43。
- 四张 P 的完整 D→P workset lease 接近其 347,392-token KV pool 上限，典型值为
  343k–347k tokens；每张 P 约 31–37 个 active lease。
- D→P `direct_bind_wait` 达到 120 秒，后续 Router 开始返回 500；不是正常的
  吞吐下降。
- P→D Shared Host 路径按消融定义未创建；D→P Direct/Slow 仍保持完整方法配置。

## 根因

该版实现把“关闭 late binding”解释为 D 在 Prefill 前完成原生 preallocation。
在 parent turn 中，D→P Direct 已经把完整 parent workset 放入 P HBM，而 P 又等待
预绑定 D 的 bootstrap/transfer slot；D 的有限 transfer slot 和 P 的 workset lease
相互等待，最终形成环形反压。该运行因此不能作为有效性能样本。

## 清理

本轮 SGLang、Router、Inference、Search 进程均已终止。消融开关默认仍为 false；
完整方法的默认 P→D late binding 与 Host staging 配置未改变。
