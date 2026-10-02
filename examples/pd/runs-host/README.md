# PD experiment results

## 显存比例公平性

从 2026-09-07 起，所有 PD 分离实验必须在每张物理 GPU 上与对应 colocated
实验使用相同的 `mem_fraction_static`。当前 A100 实验的普通 GPU 固定为 `0.80`，
承载搜索服务的 GPU 7 固定为 `0.60`；P 不再使用 `0.85`。已有文档中使用普通
P/D `0.85` 的 PD 结果保留数值用于历史诊断，但均标记为“需重跑”，不得用于最终
横向收益结论。已经使用对齐比例的 Qwen3-32B TP=2 baseline 不受显存标注影响，
但其旧版 Agentic-PD 行仍受下面的方法语义校正影响。

## 当前新方法定义

2026-09-09用户采纳 **快慢路径＋全局Slow恢复拥堵反馈重算**（c512实测5148.410 token/s）。

- 工具超过1秒未返回：D→P Slow；否则尝试Direct，建链deadline为1秒。
- 快工具Direct失败：全局Slow恢复队列拥堵时重算，否则Slow。
- Q只计工具已返回、Host durable、等待H2D worker接手的唯一parent generation；跨P不重复，等待工具不计。
- 每秒采样，连续两次Q≥32启用重算，Q≤8退出。过期超过3秒或无有效信号时保守Slow。
- 已claim失败须先通过原有物理fence；反馈不取消已有Host请求，也不抢占正在传输的KV。

完整方案复现实验须显式记录：

```text
FAST_TOOL_THRESHOLD_SECONDS=1
DIRECT_WAIT_SECONDS=1
SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=true
SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH=32
SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW=8
SGLANG_AGENTIC_KV_HOST_STAGING=true
```

底层通用开关默认关闭没有改动；必须按此完整方案配置启用，消融时显式关闭反馈。
BrowseComp/Qwen3-8B的c512已经完成；c384/c576标记需重新测试，旧数值撤出主表。
完整方案在[消融表](BROWSECOMP_QWEN3_8B_ABLATIONS.md)首行，其余为消融/基线参考。
Colocated、No-reverse及原生Mooncake已完成的对齐基线保留，不因新方法定义改变而失效。
本次仅更新记录、清理废弃实验，没有启动重测或修改传输/计算代码。

## 快慢路径统计口径

所有后续主实验 Markdown 都应记录 `D→P Direct/Slow/Recompute` 比例。正式
口径按 1,200 秒测量窗口和唯一 request-generation `snapshot_id` 去重：最终进入
Shared Host Arena 的 snapshot 计为 Slow；完成 Direct 发送的计为 Direct；
持久化为 `RECOMPUTE_REQUIRED` 的计为 Recompute。TP>1 的多个 rank 合并为一个
逻辑 snapshot。历史文档若只保存整轮计数，必须明确标注统计边界。Colocated、
No-reverse 和原生 HiCache/Mooncake 不使用当前新方法的这套 Direct/Slow
状态机，因此记为“不适用”；缺少可核验日志的历史运行记为“未记录”。

## Canonical experiment matrices

当前维护以下八个主实验矩阵；空白项表示正式实验尚未完成：

- [BrowseComp + Qwen3-8B](BROWSECOMP_QWEN3_8B.md)
- [BrowseComp + Qwen3-8B Ablations](BROWSECOMP_QWEN3_8B_ABLATIONS.md)
- [BrowseComp + Qwen3-32B TP=2](BROWSECOMP_QWEN3_32B_TP2.md)
- [Retool + BrowseComp 1:1 + Qwen3-8B](MIXED_1TO1_QWEN3_8B.md)
- [Retool + BrowseComp 1:1 + Qwen3-8B Ablations](MIXED_1TO1_QWEN3_8B_ABLATIONS.md)
- [SWE-bench Verified + Qwen3.5-27B TP=2](SWEBENCH_QWEN35_27B_TP2.md)
- [SWE-bench Verified + Qwen3.5-27B TP=2 · 双节点阶段对比](SWEBENCH_QWEN35_27B_TP2_TWO_NODE.md)
- [SWE-bench Verified + Qwen3.8-27B TP=2 · 双节点对比](SWEBENCH_QWEN38_27B_TP2.md)

重构前的新方法结果只作为 archive 历史记录，不回填到这六个矩阵；与新方法
重构无关的 colocated、No-reverse 和原生 Mooncake baseline 可以继续使用。

The result tree is split into directly comparable current checkpoints and
historical formal runs.

## Current result layout

“Current”表示目录组织和代码世代已对齐，不自动代表显存比例或新方法语义可比；
各主矩阵中的“需重跑”标记优先。BrowseComp/Qwen3-8B 与 Mixed/Qwen3-8B 的
多数既有 PD 子目录仍是普通 P/D `0.85` 或旧“Direct失败→Slow”方案的历史结果。

部分旧 experiment key 包含两个 identically scoped children：

- `baseline-colocated`: colocated SGLang baseline;
- `new-method-agentic-pd`: 目录创建时的 Agentic-PD 实现；名称本身不表示它仍是
  当前方法，必须以主矩阵中的“完成/需重跑”标记为准。

Current experiment keys:

- `current/qwen3-8b-tp1-browsecomp-c512-w300-m1200`
  - baseline: fixed source-order BrowseComp on eight colocated GPUs;
  - 当前完整方法见下列slow-congestion子目录；旧paired方法已退出主表。
- `current/qwen3-8b-tp1-browsecomp-c512-w300-m1200/current-method-slow-congestion-1s-20260909-r1`
  - 当前完整方法：5148.410 token/s，4P:4D、TP=1、c512；
  - 300+1200秒，工具/Direct阈值1秒，全局恢复队列32/8反馈。
- `current/qwen3-32b-tp2-browsecomp-c256-w300-m1200`
  - baseline: fixed source-order BrowseComp on colocated TP=2 workers;
  - new method: 2P:6D, TP=2, 300-second warmup and 1200-second measurement;
  - 当前新方法：`current-method-q32-low8-r6`，1405.70 token/s，Q high/low=32/8；
    工具/Direct阈值1秒，671条Agent完成、0失败。旧同配置结果已按用户要求清理。
- `current/qwen3-8b-tp1-mixed1to1-c512-w300-m1200`
  - baseline: fixed 1:1 Retool/BrowseComp workload on eight colocated GPUs;
  - the retained 2P:6D Agentic-PD results use the old path policy and require
    rerun under the current method.
- `current/ablations/mixed1to1-qwen3-8b-2p6d-c512/target1-spill0p5-nonstrict/full`
  - historical old-path-policy full-method checkpoint: P→D centralized fair scan with
    capacity-feasible requests allowed to bypass an infeasible/Host predecessor;
  - 0.5-second Direct grace followed by one causally fresh capacity recheck;
  - 9,799.5 Decode token/s and 100% page-aligned Parent KV reuse; requires rerun.
- `current/qwen3-8b-tp1-mixed1to1-c512-router-balanced-w300-m1200`
  - local-NUMA Host-recovery reference used to diagnose late-binding skew.
- `current/qwen3-8b-tp1-mixed1to1-c512-global-host-restore-w300-m1200`
  - latest 2P:6D checkpoint: Host-owned P-to-D snapshots may restore to any
    globally feasible Decode worker, including across NUMA nodes.

Every current result retains raw request records, two-second engine counters,
service logs, resolved workload/configuration, summary JSON, and plots.

## Archive

2026-09-09清理：撤回的旧策略结果与失败重试共15个目录、约1.95 GiB已移入回收站（可恢复），
未删除当前六组对比依据或baseline。详见[清理清单](CLEANUP_20260909.md)。

- `archive/baseline`: older formal colocated, native-PD, native-Mooncake, and
  workload-characterization results.
- `archive/new-method`: older formal agentic-PD results and ablations retained
  for regression analysis.

Superseded smoke, gate, short, diagnostic, parser-smoke, and failed diagnostic
runs were removed from this tree on 2026-08-28 and 2026-08-31. The latest
cleanup retained only runs with a formal analysis summary and moved 23
incomplete or short-run directories to the desktop trash rather than
irreversibly erasing them.
