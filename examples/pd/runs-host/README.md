# PD experiment results

## 显存比例公平性

从 2026-09-07 起，所有 PD 分离实验必须在每张物理 GPU 上与对应 colocated
实验使用相同的 `mem_fraction_static`。当前 A100 实验的普通 GPU 固定为 `0.80`，
承载搜索服务的 GPU 7 固定为 `0.60`；P 不再使用 `0.85`。已有文档中使用普通
P/D `0.85` 的 PD 结果保留数值用于历史诊断，但均标记为“需重跑”，不得用于最终
横向收益结论。已经使用对齐比例的 Qwen3-32B TP=2 baseline 不受显存标注影响，
但其旧版 Agentic-PD 行仍受下面的方法语义校正影响。

## 当前新方法定义

从 2026-09-08 起，“当前新方法”固定指 **快慢路径 + Direct失败重算**：

- 工具超过 1 秒未返回：D→P Slow；
- 工具在 1 秒内返回：尝试 D→P Direct；
- Direct admission/receiver/传输在工具返回后的统一 1 秒 deadline 内未建立：
  进入显式完整重算，不再转 Slow。

对应正式实验必须显式记录：

```text
SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD=1
SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT=1
SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=true
SGLANG_AGENTIC_KV_HOST_STAGING=true
```

除 `BROWSECOMP_QWEN3_8B_ABLATIONS.md` 中新完成的对应行，以及同步回填到
`BROWSECOMP_QWEN3_8B.md` 的 c512 行外，现有“新方法”实验均早于该定义，应标记
为旧版快慢路径并重跑。Colocated、No-reverse、原生 Mooncake 等 baseline 不因
这次新方法定义变化而失效；它们是否需要重跑仍由显存比例、数据顺序和采样参数
是否对齐决定。

## 快慢路径统计口径

所有后续主实验 Markdown 都应记录 `D→P Direct/Slow/Recompute` 比例。正式
口径按 1,200 秒测量窗口和唯一 request-generation `snapshot_id` 去重：最终进入
Shared Host Arena 的 snapshot 计为 Slow；完成 Direct 发送的计为 Direct；
持久化为 `RECOMPUTE_REQUIRED` 的计为 Recompute。TP>1 的多个 rank 合并为一个
逻辑 snapshot。历史文档若只保存整轮计数，必须明确标注统计边界。Colocated、
No-reverse 和原生 HiCache/Mooncake 不使用当前新方法的这套 Direct/Slow
状态机，因此记为“不适用”；缺少可核验日志的历史运行记为“未记录”。

## Canonical experiment matrices

当前维护以下六个主实验矩阵；空白项表示正式实验尚未完成：

- [BrowseComp + Qwen3-8B](BROWSECOMP_QWEN3_8B.md)
- [BrowseComp + Qwen3-8B Ablations](BROWSECOMP_QWEN3_8B_ABLATIONS.md)
- [BrowseComp + Qwen3-32B TP=2](BROWSECOMP_QWEN3_32B_TP2.md)
- [Retool + BrowseComp 1:1 + Qwen3-8B](MIXED_1TO1_QWEN3_8B.md)
- [Retool + BrowseComp 1:1 + Qwen3-8B Ablations](MIXED_1TO1_QWEN3_8B_ABLATIONS.md)
- [SWE-bench Verified + Qwen3.5-27B TP=2](SWEBENCH_QWEN35_27B_TP2.md)

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
  - the paired Agentic-PD child is historical and requires rerun.
- `current/ablations/browsecomp-qwen3-8b-4p4d-c512/aligned-p080-d080060-threshold1-20260908-r4/fast-direct-fail-recompute-1s`
  - current valid new-method checkpoint: aligned 4P:4D, TP=1, c512;
  - 300-second warmup + 1200-second measurement, 1-second tool/Direct policy.
- `current/qwen3-32b-tp2-browsecomp-c256-w300-m1200`
  - baseline: fixed source-order BrowseComp on colocated TP=2 workers;
  - new method: 2P:6D, TP=2, 300-second warmup and 1200-second measurement;
  - the retained Agentic-PD child uses the old Direct-failure-to-Slow policy and
    requires rerun under the current method.
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

- `archive/baseline`: older formal colocated, native-PD, native-Mooncake, and
  workload-characterization results.
- `archive/new-method`: older formal agentic-PD results and ablations retained
  for regression analysis.

Superseded smoke, gate, short, diagnostic, parser-smoke, and failed diagnostic
runs were removed from this tree on 2026-08-28 and 2026-08-31. The latest
cleanup retained only runs with a formal analysis summary and moved 23
incomplete or short-run directories to the desktop trash rather than
irreversibly erasing them.
