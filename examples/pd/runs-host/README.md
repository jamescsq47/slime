# PD experiment results

## 快慢路径统计口径

所有主实验 Markdown 都记录 `D→P Direct/Slow` 比例。该指标只统计正式
1,200 秒测量窗口，并按唯一 request-generation `snapshot_id` 去重：最终进入
Shared Host Arena 的 snapshot 计为 Slow；完成 Direct 发送且没有再转入 Slow
的计为 Direct。TP>1 的多个 rank 合并为一个逻辑 snapshot。Colocated、
No-reverse 和原生 HiCache/Mooncake 不使用当前新方法的这套 Direct/Slow
状态机，因此记为“不适用”；缺少可核验日志的历史运行记为“未记录”。

## Canonical experiment matrices

当前维护以下四个主实验矩阵；空白项表示正式实验尚未完成：

- [BrowseComp + Qwen3-8B](BROWSECOMP_QWEN3_8B.md)
- [BrowseComp + Qwen3-32B TP=2](BROWSECOMP_QWEN3_32B_TP2.md)
- [Retool + BrowseComp 1:1 + Qwen3-8B](MIXED_1TO1_QWEN3_8B.md)
- [Retool + BrowseComp 1:1 + Qwen3-8B Ablations](MIXED_1TO1_QWEN3_8B_ABLATIONS.md)
- [SWE-bench Verified + Qwen3.5-27B TP=2](SWEBENCH_QWEN35_27B_TP2.md)

重构前的新方法结果只作为 archive 历史记录，不回填到这四个矩阵；与新方法
重构无关的 colocated、No-reverse 和原生 Mooncake baseline 可以继续使用。

The result tree is split into directly comparable current checkpoints and
historical formal runs.

## Current aligned comparisons

Each experiment key contains two identically scoped children:

- `baseline-colocated`: colocated SGLang baseline;
- `new-method-agentic-pd`: the latest validated request-generation-level PD
  implementation.

Current experiment keys:

- `current/qwen3-8b-tp1-browsecomp-c512-w300-m1200`
  - baseline: fixed source-order BrowseComp on eight colocated GPUs;
  - new method: 4P:4D, TP=1, 300-second warmup and 1200-second measurement;
  - latest new-method checkpoint: the TP=1 Host-to-P lane-release run.
- `current/qwen3-32b-tp2-browsecomp-c256-w300-m1200`
  - baseline: fixed source-order BrowseComp on colocated TP=2 workers;
  - new method: 2P:6D, TP=2, 300-second warmup and 1200-second measurement;
  - latest new-method checkpoint: the TP group-owner transition run.
- `current/qwen3-8b-tp1-mixed1to1-c512-w300-m1200`
  - baseline: fixed 1:1 Retool/BrowseComp workload on eight colocated GPUs;
  - new method: 2P:6D, TP=1, 300-second warmup and 1200-second measurement.
- `current/ablations/mixed1to1-qwen3-8b-2p6d-c512/target1-spill0p5-nonstrict/full`
  - latest full-method ablation checkpoint: P→D centralized fair scan with
    capacity-feasible requests allowed to bypass an infeasible/Host predecessor;
  - 0.5-second Direct grace followed by one causally fresh capacity recheck;
  - 9,799.5 Decode token/s and 100% page-aligned Parent KV reuse.
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
