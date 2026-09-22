# P→D Host 物理终态不可倒退：r7

## r6 诊断

上一轮 `qwen35-122b-swe500-tp8-c128-socket-direct-wait-r6` 在业务启动约
504 秒后 P TP 控制连接断开，实验退出；不是一轮完成的性能/正确率结果。
历史控制服务未持久化第一个异常，不能断言历史故障已完全归因。

CPU 使用实际 Host poll/release/progress 函数和 8 个 TCP TP 客户端复现：
Host 已报告 Success，但释放仲裁遇锁竞争返回稍后重试；下一次 poll 同样
遇锁竞争却返回 Transferring，控制服务拒绝 `physical terminal report changed`
并关闭整组。r6 的 Host 完成和首次断连时间与此路径吻合。

## 本轮唯一行为修复

- Host manager 首次观察到 request-generation 的真实物理 Success/Failed
  后，在该 request 上保存不可变终态；后续查询无需重新抢锁，不再倒退。
- 缓存物理终态不等于释放许可，不设置 `_agentic_p2d_host_terminal` 或
  `_agentic_p2d_release_authorized`。原 Host ownership 仲裁、TP release-ready
  全组收齐、scheduler 释放和在途 DMA fence 保留。
- 尚未观察到完成的请求，锁竞争仍返回 Transferring；新 generation 不继承
  旧请求的终态。取消/失败清理、超时和关闭不新增状态或重试协议。
- 控制服务持久日志记录首个组失败原因，保留原 fail-closed 协议。

八项门禁：唯一所有者与 CAS 不变；P→D Direct 不变；P→D Host durable
释放不再被错误状态倒退中断；D→P Host 不变且本轮禁用；poll 保持非阻塞；
TP 原组提交不变；父 KV 复用策略不变；故障回归及独立审核后才启动 GPU。

## 重测配置

与 r6 一致：a10=P、a11=D，各 TP8/EP1，Qwen3.5-122B-A10B，静态显存
0.8、Mamba ratio 0.5；SWE Verified 500 同序各一次，c128，openai_tools，
采样 0.6/0.95/20，8192 单轮、64 轮、81920 累计输出、131072 上下文。
D→P 仅 Direct 无限等待，无 Slow 和策略重算；P→D Host 保持 8 GiB/rank。
不为了通过测试恢复 Slow 或重算，不修改容量、路由或调度。

```bash
bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 --d2p-direct-wait \
  --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-swe500-tp8-c128-socket-direct-wait-r7
```

停止通过 coordinator SIGTERM/所属 supervisor 清理，不使用宽泛 pkill。
重点监测控制首因、P→D Host durable/release、D→P Direct 进展与 D HBM 滞留。
纯 Direct 等待的容量反压是独立待测问题，不宣称本修复能消除它。

## 验证进展

- 定向 CPU 回归：54 passed，含真实 TP2/TP8 socket 的完成后锁竞争。
- 全量 CPU 回归：1208 passed、2 个 GPU 测试 skipped；79 个 launcher
  unittest 通过。
- 独立审计 GO；审核者额外运行 67 个 CPU 测试通过。真实 DMA fence、
  Host ownership 仲裁、TP release-ready 全组门槛未放宽。
- r7 两节点启动、16 rank 预热通过；即时工具和延迟 4 秒工具 smoke 均
  Direct 成功，各复用 8192 tokens，rank0–7 全组提交，输出逐 token 对齐
  完整 Prefill 参考。随后于 2026-09-21 04:09 UTC 启动同序 SWE500/c128。
- 全量评测仍在运行，尚无最终吞吐/正确率或长时间稳定性结论。

### 04:18 UTC 阶段检查（不是最终结果）

- 已经过 r6 约 504 秒故障时点；截至约 04:18，Direct group complete
  累计 2901 个 snapshot（含启动 smoke），未见 strict_direct_failed、终态
  倒退或 TP socket closed。
- P→D Host 已实际触发 9 个 snapshot；每个均观察到全部 8 rank 的 queued、
  D2H complete、prefill release 和 Host consumed/release，数量一致。
  这覆盖了 r6 出错前后的 Host 接管/释放场景，但不等于证明历史唯一根因。
- 最近 30 秒采样 Decode 703.2 token/s（整个 TP8 组）、D Forward 平均
  94.3%，瞬时 running=16、Attention KV=97.9%；P Forward=60.8%，
  Attention KV=64.0%。此前低压力区间 D Forward 约99%。
- 严格 Direct 无限等待仍会使已结束一轮的父 KV 滞留 D；高容量压力及
  性能下降需要与控制终态 bug 分开判断。本轮未增加 Slow 或策略重算。
