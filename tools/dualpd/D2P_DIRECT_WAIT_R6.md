# 严格 D→P Direct：无 Slow、无超时重算

2026-09-21 用户要求在r5a基础上关闭重算再测。r5a已由coordinator SIGTERM
按所属进程清理，cleanup.json存在，a10/a11 GPU compute进程列表均为空。
r5a是未跑完的诊断，不是500题最终结果。

## 配置与语义

- a10=P、a11=D，Qwen3.5-122B-A10B，TP8/EP1，mem_fraction_static=.8，
  mamba_full_memory_ratio=.5；SWE Verified500原顺序、openai_tools、c128。
- 采样.6/.95/20，单轮8192、最多64轮、总输出81920、上下文131072不变。
- `--d2p-direct-wait`启用`DIRECT_WAIT_ONLY`，D→P Host关闭，保留P→D Host。
- 工具迟归、Direct容量/建链等待不再触发策略重算；内部期限无限，父KV在D
  保持所有权直到P完成接管。沿用原异步队列/TP命令，不新增重试状态机。
- 真实传输失败：沿用fence收敛、保留源，报告`strict_direct_failed`并停止排查。
  不允许旧attempt的收据用于新会话。终止/取消仍能正常清理。
- 应用final后纠错及Qwen稳定checkpoint后缀计算不属于本次关闭的策略重算。
- smoke含即时工具与延迟4s工具；两者必须Direct、全8rank完成、cached_tokens
  达到8192且逐token输出与参考一致，无D→P Host条目。

## 所有权与八项门禁

1. D_HBM_OWNED保持到原Direct全组提交；无新增所有权状态。
2. P→D Direct成功释放：未修改。
3. P→D Host durable释放：未修改。
4. D→P Host在本消融禁用；默认方案代码未修改。
5. 仍由原后台I/O推进；等待不阻塞scheduler，容量反压需实测。
6. 仍由TP rank0决策，所有rank使用原真实DMA fence/组提交。
7. 不允许策略FAILED进入完整重算；发现真实FAILED保留并报错。
8. 新增迟到工具、旧arrival排队、迟到claim、TP8组启动、传输fence、
   FAILED不重算测试；完成回归与独立GO后才启动GPU。

## 入口

```bash
bash tools/dualpd/qwen35_multinode.sh run --concurrency 128 --d2p-direct-wait \
  --run-dir /homes/siqic/dualpd/slime/runs/dualpd/qwen35-122b-swe500-tp8-c128-socket-direct-wait-r6
```

此次不改变默认实验。停止仍通过coordinator TERM / owned stop，不使用宽泛pkill。
若纯保留导致容量循环、真实传输错误、重复策略重算或长时间无进展，先停止并汇报。

## 启动前验证

- 全CPU门禁：1204 passed、2个GPU测试skipped；79个launcher单元测试通过。
- 独立审核：GO；另跑110个engine测试、58个launcher测试通过。
- `strict_direct_failed`是诊断日志，不是自动kill；启动后监测该信号，出现即
  由coordinator正常TERM清理。本测试不声称无限等待必然无死锁。

## 两节点功能验证

run=`qwen35-122b-swe500-tp8-c128-socket-direct-wait-r6`。
16个rank预热完成；D每rank明确报告0个D→P Arena、0注册字节，P保持8GiB/rank。
即时工具、延迟4秒工具两项均通过：各复用8192 tokens，Direct commit包含
rank0–7，无Host条目，输出与完整Prefill参考逐token一致。
随后启动同一配置SWE500/c128全量评测，最终性能及正确率仍待完成。
