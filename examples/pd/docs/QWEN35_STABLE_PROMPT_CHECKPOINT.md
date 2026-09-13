# Qwen3.5：保持 harness 不变的 Mamba 前缀回传

2026-09-09，隔离兼容分支 `sglang-agentic-mamba` / `pd_mamba`。
不修改 `pd`、`pd_baseline` 或 SWE harness。

## 为什么不能直接复用生成结束时的 state

Miles PR51 本地 harness 把 shell 结果作为新的 user 消息提交。
Qwen3.5 原生 chat template 在下一轮会删除较早 assistant 的 reasoning。
因此“上一轮 input + output”不一定是下一轮实际 input 的前缀。
Attention KV 可以截断，Mamba recurrent/conv state 却不能由后来的状态倒推。
不能跳过 token digest 校验，也不能把末尾 state 伪装成较早位置的状态。

上一轮 PD 工程测试中，966 对相邻调用有 892 对完整生成尾前缀不兼容。
例：prompt390 / output80 / snapshot448 / next prompt621，实际公共前缀388。
可复用 page64 边界为384，必须有384处的真实 Mamba state。

## 本次 opt-in 模式

启动 P、D 时一致设置：

```bash
SGLANG_AGENTIC_KV_MAMBA_PROMPT_CHECKPOINT=true
```

默认关闭。仅在 agentic lifecycle 开启时生效。

P 选择原始 prompt 尾部 `<think>\n` 之前的 page-aligned 边界，限制原生
prefix match 不超过它，并在真实 Prefill forward 中捕获该处 state。
这只是选择 checkpoint，**不修改输入 token、模板、工具或采样参数**。
完整 prompt 仍照常计算；P→D 仍发送 active + checkpoint 两个 state 槽位。
D 正常推进 active state，但不再覆盖冻结的 prompt checkpoint。
下一轮回传的是该稳定前缀完整的 Attention KV + 同位置 Mamba state。

它不是“完整生成尾 KV 全复用”：上一轮生成的 suffix 需要重算，必须单独
统计这部分计算量，不得称为缓存损坏。只对实际发布的 stable-prefix snapshot
检查完整性、digest 和复用率。模板后续若连该前缀也改写，仍严格拒绝复用。

## 生命周期约束

复用已有两个槽位及 Radix 当前锁定节点，不新增长期 state allocation。
所有权、Direct/Host wire、CUDA fence、释放时点保持既有实现。
retraction 会使冻结 checkpoint 失效，显式重算；空稳定前缀也显式重算。
不支持本模式与 speculative decoding 或 lazy Mamba tracking buffer 同开。
P、D 必须配置一致；本轮实验 TP=1。

此分支暂保留既有 2s tool / 2s Direct→Host 调度，与并行开发的 `pd` 最新
策略不同。本修改不宣称完成后者的调度逻辑移植。

## 修改位置（隔离 SGLang 工作树内）

- `srt/environ.py`：默认关闭的 opt-in。
- `srt/disaggregation/agentic_hybrid_transfer.py`：稳定边界、锁定节点及状态校验。
- `srt/managers/schedule_batch.py`：P checkpoint 捕获；D 禁止覆盖；retraction 失效。
- `srt/managers/scheduler_components/batch_result_processor.py`：禁止 D metadata 轮转。
- `srt/disaggregation/prefill.py`：缓存命中分支使用相同稳定边界。
- `srt/disaggregation/agentic_decode_manager.py`：空/失效前缀显式重算。
- `srt/disaggregation/test_agentic_prompt_checkpoint.py`：边界、chunk、锁定及故障测试。

## 验证状态

checkpoint + Mamba + lifecycle：115 passed；TP：255 passed。
独立审计 Carver：GO（新测试独立19 passed）。
首批实际 GPU 验证：独立审计采样860个 Direct complete，860个 admitted，
0 drop。按时间排序前100个 snapshot，全都实际绑定；逐个串联旧 input+output
与下一轮真实 input IDs，100/100 checkpoint不超过真实LCP，差值全部0–63。
这是前缀匹配/实际复用证据，完整100题正确率和最终吞吐仍待评测完成。

运行目录：
`/tmp/pd-persist/qwen35-9b-tp1-swe-pd-kv-miles-first100-20260909-r2`。
相同 first100 SWE Verified 有限任务，4P:4D、Qwen3.5-9B、TP1、8k/64轮；
这是 correctness/trajectory 对比，不是300+1200秒稳态吞吐验收。
详细设置、harness hashes、源码快照及最终结果保留在运行目录。
