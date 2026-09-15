# Qwen3.5-9B：Host 恢复事件补进临时验证

2026-09-13。结论：最小改动通过 CPU 回归和 4 GPU 通路正确性验证；
尚未做同负载开关 A/B 或300+1200秒正式实验，不宣称解决了全部P空闲或提高了正式吞吐。

## 修改范围

只改融合代码 `/homes/siqic/sglang-qwen35-integration` 的两处生产逻辑：

- `python/sglang/srt/disaggregation/agentic_host_staging.py`：增加默认关闭的
  `SGLANG_AGENTIC_KV_P_HOST_EVENT_PROGRESS`，支持有界读取已有事件通知。
- `python/sglang/srt/managers/scheduler.py`：原有完成检查后，沿用原生FIFO准入补进一次，
  在同一个安全边界服务完整workset并启动已选中的恢复。共享原准入预算；
  只有真正持有完整Attention+Mamba workset的请求允许走完成交接快捷路径。

已有H2D线程仍负责后台传输，不把allocator/Radix操作放到任意后台线程。
新逻辑只对TP1、request-owned Mamba且H2D解耦已开启的路径生效；
默认、dense Qwen3、TP>1不启用。未修改安装环境或SWE/BrowseComp harness。
新增的工具、容量、路由或重算策略：无。

488项生命周期、容量、取消、TP、Mamba与事件回归通过；最后一次仅增加首次命中INFO后，
36项焦点回归再次通过。独立审计`audit_host_event_progress`给出代码及启动脚本GO。
八项设计不变量逐项说明见引擎 `validation/HOST_EVENT_PROGRESS.md`。

## R2 设置与结果

| 项目 | 设置/结果 |
|---|---|
| 模型与拓扑 | Qwen3.5-9B，TP1，2P:2D |
| GPU | P=0、6；D=1、7；其他实验及GPU7原有601MiB进程不动 |
| 运行环境 | pd_mamba_baseline解释器 + 融合源码PYTHONPATH，不修改已安装环境 |
| 静态显存 / Mamba比例 | 0.80 / 0.5 |
| page / track / context | 64 / 64 / 32768 |
| Prefill chunk / max tokens | 8192 / 8192 |
| Host arena | 每P：D2P16GiB、P2D8GiB，memfd |
| Host预注册 / 内容哈希 | 四进程预注册完成才发请求 / 关闭昂贵内容哈希 |
| 新事件开关 / 原H2D解耦 / H2D槽数 | 开 / 开 / 每P4槽 |
| Native HiCache、Mooncake、拥堵重算 | 均关闭 |
| 工具阈值 / Direct deadline | 1秒 / 1秒 |
| 测试输入 | 原有synthetic transport diagnostic，模拟0秒/3秒工具；不是SWE正确率测评 |
| 并发与调用数 | 16客户端、32条三轮轨迹；96次轨迹调用+96次独立重算参考 |
| 采样 | temperature0、top_p1、top_k-1，关闭thinking，开启确定性推理用于精确对照 |
| 输出逐token一致 | **96/96** |
| 跨轮回传 | **64/64：32 Direct +32 Host**，无缺失/重复路由 |
| 新补进逻辑实际执行 | 两个P均有首次执行日志 |
| Host上传/持久化/D释放/P恢复释放 | **32 /32 /32 /32，snapshot ID集合相同** |
| 停机前D2P / P2D活跃Host残留 | **0 /0**，不是靠停机清空；P2D有192条rejected元数据回执，不占payload |
| P→D Direct释放 / D generation释放 | 192 /192 |
| 最终应用确认释放 | 128：32条轨迹结束+96参考请求 |
| P→D Host spill | 0，本次低负载未覆盖；该分支只由既有回归覆盖 |

### 前缀复用检查

| 轮次 | 调用数 | Prompt tokens/次 | 实际命中/次 | 逻辑未命中输入/次 |
|---|---:|---:|---:|---:|
| 1 |32|2425|0|2425|
| 2 |32|2548|2368|180|
| 3 |32|2671|2496|175|

命中的是上一轮稳定Prompt的page64 checkpoint，全部与预期相等。
180/175包括上一轮输出、工具追加及page尾部；这是prompt-checkpoint模式，
不是把整个Decode末尾状态不加区别地复用。此列为逻辑Prompt减命中量，不是含padding的GPU执行tokens。

### Host阶段时延（本次短测，不是吞吐收益）

| 阶段 | 样本数 | 平均 | 最大 |
|---|---:|---:|---:|
| 被选中→获得workset |32|3.86ms|11.14ms|
| I/O启动→完整fence |32|23.89ms|50.08ms|
| fence→完成交接 |32|9.86ms|67.36ms|

未覆盖正式SWE长上下文、高并发下的Host排队，因此不能由这些数字推断P Forward能接近100%。

## 保存与复现

- R2原始结果：`/tmp/pd-persist/qwen35-9b-tp1-2p2d-host-event-smoke-20260913-r2`
- JSON统计：上述目录`smoke-summary.json`。
- 请求/响应：`raw/`；输出逐token对照：`multiturn-*.json`。
- 启停前账本：`control-before-stop/`，特别保留结束但尚未停服务时的状态。
- 配置：`launch-environment.txt`；差异/校验：`*-incremental.patch`、`code-sha256.txt`。
- 启动：`TEST_CLIENTS=16 RUN_DIR=<新目录> bash /homes/siqic/sglang-qwen35-integration/validation/run_host_event_smoke.sh`。
- 严格校验：`python /homes/siqic/sglang-qwen35-integration/validation/summarize_host_event_smoke.py <结果目录> --clients 16`。

R1保留在同名`...-r1`目录：48/48输出一致，但旧诊断脚本的reference请求未发最终ACK。
补的是`validation/check_multiturn.py`的测试清理，不是生产harness或引擎生命周期；R2已完整重跑验证。
