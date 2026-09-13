# 27B TP2 c128 baseline：配置重对齐

完整 SWE-bench Verified 500 题测评；不使用固定 1500 秒终止，不重复题目。
外部 harness、数据顺序及 8K/64 采样保持不变。主对照表见 ../../SWEBENCH_QWEN35_27B_TP2.md。

本轮 Triton attention、FlashInfer sampling、deterministic=false、允许 custom all-reduce、TP2 NUMA=0,1。
page64、Mamba track64、extra_buffer、Mamba/full=0.9、静态显存0.80、Prefill chunk/max=8192。
后续新方法 Mamba/full=0.5 是用户指定的有意差异。

Baseline 安装目录保留两个最小修复：resumed Prefill chunk 对齐；容量不足时暂停该 chunk，而非回退到8192强行准入。
本轮同时保存补丁和实际 scheduler.py/schedule_policy.py，便于复现。

## 修改门禁

本次启动配置不改变 snapshot 状态机。已暂停 chunk 的原始 Req、KV、Mamba state 仍由原所有者持有；没有新建 IO、超时、强制释放或 checkpoint 伪造。

1. 唯一所有者：本地原生缓存所有权不变；容量不足只暂停未准入 chunk。
2. P→D Direct 释放：colocated 不启用，不涉及。
3. P→D Host 释放：colocated 不启用，不涉及。
4. D→P Host 释放：colocated 不启用，不涉及。
5. 解耦：无额外传输/控制队列；非准入 chunk 不作为虚假 Forward 入账。
6. TP 原子性：相同预算作相同决策；独立测试覆盖 TP 对齐，不改 collective 协议。
7. 父 KV 正确性：保留原生 Mamba cache 断言与最终 tail，不宣称 native cache 永不淘汰。
8. 测试与审核：88 项通过，含 resumed chunk 容量/边界/取消/TP/launcher/report；独立 GO 后才启动 GPU。

性能与正确率尚待完整测评，不以 CPU 回归测试代替 GPU 验收；原生 baseline 不适用 agentic queued=durable=source_release 传输计数。

独立审计 `audit_baseline_mamba_alignment` 已给 GO，并独立复跑88项全部通过。
2026-09-12 17:58:21 UTC 启动监护 PID2987922（launcher PID2987923）；这不是业务计时起点。
原始目录：`/tmp/pd-persist/baseline-qwen35-27b-tp2-swe500-colocated-c128-20260912-r3-triton-customar`。
每30秒检查进度与致命异常，异常时只清理本轮进程组/容器标签。
自动结果汇总 PID2988769 仅在500题完整完成后写 `RESULTS.md`，失败或被取消不冒充完整结果。
保留其他用户 GPU7 进程 PID1868643。

18:02:42 UTC 四组模型及 router 就绪，inference 加载500题；启动日志确认 deterministic=False、disable_custom_all_reduce=False、Triton attention、FlashInfer sampling、Mamba/full=0.9。

首轮启动期间出现 HTTP health check 超时，同时观察到 sampling CUDA 内核编译进程；18:04 UTC 后模型已返回200并进入续轮 Prefill/Decode。未通过更换 backend 或放宽 timeout 绕过。首次编译/业务冷启动时间保留在全程墙钟中，后续比较需与中段指标区分。

2026-09-12 19:21:10 UTC 按用户要求停止并改测c64。停止阶段记录178题结束、119题通过；不是完整500题结果。原始日志与轨迹保留，停止前计数器保存在原始目录`pre_stop_metrics.json`。本轮GPU进程和标签容器已清理；未删除共享镜像或其他用户资源。
