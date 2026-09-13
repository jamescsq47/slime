# Colocated c64：只降低并发

用户于2026-09-12要求停止c128、改测c64。沿用同一份500题、原始顺序、外部OpenEnv harness及采样设置，从第一题重新运行完整500题，不续接c128的剩余题。

唯一实验参数变化：`MAX_INFLIGHT=128 → 64`。仍为8卡、四个TP2副本，组`0,4;1,5;2,6;3,7`；静态显存0.80，Mamba/full=0.9，page64、track64、extra_buffer、Triton attention、FlashInfer sampling、deterministic关闭、允许custom all-reduce、NUMA0,1。Prefill chunk/max=8192，单轮输出8192、64轮、累计输出81920、上下文131072；T=0.6/top_p=.95/top_k=20/min_p=0。PD、HiCache、Mooncake均关闭。

## 复现与门禁

启动脚本、两处baseline修复文件、SWE harness、inference、公共进程管理、workload和数据均逐字节对比c128启动快照一致；没有新增引擎代码修改。复跑88项测试全部通过，沿用相同代码既有独立审计GO。
所有权和取消协议不变：无PD传输；原生缓存锁/容量/TP逻辑不变，不新增超时、驱逐或回退路径。监护每30秒检查；停止时仅清理本轮setsid进程组与Docker run label。

原始目录：`/tmp/pd-persist/baseline-qwen35-27b-tp2-swe500-colocated-c64-20260912-r4-triton-customar`。
入口：`scripts/baseline/run_qwen35_27b_tp2_swe500_mamba_baseline.sh`，通过环境变量指定并发，不修改脚本默认值。
本轮为完整500题有限测评，非1500秒闭环压测。优先比较prefix命中、额外Prefill、Forward占比、吞吐，以及Attention/Mamba的不可驱逐、可驱逐、真正空闲三部分。

c128原始数据全部保留，其停止前计数器另存`pre_stop_metrics.json`。c128仅完成部分题目，不能与c64完整测评直接比较最终正确率或T500。

2026-09-12 19:23:01 UTC 启动监护 PID62717，launcher PID62730；模型加载不计入业务完成用时。
结果汇总等待进程 PID63545，仅在完整500题结束后写RESULTS.md。
c128首轮有sampling JIT编译，c64可能复用机器上的编译缓存；不清理共享JIT缓存以免影响其他工作，分析全程用时需注明这项冷启动差别，并同时比较共同业务窗口。

19:27:14.889 UTC inference加载500题并开始业务，四组服务已正常返回200并进入多轮推理。新旧preflight.json逐字段比较，唯一差异为concurrency（128→64）。
