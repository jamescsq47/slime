# PD Multi-node V3 解耦重构计划

本文规定 `pd_multi_node_v3` 的下一阶段实现边界。目标是使 TP=8、双节点
P/D 分离下的计算、传输和内存管理各自持续推进，同时保持 request-generation
级 KV 所有权、TP 原子性和原生 CPU overlap。

设计真源仍为 [AGENTIC_PD_DESIGN_INVARIANTS.md](AGENTIC_PD_DESIGN_INVARIANTS.md)。
本文只细化实现方式，不能覆盖其中的所有权和验收约束。

## 1. 必须满足的系统行为

1. TP 组只有 rank 0 做请求、路径、lane、attempt 和提交顺序决策。其他 rank
   未收到 rank 0 命令不得行动；完成本地 shard 操作后只上报结果。rank 0 收齐
   必要 ACK 后才能推进同一 request-generation，但等待某个 rank 不得阻塞其他
   request-generation、其他路径或 Forward。
2. P scheduler 只消费已经准备好的 workset 并执行 Prefill。D→P Direct 和
   Host→P 的路由、分配、复制、绑定及 ready 发布由后台控制器推进。
3. Prefill 完成后，scheduler 只交出不可变 snapshot 描述和计算完成 fence。
   P→D Direct/P→Host→D 由后台控制器推进；完成的 KV 不得等待后续 Prefill
   scheduler 迭代才开始交付。
4. P/D 保留 SGLang 原生 CPU overlap。禁止用全局 CUDA synchronize、大锁覆盖
   DMA/网络等待或关闭 overlap 来换取正确性。
5. D→P Direct、D→Host→P、P→D Direct、P→Host→D 是四条独立数据队列。
   源 HBM 在真实 DMA fence 完成、目标获得受保护所有权后及时释放，不等待目标
   scheduler 开始计算。只有 D HBM 与 P→D Host 同时无法接纳时，才允许完成
   Prefill 的请求反压 P HBM。

## 2. 最终职责边界

### Rank-0 group controller

- 维护唯一的 request-generation/attempt 生命周期。
- 选择 Direct/Slow、目标端和 I/O lane，并发布不可变组命令。
- 汇总各 rank 的 `PREPARED`、`DMA_DONE`、`BOUND`、`RELEASED` ACK。
- 不同步等待某次 DMA；所有推进由事件触发。

### Rank-local memory authority

- 是本 rank Attention pages、Mamba slots 和 request slots 的唯一分配/释放入口。
- P 原子申请 `parent + 新增 prompt` 的完整 workset。
- D 原子申请导入 KV 和 Decode 增长容量。
- scheduler 只能消费 authority 已授权的页，不能与后台线程分别修改同一 free list。

### I/O workers

- 独立执行 Direct、Host store 和 Host restore，并报告真实物理 fence。
- 不自行选择路径、目标或 fallback，不自行发布组级终态。
- Direct 与 Slow 使用独立执行队列和名额。

### P/D schedulers

- 从 `prefill-ready` / `decode-ready` 队列取请求并连续计算。
- 报告请求粒度的计算完成 event，随后将 snapshot 交给后台控制器。
- 不做网络轮询、Host 恢复选择、阻塞式传输或历史目录扫描。

## 3. 资源模型

必须分别管理三种资源，禁止再用一个 active/request 窗口同时代表它们：

1. **I/O lane**：传输现在能否实际启动。
2. **workset lease**：目标是否真实持有 Attention/Mamba/请求槽资源。
3. **ready item**：数据和绑定均完成、可被 scheduler 消费的请求。

目标准入顺序为：请求条件满足 → 取得对应路径 I/O lane → TP 全组尝试申请
workset → 全组成功后立即启动传输。失败时归还 lane 和所有临时预留。
Slow 没有 H2D lane 时保留 Host ownership，不提前长期占用 P HBM。

容量释放产生带递增 epoch 的事件。rank 0 每次事件推进当前可行的请求，直到
容量或 I/O lane 耗尽；一个暂时过大的请求不能阻塞后续可行请求。同一容量 epoch
下失败的申请不得立即循环重试。

## 4. TP 命令与所有权交接

每个 attempt 使用以下组级序列：

1. rank 0 发布 `PREPARE`；各 rank 本地预留并 ACK。
2. 全组成功后发布 `START`；各 rank 启动本地 shard I/O。
3. 全组真实 `DMA_DONE` 后，目标完成本地绑定并 ACK `BOUND`。
4. 目标已持有受保护所有权后，rank 0 授权源端释放并收齐 `RELEASED`。
5. 目标端统一发布 ready；scheduler 后续 adoption 不占用 I/O lane，也不阻止
   源端释放。

任一 rank PREPARE 失败时，由 rank 0 统一回滚全组临时预留。DMA 已经提交而物理
结果未知时必须 fail-closed，不能靠超时猜测完成并复用页面。

## 5. 四条路径

### D→P Direct

工具在 1 秒内返回后，P 为完整 workset 申请空间。工具返回到 Direct admission、
建链和启动共用一个 1 秒 deadline，排队不能逃逸 deadline。超时且尚未提交 DMA
时统一取消并转 Slow；已提交 DMA 必须先收敛真实 fence。

### D→Host→P

工具超时或 Direct 安全失败后，D→Host 独立推进。Host durable 后立即释放 D HBM，
不等待工具或 P。工具、Host 数据、P workset 和 H2D lane 均就绪时才取得 P HBM；
H2D 完成并绑定后释放 Host extent，进入 prefill-ready。

### P→D Direct

Prefill 完成 event 到达后台后立即 late-bind 可行 D 并传输。目标 D 全组绑定完成后
立即释放 P HBM并进入 decode-ready，不等待 D scheduler 开始 Forward。

### P→Host→D

D 暂时不可行时，snapshot 独立进入 P→D Host。Host durable 后立即释放 P HBM。
D 容量变化时后台恢复；某个 Host snapshot 不得形成严格 FIFO 队头阻塞。只有 D
无空间且 P→D Host 也无法取得完整 extent 时，才允许该 snapshot 继续占用 P HBM。

## 6. overlap 与 CUDA 依赖

- scheduler 记录请求粒度的 Forward 完成 event。
- 传输 stream 只等待对应请求的计算 event，不等待整个计算 stream。
- 接收、绑定和 ready 发布同样通过请求粒度 event 连接。
- allocator/Radix 元数据事务必须短小，不能持锁等待 TCP、TP ACK、DMA 或 Forward。
- 禁止 scheduler 中的 `cudaDeviceSynchronize()`，禁止后台与 scheduler 同时修改同一
  请求的页表或 Mamba 状态。

## 7. 代码落点

- `agentic_group_transfer.py`：组命令和 ACK barrier；拆分 I/O lane 与 ownership handoff。
- `agentic_transfer_queues.py`：四路径独立执行队列和真实 lane 容量。
- `agentic_memory_authority.py`：唯一 allocator 入口、结构化容量失败、capacity epoch。
- `agentic_native_memory_adapter.py`：受 authority 保护的 Radix/Mamba 绑定与释放。
- `agentic_multinode_policy.py`：Direct deadline、事件驱动容量恢复、路径 fallback。
- `agentic_multinode_runtime.py`：rank 0 事件循环、ready/terminal 交接。
- `agentic_memory_scheduler_bridge.py` / `agentic_decode_memory_bridge.py`：ready 输入、
  计算完成输出及请求 fence。
- `managers/scheduler.py`：只保留轻量、TP 一致的 ready 消费和计算完成发布。

## 8. 实施与验收顺序

1. 增加互斥阶段和资源账目：P/D/Host 阶段、Attention、增长预留、Mamba、lane。
2. 拆除共享 target FIFO，恢复 Direct/Slow 独立 I/O 准入。
3. 实现 capacity epoch 驱动的可行请求推进及 Direct 完整 deadline。
4. 将接收绑定和源释放从 scheduler adoption 中分离。
5. 校验 overlap 下请求级 CUDA event 和 TP8 原子性。
6. 运行生命周期、容量、取消、迟到 ACK、部分 rank 失败及 overlap 竞态测试。
7. 对照八项设计不变量并完成独立审计；审计 GO 后才能进行 GPU 验收。

GPU 验收使用同一 Qwen3.5-122B-A10B、双节点 TP=8、SWE-bench Verified c128、
500 题配置，至少满足：中段 D Forward ≥95%，Decode ≥1000 token/s，平均 running
≥50；所有未完成 request-generation 均可归入唯一阶段，且 ownership 数量守恒。

## 9. 当前实现状态（2026-09-27）

已实现并进入回归测试的部分：

- Direct、Host store、Host restore 按方向和操作使用独立 I/O 准入窗口，不再共享
  target FIFO。
- I/O lane 在全 TP shard 到达真实 `DMA_DONE` 后立即归还；绑定、ready 发布和
  scheduler adoption 继续由 lease 保护，但不占物理传输 lane。
- Direct admission deadline 覆盖 readiness、排队、PREPARE、START 以及真实 NIXL
  post；只有全 TP rank 的 sender 已 post 或 receiver 已 ready 后才停止计时。
- Forward CUDA fence、Direct shard 构造和 Host 页索引物化均在 I/O worker 中完成，
  不在串行 TP 控制线程中同步等待。
- allocator capacity edge 使用最新绝对容量快照，单次事件可批量推进可行请求，
  并允许跳过暂时不可行的大请求。
- runtime shutdown 显式取消 readiness pending、lane pending 和可安全取消的 active
  attempt；composite close 遇到在途 fence 时可重试，不会提前拆除 callback。
- no-I/O rank 的 dispatch handler 在终态显式退休；控制广播异常保留 attempt 和
  lane，采取 fail-closed，不把可能部分发布的资源重新分配。

当前 CPU 回归：`python -m pytest -q python/sglang/srt/disaggregation/test_agentic*.py`
共 937 项通过；独立复审抽查 101 项通过，并已给出进入 GPU 验收的 GO。双节点
GPU 验收仍是产出正式性能结果前的硬门槛。
