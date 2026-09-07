# Agentic PD：设计目标、生命周期与验收标准

> 本文是 `slime/examples/pd` 与修改版 SGLang disaggregation 代码的设计真源（source of truth）。
>
> 修改相关代码前必须完整阅读本文；修改方案必须明确说明影响了哪些生命周期状态和不变量；修改后必须逐条复核文末验收标准。不得为了修复局部性能或超时问题而破坏这些约束。

## 大目标

通过 PD 分离和 request-level KV cache 管理适配 multi-turn agentic 任务。

传统推理引擎通常把工具调用视为当前请求已经结束并释放 KV；工具返回后，再把完整历史作为新请求重新 Prefill。这会浪费大量算力，尤其是在工具调用较多时。控制并发数或预测工具时间只能缓解问题：许多工具等待时间不可预测（例如用户交互），限制并发也会降低吞吐。

## 1. KV cache 的管理单位

KV cache 以 **request-generation snapshot** 为基本管理单位。

- Decode 或 Prefill 可以让该 snapshot 的 KV 增长。
- 驱逐或传输时，必须驱逐或传输一个完整 request-generation snapshot，不能留下只有部分历史可用的 snapshot。
- 允许使用 SGLang Radix Cache：不同请求可以共享相同前缀，以减少物理空间使用。
- 共享前缀必须遵守引用生命周期：驱逐 Request A 时，只释放 A 独有的 KV；如果 Request B 仍引用公共前缀，则公共前缀必须保留。所有引用都结束后，公共前缀才能释放。
- 即使底层按 page 存储、共享和传输，逻辑所有权、生命周期、传输完成和驱逐决策仍以完整 request-generation snapshot 为单位。

## 2. KV cache 循环流水线

每个 request-generation 的 KV 在 P、D 和 Shared Host Arena 间循环传递。下一阶段取得完整所有权后，上一阶段必须及时释放。

### 2.1 初始 Prompt：P → D

初始 Prompt 进入 P 节点完成 Prefill，随后将生成的完整 KV 传给 D 节点 Decode。

#### 快路径

- 使用 NVLink/NIXL 将 KV 从 P HBM 直接传到 D HBM。
- D 成功接收完整 snapshot 后，P 必须立即释放对应 HBM KV。

#### 慢路径

- 如果 Prefill 产出快于 D 的接收能力，D HBM 暂时没有空间，则通过 PCIe 将完整 snapshot 放入 P→D Shared Host Arena。
- Host snapshot 达到 durable 后，P 必须立即释放对应 HBM KV，不能等待 D 后续出现空间。
- D 有空间后，从 Shared Host Arena 恢复该 snapshot。
- P→D 交付以每个 P 的完成顺序作为公平扫描顺序，但不同 request-generation 之间不设置严格 FIFO 队头门槛：暂时不可行或进入 Host 的旧 snapshot 不得阻塞后续可行 snapshot 直接进入 D。每个 snapshot 自身的所有权转移仍必须保持原子和有序。
- 这条慢路径的目的，是让已经完成 Prefill 的 KV 不阻塞后续 Prefill，也不长期占用 P HBM。

### 2.2 一轮 Decode 完成：D → P

Decode 完成一轮后，需要准备把该 request-generation 的完整 parent KV 传回 P，同时执行工具调用。工具快慢阈值默认为 2 秒，可由实验配置调整。

#### 快路径

- 如果工具在阈值内返回，且 P 可以为完整下一轮 workset 分配空间，则使用 NVLink/NIXL 直接把 parent KV 从 D HBM 传到 P HBM。
- 从工具返回时刻开始，P admission、Direct receiver 建立与传输启动共用同一个 Direct deadline，默认 2 秒；P claim 不得重新开始一个新的 deadline。deadline 内尚未成功启动 Direct 时立即转 Slow。已经提交给传输层的 DMA 必须先按物理 fence 收敛，不能在仍可能访问源页时强行切换所有权。
- 完整 workset 包括 parent KV 和新增 tool-result Prompt 所需 KV 空间；不能只接收 parent，随后因没有 suffix 空间而死锁。
- P Router 只使用 KV 容量做选择：在能容纳完整 workset 的候选中选择剩余 KV tokens 最大的 P。剩余量必须扣除尚未反映到物理采样中的 Direct/Slow reservation，以避免多个并发 snapshot 选择同一个过期容量；不再把请求队列、Host Arena 压力或加权综合分数用于 D→P 选 P。
- Direct 成功后，P 对该 snapshot 取得所有权，D 立即释放对应 HBM KV。

#### 慢路径

- 如果工具在阈值内没有返回，或者从工具返回开始的统一 Direct deadline 内仍未成功启动传输（例如 P 暂时没有完整 workset 空间），立即转入慢路径；一个失败的 Direct 不得阻塞后续 Direct。
- Slow 使用两阶段独立路由。进入 Slow 时只比较各 D→P Shared Host Arena 的可用字节数（容量减去已用空间和尚未反映到物理账本的 Host reservation），选择 Host 剩余空间最大的 Arena；此时不按 P HBM 压力选计算节点。
- D 通过 PCIe 将完整 snapshot offload 到选中的 D→P Shared Host Arena；tool 执行与 KV offload 并行。Host Arena 的物理落点不绑定后续 Prefill P；同机 P 可以通过 ledger 中的 `arena_path + offset` 读取该完整 extent。
- Host snapshot durable 后，D 必须立即释放对应 HBM KV，不能等待工具结束，也不能等待 P 有空间。
- 工具返回且 Host snapshot durable 后，Router 根据各 P 当前剩余 KV tokens 选择 H2D/Prefill 节点；Host owner 与 recovery P owner 是两个独立的 ledger 角色。只有选中的 P 能申请完整 workset 时，才把 parent KV 从原 Host extent load 到该 P HBM 并进行增量 Prefill。
- Host→P load 完成、P 已取得完整 snapshot 后，立即释放 Shared Host Arena 中该 snapshot 的空间。
- Decode 节点继续使用 SGLang Radix Cache 共享前缀，但 request-generation 的传输和释放必须保持完整、引用安全。

#### P Router

- D→P Direct 在工具返回后只根据 P 的剩余 KV tokens 选择节点：先扣除已使用 KV 和未被物理采样覆盖的 Direct reservation，再选择剩余容量最大的可行 P。
- D→P Slow 的 D2H 落点只根据 Shared Host Arena 剩余字节选择；Host durable 后的 H2D/Prefill 落点再独立根据 P 的剩余 KV tokens 选择。不得把 Host 剩余量与 P HBM 剩余量合成为加权 pressure score，也不得因为 Host 落在某个 P 管理的 Arena 就把恢复计算固定到该 P。
- 选择 P 时必须同时考虑 parent KV 和新增 tool-result tokens 的完整 workset，防止 P 被 parent KV 塞满后无法 Prefill。
- TP 场景由逻辑 TP rank 0 做一次组级决策并广播；所有 rank 必须选择同一个逻辑 P 组。
- 若 Direct 失败，TP=1 可按设计回退到同 NUMA P 的慢路径；不得因重路由造成 snapshot 丢失或多重所有权。

### 2.3 Prefill 调度

- Prefill 继续使用 SGLang 原生调度，根据 P 等待队列和可用 KV token/workset credit 执行。
- Prefill 开始前不需要在 D 预留 KV 空间。
- Prefill 完成后，完整 KV 可短暂处于 P-ready HBM 状态。
- 如果 D 暂时不能接收，可在 P HBM 中保留一个有界 Direct grace；grace 后必须用因果更新的 D 负载重新检查。仍不可行时，该 snapshot 独立进入 P→D Shared Host Arena；Host durable 后释放 P HBM，而不是让 P-ready KV 无限堆积。一个 snapshot 的 Host 决策不得迫使同一 P 的后续 snapshot 一起进入 Host。

### 2.4 Late-binding D Router

- Prefill 完成后才为 snapshot 选择 D，不提前固定 D，也不要求 D 为 Prefill 预留 KV。
- Router 只在“完整 KV 传入后不会超过可用容量”的 D 中，选择预计负载最低的节点。
- 负载应综合 running/queued/transfer requests、正在使用与已预留的 KV tokens、预计新增 Decode 工作，而不是只比较请求数量。
- P→D Direct 成功后，立即释放 P HBM。
- D 暂时无空间时，使用 P→D Shared Host Arena；Host durable 后同样立即释放 P HBM。

### 2.5 Decode Radix Cache

- Decode 继续使用原生 Radix Cache，通过共享前缀减少物理 HBM 占用。
- 一个 snapshot 对外传输时必须能够表达其完整 page-aligned KV。
- snapshot 离开 D 后，释放该 snapshot 的独有 KV；共享前缀按引用计数/原生 Radix 生命周期保留，直到没有其他 request-generation 使用。

## 3. 解耦要求

Direct 与 Slow 必须有独立 I/O 队列，并与计算进度解耦。

### D 侧

- D→P Direct 与 Slow 完全解耦。Direct 失败后立即转 Slow，不能阻塞后续 Direct。
- Slow 不做全量 candidate 扫描：每个 D rank 维护 sticky active window，并用 round-robin progress budget 推进。配置上限默认 active window=8、每轮 budget=8，但物理上限按 DMA group 数裁剪；默认 4 个 DMA group，因此同时只让 4 个 snapshot 取得 D2H pipeline，窗口外 snapshot 继续由 D HBM 完整持有，不会取得部分 Host ownership。一次 progress visit 对一个 snapshot 最多推进一个物理阶段，不能把整个 snapshot 的 chunk 全部排入 CUDA 队列。
- D→P D2H 使用独立低优先级 CUDA stream。传输 worker 从进程内有界 registered-window cache 取得最终 request extent 所覆盖的窗口，并在后台生成一次 CPU token-index mirror；CUDA 13 `cuMemcpyBatchAsync` 把相邻 allocator tokens 合并为 page/run 后批量提交 `HBM→registered final extent`，不再启动占用 SM 的 gather kernel。完成 event 同时是该 chunk 的物理完成与 Host durable fence，不再有 CPU commit memcpy。默认 batch 为 4096 tokens；窗口默认 8 GiB、通用 cache 默认 64 GiB。全局路由下一个进程可能访问 4 个 P domain，正式 Qwen3-8B 4P:4D 配置因此使用 1280 GiB 上限（`4 * (256+64) GiB`）；首次发现 arena 时只启动后台预注册线程，线程逐个 8 GiB 窗口 acquire/release，不能在传输 progress 线程内同步注册完整个 256 GiB arena。只能淘汰 refcount=0 的窗口，snapshot close 先释放窗口引用，不能逐请求 unregister。注册或 batch API 在任何 CUDA 提交前不可用时，才安全回退到原 gather→pinned-bounce/CPU-commit 路径。每条 lane、每个 snapshot 同时最多一个在途 CUDA batch；正式配置只使用两条经 Forward 隔离测试验证的 lane。
- Host durable 或 Direct 成功后，及时删除 D 上该 request-generation 的 KV，不做无必要的保守保留。
- P→D 接收、D→P Direct、D→P D2H、控制面推进均不得阻塞 Decode Forward。

### P 侧

- D→P Direct 与 Slow recovery 使用独立 I/O 队列，可并行恢复。
- D→P H2D 在 Host-copy worker 上注册最终 extent并生成目标 token-index mirror，随后用同一批量 copy-engine API 直接执行 `registered extent→paged KV`，不再经过 HBM staging/scatter kernel，也不做 pageable→bounce CPU prefetch。CUDA fence 未完成前不得 unregister/关闭 Host mapping、释放 Host extent或回收目标 HBM workset；batch API/注册不可用时沿用原 staging+scatter/双 bounce 回退语义。
- 正式 Qwen3-8B 配置在 transport worker 首次打开 arena 时预热全部 registered window；预热临时持有所有 window refs，完成后统一释放引用但保留 cache。该过程不得在 Forward/scheduler 线程执行；失败只能退回逐请求注册或既有 bounce fallback，不能改变 Host/HBM ownership 或发布任何 lifecycle 终态。
- P→D D2H/H2D 与 D→P 使用相同的 registered-final-extent + coalesced batch-DMA 语义：D2H event 完成后才能发布 Host durable并释放 P HBM，H2D event 完成且 Decode 绑定成功后才能释放 Host。四条慢路径只允许方向、owner 与 ledger 不同，不应复制出不同的数据搬运协议。
- Direct、Slow recovery 和新请求共享同一套原生 Prefill 计算准入顺序；请求类型不得形成 `Direct > Slow > New` 优先级。已经取得完整 workset 的请求可以作为物理所有权事实被及时消费，但不得再按 KV 来源类别重排。
- D→P load、P→D Direct、P→D D2H 和控制面推进均不得阻塞 Prefill Forward。
- P scheduler 应主要作为 ready workset 的消费者，不能承担重型目录扫描、阻塞式传输或长时间控制面操作。

### TP

- 上述解耦同时适用于 TP=1 和 TP>1。
- TP rank 0 维护唯一逻辑请求队列、状态机和路由决定；其他 rank 只执行广播的 shard 命令。
- 一个逻辑 snapshot 的所有 rank 必须一起 claim、一起选择路径、一起提交、一起释放；禁止部分 rank Direct、部分 rank Slow。

## 4. 物理所有权状态

每个 request-generation snapshot 在任意时刻只能处于以下一种主要物理所有权状态：

1. `P_HBM_OWNED`
2. `P2D_HOST_OWNED`
3. `D_HBM_OWNED`
4. `D2P_HOST_OWNED`
5. 正在进行受 fence 保护的原子 ownership handoff
6. `TERMINAL/RECOMPUTE_REQUIRED`

控制面 marker、ledger entry、reservation 和 tentative Host extent 不等同于 KV 所有权。只有完整 grant、物理传输 fence 和状态提交完成后，所有权才能转移。上一所有者必须在转移完成后及时释放；转移失败必须明确回到旧所有者或进入完整重算，不能形成双重所有权或无人所有。

## 5. Shared Host Arena 容量与驱逐

- 只有 D→P Shared Host Arena 允许驱逐：达到预警线（默认约 90%）时才触发 request-generation 级驱逐，驱逐到75%。
- 驱逐单位必须是完整 request-generation；TP 的所有 shard 一起驱逐。
- 正在传输、已被消费者 claim 或尚未完成物理 fence 的 snapshot 绝对不能驱逐。
- 当前没有更精确 profile 时，优先驱逐较短、重算成本较低的 snapshot；相同长度时可优先驱逐已等待更久的 snapshot。
- 驱逐后必须明确标记为 `RECOMPUTE_REQUIRED`，不能保留看似可恢复但实际缺页的 manifest。
- P→D Shared Host Arena 不设驱逐水位并默认使用 100% 物理容量。无法为完整 snapshot 分配连续 extent 时，Host 不取得 ownership，结果必须为 `RETAIN_P`：完整 KV 继续留在 P HBM 并反压该 P，等待 Host extent 或 D 容量恢复。P→D Host 一旦取得 ownership 就不能驱逐、舍弃或退回 Direct，必须完成到 D 的交付或显式 fail-closed。
- 大容量 arena 默认使用普通 CPU DRAM 的匿名 `memfd`，`/dev/shm` 仅保存小型 control ledger/marker；因此数据容量不受 `/dev/shm` mount 上限约束。下一轮 4P:4D、TP=1 实验配置为每个 P 的 D→P arena 256 GiB、P→D arena 64 GiB，两方向容量与 credit 完全独立。
- 当前新方法只使用 Shared Host Arena，不依赖原生 SGLang HiCache/Mooncake 作为生命周期兜底。

## 6. 验收标准（每次修改必须逐项检查）

1. **唯一所有者**：每个 snapshot 任意时刻只有一个物理所有者。
2. **P→D Direct 释放**：P→D Direct 成功后，立即释放 P HBM。
3. **P→D Host 释放**：P→D Host durable 后，立即释放 P HBM，不等待 D 有空间。
4. **D→P Host 释放**：D→P Host durable 后，立即释放 D HBM。
5. **进度解耦**：Direct、Slow、Decode Forward 和 Prefill Forward 互不等待；控制面或 I/O 失败不得停止另一条数据路径。
6. **TP 原子性**：TP 所有 rank 一起 claim、一起提交、一起释放，不分叉。
7. **父 KV 复用正确性**：父 KV 回传复用率应接近 100%；只允许 page 对齐尾部最多 `page_size - 1` tokens 重算，显式驱逐/失败重算必须单独统计。
8. **修改门禁**：每次修改必须先通过生命周期与故障注入测试，再由独立审计检查状态机；只有审计 GO 后才能运行 GPU 测试。

## 7. 修改前后检查模板

每次改动前必须回答：

1. 本次问题发生在哪个 snapshot ownership 状态？
2. 当前物理所有者是谁？修改后所有权何时、依据哪个 fence/CAS 转移？
3. 成功、超时、容量不足、取消、进程退出分别如何结束？
4. 是否会让 P→D、D→P、Direct、Slow、Forward/Prefill 中任一条路径等待另一条？
5. TP 的所有 rank 是否只有一个逻辑决策者，并保持完全一致？

每次改动后必须提供：

- 对应的生命周期/竞态/容量不足/取消测试；
- 上述 8 项验收标准的逐项结论；
- 独立审计 GO/NO-GO；
- GPU 测试中的 ownership 数量守恒，例如 `queued = durable = source_release`，以及未完成项的明确数量和状态。
