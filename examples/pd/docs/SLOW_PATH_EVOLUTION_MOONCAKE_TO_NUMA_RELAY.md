# Agentic PD KV 慢通路迁移报告：从 Mooncake 必经路径到 NUMA Relay

更新时间：2026-08-07  
当前实现环境：`/homes/siqic/anaconda3/envs/pd`  
实验与启动代码：`/homes/siqic/slime/examples/pd`

## 0. 给接手 Agent 的结论

请不要再把当前慢通路理解成：

```text
D Host HiCache -> Mooncake -> P Host HiCache
```

这条链路是最早的实现，也是此前带宽很低、Mooncake 容量容易被长期占满时采用的
模型。当前实现已经演进为：

```text
快通路：
D HBM --NIXL/NVLink--> P HBM

当前热慢通路：
D HBM --D GPU 的 PCIe D2H--> Shared P-side Host Arena
Shared Host Arena --P GPU 的 PCIe H2D--> P HBM

远端 NUMA D 的优化慢通路：
source D HBM --NIXL/NVLink--> Arena-local relay D 的固定 HBM slot
             --relay D 的本地 PCIe D2H--> Shared P-side Host Arena

冷溢出：
Shared Host Arena --仅在 Host 水位压力下--> Mooncake
```

因此，Mooncake **没有被删除**，但已从“每个 multi-turn generation 的必经回传数据
面”降级为“Shared Host Arena 的冷溢出层”。D 侧原有本地 Host HiCache 在
`D-hostless` 模式下也不再为每个 D 初始化和占用大块内存。

下面按三代实现解释为什么要这样改、现在每一步如何工作，以及哪些测试已经完成。

## 1. 第一代：Mooncake 是 D→P 回传的必经慢通路

### 1.1 原始数据通路

最早为实现 multi-turn 请求的 D→P KV 回传，完整路径是：

```text
┌──────────────── Decode worker ────────────────┐
│ D GPU KV pool                                 │
└──────────────────┬────────────────────────────┘
                   │ CUDA stream，GPU→Host
                   │ D GPU 的 PCIe D2H
                   ▼
          D Local Host HiCache（L2）
          每个 D 约 56 GiB，CPU DRAM
                   │
                   │ SGLang MooncakeStore.batch_put_from()
                   │ Mooncake Transfer Engine，protocol=tcp
                   ▼
       Mooncake Shared Segment（L3）
       约 256 GiB，共享 CPU DRAM 存储段
                   │
                   │ SGLang MooncakeStore.batch_get_into()
                   │ Mooncake Transfer Engine，protocol=tcp
                   ▼
          P Local Host HiCache（L2）
          约 128 GiB，CPU DRAM
                   │
                   │ P GPU 的 PCIe H2D / async load
                   ▼
┌──────────────── Prefill worker ───────────────┐
│ P GPU KV pool                                 │
│ 命中历史 KV，只计算新增 tool-result tokens    │
└───────────────────────────────────────────────┘
```

Mooncake Master 只负责元数据和控制；KV 数据由 Mooncake Transfer Engine 在注册的
内存段之间移动，不经过 Master 进程。这里逻辑上和物理上都有多份副本：D Local Host、
Mooncake L3、P Local Host。

### 1.2 当时测到的带宽

以下是旧链路上的历史实测，不是硬件理论上限：

| 阶段 | 历史实测结果 |
|---|---:|
| 3 个 D 聚合 `batch_put_from` | 平均 0.836 GiB/s，P90 1.148 GiB/s，最大 1.397 GiB/s |
| Mooncake→P `batch_get_into` | 平均 0.0757 GiB/s（约 77.5 MiB/s），P90 0.164 GiB/s，最大 0.582 GiB/s |
| 7 个 D 与 P GET 同时竞争时的 PUT | 约 0.55 GiB/s |
| 7 PUT 竞争时的 P GET | 约 0.08 GiB/s |

这里“聚合”表示同一观测窗口内所有 worker 完成的总字节数除以墙钟时间，不是每个 D
都各自达到该速率。它反映的是整条 Mooncake Store 软件路径在当时配置和竞争条件下
提供的有效吞吐。

### 1.3 第一代为什么成为瓶颈

1. 每轮反向 KV 都强制经过 D Host、Mooncake、P Host 三层，复制次数多。
2. PUT 与 GET 共享 Mooncake Store/Transfer Engine 资源；多个 D PUT 时，P 的关键 GET
   会被明显挤压。
3. 每个 D 都初始化大块本地 Host HiCache，D 数量增加时 Host DRAM 占用线性增长。
4. Mooncake 的原生被动驱逐在高水位后才批量清理；长时间重载下容易积累大量等待
   tool 的 snapshot。
5. 原生物理 page 驱逐和 agentic 请求的完整 generation 生命周期不完全一致，可能
   出现某个 generation 的一部分仍占空间、但已无法形成可复用的完整历史。
6. 新 KV 在接近高水位时写入后可能很快被驱逐，产生“写进去了但下一轮尚未来得及
   使用”的 thrashing。

这促成了核心设计变化：热路径不应该强制经过 Mooncake；Mooncake 更适合保留等待时间
长、暂时不会立即被 P 使用的冷 snapshot。

## 2. 第二代：Shared P-side Host Arena 取代 Mooncake 必经链路

### 2.1 快慢分流

Decode 一轮结束后，工具执行与 KV 搬运可以并行。当前设计把路径分为：

```text
若 Tool 在阈值内返回（设计默认约 0.2 s），且 P HBM 预留成功：
  D HBM --NIXL/NVLink--> P HBM                  [快通路]

若 Tool 超时，或 P HBM 暂时没有可承诺空间：
  D HBM --PCIe D2H--> Shared P-side Host Arena [热慢通路]
```

快通路绕开 Host 和 Mooncake。热慢通路也不再先写 D Local Host，再 PUT Mooncake，再
GET 到 P Host，而是直接写 P、D 进程共同映射的一块 Shared Host Arena。

### 2.2 Shared Host Arena 的数据流

```text
D GPU KV pool
    │
    │ D GPU 自己的 PCIe D2H
    ▼
Shared P-side Host KV Arena
CPU DRAM；/dev/shm 映射；P 负责容量与生命周期
    │
    │ 等待 Tool 完成
    │
    │ P GPU 自己的 PCIe H2D
    ▼
P GPU KV pool
    │
    │ Prefill 新增 prompt/tool-result
    ▼
P HBM --原生 PD NIXL/NVLink--> 负载合适的 D HBM
```

这里 Host 可以共享，因为它是同一台主机上的 CPU DRAM，P 和所有 D 进程映射同一块
物理内存。D 写 Host 使用的是各自 D GPU 的 D2H DMA/PCIe 链路；P 读 Host 使用的是
P GPU 的 H2D DMA/PCIe 链路。二者不会争用同一块 GPU 的 PCIe 方向，但会受 CPU NUMA
拓扑、DRAM 带宽和 Host Arena 所在 NUMA 位置影响。

### 2.3 request-generation 完整性和释放顺序

当前逻辑单位是 request-generation 的完整 snapshot；物理数据仍按 page/chunk 搬运，
但不会把部分 snapshot 发布给下一阶段。

```text
D 完成一轮 Decode
  -> 发布完整 snapshot OFFER
  -> P 为完整 snapshot 分配 Host extent
  -> D 写入全部 KV
  -> 全部 D2H CUDA event 完成
  -> 发布 HOST_READY
  -> D 收到完整 Host ACK 后才释放原 D HBM KV

下一轮到达 P
  -> P 将完整 snapshot 从 Host load 到 P HBM
  -> H2D 完成并把 snapshot 插入/固定到 P Radix Cache
  -> 才释放 Shared Host extent
  -> P 只计算本轮新增 tokens
  -> 完整 KV 按正常 P→D 路径传到 D
```

最终答案不会再有下一轮，因此 terminal generation 不进入慢通路。

`D-hostless` 模式下，D 不再为这条反向路径初始化原来的大容量 D Local Host HiCache；
D 的快照直接落入共享 Arena，从而避免 `N × D Host HiCache` 的内存占用。

### 2.4 Mooncake 在第二代之后的位置

Mooncake 仍保留，但只作为 P-side Shared Host Arena 的冷溢出层：

```text
Shared Host Arena 使用率达到 high watermark（当前默认 80%）
  -> 按完整 request-generation 选择冷 snapshot
  -> 完整写入 Mooncake
  -> Mooncake commit 成功是“替代副本完整”的 ACK
  -> 才释放原 Shared Host extent

下一轮需要冷 snapshot
  -> 从 Mooncake 完整恢复
  -> P HBM load、pin 成功
  -> 删除/释放旧层副本
```

当前默认水位为 low/high/hard = 70%/80%/90%。Mooncake 不是消失了，而是只有 Host
压力下的少量冷 snapshot 才使用它。这样把低带宽 Store 路径从正常关键路径上移走。

## 3. 第三代：解决 Shared Host Arena 的跨 NUMA 写入

### 3.1 新问题

Shared Host Arena 位于 P 所在 NUMA 的 CPU DRAM。与 Arena 同 NUMA 的 D 可以本地
D2H；另一个 NUMA 上的 D 直接写同一 Arena 时，要经过 CPU socket/NUMA interconnect，
带宽明显下降。

历史带宽测试展示了这一差异：

| 拓扑 | 聚合 D→Host | Host→P HBM |
|---|---:|---:|
| 1P:2D，P0+D1+D2 均在 NUMA0 | 43.89 GiB/s | 15.71 GiB/s |
| 1P:3D，另增加一个跨 NUMA 的 D4 | 21.41 GiB/s | 16.88 GiB/s |

第二行 D→Host 下降的原因不是 D 数量越多 PCIe 越慢，而是 D4 的写目标仍在 NUMA0，
跨 socket 写入使聚合路径受到 NUMA fabric、远端内存访问和调度竞争影响。

### 3.2 当前 NUMA-aware relay 通路

当前实现按 source D 所在 NUMA 选择路径：

```text
情况 A：source D 与 Arena 同 NUMA

source D HBM --本地 PCIe D2H--> Shared Host Arena


情况 B：source D 位于远端 NUMA，relay 更快

source D HBM
   --NIXL/NVLink-->
Arena-local relay D 的固定 HBM slot
   --relay D 本地 PCIe D2H-->
Shared Host Arena


情况 C：relay 不可用、过载、心跳过期，或预计不如直写

source D HBM --跨 NUMA D2H--> Shared Host Arena
```

P 不充当 relay。这样慢通路写入不会占用 P 的 KV HBM 和计算调度，P 的 PCIe/H2D
继续服务更关键的 `Shared Host Arena -> P HBM` 恢复。

### 3.3 Relay 调度

只考虑满足下列条件的 relay：

- 与 Shared Host Arena 位于同一 NUMA；
- 不是 source D 自己；
- heartbeat 未过期；
- 已成功发布固定 HBM slot。

在 `/dev/shm` ledger 文件锁内，以字节而不是请求数做原子排队：

```text
relay_eta = (relay_queued_bytes + snapshot_bytes) / relay_local_d2h_bw
            + snapshot_bytes / nvlink_bw

direct_eta = snapshot_bytes / direct_cross_numa_d2h_bw
```

选择 ETA 最小的本地 relay；只有 `relay_eta < direct_eta` 才使用 relay，否则 source D
直接跨 NUMA 写 Host。当前默认估算输入为：

| 参数 | 默认调度值 |
|---|---:|
| relay-local D2H | 21.0 GiB/s |
| 直接跨 NUMA D2H | 7.45 GiB/s |
| source→relay NVLink/NIXL | 220.0 GiB/s |

这些是路径选择模型的校准值，不是每条请求实时测得的带宽；后续可以改为在线 EWMA。

### 3.4 固定 relay slot 与正确性

relay 不为每个 snapshot 预留完整 HBM，只永久保留少量固定 slot。默认请求
`2 × 64 MiB`；Qwen3-8B、page size 64 的实测几何中，两槽实际总计约 126 MiB，
每槽容纳 448 KV tokens。

```text
source D 保留完整原 KV
  -> chunk 0 --NIXL--> relay slot 0 --local D2H--> Host range 0
  -> chunk 1 --NIXL--> relay slot 1 --local D2H--> Host range 1
  -> 后续循环复用 slot
  -> 全部 chunk 的 D2H event 完成
  -> HOST_READY
  -> source D 才释放完整原 KV
```

当前两个 slot 以顺序状态机交替复用，尚未真正重叠“slot 0 D2H”和“slot 1 NIXL”。
因此它目前主要限制 relay HBM 占用，还没有完全实现双缓冲流水化带宽收益。

relay 中途失败时，必须先停止/排空在途 DMA；因为 source D 仍持有完整原 KV，可以
回退为 direct cross-NUMA，重新写完整 snapshot，不会让部分 Host 数据被误认为完整。

## 4. 三代慢通路对照

| 项目 | 第一代：Mooncake 必经 | 第二代：Shared Host | 当前：Shared Host + NUMA relay |
|---|---|---|---|
| 正常 D→P 路径 | D Host→Mooncake→P Host | D HBM→Shared Host→P HBM | 本地直写；远端经 local-D relay 或直写兜底 |
| Mooncake 是否必经 | 是 | 否 | 否 |
| 热路径 Host/Store 副本 | D Host、L3、P Host | 一份 Shared Host | 一份 Shared Host；relay slot 仅瞬时分块 |
| D 本地大容量 HiCache | 每个 D 都需要 | D-hostless 下不需要 | D-hostless 下不需要 |
| 主要瓶颈 | Store PUT/GET 及竞争 | 跨 NUMA D2H | relay 排队、P H2D、静态路径估计 |
| Mooncake 的职责 | 所有 reverse KV | Host 压力下冷 spill | Host 压力下冷 spill |
| 完整性单位 | 物理 page 为主 | 完整 request-generation manifest | 完整 request-generation manifest |
| D HBM 释放点 | 远端副本确认后 | 完整 HOST_READY 后 | 完整 HOST_READY 后 |

最重要的架构变化不是“给 Mooncake 加了一个 relay”，而是：

1. 把 Mooncake 从热慢通路移出；
2. 用一份 P-side Shared Host snapshot 作为正常慢路径落点；
3. 再用 NUMA-aware GPU relay 优化远端 D 写这份 Shared Host snapshot 的方式；
4. Mooncake 仅在 Shared Host Arena 水位压力下接收冷 snapshot。

## 5. 当前代码位置

修改的是 `pd` 环境内的 SGLang：

| 文件 | 当前职责 |
|---|---|
| `sglang/srt/disaggregation/agentic_host_staging.py` | request-generation ledger、Shared Host Arena、水位/冷 spill、路径选择、relay worker、D source client、P H2D |
| `sglang/srt/disaggregation/decode_kvcache_offload_manager.py` | D staging client、relay 挂接、慢通路进度以及 source KV 的安全释放 |
| `sglang/srt/managers/scheduler.py` | P Host manager、复用 Decode NIXL manager、relay metadata mailbox |
| `sglang/srt/managers/scheduler_runtime_checker_mixin.py` | 将固定 relay HBM slot 纳入严格 KV 内存核算 |
| `sglang/srt/environ.py` | agentic slow-path 和 relay 环境变量 |

仓库中的入口和测试：

| 文件 | 当前职责 |
|---|---|
| `examples/pd/scripts/new_method/internal/run_agentic_pipeline.sh` | 开启 Shared Host、D-hostless、Mooncake cold spill 和 relay 参数 |
| `examples/pd/scripts/new_method/internal/run_pd_servers.sh` | 探测 GPU NUMA、分配 relay ID/端口并启动 1P:nD |
| `examples/pd/tests/test_agentic_host_staging.py` | manifest、ACK/释放、水位 spill、relay 选择和 fallback 单测 |
| `examples/pd/docs/NUMA_RELAY_SLOW_PATH_REPORT.md` | 只聚焦第三代 NUMA relay 的实现细节 |

## 6. 已完成验证

真实链路验证拓扑：

```text
P GPU0 / NUMA0
relay D GPU2 / NUMA0
remote source D GPU4 / NUMA1
```

验证结果：

- snapshot：576 tokens，84,934,656 bytes；
- 调度预测 relay 4.126 ms，direct cross-NUMA 10.618 ms，自动选择 relay；
- 分块：448 + 128 tokens；
- source→relay NIXL、relay-local D2H、完整 ACK 均完成；
- ledger 最终 `relay_completed_tokens=576/576`、`state=consumed`；
- source D 只在 `relay_host_complete` 后释放；
- 下一轮 `cached_tokens=576`，完整命中 page-aligned 父代 KV；
- terminal generation 未进入慢通路；
- Python 编译、Bash 语法、38 项测试和真实 1P:2D serving 链路通过；
- 测试服务已停止，GPU 已释放。

这轮属于历史链路验证而非正式吞吐结果；清理时已删除其运行目录，只在本文保留结论。

## 7. 当前限制与不能误解的地方

1. 当前 V1 是 TP=1、同节点 `/dev/shm` 控制面、NIXL direct/relay；不支持 EAGLE
   direct transfer。
2. Mooncake cold spill 仍然存在，不能宣称 Mooncake 已被完全移除。
3. 第一版 Mooncake snapshot 独立保存完整 KV，尚未实现跨 request 的 page 共享；本地
   SGLang Radix Cache 仍保持原生行为。
4. relay 必须是 Arena-local D；若 P 所在 NUMA 没有 D，只能 direct cross-NUMA。
5. relay 带宽目前是静态校准值，尚未按实际 CUDA/NIXL 完成时间维护在线 EWMA。
6. 两个 relay slot 当前不是完全重叠的双缓冲流水线。
7. source D 在完整替代副本 ACK 前仍需临时保留原 KV；这不是长期预留，而是正确性
   所需的短生命周期 ownership。
8. P HBM load 完成并 pin 前不能释放 Host；Mooncake spill commit 前也不能释放被替代
   的 Shared Host extent。

## 8. 后续修改必须保持的四条不变量

1. 任意时刻必须至少存在一份完整、可恢复的 request-generation KV snapshot。
2. `HOST_READY` 只能在完整 snapshot 的所有 D2H event 完成后发布。
3. relay 失败切换 direct 前，必须确认旧 relay DMA 已停止或排空。
4. Host→Mooncake 的冷 spill 只有在 Mooncake 完整 commit 后才能释放 Host 原副本。
