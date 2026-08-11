# Agentic PD 慢通路 NUMA Relay 改动交接报告

> 注意：本文只描述最终一层 NUMA relay 的实现。若要理解从最早
> `D Host HiCache -> Mooncake -> P Host HiCache` 到当前 Shared Host + relay 的完整
> 迁移历史，请以
> [`SLOW_PATH_EVOLUTION_MOONCAKE_TO_NUMA_RELAY.md`](./SLOW_PATH_EVOLUTION_MOONCAKE_TO_NUMA_RELAY.md)
> 为准。

更新时间：2026-08-07  
实现环境：`/homes/siqic/anaconda3/envs/pd`  
实验代码：`/homes/siqic/slime/examples/pd`

## 1. 改动目标

原有慢通路是：

```text
任意 D HBM --该 D 的 PCIe D2H--> P 所属 NUMA 上的 Shared Host Arena
```

Shared Host Arena 位于 P 所属 NUMA 的 CPU DRAM，并通过 `/dev/shm` 文件让 P、D
映射同一份物理 Host 内存。D 与 Arena 同 NUMA 时路径正常；D 位于另一个 NUMA
时，D2H 写入需要跨 CPU socket/NUMA fabric，吞吐明显下降。

本次改成：

```text
Arena-local D:
  D HBM --本地 PCIe D2H--> Shared Host Arena

Remote-NUMA D:
  source D HBM --NIXL/NVLink--> Arena-local relay D 固定 HBM slot
               --relay 本地 PCIe D2H--> Shared Host Arena

Relay 不可用、已过载或预计不如直接写快：
  source D HBM --跨 NUMA D2H--> Shared Host Arena
```

P 不作为 relay，避免慢通路写入占用 P 的计算和 Host→P HBM 关键读取路径。

## 2. 完整生命周期

### 2.1 P 分配完整 Host extent

1. Decode 一轮结束后，source D 仍持有该 request-generation 的完整 KV。
2. D 在 `/dev/shm` ledger 中发布 `OFFERED`，包含 snapshot ID、token 数、字节数、
   source NUMA、Arena NUMA、NIXL bootstrap 地址等元数据。
3. P 检查 Shared Host Arena 水位并为完整 snapshot 分配一个连续 extent。
4. P 发布 grant，状态进入 `HOST_WRITING`。这里没有 page 级半成品可见性：只有完整
   snapshot 全部写完后才发布 `HOST_READY`。

### 2.2 路径选择

路径选择在 ledger 文件锁内原子完成，避免多个 source D 同时看到相同的 relay
队列长度。

- source NUMA 等于 Arena NUMA：直接选择 `direct_local`。
- source 位于远端 NUMA：只考虑以下 relay：
  - relay NUMA 等于 Arena NUMA；
  - relay 不是 source 自己；
  - heartbeat 未超过 stale timeout；
  - relay 已发布合法固定 slot。

估算公式为：

```text
relay ETA = (relay_queued_bytes + snapshot_bytes) / relay_local_D2H_bw
            + snapshot_bytes / NVLink_bw

direct ETA = snapshot_bytes / direct_cross_NUMA_D2H_bw
```

选择 relay ETA 最小的 relay。只有 `relay ETA < direct ETA` 才走 relay，否则直接
跨 NUMA 写 Host。选中 relay 后，snapshot 字节数立即原子加入其 `queued_bytes`；
完成或失败时扣除。

默认估算参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `SGLANG_AGENTIC_KV_RELAY_D2H_GIBPS` | 21.0 GiB/s | Arena-local relay D2H |
| `SGLANG_AGENTIC_KV_DIRECT_CROSS_NUMA_GIBPS` | 7.45 GiB/s | source 直接跨 NUMA D2H |
| `SGLANG_AGENTIC_KV_RELAY_NVLINK_GIBPS` | 220.0 GiB/s | source D 到 relay D |
| `SGLANG_AGENTIC_KV_RELAY_STALE_SECONDS` | 5 s | relay heartbeat 失效阈值 |

这些数值是调度模型输入，不是本次 snapshot 的实测完成时间；后续可以替换成在线
EWMA 带宽。

### 2.3 Relay 分块传输

Arena-local D 启动时注册为 relay，并永久预留固定数量的小 HBM slot，而不是为每个
等待 snapshot 预留完整 KV：

```text
source D 完整 KV（仍保留）
      │
      ├─ chunk 0 ─NIXL/NVLink─> relay slot 0 ─local D2H─> Host range 0
      ├─ chunk 1 ─NIXL/NVLink─> relay slot 1 ─local D2H─> Host range 1
      └─ 后续 chunk 循环复用 slot
```

默认请求 2 个 64 MiB slot。Qwen3-8B、page size 64 的验证配置中，受 KV page
几何对齐影响，每个 slot 是 448 tokens，两个 slot 实际共约 126 MiB。

relay 的工作过程：

1. 从分配给自己的队列中按创建时间 claim 一个 snapshot。
2. 映射 P 已分配的同一个 `/dev/shm` Host extent。
3. 为当前 chunk 发布 receiver-ready 元数据。
4. source 使用反向 NIXL sender，把对应完整 KV page 传入 relay 固定 slot。
5. relay 等待 NIXL 完成，再用本地 D2H stream 把 slot 写入 Host extent 对应 range。
6. CUDA event 完成后 ACK 当前 chunk，再复用 slot 处理下一块。
7. 最后一块完成后原子发布 `HOST_READY`，清空 relay queue accounting。

当前实现虽然有两个轮换 slot，但 chunk 处理仍是顺序状态机；尚未实现“slot 0 D2H
时同时向 slot 1 做 NIXL”的真正双缓冲流水线。固定 slot 的主要收益目前是限制 HBM
占用，而不是隐藏全部传输延迟。

### 2.4 Source D 释放条件

source D 在以下条件之前始终保留完整原始 KV：

```text
所有 chunk NIXL 完成
→ relay 所有 D2H CUDA event 完成
→ ledger 原子发布 HOST_READY（或更晚状态）
→ source D 收到 host_ready
→ source D 才释放原 KV
```

因此 relay 的中间 slot、部分 Host range 或单个 chunk ACK 都不能触发 source KV
释放。relay 失败时先停止/排空可能在途的 DMA，再把 write mode 改为
`direct_cross_numa`，由仍持有完整 KV 的 source D 兜底重写整个 snapshot。

### 2.5 P 读取与 Host 释放

下一轮 request 到达 P 后：

1. P 根据 request-generation 查到 `HOST_READY` snapshot。
2. 从 Shared Host Arena load 完整 snapshot 到 P HBM。
3. H2D 完成后插入并 pin P 的 Radix Cache。
4. 确认 GPU snapshot 可用后立即释放 Shared Host extent。
5. P 完成本轮新增 prompt/tool-result prefill，再按原生 PD 路径发送给负载合适的 D。

Shared Host Arena 是 CPU DRAM 热层；Mooncake 不再是每轮 D→P 回传的必经路径。
只有 Host Arena 达到水位线时，少量冷 snapshot 才继续 spill 到 Mooncake。

## 3. Relay 负载均衡和降级

目前负载信号是每个 relay 的 `queued_bytes`，不是请求数量。这样大 snapshot 会比小
snapshot 产生更高压力。

行为如下：

- 多个 Arena-local relay：选预计完成时间最短的一个；相同 ETA 时按 queued bytes、
  relay ID 稳定打破平局。
- relay 过载：当排队后的 relay ETA 不再优于直接跨 NUMA ETA，source 自动直写。
- relay 心跳过期：不再分配新任务。
- relay 处理中失败：完整 snapshot 回退给 source 直写。
- source 已在 Arena NUMA：永远不增加 NVLink relay hop。

## 4. 内存和元数据改动

### 4.1 固定 HBM slot 计入严格内存核算

relay slot 从正常 KV allocator 中永久分配。为避免 SGLang 把它误判为 KV 泄漏，
`reserved_token_count` 已纳入 idle/busy scheduler memory checker。没有关闭严格内存
检查。

### 4.2 Relay-only metadata mailbox

relay receiver 复用原生 P→D NIXL manager，避免同一完整 KV pool 被 UCX/NIXL 注册
第三次。原生 manager 有 10 组 request metadata，而反向 relay sender 只需要一个
完成通知。

现在从 `ReqToMetadataIdxAllocator` 永久拿出一行作为 relay-only mailbox，正常请求永远
不会拿到该行。反向 sender 只向该保留行写 1-byte sentinel，因此不会覆盖 live
request metadata。

### 4.3 控制面大小

ledger 位于 `/dev/shm`，只保存 request-generation manifest、relay registry、队列字节
数、chunk 序号和状态，不在 JSON 中保存 KV 数据。更新使用 `flock`，同节点多进程
原子可见。

## 5. 主要代码位置

修改后的 SGLang 位于 `pd` conda 环境：

| 文件 | 主要内容 |
|---|---|
| `sglang/srt/disaggregation/agentic_host_staging.py` | ledger、路径选择、Shared Host extent、relay worker、source sender、P H2D |
| `sglang/srt/disaggregation/decode_kvcache_offload_manager.py` | 创建 D staging client、挂接 relay、轮询和 source KV 释放 |
| `sglang/srt/managers/scheduler.py` | 复用原生 Decode NIXL manager 并保留 relay metadata row |
| `sglang/srt/managers/scheduler_runtime_checker_mixin.py` | relay 固定 KV slot 内存核算 |
| `sglang/srt/environ.py` | relay 环境变量定义 |

启动和测试代码：

| 文件 | 主要内容 |
|---|---|
| `examples/pd/scripts/new_method/internal/run_agentic_pipeline.sh` | 默认启用 relay 及带宽/slot 参数 |
| `examples/pd/scripts/new_method/internal/run_pd_servers.sh` | 探测 GPU NUMA、给每个 D 配 relay ID、使用独立反向 NIXL 端口 |
| `examples/pd/tests/test_agentic_host_staging.py` | ledger、完整 ACK、relay 选择和 direct fallback 测试 |

反向 NIXL listener 默认从 61000 开始分配，避开常见 Linux ephemeral port 范围
32768–60999。仍支持显式 `AGENTIC_DIRECT_PORT_OFFSET` 兼容旧启动方式。

## 6. 真实验证结果

验证拓扑：

```text
P GPU0 / NUMA0
relay D GPU2 / NUMA0
source D GPU4 / NUMA1
```

验证 snapshot：

- 576 tokens；
- 84,934,656 bytes；
- relay 预测 4.126 ms；
- 直接跨 NUMA 预测 10.618 ms；
- 自动选择 `decode-0-gpu-2`；
- 分块为 448 + 128 tokens；
- 两块 NIXL、relay-local D2H 和 ACK 全部完成；
- ledger 最终 `relay_completed_tokens=576/576`、`state=consumed`；
- source D 在 `relay_host_complete` 后释放；
- 下一轮返回 `cached_tokens=576`，完整命中 page-aligned 父代 KV；
- P、relay D、source D 在清理后均保持健康；
- terminal generation 没有进入慢通路。

验证覆盖：Python 编译通过、Bash 语法检查通过、38 项测试通过、真实 1P:2D serving
链路通过。

这轮属于历史链路验证而非正式吞吐结果；清理时已删除其运行目录，只在本文保留结论。

## 7. 当前明确限制和后续建议

1. V1 仍要求 TP=1、同节点 `/dev/shm` 控制面、NIXL direct/relay；不支持 EAGLE
   direct transfer。
2. 只有与 Shared Host Arena 同 NUMA 的 D 可以成为 relay；如果该 NUMA 没有 D，
   只能直接跨 NUMA 写 Host。
3. 选择公式使用静态配置带宽。建议后续根据每个 relay 的 NIXL、D2H CUDA event
   实测值维护 EWMA，并把 `queued_bytes / bandwidth` 替换成更准确的 remaining ETA。
4. 每个 relay 当前一次只处理一个 snapshot，chunk 顺序执行。后续可实现真正双缓冲，
   重叠 source→relay NIXL 与上一块 relay→Host D2H。
5. relay crash 的新任务由 heartbeat 排除；已 claim 任务目前依赖接收超时后回退。后续
   可增加更短的 lease/watchdog。
6. 第一版 Mooncake snapshot 仍独立存储完整 KV，没有实现跨 request page 共享；本地
   D Radix Cache 保持原生行为。
7. 带宽模型必须按机器重新校准，尤其是 GPU/CPU NUMA 拓扑不同或 NVLink 不完整的
   节点。

## 8. 交接时最重要的正确性约束

后续修改时不要破坏以下四条：

1. source D 只有在完整非 D 副本可见后才能释放原 KV。
2. `HOST_READY` 只能在所有 chunk 的 D2H event 完成后发布。
3. relay 失败切 direct 前必须确认没有 DMA 仍在写旧 Host extent。
4. P 只有在 Host→HBM 完成并 pin GPU snapshot 后才能释放 Host extent。

这四条分别保证 source、Host、relay 和 P 之间不会出现部分 snapshot、use-after-free
或下一轮重新 prefill。
