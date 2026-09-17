# 多节点开发入口（未完成实机验收）

本入口不修改既有单节点 launcher，不自动 SSH、不停止他人的服务；运行前请确认所选GPU空闲。
提供计划、元数据共享检查、独立 worker/router supervisor、两轮 Direct/Slow smoke 和
显式 workload 启动。worker/router 仍受引擎集成能力 gate 保护；未集成的代码版本不会
因有启动脚本就被绕过。不能将 dry-run/CPU 测试当作 TP8 KV 正确性、RDMA 带宽或正式吞吐验收。

## V1 边界

- 一台机器一个逻辑 P 或 D TP 组；TP=1/2/4/8，整组 GPU 必须同节点。
- V1 一个逻辑 P TP 组，可有一个或多个逻辑 D TP 组。暂不接受多个 P 组，避免沿用
  单节点 NUMA 分区逻辑错误地约束异节点路由。模型支持 dense Qwen3，以及显式
  `model_family=minimax_m2` 的 MiniMax-M2.7 普通 GQA MoE（专家 EP=TP，组内通信）；
  不允许 Mamba、MLA、其他未经适配的 MoE、DP attention、TP组跨主机。
  MiniMax 的配置、CPU检查与远端 colocated SWE500 入口见 `MINIMAX_M27.md`；尚未GPU验收。
- P、D 的 TP 一致，各 rank 必须采用相同的模型/revision/dtype/page layout。
- 所有 engine ID 和 node ID 全局唯一。TP 决策和所有 shard 的 fence/ownership
  必须组级一致；不得把 TP rank 的 HBM 指针直接当作另一 rank 的地址。
- 快路径 HBM→跨节点 NIXL/UCX RDMA→HBM；慢路径先在**源节点**
  HBM→CPU DRAM，之后源 Host→跨节点 RDMA→目标 HBM。
- `control_root/run_id` 是跨节点可见的小型 POSIX 元数据目录，不放 KV payload。
  必须支持跨客户端 flock、O_EXCL、原子 hardlink/rename 及及时一致的读；普通 NFS 的远端修改
  不会触发本地 inotify，必须使用引擎 multi-node 控制面实现。只共享 `/dev/shm` 路径名无效。
- `local_root/run_id/engine_id` 是节点本地目录。Arena 默认 memfd DRAM，不应把
  KV 写入共享文件系统。远端必须使用 transport descriptor/endpoint，而不是
  在另一节点直接打开 `/proc/<pid>/fd/<fd>`。
- 所有 mailbox 先保留 shared control root；其中包含跨 P/D 的 TP fence，不能
  因 TP ranks 同节点就把整个 mailbox 目录全部改成本地目录。
- 仅 native HiCache/Mooncake 禁用；两条自定义 Host 通路是独立容量/队列。

## 配置

复制并编辑 `multinode.example.json`。示例用两个真实目标 IP，但**不代表确认空闲**。
修改模型/Python/仓库路径与 run ID；`host_ip` 必须是相互可达的控制/传输通告地址。
`ib_device` 可按各节点真实 NIC 显式填写，不从管理 IP 猜测 RDMA NIC。
`numa_nodes` 按每个 GPU 所属 NUMA 填写，留空不强行错误绑定。

示例 Host 容量是保守工程 smoke 值：每 rank D→P 8 GiB、P→D 4 GiB。
TP8 总量分别64/32 GiB，正式实验应依据实际 DRAM预算统一修改并记录。
所有 rank 使用组内统一的 `mem_fraction_static`；若有搜索等共驻服务，需降到该组最紧张
GPU可承受的值并与 baseline 对齐。不要自动启动搜索占满8张模型卡的某一张。

保留当前完整方法的拥堵反馈重算：`slow_congestion_recompute=true`，high=32/low=8，
显式记录在配置中。这是沿用策略参数，**没有声称已针对TP8调优**；快工具Direct失败时按Q
选择Slow/重算，工具本身慢仍走Slow。单独验证恢复链路时可以显式设false，但正式对比需
使用一致参数，不能将不同策略混成同一组结果。

```bash
# 在 slime 仓库运行；只生成计划，不初始化 CUDA。
bash tools/dualpd/multinode.sh plan --config /path/experiment.json

# 分别在 P、D 节点运行，建立极小的共享文件可见性探针。
bash tools/dualpd/multinode.sh fs-publish --config /path/experiment.json --node-id node-p
bash tools/dualpd/multinode.sh fs-publish --config /path/experiment.json --node-id node-d
# 两个节点均执行；检查同一配置/不同主机、远端O_EXCL/hardlink创建及rename记录可见。
bash tools/dualpd/multinode.sh fs-verify --config /path/experiment.json

# P 节点持锁30秒，看到 LOCK_HELD 后立即在 D 节点运行 probe。
bash tools/dualpd/multinode.sh fs-lock-hold --config /path/experiment.json
bash tools/dualpd/multinode.sh fs-lock-probe --config /path/experiment.json
```

每个node只publish一次；重复publish拒绝覆盖，请为下一次检查使用新run ID。
探针仅检查当时远端O_EXCL、hardlink/rename可见性及一次锁冲突，不代表 NFS 性能或所有故障语义已经可靠；慢或不可靠的
共享控制面不适合1秒deadline。不要用“关闭锁”绕过失败。
工具/Direct marker使用壁钟时间，节点需要时钟同步；此处没有测量时钟偏差，不保证1秒期限的跨机精度。

## Worker/Router启动与停止

引擎 `sglang.srt.disaggregation.agentic_multinode.capabilities()` 必须报告
`integrated=true` 且列出 source-local Host RDMA、shared control polling、TP atomicity、
remote Host fence release、P→D/D→P integration 等全部能力。缺项时拒绝启动 GPU。
**不要为了启动实验手动绕过这个 gate。** 只有源 Host 导出/目标恢复和完整生命周期接线
完成并通过独立审计后，才允许放行。`integrated=true`也只代表代码接线能力，
不代表硬件、吞吐、长期生命周期已远端验收。

每个节点先激活准备好的环境，确保其中 `sglang` / `slime` 指向本次两个仓库，
NIXL UCX backend及其依赖可用。配置的 `python` 必须是这个环境的绝对路径。
不要使用历史单节点脚本（它们会覆盖路径/端口），也不要在这些节点同时跑带宽基准和性能实验。

```bash
# P节点，前台运行。若集成能力未通过，明确报错，不启动GPU。
bash tools/dualpd/multinode.sh start-worker --config /path/experiment.json --node-id node-p
# D节点，另一个终端运行。
bash tools/dualpd/multinode.sh start-worker --config /path/experiment.json --node-id node-d

# Router所在节点的新终端；这里只探测/model_info，不用可能生成请求的/health做模型启动检查。
bash tools/dualpd/multinode.sh wait-ready --config /path/experiment.json --timeout 1800
bash tools/dualpd/multinode.sh start-router --config /path/experiment.json
```

需要后台运行时，可在**对应节点**使用 `nohup bash tools/dualpd/multinode.sh ... >launch.log 2>&1 </dev/null &`。
脚本没有自动SSH或跨机kill，也不擅自判断卡空闲。启动前会探测本节点的 HTTP、bootstrap、reverse bootstrap 或 Router 端口，已有服务占用则拒绝启动，不会误用或终止另一组实验。探测后会关闭临时 socket，不能排除探测与启动之间另一进程抢占的竞争，操作者仍需确保独占这些端口及GPU。

本地 `${local_root}/${run_id}/${engine_id}/` 保存 `config.json`、`launch.json`、
`revisions.json`、`service.log`、`process.json`、`stopped.json`。配置文件和计划记录
最终显存比例、TP、阈值、Host容量、环境路径。不要在一个run ID下改配置再重启；
用新run ID避免旧marker与新的引擎混用。`launch.once`会拒绝相同组件在同一run下二次
启动（即便上次异常退出）；不要删掉它强行复用旧ledger，应使用新run ID。

```bash
# 在对应worker所在节点执行，只联系此配置拥有的supervisor。
bash tools/dualpd/multinode.sh status --config /path/experiment.json --node-id node-d
bash tools/dualpd/multinode.sh stop --config /path/experiment.json --node-id node-d
# Router所在节点：
bash tools/dualpd/multinode.sh stop --component router --config /path/experiment.json
```

supervisor为子进程创建单独session/process group，通过带随机token的本地Unix socket
处理stop，不根据过期PID文件kill。TERM最多等30秒再KILL；启动后写状态失败也进入清理。
Linux subreaper回收同组后代，先清理再reap session leader，避免PID重用竞争。
**保证仅覆盖继承该进程组的子进程**；不要在workload里再调用daemonize/setsid，
脱离进程组的第三方服务不在此保证内。SIGKILL杀supervisor/主机崩溃也不能保证优雅清理。
停止完整实验时先停workload、Router，再分别停P/D；所有worker结束之前不要删除控制目录或Host数据。

## 两轮传输验证

Router启动成功后在Router节点执行：

```bash
bash tools/dualpd/multinode.sh smoke --config /path/experiment.json
```

脚本串行测试fast-tool和slow-tool两条轨迹。第一轮约数百token输入，约束模型输出`TOOL`
并发布真实tool ACK；分别立即返回、等待大于工具阈值再返回。第二轮使用**精确token IDs**
拼接父输入/输出和工具结果，不将模型文本重新tokenize。对比同样第二轮输入的完整重算
输出token IDs，并记录actual route、page-aligned父长度、`cached_tokens`及Host ledger状态。
为了得到生成token IDs，smoke仅对最多32个生成token请求output logprobs，不请求整段Prompt logits；
正式吞吐实验不应沿用这个小样本诊断设置。

`${local_root}/${run_id}/smoke/smoke.json`记录每条轨迹原始响应和检查结果。缺少reuse证据、
Direct误走Slow、Slow未恢复或输出不一致都不标为通过。输出不一致需再区分数值/batching
差异与KV错误，不能单凭本测试直接下“KV损坏”的结论。最终ACK负责清理所有尝试的generation。

这不是300+1200秒性能测试，也不覆盖P→D Host背压、多个snapshot并发、故障注入；这些仍需
按设计invariants单独验收。首次smoke成功后，先增加并发观察所有权守恒，再做正式测试。

## 正式workload

可在JSON `workload_command`填入**argv数组**（不是shell拼接字符串），例如使用现有
`examples/pd/inference.py`，显式给模型、数据配置、固定顺序、远端Router/P/D地址、温度0、
并发、300秒预热、1200秒测量以及输出目录。不自动选择数据集、不启动搜索服务。
之后执行：

```bash
bash tools/dualpd/multinode.sh run-workload --config /path/experiment.json
bash tools/dualpd/multinode.sh stop --component workload --config /path/experiment.json
```

workload继承相同snapshot/ledger命名空间并在Router节点运行。现有`inference.py`的
`--decode-host`只有一个host地址：单D TP组可直接用；多个D节点时不要把同一个host的
metrics当作全体D数据，应使用跨endpoint采集器或后续扩展指标采集后再报告整体吞吐。

## Router 最小迁移点

- `LateBindingMiniLoadBalancer` 可接受外部 `--prefill http://P:port bootstrap-port` 和
  `--decode http://D:port`；历史 shell 中 loopback URL 必须由拓扑配置生成。
- 单逻辑P V1设 NUMA_DOMAINS=0、DYNAMIC_PREFILL_DOMAINS=0、GLOBAL_DECODE=1。
  源 Host 位置是 node ID + TP shard；不能把不同机器同名 NUMA0 视为同一存储域。
- Router、harness 与 engine 必须共享 run 控制命名空间，Host 选择/恢复不能直接 mmap
  异节点进程的 memfd；这是必须接线的 backend 能力，不是改 host IP 就能解决。
- HTTP、bootstrap、NIXL动态端口需可达。初期不要宣称开放列出的3个固定端口就足够。

## 本地验证

```bash
python3 -m unittest discover -s tools/dualpd -p 'test_multinode*.py'
python3 -m unittest discover -s tools/dualpd -p test_process_supervisor.py
bash -n tools/dualpd/multinode.sh
```

这些是 CPU 单元测试；没有运行任何 GPU/跨节点传输。
