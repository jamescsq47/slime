# DualPD 远端执行入口

开发在本地 `dualpd` 工作区进行。用户在目标机器执行 Bash 脚本，将原始报告回传；不要求远端安装 Codex。

## 硬件和拓扑采集

在每个计划参与实验的节点运行一次。最好在实际运行推理服务的容器和 Python 环境内执行；如怀疑容器隐藏了 RDMA/NUMA 信息，可在宿主机再采一份并注明区别。

```bash
bash tools/dualpd/collect_hardware_topology.sh
```

可选参数示例（IP 请替换为另一节点的地址）：

```bash
bash tools/dualpd/collect_hardware_topology.sh \
  --output /tmp/dualpd-node-a-report \
  --peer 10.0.0.2 \
  --python /path/to/conda/env/bin/python
```

输出目录必须尚不存在，防止覆盖旧结果。默认创建 `/tmp/dualpd-hardware-时间-随机后缀`。脚本最后输出报告和压缩包的绝对路径，将每个节点的 `.tar.gz` 回传即可。

采集内容：

- GPU 型号、容量、即时占用、驱动、MIG、NVLink 状态及 P2P 拓扑。
- CPU/NUMA、CPU 亲和性、内存/cgroup/锁页限制。
- GPU 和网卡的 PCIe/NUMA 归属、PCIe 当前/最大链路速率与宽度。
- NIC 驱动、MTU、IP、路由；RDMA 端口、InfiniBand/RoCE、GID 与 netdev 对应关系。
- UCX/OFED/CUDA 编译器版本及指定 Python 环境的相关包版本/源码位置。
- 每个命令的输出、退出码以及缺失/不支持/超时记录。

脚本不使用 sudo、不安装依赖、不改配置、不启动模型、不跑带宽测试、不修改或终止已有服务。需要 Bash、Linux、GNU timeout、tar；其他工具缺失会记录并继续。每条探测默认最多20秒，超时后终止的只是本脚本启动的探测命令。不会扫描 NFS 数据目录。

`--peer` 只查询本地路由，不验证网络连通性。速率字段是链路信息，不是实测吞吐；硬件清单也不能单独证明 GPUDirect RDMA 已可用。

报告包含主机名、内网 IP、GPU UUID 等拓扑标识，分享前可检查；不采集完整环境变量、密钥、访问令牌或任意进程命令行。
