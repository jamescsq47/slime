# MiniMax-M2.7：TP8 多节点适配与 colocated SWE500 验证

当前只是代码/CPU 验证；没有在本机或远端运行 GPU，不提供虚构吞吐/正确率。
先运行单节点 colocated，再使用 `multinode.sh` 验证跨节点 Direct/Slow。

## 固定设置

`minimax_swe.json` 是本轮配置源：MiniMaxAI/MiniMax-M2.7，固定 revision
`d494266a4affc0d2995ba1fa35c8481cbd84294b`；官方 FP8 权重，BF16 KV。
Attention TP=8、专家 EP=8，均限同一个节点；不启用 DP attention 或 MTP/speculation。
EP=8 不改变 KV 的 TP8 切分。纯专家 TP8 的 intermediate1536/8=192
不满足128×128 FP8 block，故不用 EP=1，不新增 padding/kernel 补丁。

- 一组8卡 collocated；所有 rank `mem_fraction_static=0.8`；c64。
- 原 SWE-bench Verified 500题，source-order，各执行一次；不是64题。
- 现有 Miles fenced-shell harness + Docker + inline 隐藏 verifier；不修改提示词或评分器。
- 8192 tokens/turn、最多64轮、context131072；temperature0、top_p1、top_k=-1、seed2026。
- 每容器2CPU/4GiB；工具600秒；verifier2400秒/并发16。
- finite evaluation，无300+1200秒稳态窗口，不称为正式吞吐对比；保留全程吞吐/轮次/长度/评分。
- 没有 custom PD、HiCache、Mooncake、Mamba 开关；原生 Radix 正常使用。

Qwen3.6 只完成过只读审查，没有引擎改动可撤销。历史单节点 Qwen/Mamba 支持保留，
不是本轮残留；不删除其他模型代码。多节点仍拒绝 hybrid/MLA，仅显式添加普通GQA MiniMax。

## 远端运行

在 `dualpd/slime` 执行，Python 环境必须已安装本工作区两个 editable 包及推理依赖。
脚本不自动安装/升级 Torch、CUDA 或 SGLang，不自动 SSH，不强占 GPU。

```bash
export DUALPD_PYTHON="$(command -v python)"
bash tools/dualpd/minimax_swe.sh plan

# 一次性准备；权重约230GB，Docker images可能占用更多空间，请先确认磁盘。
bash tools/dualpd/minimax_swe.sh download --model /path/MiniMax-M2.7
bash tools/dualpd/minimax_swe.sh prepare-data --data-root /path/pd-data
# verifier 依赖；仅在你选定的实验环境安装。
"$DUALPD_PYTHON" -m pip install -r examples/pd/requirements-swe-bench.txt
"$DUALPD_PYTHON" examples/pd/scripts/tools/prefetch_swe_bench_images.py \
  --dataset /path/pd-data/swe-bench-verified/test.jsonl \
  --log /path/pd-data/images-prefetch.jsonl --concurrency 4

bash tools/dualpd/minimax_swe.sh check-model --model /path/MiniMax-M2.7
bash tools/dualpd/minimax_swe.sh preflight \
  --model /path/MiniMax-M2.7 --data-root /path/pd-data
bash tools/dualpd/minimax_swe.sh run \
  --model /path/MiniMax-M2.7 --data-root /path/pd-data \
  --run-dir /path/local-results/minimax-m27-tp8-c64-unique-run
```

模型路径必须由 download 命令固定版本并验证完整性；`check-model` 支持只有 metadata
的CPU验证，但不能把它当权重完整性或GPU加载验证。首次 `trust_remote_code` 用于官方
配置类，执行前应审阅该固定revision的配置代码；模型实现来自本地SGLang。

run会拒绝已有run目录、被占用端口/GPU、缺失镜像/权重。模型与harness用独立supervisor
进程组，异常/退出只终止本次会话；Docker只清理本次唯一run标签，不做全机prune/kill。
输出：`model/service.log`、`inference/service.log`、逐题 `requests.completed.jsonl`、
`requests.jsonl`、metrics、verifier记录、`swe_bench_profile_summary.md/json`、源码commit和diff。
发生失败保留已完成题，不能把不完整500题的正确率当完整Verified得分。

## 后续多节点

现有 `multinode.example.json` 增加 `"model_family": "minimax_m2"`，改为本模型路径，
P和D均TP8、各在独立节点。launcher自动设置EP=TP、MiniMax parser、BF16 KV。
其余容量/路由/所有权/超时不变；只传普通K/V，复用现有NIXL和源节点Host RDMA。
先完成既有共享控制面检查和 `multinode.sh smoke`，不自动启动跨节点正式实验。
代码放行不代表网络/GDRDMA/TP正确性已验收。

## 本地验证记录（2026-09-17）

- 既有多节点/生命周期/TP/路由回归：741 passed，2 GPU tests skipped。
- 新增 MiniMax GQA TP1/2/4/8 全rank字节映射：4 passed（CPU，不是RDMA实测）。
- SWE fenced-shell/模板/终止/超时回归：39 passed。
- 新增配置隔离/TP8+EP8/非法布局/失败清理测试：6 passed。
- 固定revision真实config/tokenizer/harness CPU检查通过，shell语法与diff检查通过。
- 独立agent审查GO；发现的部分清理失败跳过其他supervisor的问题已修正并补测。
  Docker归属另外改为每次生成的随机UUID，避免手工run目录同名时互相清理。
- 未下载本地完整230GB权重，未运行GPU、Docker题目或verifier；不填写测试分数。
