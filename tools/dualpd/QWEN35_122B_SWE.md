# Qwen3.5-122B-A10B：本地 collocated 功能与 SWE500 评测

## 固定配置

- BF16 官方权重，revision `dc4d348443bc740c68e2d77492492c11606384d5`。
- TP=8、EP=1、8 张 A100；所有 rank `mem_fraction_static=0.8`。
- SWE-bench Verified 原始 500 道各一次，c64、seed=2026。
- **所有后续 SWE-bench 实验固定 temperature=0.6、top_p=0.95、top_k=20**。
  JSON配置、Bash入口和Python启动校验均明确此要求，不沿用通用temperature默认值。
- 2026-09-18起直接复用27B的
  `swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml`，不复制一套不同默认值。
  `openai_tools`、thinking开启、单轮8192、最多64轮、累计输出81920、
  上下文131072、history12、observation12000；inline verifier。
- 复用原工具定义、system prompt、解析/终止主循环，不增加重试或纠错规则。
  Docker不额外设置CPU/内存限制、使用镜像默认用户、不启用超时归一化；
  这是27B实际快照的行为，旧结果文档中“2 CPU/4 GiB”不是实际设置。
- shell/容器启动600秒，verifier2400秒、最多16并行，container_network=none。
- Prefill chunk和batch上限均8192；Mamba ratio=0.9、page/track=64；
  parser沿用glm45/qwen3_coder，Attention/Triton和FlashInfer采样不变。
- 使用 `pd_multi_node` 和 `dualpd/sglang` 的原生collocated路径，不切换baseline环境，
  不启用自定义PD、原生HiCache/Mooncake或推测解码。环境/源码版本与27B不同须单列。
- GPU7 的现有 ComfyUI 进程获用户明确授权共享，不终止它；本次是功能与
  评测验证，不作为独占GPU吞吐验收，也不是300+1200秒稳态性能测试。

```bash
export DUALPD_PYTHON=/homes/siqic/anaconda3/envs/pd_multi_node/bin/python
bash tools/dualpd/qwen35_swe.sh download
bash tools/dualpd/qwen35_swe.sh check-model
# GPU:PID必须以运行前实际确认的进程为准，不使用广泛忽略GPU占用的开关。
bash tools/dualpd/qwen35_swe.sh preflight --allow-gpu-process 7:1868643
bash tools/dualpd/qwen35_swe.sh run --allow-gpu-process 7:1868643
```

入口先验证全部权重大小、revision、配置与tokenizer、500道唯一题目、
Docker镜像、GPU进程和端口。模型启动后先做32-token HTTP生成检查，再启动
评测；此检查不计入500题。监督进程仅回收本次进程组和唯一Docker标签。
每轮保存plan、数据与配置副本、git diff、runner源码、smoke响应和服务端raw请求日志。
启动校验workload SHA256=`9796e2d014cc7cdf49addddfb9b2f3711d7033636c98ea17a45dd30857fd7348`，
数据 SHA256=`f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c`，
以及采样/预算/Prefill上限，配置漂移直接拒绝运行。

对齐依据：27B collocated c64实际快照
`/tmp/pd-persist/swe-aiohttp-matrix-20260915-r1/qwen35-27b-tp2-c64-extension/qwen35-27b-tp2-c64-extension-01-27b-colocated-c64`。
27B是4个TP2副本，122B是单个TP8/EP1 MoE副本；两者同为全局c64。
122B显式BF16和Triton MoE后端，GPU7共享获授权；不能将吞吐当作同构实例比较。

## 本轮适配与验证边界

SGLang已有该MoE模型实现。当前Transformers的子类初始化行为会绕过
Qwen3.5部分配置子类继承的初始化函数，导致嵌套配置未类型化或默认字段丢失；
本轮通过在相应子类显式调用父类初始化修复，并覆盖dense/MoE配置回归。
TP8下2个Attention KV heads会复制到rank，不能误记为模型本身有8个KV heads。

另外为未来多节点准备了Attention+GDN复合Host传输adapter，覆盖完整Host映射、
卷积/时序state、TP各shard完成凭据和失败重试。不只传Attention KV。
该多节点hybrid入口仍受运行门禁保护，**没有宣称真实跨节点RDMA已经验收**；
本次collocated不经过这些自定义路径。

启动前：launcher/supervisor 12项通过；remote Host/hybrid/TP故障及配置回归
70项通过。独立审核已对collocated启动给出GO，对受门禁保护的remote实现
给出代码审核GO。GPU生成、长上下文、工具和verifier结果以实际运行输出为准。

## 所有权验收检查

1. 复合snapshot由源Host保留到全TP物理完成；没有用marker替代所有权。
2. P→D Direct释放协议不变。
3. P→D Host仍在源D2H完成后取得所有权；远端加载含两份state slot。
4. D→P Host仍在源D2H完成后取得所有权；远端加载含父checkpoint。
5. 网络Host操作仍在现有I/O worker，不新增Forward等待。
6. TP1/2/8测试按同一attempt与全组receipt提交；单rank校验失败可安全全组重试。
7. 复合数据校验包括Attention和全部state；padding不作为模型state恢复。
8. 相关CPU测试和独立审核已执行；真实PD GPU/RDMA验收仍待后续单独进行。

## 2026-09-17 首次本机运行（已按用户要求停止）

- 运行目录：`runs/dualpd/qwen35-122b-swe500-tp8-c64-20260917T215200Z`。
- 全部权重下载并校验，TP8加载及CUDA Graph成功；32-token生成检查通过。
- 22:07 UTC开始500题评测，配置c64；22:10 UTC已完成10题，全部verifier
  正常返回，其中3题resolved。早期完成集合偏短，**不是最终正确率**。
- 已观察到实际工具执行、非空模型patch、verifier通过；无OOM/TP异常。
- 原生pool报告Attention容量1,522,240 tokens、max_running_requests=175；
  c64是本轮Agent并发设置，不是引擎的max_running_requests。
- 成功生成不等于所有题目正常解题：早期7题以no_command结束，3题以
  task_complete结束，保留原harness语义，不按得分修改终止规则。
- 此轮实际采样为temperature=0/top_p=1/top_k=-1，已停止并保留诊断数据，
  不作为完整500题正确率或本次采样设置的结果。所属模型/评测进程已退出，
  未停止GPU7原有ComfyUI。新参数从500题第一题重新开始，不拼接旧结果。

## 仅改采样的旧轮（已停止，非27B对齐结果）

- 目录：`runs/dualpd/qwen35-122b-swe500-tp8-c64-t06-p095-k20-r1`。
- 唯一评测变量：temperature=0.6/top_p=0.95/top_k=20；工具协议及全部预算不变。
- 本地46项回归通过；独立审核GO，已核实三个采样参数实际写入HTTP payload。
- 启动前权重、数据500题、Docker镜像、端口和共享GPU允许名单检查通过。
- 此次只改采样与启动校验，不改引擎或PD状态机；上述所有权验收1–7均无改动，
  第8项已完成相关测试与独立审核。此轮为collocated评测，不声称PD路径验收。
- 按用户要求停止：498/500题收尾、182题resolved；2题未收尾。
  该轮仍是fenced-shell/524288累计预算，不能当作27B对齐结果或完整500题正确率。
  已确认两组监督进程停止，所属Docker标签无残留，ComfyUI未受影响。

## openai_tools对齐重测

- 新目录：`runs/dualpd/qwen35-122b-swe500-tp8-c64-openai27b-r1`；已提交启动，结果待完成。
- 从500题第一题重跑，不续接或拼接旧记录。
- 本轮仅修改启动配置、校验和记录，不改harness业务/引擎/TP状态机；
  snapshot验收1–7不受本轮改动影响，第8项按要求测试及独立审核后启动。
- 本地56项测试通过；独立审核GO，确认27B workload/数据哈希以及harness源码一致。
  preflight权重、结构化工具history渲染、数据/镜像、GPU允许名单和端口检查通过。
- 2026-09-18 00:06:25 UTC开始500题评测。模型加载及32-token生成检查通过。
  服务端raw日志确认采样为0.6/0.95/20、tool_choice=auto；已观察到shell tool_call、
  exit_code=0的tool结果及后续模型请求。早期已超过400次携带工具结果的续轮请求。
  这是工具链功能验证，不是最终500题正确率；完整结果待收尾。
