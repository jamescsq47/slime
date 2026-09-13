# SWE harness：Miles PR51 对齐与修复（2026-09-08）

## 来源与范围

上游本地镜像 `/homes/siqic/miles-coding-rl-core`，固定提交
`e2e516603ad6688f3ba7a5e5e8d2b5a7606fa849`（`origin/pr/51`），文件
`examples/experimental/swe-agent-v2/eval_swebench_daytona.py`。
GitHub PR页面在此次读取时返回404，不声称已获取最新远程版本。

该文件是直接调用Daytona的SWE evaluator，不是仓库中面向Terminal-Bench的
OpenEnv客户端。当前本地实现是其适配版，历史`openenv-swebench-pr51`标识保留兼容。
本轮只改SWE harness、专用Qwen3.5-9B入口、测试与此文档，不修改任何SGLang环境。

## 默认协议

新配置：`configs/experiments/swe_bench_verified_miles_pr51_8k_t64.yaml`。
9B colocated与4P:4D入口默认使用该配置；`WORKLOAD_CONFIG`可显式覆盖以复现旧设置。
历史structured-tool YAML与所有实验原始结果保持不变。

- 上游原始system prompt；输出一个bash代码块，从可见content提取shell。
- 按上游保留assistant消息及reasoning字段；工具观察作为user消息。
- TASK_COMPLETE按上游控制token规则结束；普通总结无命令记no_command。
- 不增加恢复prompt、不新增submit工具、不启用review协作；REQUEST_REVIEW不能作为shell执行。
- Docker代替Daytona；现有PD入口、隐藏verifier、轨迹与时间记录保留。
- 用户指定采样仍为temperature .6、top_p .95、top_k20、min_p0；thinking开启，
  单次8192、最多64轮。上游示例默认temperature .2/top_p1，不直接覆盖用户设置。
- 上游没有独立累计输出上限，新配置使用64×8192=524288的非额外限制；旧structured配置
  明确配置的81920现在会严格计费并缩减最后一轮预算。
- 两个9B启动入口默认使用环境内已有qwen3 reasoning parser；没有改parser源代码。

保留的本地差异：单次输出截断时拒绝执行可能不完整的命令；按本地context容量缩减
输出/必要时压缩（并非上游收到context错误后回退）；Docker启动600秒、无新增episode
总时限，verifier并发16。工具600秒、verifier2400秒、观察12000字符、重复结果4次与
上游默认相同。第一版不实现上游review gateway与其恢复逻辑。

## 明确修复

1. 旧structured工具协议中，残缺工具XML不能被当作final_answer；reasoning中的
   TASK_COMPLETE不再被当作正式提交。仍保留既有普通可见总结结束语义用于旧协议。
2. 既有允许的工具别名在执行后写回历史时规范为shell；原始tool_calls留在turn_events，
   不扩大可执行别名集合。
3. finish_reason=length的输出不执行工具、不认定成功提交，记录独立截断原因。
4. 明确配置的累计token预算现在生效，包括reasoning/output usage，不仅计算可见文本。
5. 工具失败、取消、轮次/预算耗尽也尝试应用层final ACK；下一轮渲染失败时同时通知
   上一已完成generation与当前尝试generation，正常结束在verifier前通知。
6. Miles Chat 路径显式使用 `skip_special_tokens=true`、`no_stop_trim=false`。
   原先继承的 Slime 返回设置会把 `<|im_end|>` 留在闭合代码块行，导致上游锚定
   正则拒绝有效命令。只修复传输设置，不放宽上游命令语法。
7. 本地 Docker 显式对齐上游 Daytona 的 `os_user=root`、2 CPU、4 GiB RAM。
   镜像默认的 openhands 用户不能直接写 root 所有的 0644 源码，虽然目录可写。
   新增 Docker 选项默认不启用，仅 Miles 配置选择它们；其他数据集保持原行为。
8. r3最终审计发现 GNU `timeout --signal=KILL` 返回137，而 loop只识别124，
   两题工具超时后错误继续。因此 Miles 配置显式启用容器内 Python 超时 supervisor：
   截止时杀该 shell 进程组并返回124，独立的 SIGKILL/自然137仍保留137。只影响
   opt-in 的 agent_tool，不修改 verifier 或其他配置。增加测试后53 passed/1 deselected。
9. r4两题运行完整 Django 测试集时出现外层 Docker TimeoutError。进一步将工具
   输出落在容器本地临时普通文件，防止后代持有 attach pipe；pread读取固定大小
   输出快照。确认工具超时后，在该题专属 PID namespace 清理后代，并确认非
   zombie 进程（除PID1/清理器）清零，才捕获patch/验证；未清零或未知外层
   transport超时仍fail-closed，不把它们伪装成正常工具超时。56测试通过，真实
   Docker验证setsid后代输出不阻塞、超时writer停止；独立审计GO。

## 新发现：模板差6 tokens

对旧colocated first100 r2的100个终止前对话，使用当前SGLang真实
`OpenAIServingChat._apply_jinja_template`作CPU对照：修复前100/100不一致，
原生多6tokens，首差异位置98。原因是Tool schema序列化新增外层
`defer_loading: null`，本地手写schema未包含。

`_render_prompt`现在使用实际环境的`Tool.model_validate(...).model_dump()`，
不再硬编码某个版本的所有默认字段。这个问题影响工具schema的token一致性，
不是历史正文被截掉的证据，也没有证明它解释99题的格式异常。

## 验证

- 25个直接相关CPU测试通过：test_swe_bench_openenv_harness.py与
  test_swe_miles_alignment.py（终止、截断、预算、别名、历史、异常、取消、schema）。
- 扩展test_swe_bench_harness.py后共42通过、1失败；失败是此环境未安装
  minisweagent的旧mini-SWE测试，不是修改文件的回归，未为此安装/修改环境。
- 通用test_agentic_kv_request.py不能在此pd_mamba收集：它依赖另一个pd版本的
  kv_to_page_indices接口。本轮不修改这份跨环境测试或SGLang。
- 两份shell入口语法检查、git diff --check通过。
- 固定上游system prompt一致，8个上游命令提取案例一致。
- 修复后100旧structured终止prompt + 100基于相同工具记录构造的fenced模板探针，
  共200个与真实原生模板逐token一致；后100个不是新模型生成轨迹。
- 独立agent复核上述200个prompt与源代码，CPU适配审计GO。
- EOS 与 Docker 对齐后最终相关测试为 **45 passed / 1 deselected**；跳过的是
  当前环境没有安装 minisweagent 的既有测试。真实 Docker 探针验证 uid=0、
  源码可写、`cpu.max=200000 100000`、`memory.max=4294967296`。

可复现CPU检查（在仓库根目录）：

```bash
PYTHONPATH="$PWD/examples/pd:$PWD" /homes/siqic/anaconda3/envs/pd_mamba/bin/python \
  examples/pd/scripts/tools/audit_swe_miles_alignment.py \
  --model /homes/siqic/Qwen3.5-9B \
  --traces /tmp/pd-persist/qwen35-9b-tp1-swe-colocated-first100-20260908-r2/requests.jsonl \
  --miles-reference /homes/siqic/miles-coding-rl-core
```

## PD生命周期边界与验收

本次仅在agent确定不再继续后发布现有best-effort final标记，不增加reservation、
credit、超时回退或新的物理owner。D_HBM/D2P_HOST/P_HBM等物理所有者仍由现有
状态机依据fence/CAS释放。八项标准中：1–4、6不改物理状态机；5不加入新的I/O依赖；
7修复模板一致性但不证明KV/Mamba数值正确；8独立审计 GO 后运行下述原生
colocated GPU 测评，没有运行新的 PD ownership 验收。

取消发生在远端manifest建立前时，final ACK时间戳可能早于manifest；ACK失败后
当前通知也不是有保证的远端取消协议。mock的final调用次数不能当作物理释放证明。
因此下述 colocated 结果不能证明 PD ownership 守恒、KV/Mamba 回传数值正确，
也不是 300+1200 秒闭环稳态吞吐验收。

## 实际模型测评（CPU 检查之后）

全部使用 Qwen3.5-9B、8 个 colocated TP=1 副本、首 100 道唯一题各执行一次，
保留原始模型输出、工具轨迹、patch、隐藏 verifier 报告和时间。不是 Verified
完整 500 题，也不是 100 题重复五次。采样和 8k/64 轮预算保持上述设置。

路径共同前缀：
`/tmp/pd-persist/qwen35-9b-tp1-swe-colocated-miles-pr51-first100-20260908-`。

- `r1`：发现并复现 EOS 污染命令提取，保留失败数据，不作为正常质量结果。
- `r2`：EOS 修复后完整结束 100 题，29 题通过；17 题实际遇到 Python 源码写
  PermissionError，因而发现 Docker 默认用户与上游不一致。保留此诊断结果，
  不把它当作环境对齐后的最终结果。
- `r3`：增加 root/CPU/RAM 对齐后重新完整评测同一 100 题，**27/100 通过**，
  全部完成耗时 2062.946 秒（34.38 分钟），平均 42.64 轮、42.11 次 shell、
  13862.06 输出 tokens（含 reasoning）。写权限错误和 verifier 基础设施错误均为0；
  2 次 shell 触及600秒超时，任务继续。31题轮数耗尽、26题单轮8k耗尽、25题
  TASK_COMPLETE、16题重复结果、2题no_command。详细配置、口径与原始路径见
  该目录 `RESULTS.md`；保留全部100题轨迹和验证报告。
  最终审计确认两次超时后继续与上游不一致（其中一题通过），所以27%是r3实际代码
  的有效评分，不是严格对齐的验收结果。不能将后续重测的两条直接替换进这100条。
- `r4`：完整100题33通过，但2题外层Docker超时且未评分；作为诊断结果保留。
- `r5`：加入本地文件输出/超时进程收敛后，**100题全部执行并评分，32通过**，
  基础设施失败0。总耗时2760.923s（46.02min），平均44.17轮、43.65次工具、
  13945.91输出tokens。正式任务12039真实触及600s工具上限，清理收敛后停止并评分，
  验证了不只是mock可用。详细终止、时间、长度、限制见r5的RESULTS.md；源码快照
  和全部原始轨迹/patch/verifier报告均保留。

本地仍有已明确记录的CPU可见性差异：Docker quota为2CPU，但Django自动看到256核，
完整测试集可能启动约256worker，导致12050工具累计2327s的长尾。OMP/BLAS=2不限制
Django测试worker。本轮未中途调参，不声称与Daytona性能等价；后续CPU/Daytona
对比应控制此差异。此限制不隐去，本轮32%只代表记录的实际本地配置。

Docker 是本地后端，不声称与 Daytona 所有条件相同：本地禁网、没有同样的
10 GiB 磁盘配额，swap 语义也不完全一致。不额外引入恢复 prompt 或 submit 工具。
