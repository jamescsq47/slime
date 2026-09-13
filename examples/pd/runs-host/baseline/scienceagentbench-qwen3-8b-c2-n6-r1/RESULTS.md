# ScienceAgentBench / Qwen3-8B：6道原题功能试跑

日期：2026-09-08。**非正式吞吐实验，不是官方正确率评估。**

结论：原始 Self-debug 流程可以接入本地模型；本轮5道题成功生成目标文件，
另1道题模型错误地计算超大相关矩阵，触发4GiB沙箱OOM，重复尝试被人工终止。
本轮不足以证明此子集适合“有效计算本身稳定超过2秒”的工作负载。

## 数据、环境和复用

| 项目 | 设置 |
|---|---|
| 数据 | ScienceAgentBench verified，共102题；本轮IDs 5/6/7/8/9/20，原顺序 |
| 选择依据 | CPU可执行的机器学习、统计及绘图任务；按依赖选择，未按参考答案选择 |
| 原题/输入 | 保留官方题目、数据预览、数据文件；路径映射到沙箱 |
| 模型 | `/dataset/model/qwen3/Qwen3-8B`，GPU0，TP=1，colocated |
| 环境 | `pd_baseline`，SGLang0.5.10.post1；未修改SGLang/PD |
| 并发 | 2个agent，有限6题，不是闭环稳态 |
| 采样 | temperature0，top_p1，每调用最多8192tokens |
| 上下文 | 40960tokens，mem_fraction_static0.80，page_size64 |
| Agent | 官方Self-debug：最多10次程序执行；错误反馈后重新生成完整程序 |
| 工具 | 隔离Docker内执行完整Python程序；复用DABstep ContainerPython |
| CPU约束 | 每沙箱2CPU、4GiB RAM、无网络、无GPU，BLAS/OMP单线程 |
| 工具超时 | 900秒；本轮未达到该超时，ID9因内存问题提前失败 |
| 依赖 | 基础镜像科学库+scikit-learn1.7.2、statsmodels0.14.6；预安装不计工具时间 |

官方来源：
[代码与说明](https://github.com/OSU-NLP-Group/ScienceAgentBench)，
[verified数据](https://huggingface.co/datasets/osunlp/ScienceAgentBench)。

代码commit：`c26e151ed601ba109dc4d35e057ff8e73fec469d`。
数据revision：`9c6e96c9e74572e979b0930ee735041cef528cb7`。
Parquet SHA256：`c6f937863a220bd1762a00c20a0f79cc8dfca900b819bdb552150310731ae147`。
实际Docker image：`sha256:22317581794a1c969d7de3fc046768e094da7320a55b8aeb17fde68c3ba7f703`。

直接复用官方prompt常量及get_sys_msg/write_program/solve_task，替换cloud模型接口、
依赖安装及程序执行边界。模型输入不包含gold程序、评分代码或domain knowledge。
只解压输入数据目录（约52MiB）供沙箱读取。官方zip保留在本机，不上传解压数据。

## 逐题结果

“成功”仅表示程序退出0且新生成预期文件，不代表科学结果正确。

| ID | 原始任务 | 模型调用 | Decode tokens总计 | 完整程序执行情况 | 最后成功执行时间 |
|---|---|---:|---:|---|---:|
|5|DKPES随机森林分类|4|11,584|3次错误后生成CSV；但最终写成回归，偏离题意|1.403s|
|6|信号抑制分布/相似性绘图|2|2,496|缺seaborn，改用matplotlib后生成图|0.890s|
|7|高/低活性分子官能团分布|1|4,482|生成图；未做官方图像评分|1.286s|
|8|逻辑回归后向特征选择|3|12,741|2次错误后生成图；未做官方评分|1.847s|
|9|FACTORS任务间相关性|2|4,739|首次381.592s后SIGKILL/OOM，重复同类算法的第二次执行人工终止|—|
|20|TDC单/多任务R²绘图|2|2,297|修复range与float运算错误后生成图|0.846s|

ID5和ID8各有一次模型输出达到8192上限且没有完整可执行程序。
官方write_program据此写入`ERROR`占位符，执行产生NameError并进入纠错。
这两次不能视为有效科学计算。

| 汇总 | 数值 |
|---|---:|
| 实际尝试原题 |6|
| 生成新目标文件 |5/6（不是正确率）|
| 模型调用总数 / 平均每题 |14 / 2.33|
| Decode tokens总计 / 平均每题 |38,339 / 6,389.8|
| 累计输入tokens / 平均每题 |14,541 / 2,423.5|
| 程序执行尝试 |14：13次已返回，1次人工终止 |
| 已返回尝试分类 |5次输出成功、7次非零退出、1次SIGKILL；7次中含2次ERROR占位执行 |
| 5次成功执行平均耗时 |1.254s|
| 5次成功执行最长耗时 |1.847s|
| 成功执行 >1秒 / >2秒 |3/5 / 0/5|

时间在容器内部围绕Python子进程测量，包含解释器启动、库导入、数据读取和程序计算，
不含模型推理、Docker创建、排队或依赖安装。
不要把包含OOM长调用的总均值当作有效工具耗时。

## 长调用的实际原因

FACTORS训练文件是1118行、14306列：Molecule、14293个D_特征、12个T_任务。
模型仅删除Molecule，就对其余14305列计算相关矩阵，而不是只计算任务间相关性。
这产生约2.05亿个矩阵元素，还要创建mask、masked DataFrame和展开数组。
首次程序执行381.592秒后退出-9；现场Docker State.OOMKilled=true。
随后模型仅修改输出路径检查，再次运行同一个过大矩阵计算。
为避免重复浪费，人工只删除该题的唯一沙箱容器；probe保存已有轨迹后退出。

`task-9.json`因此记录state=error、`Python sandbox exited`，不是900秒timeout；
第二次中断的执行没有完成时间，不能混入已返回13次的耗时统计。
这是真实模型选择的错误算法，不是我们添加sleep、加大数据或强迫调用慢工具。

另外进行了独立CPU导入开销检查（非题目结果，每项2次）：

| 只启动/导入的程序 | 耗时 |
|---|---|
|pass|0.031 / 0.028s|
|numpy+pandas+matplotlib.pyplot|0.948 / 0.665s|
|pandas/numpy+RandomForestClassifier|1.860 / 1.393s|
|pandas/numpy+LogisticRegression|1.287 / 1.284s|

因此本轮成功调用的秒级时间可能有很大一部分来自库导入；未逐条分解计算时间，
不能简单相减后宣称精确的纯计算耗时。

## 发现并修复的执行反馈问题

r1运行时，非零退出且stderr为空被描述成“没有生成文件”。这让ID9在OOM后进行
了无意义的输出路径修复。运行后已修复：

- 信号终止：记录program_signal和signal值，终止该功能probe题并清理整个沙箱；
- 普通非零exit无stderr：反馈实际exit code；
- 仅在退出0但缺少输出时反馈缺少输出；
- 不将SIGKILL一概称为OOM，OOM需另有容器/系统证据。

原始配置及task记录不追改为修复后结果。修复后未再次用GPU重跑相同6题。
新增signal/empty-stderr故障测试后，SAB6项+DAB9项=15 passed；独立agent审核GO。
运行前原13项测试及独立审核也均通过。

## 对后续PD研究的适配性

官方Self-debug每轮仅传“原始题目+上一版程序+最新错误”，不累积完整历史；
下一轮prompt通常不是上轮完整KV对应token序列的延伸。
本轮忠实保留该语义，不能据此测试你设计的append-only父KV回传收益。
如继续使用ScienceAgentBench，需要选择累积历史的agent流程，并在colocated和PD
两侧使用完全相同的harness；不能只改一边。

当前结论：数据/执行接入可行，但所选6题没有验证出稳定、有效的>2秒计算工具调用；
Qwen3-8B的任务正确性及长调用真实性仍是主要风险，不能宣称整个数据集不适合。

## 清理与代码边界

本轮模型进程、子进程及所有本轮沙箱已退出；27700端口关闭，GPU0回到599MiB原占用。
其他用户/agent的容器与GPU进程未操作。唯一旧代码改动是ContainerPython可选image参数，
默认行为不变；新增代码均位于`examples/pd/data/scienceagentbench`及对应配置/启动/测试文件。
无SGLang、PD Router、KV allocator或TP状态机修改，无Git推送。
