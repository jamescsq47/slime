# MiniMax-M2.7 A100 本地预检（2026-09-17）

请求：TP=8、EP=8、collocated、mem_fraction_static=0.8、c64，
SWE-bench Verified 500题各一次，包含 inline verifier。

## 结论

**未启动模型服务或500题评测。当前官方FP8 checkpoint的原生MoE路径不支持本机SM80。**
这不是TP原子性、PD数据通路或HBM容量测试失败；不能报告吞吐或正确率。
未修改引擎或偷偷改变权重量化配置。

## 已验证

- 本机8×NVIDIA A100-SXM4-80GB，GPU0–6检查时空闲。
- GPU7有其他用户ComfyUI进程，PID1868643，约0.6GiB；没有终止或修改。
- `swebench==4.0.3` 已安装；Docker可访问，500题所需镜像均已存在。
- 原始数据500行、500个唯一instance_id，source-order固定。
- 数据revision：`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`。
- 导出JSONL SHA256：`f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c`。
- 模型下载已发起；本记录不代表完整权重下载完成。完整性以
  `downloads/MiniMax-M2.7/dualpd-download.json`及preflight校验为准。

## 原生算子最小复现

独立审计允许在空闲GPU0做有120秒超时的原生MoE算子测试。
没有加载模型或启动TP服务；使用BF16输入、e4m3fn权重、128×128 scales，
调用现有`fused_moe(..., use_fp8_w8a8=True, block_shape=[128,128])`。
尺寸为M=1、hidden=3072、intermediate=1536、8个专家/topk8，
用于覆盖同样的dtype/backend约束，不声称是完整EP8验证。

第一次运行缺少PATH中的ninja；将pd_multi_node/bin加入PATH后重新运行，
CUDA量化准备通过，原生Triton MoE编译明确失败：

```text
DEVICE NVIDIA A100-SXM4-80GB CAPABILITY (8, 0)
triton.compiler.errors.CompilationError: at 1:0:
def fused_moe_kernel(
^
ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

进程以exit=1退出，不遗留模型/GPU worker。
审计同时确认：dense Linear有Marlin回退，但当前Fp8MoEMethod没有相应自动回退；
`--dtype bfloat16`只设置运行dtype，不等于正确解量化官方FP8权重。

## 后续选择

1. 保留官方FP8设置，在H100上做同一组评测。
2. 若必须本机运行，需要另行确认正确解量化为BF16并核算0.8显存容量，
   或实现并验证受支持的MoE权重后端；不能直接忽略FP8 scale或强行改dtype。

本轮没有进行上述变更。下载与数据准备不等于500题/verifier已执行。
