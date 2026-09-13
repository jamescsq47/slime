# c512：所有快工具 Direct 失败重算

工具>1s仍走Slow；快工具失败含claim后失败均显式重算；DMA fence未放松。
Qwen3-8B / BrowseComp source-order n680 / TP1 4P:4D / c512 / t0。
全部Host预注册完成后，300秒业务预热+1200秒测量；原生HiCache/Mooncake关闭。

| 指标 | 本轮 | 4888参考 |
|---|---:|---:|
| Decode token/s | 4668.150 | 4888.225 |
| P Forward/卡 | 98.10% | 97.04% |
| D Forward/卡 | 99.64% | 99.62% |

完整资源、正式窗口Direct/Slow/重算事件见comparison.json。
正式窗口边缘未完成snapshot不等于丢失；最终结论仍需逐ID守恒审核。
