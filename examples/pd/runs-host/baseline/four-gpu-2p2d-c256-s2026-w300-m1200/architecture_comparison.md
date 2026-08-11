# Four-model-GPU 2P:2D mixed-agent baseline comparison

Both cases use the same fixed Mixed 1:1 schedule, seed, c256, 300-second warmup, and 1200-second measurement window.

| Case | Agent/s | P compute token/s | P cache token/s | P busy/GPU | P KV | D token/s | D token/s/GPU | D active token/s/GPU | D busy/GPU | D running/GPU | D KV | D prealloc/GPU | D transfer/GPU | Engine cache fraction | Mooncake mean/max | Evictions / GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2P:2D, no reverse KV | 0.705 | 22798 | 0 | 96.6% | 6.8% | 2765 | 1383 | 1384 | 99.9% | 49.6 | 88.9% | 61.9 | 16.4 | 0.0% | 0.0%/0.0% | 0 / 0.0 |
| 2P:2D, native Mooncake | 0.692 | 8349 | 13834 | 47.9% | 5.7% | 2795 | 1397 | 1402 | 99.7% | 68.4 | 87.9% | 56.0 | 3.2 | 62.4% | 80.7%/85.0% | 46 / 1009.7 |
