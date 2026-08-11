# Four-model-GPU mixed-agent baseline comparison

All cases use the same fixed Mixed 1:1 schedule, seed, concurrency, warmup, and measurement window.

| Case | Agent/s | Prefill token/s | Decode token/s | Decode token/s/GPU | D busy/GPU | D running/GPU | Prefix hit | Reverse KV reuse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4× Colocated | 1.221 | 10068 | 4826 | 1207 | 73.4% | 63.2 | 92.1% | 52.3% |
| 1P:3D, no reverse KV | 0.392 | 11548 | 1729 | 576 | 99.2% | 8.9 | 0.0% | 0.0% |
| 1P:3D, native Mooncake | 0.472 | 8657 | 1941 | 647 | 99.6% | 11.6 | 60.6% | 30.7% |
