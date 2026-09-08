# P-ready P→D staging refactor: formal result

- Workload: BrowseComp source-order n680, Qwen3-8B, TP=1, 4P:4D, c512
- Window: 301.1 s warmup + 1200.0 s measurement
- Completed: 2653 agents, 0 failures, 2.211 agents/s
- Prefill compute throughput: 30,221 token/s
- Decode throughput: 4,557.6 token/s total, 1,139.4 token/s/D
- P Forward: 315.2% aggregate, 78.8%/P
- D Forward: 394.6% aggregate, 98.66%/D
- Page-aligned parent KV reuse: 100%
- Extra Prefill caused by parent-KV loss: 0 tokens

The refactor separates bounded P→D Host preparation from FIFO Decode
admission. A completed Prefill generation releases P HBM after either Decode
accepts its KV or the complete snapshot is durable in the P→D Shared Arena.
Across the formal run, P→D Host source release tracked D2H completion exactly;
there was no completed Host copy left retaining P HBM. The four 32 GiB P→D
arenas peaked at about 28.8 GiB each and safely fell back to retaining P KV
when an Arena could not reserve the whole request-generation snapshot.

The four 300-second Decode-throughput quarters were approximately 4,321,
4,922, 4,756, and 4,233 token/s. This source-order phase variation explains
the difference from the 300-second short validation (5,043 token/s); it was
not a monotonic pipeline collapse. No Router/Host lifecycle exception or KV
correctness failure occurred.
