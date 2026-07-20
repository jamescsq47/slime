# Slime dashboard

The dashboard records GPU utilization, SGLang request concurrency, existing
training metrics, and per-sample generation/tool/lifecycle events to
append-only hourly JSONL files. It is disabled unless
`--use-slime-dashboard` is provided.

For the mixed training script:

```bash
ENABLE_SLIME_DASHBOARD=1 \
SLIME_DASHBOARD_DIR=/shared/run/dashboard \
bash examples/mixed/hybrid_qwen3_4b_multi.sh
```

View a live or completed run from any machine that can read the directory:

```bash
python -m slime.dashboard.serve \
  --dashboard-dir /shared/run/dashboard \
  --host 0.0.0.0 \
  --port 7788
```

The browser polls the file-backed API every five seconds. Training processes
never wait for dashboard writes or HTTP scraping. Collector, NVML, scraper,
and disk failures only leave telemetry gaps.

This implementation is adapted from
[`radixark/miles#1654`](https://github.com/radixark/miles/pull/1654), pinned at
`d9189010bc3ba407cf0189389015032096a7c725`, with Slime-specific trace and
fully-async lifecycle integration.

