# Request-generation KV pipeline

These launchers use the modified `pd` environment.  Every launcher validates
the environment fingerprint before allocating GPUs.

Supported entry points:

- `run_1p3d_case.sh`: current validated formal configuration; Mixed 1:1,
  c256, seed 2026, 300-second warmup and 1200-second measurement.
- `run_1p5d_case.sh`: scale-out 1P:5D configuration.
- `run_2p6d_numa_case.sh`: two-NUMA-domain 2P:6D configuration.
- `internal/`: Mooncake, lifecycle and model-service implementation used by
  the three entry points.  Do not launch these files directly.

The current policy uses a 2-second fast-tool threshold, a 2-second Direct
handshake bound, exact-size Direct receive pages, Direct > slow recovery > new
priority, late Decode binding and P-ready compute-ahead backpressure.

```bash
cd /homes/siqic/slime
bash examples/pd/scripts/new_method/run_1p3d_case.sh
```

Override `RUN_DIR`, GPU/port variables, `MAX_INFLIGHT`, `WARMUP_SECONDS`, or
`MEASURE_SECONDS` through the environment.  The retained reference result is
`runs-host/new-method/1p3d-c256-s2026-w300-m1200`.
