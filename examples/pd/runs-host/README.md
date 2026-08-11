# Retained host results

Only formal baselines and the latest validated request-generation KV result
are retained here.  Smoke tests, failed attempts, superseded revisions and
historical parameter sweeps were deleted on 2026-08-11.

## Baseline

- `baseline/four-gpu-1p3d-c256-s2026-w300-m1200`: colocated, 1P:3D without
  reverse KV, and fixed stock-SGLang 1P:3D Mooncake.
- `baseline/four-gpu-2p2d-c256-s2026-w300-m1200`: 2P:2D without reverse KV
  and stock-SGLang Mooncake.
- `baseline/current-workload-colocated-4gpu-c256-s2026-w300-m1200`: latest
  clean colocated reference using the current workload implementation.

Every baseline environment fingerprint records `expect=baseline` and points
to the unmodified `pd_baseline` environment.

## New method

- `new-method/1p3d-c256-s2026-w300-m1200`: latest completed 300+1200-second
  Mixed 1:1 run.  It completed 1088 agents in the formal window and measured
  4408.7 Decode token/s total, or 1469.6 token/s per D GPU.

Each run keeps raw requests, two-second engine counters, service logs,
configuration, environment fingerprint, summary JSON and plots.  Historical
`config.json` files record the schedule path used at run time; the maintained
copies now live under `configs/workloads/`.
