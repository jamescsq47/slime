# Unmodified SGLang baselines

All launchers in this directory default to
`/homes/siqic/anaconda3/envs/pd_baseline`.  Before allocating a GPU they run
`check_environments.py`; the run is rejected if any custom agentic KV module
is present.  The resulting fingerprint is saved as `environment.json`.

The reusable baseline entry points are:

| Directory | Layout | Reverse KV behavior |
|---|---|---|
| `run_four_gpu_comparison_suite.sh` | 4 colocated / 1P:3D | colocated, no reverse KV, native Mooncake |
| `run_two_p_two_d_comparison_suite.sh` | 2P:2D | no reverse KV, native Mooncake |
| `run_comparison_suite.sh` | 6 colocated / 1P:5D / 2P:4D | larger architecture sweep |

The four-GPU formal cases use Mixed 1:1, fixed seed 2026, the same c256 request
sequence, a 300-second warmup, and a 1200-second measurement.

Run the complete suite with:

```bash
bash examples/pd/scripts/baseline/run_four_gpu_comparison_suite.sh
```

Individual cases can override `RUN_DIR`, `MAX_INFLIGHT`, `WARMUP_SECONDS`, and
`MEASURE_SECONDS`.  For example:

```bash
CASE_MODE=no_reverse PREFILL_GPUS='0 1' PREFILL_PORTS='27500 27501' \
PREFILL_BOOTSTRAP_PORTS='28500 28501' DECODE_GPUS='2 3 4 5' \
DECODE_PORTS='27502 27503 27504 27505' \
bash scripts/baseline/run_pd_case.sh
```
