# Experiment tools

- `check_environments.py`: enforce clean-baseline versus modified environment.
- `analyze_pd_offload.py` / `analyze_run.sh`: summarize one formal run.
- `compare_*_baselines.py`: render retained baseline comparison tables/plots.
- `capture_*` and `compare_pd_correctness.py`: deterministic correctness checks.
- `analyze_agentic_kv_v1.py`: summarize request-generation lifecycle events.
- `validate_agentic_kv_v1_runtime.py`: small serving-level path validation.
- `plot_direct_pipeline_summary.py`: visualize Direct/slow pipeline counters.
- `select_rate.py`: select the best point from an arrival-rate sweep.
- `prepare_workloads.py`: convert the FoldAgent BrowseComp parquet files into
  the JSONL schema used by this example.
- `hold_node.sh`: optional bounded experiment holder; not part of formal tests.

Analysis utilities are safe to run on completed result directories.  Capture,
validation and holder tools may allocate GPUs; inspect their arguments first.
