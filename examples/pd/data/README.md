# Workload harnesses

`data/` contains inference workload plugins, not the raw datasets. Raw JSONL
or Parquet files remain outside the repository and are referenced by a
workload YAML/JSON file.

Each `data/<harness>/` package exports one `HARNESS` from `__init__.py`. A
harness contains only two operations:

1. `load_samples(context, dataset_spec)` converts that dataset into Slime
   `Sample` objects.
2. `generate(args, sample, sampling_params)` runs one complete inference
   trajectory, including its tools.

The registry imports a harness lazily. Importing a harness must never launch a
GPU service, download a corpus, or create an external environment. Services
listed in `HARNESS.required_services` are started and supervised separately.

## Workload configuration

An experiment config declares dataset identities, harnesses, source paths,
mixture weights and the sampling rule:

```yaml
schema_version: 1
datasets:
  - id: math
    harness: retool
    path: ${PD_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl
    weight: 1
    options:
      max_response_tokens: 8192
  - id: qa
    harness: browsecomp
    path: ${PD_DATA_ROOT}/browsecomp/bc_train.jsonl
    weight: 1
    options:
      search_url: http://127.0.0.1:8000
sampling:
  policy: random
  seed: 2026
  preserve_source_order: false
  shuffle_algorithm: legacy_two_stage_v1
  count_algorithm: legacy_two_dataset_round_v1
  pool_reuse_algorithm: cover_all_cycle_v1
```

`legacy_two_stage_v1` reproduces the original Retool/BrowseComp shuffle and
`legacy_two_dataset_round_v1` preserves its exact two-pool rounding. New
multi-dataset workloads should use `largest_remainder_v1`.
`cover_all_cycle_v1` also preserves the old behavior where a smaller dataset
is deterministically recycled until the mixed epoch covers every row of the
largest dataset. `cycle_as_needed_v1` only repeats rows when the requested run
is longer than a source.
Every run writes both `resolved_workload.json` and the exact reusable
`dispatch_sequence.json`; use the latter as a fixed schedule when an A/B test
must consume identical sample identities in identical order.

To replay a run, copy its workload config, change `sampling.policy` to
`fixed`, and set `sampling.schedule_file` to that run's
`dispatch_sequence.json`. Keep the same seed, shuffle algorithm, count
algorithm, dataset order and source files.

## Adding a dataset

Create `data/<name>/__init__.py`, `loader.py`, and `harness.py`, then export a
`HarnessSpec`. No central `if dataset == ...` branch is needed. Add the source
to a workload config and select its mixture using `weight`. Dataset-specific
settings belong under `options`; generic serving settings remain command-line
arguments.

The `terminal_bench` package is a concrete third example. Its OpenEnv service
must be started manually and its endpoint passed as `environment_url`.
