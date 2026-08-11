# Agentic PD serving experiments

This directory contains the mixed Retool/BrowseComp serving workload, the
clean SGLang baselines, and the request-generation KV pipeline.  Runtime
modules stay at the directory root because the rollout workers import them as
top-level modules; experiment launchers and one-off utilities do not.

## Layout

```text
examples/pd/
├── configs/                 environment and fixed workload schedules
├── docs/                    design and validation notes
├── patches/                 SGLang patch kept for reproducibility
├── scripts/
│   ├── baseline/            unmodified pd_baseline experiments
│   ├── new_method/          modified pd request-generation KV experiments
│   │   └── internal/        service lifecycle implementation; do not run directly
│   ├── bandwidth/           isolated NVLink/PCIe/Host/Mooncake microbenchmarks
│   ├── tools/               analysis, validation and correctness utilities
│   └── common/              shared process cleanup helpers
├── tests/                   focused unit tests for the PD extensions
├── runs-host/
│   ├── baseline/            retained formal baseline results only
│   └── new-method/          retained latest-method result only
└── *.py                     serving workload/runtime modules
```

Historical sweep, smoke-test, superseded launcher and temporary visualization
scripts have been removed.  See each `scripts/*/README.md` for supported entry
points and `runs-host/README.md` for the retained result inventory.

## Prerequisites

The validated setup uses Linux, NVIDIA GPUs with peer-to-peer access, Conda,
and a CUDA 12.x-compatible driver.  The launchers assume that the repository
root is on `PYTHONPATH`; Slime does not need to be installed as a wheel.

Set reusable locations before preparing the model and datasets:

```bash
export SLIME_ROOT=/path/to/slime
export PD_DATA_ROOT=/path/to/pd-data
export PD_MODEL_ROOT=/path/to/models
mkdir -p "${PD_DATA_ROOT}" "${PD_MODEL_ROOT}"
cd "${SLIME_ROOT}"
```

If Hugging Face requires authentication, run `hf auth login` first.  Keep the
model, datasets, Hugging Face cache and experiment outputs on local storage
rather than NFS when possible.

## Model and datasets

Download the generation model:

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-8B \
  --local-dir "${PD_MODEL_ROOT}/Qwen3-8B"
```

Download the exact Slime-ready DAPO-Math-17k JSONL used by these experiments:

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "${PD_DATA_ROOT}/dapo-math-17k"
```

Do not substitute the current default split of
`BytedTsinghua-SIA/DAPO-Math-17k`: its upstream contents have changed and no
longer represent the fixed 17k-row JSONL used in the retained runs.

The BrowseComp workload used by the retained experiments is the FoldAgent
`bc_train.parquet`, not the test-only `Tevatron/browsecomp-plus` dataset.  Fetch
the public FoldAgent archive and convert it as follows:

```bash
python -m pip install gdown pandas pyarrow
mkdir -p "${PD_DATA_ROOT}/browsecomp/raw"
python -m gdown 1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM \
  -O "${PD_DATA_ROOT}/browsecomp/raw/browsecomp_data.zip"
unzip -o "${PD_DATA_ROOT}/browsecomp/raw/browsecomp_data.zip" \
  -d "${PD_DATA_ROOT}/browsecomp/raw"

python examples/pd/scripts/tools/prepare_workloads.py browsecomp \
  --input "${PD_DATA_ROOT}/browsecomp/raw/bc_train.parquet" \
  --output "${PD_DATA_ROOT}/browsecomp/bc_train.jsonl"
```

Some versions of the archive contain an extra directory level.  In that case,
locate the file with `find "${PD_DATA_ROOT}/browsecomp/raw" -name
'bc_train.parquet'` and pass the returned path to `--input`.

BrowseComp search additionally uses:

- `Qwen/Qwen3-Embedding-8B` for query embeddings;
- `Tevatron/browsecomp-plus-corpus` for the 100k-document corpus;
- `miaolu3/browsecomp-plus` for precomputed corpus embeddings.

`search_server.py` downloads these resources on first launch.  To prefetch
them into a persistent cache:

```bash
export HF_HOME="${PD_DATA_ROOT}/huggingface-cache"
hf download Qwen/Qwen3-Embedding-8B
hf download --repo-type dataset Tevatron/browsecomp-plus-corpus
hf download --repo-type dataset miaolu3/browsecomp-plus
```

All launchers accept `MODEL_PATH`, while `inference.py` accepts `--math-data`
and `--qa-data`.  For example:

```bash
export MODEL_PATH="${PD_MODEL_ROOT}/Qwen3-8B"
export MATH_DATA="${PD_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
export QA_DATA="${PD_DATA_ROOT}/browsecomp/bc_train.jsonl"
```

The maintained formal launchers already pass their workload through
`inference.py`; override its defaults through launcher environment variables
or edit a copied experiment launcher when using non-default dataset paths.

## Baseline environment

Create the clean `pd_baseline` environment from the maintained package list:

```bash
cd "${SLIME_ROOT}"
conda env create -n pd_baseline -f examples/pd/configs/environment-pd.yml
conda run -n pd_baseline python -m pip install -r requirements.txt gdown pyarrow
```

The validated baseline package versions are Python 3.12, SGLang
`0.5.10.post1`, SGLang Router `0.3.2`, Mooncake Transfer Engine
`0.3.12.post1`, NIXL `1.3.2`, and PyTorch `2.9.1`.  Verify that no custom
agentic modules leaked into it:

```bash
conda run -n pd_baseline python \
  examples/pd/scripts/tools/check_environments.py --expect baseline
```

The scripts default to `/homes/siqic/anaconda3/envs/pd_baseline/bin` and
`/dataset/model/qwen3/Qwen3-8B`.  On another machine, provide both overrides:

```bash
PD_ENV_BIN="$(conda run -n pd_baseline which python | xargs dirname)" \
MODEL_PATH="${PD_MODEL_ROOT}/Qwen3-8B" \
MATH_DATA="${PD_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
QA_DATA="${PD_DATA_ROOT}/browsecomp/bc_train.jsonl" \
bash examples/pd/scripts/baseline/run_four_gpu_comparison_suite.sh
```

## Modified environment

- `pd_baseline`: clean SGLang 0.5.10.post1 baseline.
- `pd`: modified SGLang environment containing the agentic KV lifecycle.
- `configs/environment-pd.yml`: reproducible modified-environment package list.

The YAML recreates the common binary/package layer for both environments.  The
new method also requires the modified SGLang sources currently installed in
the original `pd` Conda environment.  The patch under `patches/` records only
an earlier subset and is not yet a complete standalone installer; copying this
directory alone is therefore sufficient for baseline experiments, but not yet
for reproducing the complete request-generation KV pipeline on a new node.

Validate without allocating GPUs:

```bash
cd "${SLIME_ROOT}"
conda run -n pd python \
  examples/pd/scripts/tools/check_environments.py --expect modified
conda run -n pd_baseline python \
  examples/pd/scripts/tools/check_environments.py --expect baseline
```

## Formal experiments

Run the four-GPU colocated/1P:3D baseline comparison:

```bash
bash examples/pd/scripts/baseline/run_four_gpu_comparison_suite.sh
```

Run the four-GPU 2P:2D baseline comparison:

```bash
bash examples/pd/scripts/baseline/run_two_p_two_d_comparison_suite.sh
```

Run the current 1P:3D request-generation KV method:

```bash
bash examples/pd/scripts/new_method/run_1p3d_case.sh
```

All formal launchers default to a 300-second warmup and 1200-second
measurement window.  They use process groups and bounded TERM-to-KILL cleanup
so a failed run does not intentionally leave model services behind.

Runtime `.log` files, local search artifacts, profiler databases and temporary
`runs/` directories are ignored by Git.  Retained result summaries and plots
under `runs-host/` remain versionable.

## Tests

```bash
cd /homes/siqic/slime
PYTHONPATH=examples/pd \
  /homes/siqic/anaconda3/envs/pd/bin/python -m pytest -q \
  -p no:cacheprovider examples/pd/tests
```

Serving-level validation and bandwidth tests are intentionally separate from
unit tests; they require idle GPUs and are documented under `scripts/tools/`
and `scripts/bandwidth/`.
