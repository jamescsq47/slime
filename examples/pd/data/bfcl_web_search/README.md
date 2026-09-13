# BFCL V4 Web Search — keyless serving adapter

This adapter uses the original BFCL questions and two tool signatures:
`search_engine_query` and `fetch_url_content`. It searches public web pages
through DDGS's public-web metasearch (`search_backend: auto`); it needs no SerpAPI key or
paid search subscription. No unrelated local corpus, artificial delay, or
search-result cache is used. Network access is still required.

`auto` allows the library to choose public providers, including Wikipedia and
general web engines. It is not a fixed-provider benchmark. Disabled or unknown
provider names are rejected rather than silently mislabeled as that provider.
BFCL's unspecified `wt-wt` region is mapped to DDGS's `us-en` English default;
passing `wt-wt` verbatim makes some DDGS providers use an invalid language code.

Data: `/homes/siqic/data/bfcl-v4-web-search`. Source revision and checksums are
in that directory's README. The loader never opens `answers.jsonl`.

Configuration: `configs/experiments/bfcl_web_search.yaml`.

- Default: 100 questions, source-order cycling, base/snippets condition.
- `variants: [base, no_snippet]` runs the two conditions as 200 cases, not
  200 independent questions. Keep the selected condition fixed for comparisons.
- Native Qwen tool-call JSON; exact output token IDs retained across turns.
- 8192 output tokens/turn, 20 turns, 32768 cumulative response tokens
  (model output + tool observation), 40960 context with 512-token margin.
- Web fetch modes raw/markdown/truncate; 2 MB downloaded-body limit and
  40000-character returned-content limit. Returned content indicates truncation.
- At most 4 simultaneous searches and 8 fetches per inference process.
  `web_tool_events.elapsed_seconds` includes semaphore wait as well as network
  service time; it must not be labeled pure network latency.
- Backend failures return an error observation to the model, as in the
  reference tool; the third failed tool call aborts the trajectory. Every
  failed call remains recorded even if the agent later answers. Malformed
  tool arguments receive a repair observation. No infinite retries.
- Colocated only. PD lifecycle use fails explicitly before generation until a
  separate integration/audit adds application lifecycle acknowledgements.

This is a **serving workload adapter**, not the official BFCL accuracy runner:
the search provider, template, budgets, fetch limits and error handling differ
from the upstream SerpAPI harness. Predictions and tool events are retained,
but no official BFCL score is claimed. Live results also change over time.

## Run

Install `data/bfcl_web_search/requirements.txt` into the baseline environment.
Then, from `examples/pd`:

```bash
bash scripts/baseline/run_bfcl_web_colocated.sh
```

The launcher first checks two factual search queries and one real webpage
fetch. Failure exits before allocating GPUs. Default trial configuration is
Qwen3-8B, one GPU, TP=1, c8, temperature=0, mem_fraction_static=0.80,
300 seconds warmup + 1200 seconds measurement. Public search should pass a
low-concurrency trial before scaling; do not load-test a rate-limited endpoint
or reinterpret failed tool calls as successful high-throughput agents.

No SGLang files, existing harnesses, KV ownership transitions, TP behavior,
or existing experiment defaults are changed by this adapter.
