# Validation notes

Validated on 2026-07-29 in the `search_r1` container:

- Standalone syntax/metric tests: 4 passed.
- Real workload-plugin dry run: the legacy CLI and the new 1:1 YAML config
  produced byte-identical 20-entry `dispatch_sequence.json` files. Dataset
  identity, source row and ordering therefore remain unchanged.
- Inference-only Retool, BrowseComp and Terminal-Bench harness tests explicitly
  disable token logprobs. The Terminal-Bench test executes a fake shell turn,
  terminates, evaluates and closes the externally managed environment.
- Qwen3-8B SGLang NIXL PD smoke on GPU2=P and GPU3=D: four capped requests at
  0.2 request/s reached the PD router with p95 queue delay below 0.02 ms.
  Measured engine counter rates were about 281.5 prefill prompt token/s and
  53.5 decode generation token/s. This short run deliberately capped outputs,
  so it validates plumbing rather than final capacity.
- BrowseComp search-only test on GPU2: query returned 3 results (top docid
  `14721`) in 6596 ms.
- QA generated real `/search` calls in a co-location probe. Search timed out
  when sharing GPU2 with prefill, confirming that search must use a third,
  dedicated GPU for valid agentic performance results.
- GPU occupancy protection refused to start on an occupied GPU. All experiment
  ports were closed and GPU2/3 memory returned to baseline after cleanup.

The node did not have a third idle GPU during validation; GPU0/1/4/5/6/7 were
running unrelated jobs and were not disturbed. Therefore 0.05 request/s is a
provisional default, not a claimed capacity result. Run the documented
`ARRIVAL_RATES="0.025 0.05 0.1"` sweep when three GPUs are idle; the launcher
will write the selected rate to `rate_sweep_summary.json`.

## Agentic request-generation KV lifecycle V1 (2026-08-07)

Implemented in the `pd` conda environment and disabled by default.  The
launcher `scripts/new_method/internal/run_agentic_pipeline.sh` opts in explicitly.

- One manifest owns one complete, page-aligned request-generation snapshot.
- D publishes `OFFLOADING` before Put and `MOONCAKE_READY` only after every
  physical K/V page exists. Final answers bypass Mooncake.
- P claims a snapshot once, requires an all-or-nothing GET, deletes Mooncake
  only after Host-to-GPU completion, and drops its private GPU/Host branch only
  after the P-to-D ACK.
- Abort, partial GET, leased delete, Put/commit failure and stale owner paths
  retain tombstones and retry without force-removing an active Mooncake page.
- Admission evicts complete READY snapshots by request-level cost; physical
  storage remains page based. All D workers share one locked `/dev/shm`
  reservation/residency ledger, so the configured 90% agentic budget is no
  longer divided into independent per-D estimates.
- A 0.2-second fast-tool window publishes a metadata-only offer. If the next
  P turn has enough HBM it reserves the complete destination and uses a
  role-reversed NIXL GPU-to-GPU transfer; failed or expired handshakes fall
  back to the complete Mooncake snapshot path.
- Retool and BrowseComp both send stable request id, generation, parent and
  tool classification metadata. BrowseComp also sends terminal markers so a
  `<function=finish>` answer is not backed up as another tool generation.
- Dataset markers are Unicode-text canonical and travel in a bounded Base64
  JSON envelope. Single-string markers, punctuation, newlines, slashes, emoji,
  and combining characters are handled without relying on tokenizer-specific
  delimiter tokens.
- A P cache cleanup that races with asynchronous Host write-through is retried
  after HiCache ACK processing rather than leaking the request branch.
- 35 isolated lifecycle/request-metadata tests passed. Modified SGLang modules
  passed `py_compile`.

Real 1P:2D GPU validation completed with Qwen3-8B on GPU0/1/2. The direct NIXL
path and the Mooncake fallback path both reused a complete 576-token parent
snapshot; the terminal path skipped reverse offload. A special-character id
`新数据集:QA/🔧-é-v13` survived the router and reused 640 cached tokens. Four
concurrent validators also completed across both D workers. The final shared
ledger had no reservations or residents and the logs had no metadata loss,
scheduler exception, deferred-release timeout, or traceback. This was a
plumbing-validation run rather than a retained formal throughput result; its
obsolete run directory was removed during repository cleanup.

The shared ledger governs the agentic namespace; native HiCache objects, old
foreign keys and other Mooncake writers are still outside that logical
accounting and remain protected by Mooncake's physical watermark.

The complete `examples/pd/inference.py` source was reconstructed from the
original file plus its successful patch history and replaces the former
CPython-3.12-only bytecode wrapper. It compiles and runs directly; no runtime
payload under `__pycache__` is required.
