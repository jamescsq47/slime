# Native Mooncake c576 — retry r5

This retry completes the baseline matrix using the untouched `pd_baseline`
SGLang 0.5.10.post1 engine and native HiCache/Decode-offload/Mooncake path.
The other agent's custom engines and shared launcher/harness files are untouched.

Configuration: Qwen3-8B, TP1, P=0/2/4/6, D=1/3/5/7, BrowseComp source-order
n680 cycling, c576, seed2026, temperature0/top-p1/top-k-1, context40960,
response budget36864, 2048 tokens/turn, max100 turns. P/D GPU fractions0.80,
except D7=0.60 with the search service. P Host128 GiB each, D Host56 GiB each,
Mooncake256 GiB. Full 300-second business warmup + 1200-second measurement.

The local `baseline_case.sh` is copied from the existing baseline launcher;
the only behavioral additions are explicit physical-GPU-local NUMA binding
for TP1 and a 1200-second scheduler watchdog. The shared launcher and the
installed engine are not edited. These startup/resource-placement differences
are recorded because older runs automatically bound every TP1 worker to NUMA0.
The watchdog setting also applies during serving; no claim is made that only
the startup watchdog changed. Healthy scheduling itself is unchanged.

Search-only preflight and identical embedding-cache hashes are recorded in
the adjacent r4 directory. That run failed before business traffic due to the
300-second NIXL initialization watchdog. All its GPU services were gone before
this retry started. The existing process-group cleanup traps remain enabled.
