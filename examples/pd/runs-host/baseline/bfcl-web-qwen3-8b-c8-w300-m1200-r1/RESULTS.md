# Terminated during warmup — no formal performance result

Live Qwen3-8B colocated c8 ran real BFCL questions and web tools. During warmup
we found the BFCL default region `wt-wt` was being passed directly to DDGS,
which interpreted `wt` as a language and attempted `wt.wikipedia.org`.
This added avoidable failed-provider work. Stopped the inference client and
let the launcher's tracked-service cleanup terminate model/router workers.

Fix: translate unspecified region `wt-wt` to DDGS English default `us-en`.
No generation budget, model setting, or engine lifecycle change.
This run is engineering validation only; no throughput is accepted.
