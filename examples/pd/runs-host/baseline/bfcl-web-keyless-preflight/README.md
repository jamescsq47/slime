# BFCL keyless integration status

Status: initial Yahoo-only preflight failed; subsequent explicit **auto
metasearch preflight passed**. Colocated GPU validation is the next step.

Latest check (`auto_search_and_fetch_check.jsonl`): both search relevance
checks passed (15.726 s and 16.388 s), webpage fetch passed (0.449 s). This
includes metasearch internal waits/timeouts, not artificially inserted sleep.
Latest independent review: GO; 40 BFCL/workload tests plus 11 custom-environment
lifecycle tests passed. No PD state machine was changed or exercised.

Completed:

- Downloaded 100 BFCL V4 Web Search questions, separate gold answers and tools.
- Added independent colocated harness and repeatable configuration.
- 39 CPU tests passed: 18 new harness tests and 21 existing workload tests.
- 11 existing request-generation lifecycle tests passed in the custom `pd`
  environment (these intentionally cannot import custom modules from the clean
  `pd_baseline` environment). Total: 50 passing tests.
- Real Qwen3-8B tokenizer validated (including tool-message boundaries).
- Independent revised-code audit: GO for colocated only, conditional on live
  search/fetch preflight passing before a GPU run.

Earlier Yahoo-only preflight (`search_and_fetch_check.jsonl`):

| Probe | Result | Tool elapsed time |
|---|---|---:|
| Marie Curie first Nobel prize year | Correct search results | 0.856 s |
| Super Bowl 2024 halftime performer | Search library returned no results | 0.622 s |
| Fetch Marie Curie Wikipedia page | Successful content retrieval | 0.512 s |

These three probes do not establish a latency distribution or sustained
capacity. A previous search of the same Curie question took 1.130 s.
Do not infer that BFCL tools consistently exceed 1 second from these samples.

Other public HTML probes encountered DuckDuckGo/Brave challenges or rate
limits, Google JavaScript-only responses, and Bing query/result mismatches.
No CAPTCHA bypass, credential acquisition, mock search, artificial tool sleep,
or substitution with BrowseComp's local corpus was performed.

Re-run the baseline launcher once keyless web search is reliably reachable;
it will repeat the mandatory search/fetch gate before GPU initialization.
