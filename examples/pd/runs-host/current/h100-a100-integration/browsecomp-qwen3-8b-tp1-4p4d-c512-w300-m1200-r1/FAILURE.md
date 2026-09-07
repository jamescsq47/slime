# C512 integrated run: terminated, not a formal throughput result

SGLang integration commit: `20849bd039`; Slime launch commit: `4576da5`.
Qwen3-8B / BrowseComp source-order n680 / TP1 / 4P:4D / c512 / temperature 0.
Intended timing: 300 seconds warmup + 1200 seconds measurement.

## Observed sequence (UTC, 2026-09-07)

- 04:02:36: workload loaded and closed-loop traffic began shortly afterward.
- Until about 04:03:48: Decode progressed normally with roughly 83–99 running
  requests per D at the first manual poll.
- About 04:03:49 onward: first-touch eager registration of complete Host arenas
  caused long interruptions to CUDA submissions. D0 logged 128 GiB arena
  prewarm times of 62.717 / 62.321 / 68.605 seconds. Its slow progress maximum
  durations closely matched the first two prewarm times (62.743 / 62.342 s).
  Other D workers showed the same pattern. Individual extent registration
  calls also took about 3–8 seconds.
- Python stack samples found the D main thread in CUDA tensor clone, prefix
  match, or event operations while a background thread was registering memory.
  This supports registration-related submission blocking; a specific CUDA
  driver mutex was not proven. The small index-clone safety fix was not
  removed. Independent read-only audit agreed with this interpretation.
- By roughly 04:09–04:10, several workers finished most arena registrations;
  Host transfers and Decode resumed, but the startup disturbance had already
  extended past the intended warmup. This is not a stable comparison window.
- 04:10:40: D3 (physical GPU7, process-local GPU0) exited with CUDA OOM while
  allocating a 2 MiB tensor in `clamp_position_cuda`. The error reported a
  79.25 GiB device, 2.62 MiB free, D process usage 62.42 GiB and the colocated
  search process usage 15.58 GiB. D3 used mem_fraction_static=0.74 to retain
  the previous A100 configuration; the H100 wrapper default is lower.
- Router subsequently returned HTTP 500 / ServerDisconnectedError.
- The experiment supervisor was terminated after diagnosis. No valid formal
  throughput, parent reuse percentage or completed-window summary is claimed.

## Evidence

- `logs/decode-0.log`: eager registration timing and progress interruptions.
- `logs/decode-3.log`: first fatal OOM at 04:10:40, followed by SIGQUIT.
- `logs/router.log`: downstream HTTP 500 after D3 exit.
- `monitor_samples.json`: raw live metric counter samples. Missing endpoint
  samples are errors, NOT zeros. Counter updates can be delayed across long
  CUDA stalls, so short-window Forward deltas are not exact kernel occupancy.

## Required follow-up before another comparable formal run

1. Complete Host registration before timed workload warmup, or verify a
   bounded registration approach that does not cause minute-long submission
   stalls. Merely using a background Python thread was insufficient here.
2. Budget for the search process on GPU7 (or reduce its peak memory use).
   Changing D3's static KV fraction changes its KV capacity and must be
   explicitly recorded; do not silently call it unchanged configuration.
3. Re-run the full warmup + measurement after independent review of changes.

No serving code or parameters were changed during this monitoring turn.
