# Agentic PD modification gate

Before modifying any file under this directory, read
`docs/AGENTIC_PD_DESIGN_INVARIANTS.md` completely.

The design document is the source of truth. Every implementation plan must map
the proposed change to the snapshot ownership transition it affects. Do not
introduce a timeout, credit, reservation, cache, retry, or fallback that violates
the eight acceptance criteria in that document.

Before running a GPU experiment after code changes:

1. Run the relevant lifecycle, cancellation, capacity, TP, and fault-path tests.
2. Re-read and explicitly check all eight acceptance criteria.
3. Obtain an independent code/state-machine audit and require a GO result.
4. In the experiment, verify ownership conservation such as
   `queued == durable == source_release` and account for every outstanding item.

If context has been compacted or the current code baseline is unclear, stop and
re-read the design document before continuing. Do not infer the design from the
most recent failure or patch alone.

Final performance acceptance always requires a closed-loop run with at least
300 seconds of steady-state warmup followed by 1200 seconds of measured time.
Shorter runs are engineering smoke tests only and must not be reported as final
performance results.
