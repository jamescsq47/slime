# BFCL r2 — diagnostic only, superseded by r3

20 cases reached terminal state in 113.57 seconds: 7 official checker passes,
7 adapter parse-error task exits. Not the reported final probe result.

The adapter differed from BFCL in two ways: bare argument names were rejected
instead of becoming strings; malformed calls aborted the entire task instead
of just the current user turn. Both were aligned with official source before
r3, with tests and independent GO. No PD/SGLang code was modified.
