# Diagnostic run: action boundary bug, not accepted

Six tasks ran but the RecordedModel adapter ignored CodeAgent's stop strings
after stripping Qwen reasoning. A model could produce code, fabricate an
Observation, and produce more code in the same turn; CodeAgent then combined
multiple blocks. This is not the intended one-action/real-observation workflow.

Corrected by applying the exact CodeAgent stop strings locally after `</think>`;
incomplete reasoning cannot execute any code. Original raw content/token counts
remain recorded, so unused generation is explicit. No scientific/data answers
or model settings were changed. Rerun the same first six dev tasks as r3.

All containers and GPU processes were cleaned. Preserve these diagnostic
trajectories to explain the change, not as a benchmark/accuracy result.
