# Fixed-trace KL matrix

## Scope

This analysis uses fixed Base-Qwen3-8B correct trajectories as a common probe.
It is a conditional-distribution comparison, not an on-policy Fisher estimate
and not a task-accuracy evaluation.

- Reference model: `/workspace/Qwen3-8B`
- Math checkpoints:
  `/workspace/Qwen3-8B-retool-mask51200-collocate-partial/iter*`
- QA checkpoints:
  `/workspace/Qwen3-8B-browsecomp-mask51200-collocate-partial/iter*`
- Math probe: 16 fixed correct trajectories, 67,788 assistant tokens
- QA probe: 16 fixed correct BrowseComp trajectories, 3,723 assistant tokens
- Context: contiguous prefix, up to 8,192 tokens; no head/tail splicing
- Metric: Base-to-checkpoint forward KL, averaged per assistant token, in nats

The labels mean:

- Math -> Math: Math-only checkpoint evaluated on fixed Math contexts
- Math -> QA: Math-only checkpoint evaluated on fixed QA contexts
- QA -> QA: BrowseComp-only checkpoint evaluated on fixed QA contexts
- QA -> Math: BrowseComp-only checkpoint evaluated on fixed Math contexts

## Forward-KL table

| Step | Math -> Math | Math -> QA | QA -> QA | QA -> Math |
|---:|---:|---:|---:|---:|
| 4 | 0.000627 | 0.000766 | 0.001188 | 0.000600 |
| 9 | 0.000648 | 0.000877 | 0.004231 | 0.000611 |
| 14 | 0.000708 | 0.000888 | 0.011216 | 0.000608 |
| 19 | 0.000787 | 0.000890 | 0.019156 | 0.000614 |
| 24 | 0.000874 | 0.000925 | 0.027114 | 0.000621 |
| 29 | 0.000943 | 0.001101 | 0.036070 | 0.000639 |
| 34 | 0.001021 | 0.000883 | 0.046018 | 0.000644 |
| 39 | 0.001114 | 0.001253 | 0.057771 | 0.000634 |
| 44 | 0.001202 | 0.001133 | 0.068484 | 0.000660 |
| 49 | 0.001301 | 0.001154 | — | — |
| 54 | 0.001377 | 0.001293 | — | — |
| 59 | 0.001439 | 0.001490 | — | — |
| 64 | 0.001530 | 0.001449 | — | — |
| 69 | 0.001634 | 0.001294 | — | — |

## Fixed-correct-token NLL change

Negative values mean that the checkpoint assigns higher probability than Base
to the fixed correct continuation. Positive values mean lower probability.

| Step | Math -> Math | Math -> QA | QA -> QA | QA -> Math |
|---:|---:|---:|---:|---:|
| 4 | -0.000027 | +0.001540 | +0.000206 | +0.000135 |
| 9 | -0.000217 | +0.001665 | +0.005161 | +0.000193 |
| 14 | -0.000323 | +0.001877 | +0.009893 | +0.000207 |
| 19 | -0.000730 | +0.000248 | +0.015810 | +0.000321 |
| 24 | -0.000915 | +0.003516 | +0.019470 | +0.000251 |
| 29 | -0.001314 | +0.003522 | +0.028240 | +0.000496 |
| 34 | -0.001422 | +0.002357 | +0.035942 | +0.000367 |
| 39 | -0.002161 | +0.001465 | +0.042846 | +0.000470 |
| 44 | -0.002483 | +0.001027 | +0.047912 | +0.000336 |
| 49 | -0.002882 | -0.004359 | — | — |
| 54 | -0.003177 | -0.002741 | — | — |
| 59 | -0.003308 | -0.005814 | — | — |
| 64 | -0.003647 | -0.003725 | — | — |
| 69 | -0.003888 | -0.008304 | — | — |

## Endpoint uncertainty

Trace-level bootstrap 95% confidence intervals:

| Comparison | KL | 95% CI |
|---|---:|---:|
| Math 44 -> Math | 0.001202 | [0.001031, 0.001402] |
| Math 44 -> QA | 0.001133 | [0.000854, 0.001484] |
| QA 44 -> QA | 0.068484 | [0.057456, 0.079711] |
| QA 44 -> Math | 0.000660 | [0.000591, 0.000726] |
| Math 69 -> Math | 0.001634 | [0.001406, 0.001884] |
| Math 69 -> QA | 0.001294 | [0.001080, 0.001540] |

## Conclusions

1. Math-only training changes Math and QA conditional distributions on a
   similar order of magnitude. At step 44, Math -> Math is 0.001202 and
   Math -> QA is 0.001133; their bootstrap intervals overlap.
2. Math-only drift is slow. By step 69, Math -> Math reaches 0.001634 and
   Math -> QA reaches 0.001294. The cross-domain curve is noisier but does not
   show runaway growth.
3. BrowseComp-only training is highly domain-local in this fixed probe.
   QA -> QA grows monotonically from 0.001188 to 0.068484 by step 44, while
   QA -> Math stays nearly flat from 0.000600 to 0.000660.
4. At step 44, QA -> QA KL is about 104 times QA -> Math KL. BrowseComp
   training substantially restructures the QA policy distribution without
   comparably restructuring the Math conditional distribution.
5. KL magnitude alone does not determine whether a change is beneficial.
   Math-only lowers Math fixed-trace NLL throughout. Its QA NLL is slightly
   worse through step 44 but becomes better than Base from step 49 onward.
   This is compatible with positive transfer on these fixed QA continuations,
   but it is not an accuracy result.
6. BrowseComp-only increases QA fixed-trace NLL strongly while its QA KL grows.
   This means it moves away from the Base-generated correct SFT trajectories;
   it does not by itself prove worse BrowseComp behavior, because RL may learn
   different successful search trajectories.

## Important limitation

The contexts come from Base-generated correct trajectories. These results
answer how each checkpoint changes its conditional next-token distribution on
a common, controlled state set. They do not estimate the state distribution
that each checkpoint would visit on-policy.
