# Qwen3-8B domain-interference analysis

This report intentionally excludes task accuracy and online sampling. It
measures parameter geometry, fixed-trace distribution drift, and matrix-free
empirical-Fisher interactions.

## Models

- Base: `/workspace/Qwen3-8B`
- Math-only: `/workspace/Qwen3-8B-retool-mask51200-collocate-partial`
- BrowseComp-only: `/workspace/Qwen3-8B-browsecomp-mask51200-collocate-partial`
- Mixed 50/50:
  `/workspace/Qwen3-8B-mixed-browsecomp-retool0.5-mask51200-51200-collocate-partial`

## Definitions

All gradients below use the mean assistant-token negative log likelihood on
fixed correct traces. Prompt and observation tokens are excluded.

- `QA -> Math`: `g_QA^T F_math g_QA`
- `Math -> QA`: `g_math^T F_QA g_math`
- Normalized risk: the quadratic form divided by the squared source-gradient
  norm.
- Euclidean cosine: ordinary cosine between `g_math` and `g_QA`.

The Fisher matrix is never materialized. For target-domain per-trace score
`s_i`, the quadratic is estimated as `mean_i (s_i^T g)^2`.

The final Fisher table uses 8 fixed traces per domain and 512 tokens per
trace. Confidence intervals are trace-level bootstrap 95% intervals with
20,000 resamples. KL uses 16 fixed traces per domain and up to 2,048 tokens.

## 1. Exact parameter-update geometry

Cosine between the two single-domain cumulative checkpoint deltas relative
to Base:

| Step | Math delta norm | QA delta norm | Euclidean cosine |
|---:|---:|---:|---:|
| 4 | 0.05332 | 0.05679 | 0.000632 |
| 9 | 0.09367 | 0.09725 | 0.000476 |
| 14 | 0.13909 | 0.14315 | 0.000160 |
| 19 | 0.16932 | 0.17639 | -0.000200 |

The realized single-domain training trajectories are therefore almost
exactly Euclidean-orthogonal at every matched checkpoint.

## 2. Cross-domain distribution drift

### Math-only update measured on BrowseComp traces

| Math checkpoint | Forward KL from Base | BrowseComp NLL change |
|---:|---:|---:|
| 4 | 0.000734 | +0.00117 |
| 9 | 0.000722 | +0.00375 |
| 14 | 0.000795 | +0.00484 |
| 19 | 0.000842 | +0.00500 |

### BrowseComp-only update measured on Math traces

| QA checkpoint | Forward KL from Base | Math NLL change |
|---:|---:|---:|
| 4 | 0.000548 | -0.00036 |
| 9 | 0.000562 | +0.00011 |
| 14 | 0.000561 | +0.00053 |
| 19 | 0.000558 | +0.00053 |

The same-model self-check gives exactly zero KL and zero NLL difference, so
the non-zero values are not an implementation floor.

The Math-only trajectory produces a small but increasing change in QA NLL.
The QA-only trajectory causes a similarly small Math-distribution change,
but it is nearly flat over steps 4-19.

## 3. Cross-domain Fisher interactions

Raw values are followed by trace-bootstrap 95% confidence intervals.

| Checkpoint | Gradient cosine | QA -> Math raw | QA -> Math normalized | Math -> QA raw | Math -> QA normalized |
|---|---:|---:|---:|---:|---:|
| Base | 0.2924 | 38,697 [22,247, 54,866] | 15.39 | 34,669 [28,181, 42,810] | 219.68 |
| Math 19 | 0.2981 | 40,736 [23,865, 57,696] | 15.57 | 36,610 [29,999, 45,041] | 236.97 |
| QA 19 | 0.2941 | 42,496 [24,415, 59,929] | 15.53 | 38,011 [31,292, 46,112] | 241.04 |
| Mixed 99 | 0.2945 | 21,268 [12,160, 30,310] | 14.93 | 18,913 [16,059, 22,368] | 125.13 |
| Mixed 199 | 0.2932 | 33,427 [18,722, 48,517] | 15.57 | 29,425 [24,733, 34,913] | 187.52 |
| Mixed 299 | 0.2553 | 103,161 [58,089, 146,992] | 12.79 | 90,317 [76,073, 103,862] | 532.30 |

Interpretation:

- Ordinary gradient cosine remains around 0.29 through Mixed 199 and therefore
  does not reveal the large changes in Fisher risk.
- At Mixed 99 both raw cross-domain risks fall substantially. Much of this is
  caused by smaller gradients, but normalized `Math -> QA` risk also falls
  from 219.68 to 125.13.
- At Mixed 199 the risks partially return toward Base.
- At Mixed 299 both raw risks increase sharply. The two directions are not
  geometrically equivalent:
  - normalized `QA -> Math` falls to 12.79, so the QA direction is not moving
    into a more curved Math direction; its raw risk rises because the QA
    gradient norm becomes much larger;
  - normalized `Math -> QA` rises to 532.30, 2.42 times Base, which is direct
    evidence that the Math gradient points into a much more QA-sensitive
    direction at this checkpoint.

## 4. Mixed-model KL

| Mixed checkpoint | Math forward KL | Math NLL change | QA forward KL | QA NLL change |
|---:|---:|---:|---:|---:|
| 99 | 0.001190 | -0.00232 | 0.06315 | -0.06967 |
| 199 | 0.002268 | -0.00261 | 0.17754 | +0.16598 |
| 299 | 0.003697 | -0.00325 | 0.43203 | +0.57222 |

Across the three Mixed checkpoints, both Fisher raw risks and target-domain KL
increase monotonically. The QA side changes far more strongly than the Math
side. With only three Mixed points this is a trajectory observation, not a
general statistical law.

## Conclusions

1. The two realized single-domain parameter updates are almost perfectly
   Euclidean-orthogonal, yet both produce non-zero target-domain KL and
   non-zero cross-domain Fisher quadratic forms. Euclidean orthogonality does
   not imply distributional independence in this experiment.
2. Early single-domain cross-effects are small. Math-only training shows a
   gradual QA NLL drift through step 19; QA-only training has an almost flat
   effect on Math traces over the same interval.
3. Mixed training initially reduces cross-domain risk, especially normalized
   `Math -> QA` risk, but this compatibility is not stable.
4. Mixed 299 is strongly asymmetric: the large `QA -> Math` raw risk is mainly
   gradient-magnitude driven, while `Math -> QA` exhibits a genuine increase
   in Fisher-normalized curvature sensitivity.
5. Ordinary gradient cosine changes only modestly and misses the risk reversal
   between Mixed 99 and Mixed 299. Fisher geometry and actual KL expose it.

## Limits

- Fisher estimates use 8 fixed traces per domain; confidence intervals remain
  broad on the Math-target side.
- The traces are correct SFT/rollout trajectories, not fresh on-policy samples.
- NLL changes describe likelihood on fixed traces, not task accuracy.
- Mixed checkpoint numbers are treated as saved trajectory points; no claim is
  made that they have equal domain exposure to same-numbered single-domain
  checkpoints.
