# 64-Trace Fixed-Context KL Matrix

## Definition

- Reference distribution: base `Qwen3-8B`.
- Candidate distributions: Math-only, BrowseComp-only, or Mixed checkpoints.
- Probe sets: 64 fixed, base-generated correct Math trajectories and 64 fixed,
  base-generated correct BrowseComp (QA) trajectories.
- All checkpoints use exactly the same selected records (seed `20260724`).
- Sequences longer than 8192 tokens are truncated to their contiguous prefix.
- Reported value is the assistant-token-weighted, full-vocabulary forward KL
  `KL(P_base(.|x,y_<t) || P_checkpoint(.|x,y_<t))`.
- `A→B` means a checkpoint trained on A evaluated on fixed B trajectories.

The Math probe contains 242,378 scored assistant tokens. The QA probe contains
15,844 scored assistant tokens. This is a fixed-context finite-displacement
measurement related to the Fisher quadratic form; it is not an on-policy
behavioral evaluation or a direct infinitesimal FIM-vector product.

## Forward KL

| Iter | Math→Math | Math→QA | QA→QA | QA→Math |
|---:|---:|---:|---:|---:|
| 004 | 0.000658 | 0.000678 | 0.000989 | 0.000644 |
| 009 | 0.000681 | 0.000719 | 0.003843 | 0.000645 |
| 014 | 0.000757 | 0.000772 | 0.009927 | 0.000646 |
| 019 | 0.000825 | 0.000821 | 0.017626 | 0.000661 |
| 024 | 0.000919 | 0.000717 | 0.025496 | 0.000666 |
| 029 | 0.001002 | 0.000945 | 0.034025 | 0.000682 |
| 034 | 0.001073 | 0.000991 | 0.044017 | 0.000700 |
| 039 | 0.001166 | 0.000925 | 0.056308 | 0.000691 |
| 044 | 0.001301 | 0.001126 | 0.068096 | 0.000707 |
| 049 | 0.001390 | 0.001044 | — | — |
| 054 | 0.001450 | 0.001198 | 0.088131 | 0.000713 |
| 059 | 0.001560 | 0.001145 | 0.095251 | 0.000732 |
| 064 | 0.001616 | 0.001243 | — | — |
| 069 | 0.001720 | 0.001363 | — | — |
| 074 | 0.001816 | 0.001390 | — | — |
| 079 | 0.001886 | 0.001398 | — | — |
| 084 | 0.002021 | 0.001415 | — | — |
| 089 | 0.002090 | 0.001657 | — | — |

## Fixed-continuation NLL change

`ΔNLL = NLL_checkpoint - NLL_base`; negative means that the checkpoint assigns
more probability to the fixed correct continuation.

| Iter | Math→Math | Math→QA | QA→QA | QA→Math |
|---:|---:|---:|---:|---:|
| 004 | -0.000175 | +0.002845 | -0.000776 | +0.000233 |
| 009 | -0.000380 | +0.000256 | +0.004638 | +0.000251 |
| 014 | -0.000311 | +0.001402 | +0.010227 | +0.000322 |
| 019 | -0.000562 | +0.001494 | +0.015775 | +0.000444 |
| 024 | -0.001037 | +0.001776 | +0.022271 | +0.000498 |
| 029 | -0.001388 | +0.000634 | +0.033095 | +0.000476 |
| 034 | -0.001556 | +0.000550 | +0.039562 | +0.000543 |
| 039 | -0.002117 | -0.000826 | +0.049286 | +0.000685 |
| 044 | -0.002574 | -0.001585 | +0.056434 | +0.000659 |
| 049 | -0.003071 | -0.004363 | — | — |
| 054 | -0.003456 | -0.004334 | +0.068953 | +0.000859 |
| 059 | -0.003616 | -0.007289 | +0.077632 | +0.001056 |
| 064 | -0.003790 | -0.005653 | — | — |
| 069 | -0.004072 | -0.007717 | — | — |
| 074 | -0.003859 | -0.007551 | — | — |
| 079 | -0.004050 | -0.008456 | — | — |
| 084 | -0.004346 | -0.009208 | — | — |
| 089 | -0.004635 | -0.008920 | — | — |

## Mixed checkpoints

| Candidate | Probe | Forward KL | 95% trace-bootstrap CI | ΔNLL |
|---|---|---:|---:|---:|
| Mixed iter099 | Math | 0.001092 | [0.001033, 0.001160] | -0.000302 |
| Mixed iter099 | QA | 0.059205 | [0.053023, 0.066052] | +0.020532 |
| Mixed iter199 | Math | 0.002093 | [0.001981, 0.002222] | +0.000288 |
| Mixed iter199 | QA | 0.191180 | [0.175379, 0.208135] | +0.171034 |
| Mixed iter299 | Math | 0.003432 | [0.003252, 0.003639] | -0.000644 |
| Mixed iter299 | QA | 0.403642 | [0.372927, 0.435946] | +0.392401 |

The QA/Math KL ratio grows from about 54.2 at iter099, to 91.3 at iter199, and
117.6 at iter299. Mixed checkpoint iterations should not be treated as directly
budget-matched to single-domain checkpoint iterations without checking the
number of examples/tokens contributed by each domain.

## Uncertainty at representative endpoints

The intervals below resample whole trajectories with replacement and recompute
the token-weighted mean; tokens are not treated as independent observations.

| Direction | Iter | Forward KL | 95% CI |
|---|---:|---:|---:|
| Math→Math | 044 | 0.001301 | [0.001195, 0.001444] |
| Math→QA | 044 | 0.001126 | [0.000901, 0.001439] |
| QA→QA | 044 | 0.068096 | [0.060710, 0.076436] |
| QA→Math | 044 | 0.000707 | [0.000679, 0.000734] |
| Math→Math | 069 | 0.001720 | [0.001616, 0.001845] |
| Math→QA | 069 | 0.001363 | [0.001209, 0.001541] |

## Stability relative to the 16-trace probe

Across checkpoints, the Pearson correlations between the 16- and 64-trace
curves are 0.999 (Math→Math), 0.838 (Math→QA), 1.000 (QA→QA), and 0.965
(QA→Math). Thus the central structural result is stable, although the small
Math→QA curve remains the noisiest one.

## Conclusions

1. BrowseComp-only training causes a large displacement on its own QA
   conditional distribution: KL rises from 0.000989 to 0.095251 by iter059.
2. The same BrowseComp updates barely move the Math conditional distribution:
   QA→Math stays between 0.000644 and 0.000732 through iter059. At iter059,
   QA→QA is about 130 times QA→Math.
3. Math-only updates move the two probe distributions by similar small
   magnitudes. At iter089, Math→Math is 0.002090 and Math→QA is 0.001657.
4. A negative fixed-trace ΔNLL is not proof of improved task accuracy. It only
   says the stored correct continuation became more likely under teacher-forced
   contexts. Actual generation/evaluation was intentionally not measured here.
5. These results support strongly anisotropic probability-space sensitivity,
   which Euclidean gradient cosine alone cannot characterize. Establishing the
   local quantity `g_A^T F_B g_A` directly would additionally require measuring
   sufficiently small updates (or an explicit FIM-vector product) at matched
   parameter points.
