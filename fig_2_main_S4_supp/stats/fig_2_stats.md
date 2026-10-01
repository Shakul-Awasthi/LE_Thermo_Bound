# Figure 2: numbers for the manuscript

Input: `data_fig2/fig_2_inputs.csv`

## Quotable numbers (LaTeX macros in `fig_2_numbers.tex`)

| macro | value |
|---|---|
| `\FigTwoInputs` | 800,000 |
| `\FigTwoPerCase` | 100,000 |
| `\FigTwoFixed` | 292,382 |
| `\FigTwoCycles` | 441,964 |
| `\FigTwoChaos` | 65,654 |
| `\FigTwoMainFixedPlotted` | 205,371 |
| `\FigTwoMainFixedOutside` | 87,011 |
| `\FigTwoMainMovingOutside` | 10 |
| `\FigTwoMainMovingPlotted` | 2,000 |
| `\FigTwoMainCyclesPlotted` | 1,000 |
| `\FigTwoMainChaosPlotted` | 1,000 |
| `\FigTwoMainMovingAvailable` | 507,618 |
| `\FigTwoMainCellDecades` | 0.2 |
| `\FigTwoMainCap` | 35 |
| `\FigTwoMainCells` | 156 |
| `\FigTwoSatMedian` | 0.228 |
| `\FigTwoSatNinety` | 0.3 |
| `\FigTwoSatMax` | 0.53 |
| `\FigTwoSatChaosMax` | 0.345 |
| `\FigTwoSatCycleMax` | 0.53 |
| `\FigTwoViolations` | 0 |
| `\FigTwoBoundRatioMax` | 0.942 |
| `\FigTwoBudgetMedian` | 0.057 |
| `\FigTwoBudgetMax` | 0.1 |
| `\FigTwoPsRatioMovingMedian` | 0.293 |
| `\FigTwoIdentityResidual` | 3.5e-14 |
| `\FigTwoLiouvilleError` | 7.4e-07 |
| `\FigTwoCycleClosure` | 2.0e-08 |
| `\FigTwoFloquetMax` | 1 |
| `\FigTwoChaosLambdaRelErr` | 0.0422 |
| `\FigTwoAttempts` | 898,992 |
| `\FigTwoAcceptance` | 89 |
| `\FigTwoVerifyCyclesChecked` | 200 |
| `\FigTwoVerifyCyclesPassed` | 200 |
| `\FigTwoVerifyCycleClosure` | 2.0e-08 |
| `\FigTwoVerifyChaosChecked` | 200 |
| `\FigTwoVerifyChaosPassed` | 198 |
| `\FigTwoVerifyChaosLambdaDiff` | 0.0349 |

## Inputs per case

| case | n | fixed | cycle | chaos |
|---|---|---|---|---|
| wr/ideal | 100000 | 78054 | 21533 | 413 |
| wr/nonideal | 100000 | 80937 | 17325 | 1738 |
| brusselator/ideal | 100000 | 26192 | 73808 | 0 |
| brusselator/nonideal | 100000 | 35525 | 64475 | 0 |
| oregonator/ideal | 100000 | 10512 | 89488 | 0 |
| oregonator/nonideal | 100000 | 13026 | 86974 | 0 |
| cs14/ideal | 100000 | 22101 | 45844 | 32055 |
| cs14/nonideal | 100000 | 26035 | 42517 | 31448 |

## Sampling (attempts up to the last selected proposal index)

| case | attempts | accepted | acceptance | rejected |
|---|---|---|---|---|
| wr/ideal | 102271 | 100000 | 97.8% | unresolved 2271 |
| wr/nonideal | 102140 | 100000 | 97.9% | unresolved 2139, stiff (step budget) 1 |
| brusselator/ideal | 109657 | 100000 | 91.2% | unresolved 9374, stiff (step budget) 276, audit failed 7 |
| brusselator/nonideal | 109789 | 100000 | 91.1% | unresolved 9262, stiff (step budget) 515, audit failed 12 |
| oregonator/ideal | 101044 | 100000 | 99.0% | stiff (step budget) 844, unresolved 200 |
| oregonator/nonideal | 100397 | 100000 | 99.6% | stiff (step budget) 202, unresolved 195 |
| cs14/ideal | 125089 | 100000 | 79.9% | unresolved 21426, stiff (step budget) 3663 |
| cs14/nonideal | 148605 | 100000 | 67.3% | stiff (step budget) 5810, unresolved 42795 |

## Main figure

- fixed points (gray density): 205,371 inside the axes, 87,011 outside (not drawn), 292,382 in total
- cycles and chaos plotted: 2,000 of 507,618 available (1,000 cycles, 1,000 chaos); 505,618 not shown because of the per-cell cap
- density cells: 0.2 decade wide, 156 occupied, 1 to 35 points per cell (cap 35)
- axes [0.01, 1000.0]; cycles/chaos outside the axes 10; cycles/chaos under the inset 0 (fixed points under it 49); inset histogram built from 507,618 inputs

| network | cycles | chaos | ideal | nonideal |
|---|---|---|---|---|
| WR | 268 | 130 | 199 | 199 |
| Brusselator | 38 | 0 | 19 | 19 |
| Oregonator | 574 | 0 | 287 | 287 |
| CS14 | 120 | 870 | 495 | 495 |

## Supplementary figure (every input plotted)

| panel | network | mixing | plotted | fixed | cycle | chaos |
|---|---|---|---|---|---|---|
| (a) | WR | ideal | 100,000 | 78,054 | 21,533 | 413 |
| (a) | WR | nonideal | 100,000 | 80,937 | 17,325 | 1,738 |
| (b) | Brusselator | ideal | 100,000 | 26,192 | 73,808 | 0 |
| (b) | Brusselator | nonideal | 100,000 | 35,525 | 64,475 | 0 |
| (c) | Oregonator | ideal | 100,000 | 10,512 | 89,488 | 0 |
| (c) | Oregonator | nonideal | 100,000 | 13,026 | 86,974 | 0 |
| (d) | CS14 | ideal | 100,000 | 22,101 | 45,844 | 32,055 |
| (d) | CS14 | nonideal | 100,000 | 26,035 | 42,517 | 31,448 |

## Tightness

Saturation s = Δ̄₁²/(B̄₁σ̄_ps) of cycles and chaos:

| case | n | median | 90% | max |
|---|---|---|---|---|
| wr/ideal | 21946 | 0.357 | 0.376 | 0.427 |
| wr/nonideal | 19063 | 0.29 | 0.321 | 0.373 |
| wr/both | 41009 | 0.312 | 0.373 | 0.427 |
| brusselator/ideal | 73808 | 0.244 | 0.276 | 0.53 |
| brusselator/nonideal | 64475 | 0.256 | 0.286 | 0.494 |
| brusselator/both | 138283 | 0.249 | 0.282 | 0.53 |
| oregonator/ideal | 89488 | 0.117 | 0.216 | 0.365 |
| oregonator/nonideal | 86974 | 0.169 | 0.236 | 0.363 |
| oregonator/both | 176462 | 0.148 | 0.229 | 0.365 |
| cs14/ideal | 77899 | 0.253 | 0.31 | 0.385 |
| cs14/nonideal | 73965 | 0.249 | 0.3 | 0.361 |
| cs14/both | 151864 | 0.251 | 0.305 | 0.385 |
| all/ideal | 263141 | 0.223 | 0.312 | 0.53 |
| all/nonideal | 244477 | 0.233 | 0.294 | 0.494 |
| all/both | 507618 | 0.228 | 0.3 | 0.53 |

Full bound ratio r = (λ₁+Δ̄₁)²/(σ̄_ps B̄₁), all networks:

| class | n | median | 90% | max |
|---|---|---|---|---|
| fixed | 292382 | 0.254 | 0.554 | 0.942 |
| cycle | 441964 | 0.241 | 0.305 | 0.53 |
| chaos_candidate | 65654 | 0.224 | 0.237 | 0.359 |

Violations: motion cost 0, full bound 0; smallest relative margins 0.47 and 0.058.

Chaos, share of the allowed growth rate used, f = λ₁/(√(σ̄_ps B̄₁) − Δ̄₁):

| case | n | median | 90% | max |
|---|---|---|---|---|
| wr/ideal | 413 | 0.0291 | 0.0396 | 0.0562 |
| wr/nonideal | 1738 | 0.0319 | 0.0448 | 0.07 |
| wr/both | 2151 | 0.0314 | 0.0439 | 0.07 |
| cs14/ideal | 32055 | 0.0608 | 0.0742 | 0.1 |
| cs14/nonideal | 31448 | 0.0543 | 0.0715 | 0.0974 |
| cs14/both | 63503 | 0.0576 | 0.073 | 0.1 |
| all/ideal | 32468 | 0.0606 | 0.0741 | 0.1 |
| all/nonideal | 33186 | 0.0534 | 0.0711 | 0.0974 |
| all/both | 65654 | 0.057 | 0.0728 | 0.1 |

σ̄_ps/σ̄ (all networks):

| class | n | median | min | max |
|---|---|---|---|---|
| fixed | 292382 | 0.277 | 0.08 | 0.826 |
| cycle | 441964 | 0.288 | 0.182 | 0.797 |
| chaos_candidate | 65654 | 0.318 | 0.195 | 0.618 |

## What controls the saturation (Spearman ρ, cycles and chaos)

| network | variable | n | ρ | p |
|---|---|---|---|---|
| wr | drive | 41009 | -0.30 | 0.0e+00 |
| wr | oscillation_decades | 41009 | -0.16 | 6.8e-220 |
| wr | nontrivial_floquet_modulus | 38858 | -0.01 | 8.3e-02 |
| wr | Delta1 | 41009 | -0.57 | 0.0e+00 |
| wr | sigma_ps_over_sigma | 41009 | 0.20 | 0.0e+00 |
| brusselator | drive | 138283 | -0.08 | 3.7e-208 |
| brusselator | oscillation_decades | 138283 | -0.53 | 0.0e+00 |
| brusselator | nontrivial_floquet_modulus | 138283 | 0.50 | 0.0e+00 |
| brusselator | Delta1 | 138283 | 0.66 | 0.0e+00 |
| brusselator | sigma_ps_over_sigma | 138283 | 0.05 | 1.1e-79 |
| oregonator | drive | 176462 | -0.34 | 0.0e+00 |
| oregonator | oscillation_decades | 176462 | -0.86 | 0.0e+00 |
| oregonator | nontrivial_floquet_modulus | 176462 | 0.87 | 0.0e+00 |
| oregonator | Delta1 | 176462 | 0.59 | 0.0e+00 |
| oregonator | sigma_ps_over_sigma | 176462 | 0.31 | 0.0e+00 |
| cs14 | drive | 151864 | -0.28 | 0.0e+00 |
| cs14 | oscillation_decades | 151864 | -0.65 | 0.0e+00 |
| cs14 | nontrivial_floquet_modulus | 88361 | 0.18 | 0.0e+00 |
| cs14 | Delta1 | 151864 | 0.02 | 3.0e-10 |
| cs14 | sigma_ps_over_sigma | 151864 | 0.31 | 0.0e+00 |
| all | drive | 507618 | -0.16 | 0.0e+00 |
| all | oscillation_decades | 507618 | -0.76 | 0.0e+00 |
| all | nontrivial_floquet_modulus | 441964 | 0.85 | 0.0e+00 |
| all | Delta1 | 507618 | 0.51 | 0.0e+00 |
| all | sigma_ps_over_sigma | 507618 | -0.00 | 1.2e-01 |

## Ideal vs nonideal (two-sample KS test on the saturation)

| network | n ideal | n nonideal | median ideal | median nonideal | KS | p |
|---|---|---|---|---|---|---|
| WR | 21946 | 19063 | 0.357 | 0.29 | 0.656 | 0.0e+00 |
| Brusselator | 73808 | 64475 | 0.244 | 0.256 | 0.284 | 0.0e+00 |
| Oregonator | 89488 | 86974 | 0.117 | 0.169 | 0.588 | 0.0e+00 |
| CS14 | 77899 | 73965 | 0.253 | 0.249 | 0.137 | 0.0e+00 |

## Independent re-check of cycles and chaos (verify_attractors.py, scipy stiff solvers)

- cycle: 200 of 200 verified; max_closure 2.0e-08, max_nontrivial_floquet 0.994, max_drift_20_periods 5.7e-07
- chaos: 198 of 200 verified; median_abs_rel_diff_lambda1 0.0349, max_abs_rel_diff_lambda1 0.807, min_window_lambda1 -4.0e-04

## Numerical quality

- max_identity_residual: 3.5e-14
- max_liouville_error: 7.4e-07
- max_stretch_integral_error: 2.6e-05
- max_cycle_closure: 2.0e-08
- max_nontrivial_floquet_modulus: 1
- max_fixed_point_root_residual: 1.0e-09
- chaos_confirmed_in_two_windows: 65,654
- chaos_total: 65,654
- chaos_lambda1_relative_standard_error: median 0.0422, max 0.25
- max_integration_steps: 2,999,435
- mean_seconds_per_accepted_input: 2.32
