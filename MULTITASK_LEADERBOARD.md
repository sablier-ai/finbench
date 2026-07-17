# FinBench Leaderboard

Benchmark for multivariate financial time-series generation. Every model generates 200 paths × 60 days × 7 features from the same frozen out-of-sample panel, then gets scored on 7 independent tasks under identical conditions.

**Panel** `us_equities_macro` · 7 features · N=200 paths · H=60 days · out-of-sample &nbsp;|&nbsp; **Models** 18 &nbsp;|&nbsp; **Tasks** 7

Every task reports a real-vs-real noise floor (real data scored against a calendar-disjoint draw of itself) so you can tell whether score differences are real or measurement noise.

## Summary of results

Sablier-Flow and Sablier-Flow-Next lead 4 of the 7 tasks: overall synthetic-data quality (F1), distributional distance (F4), options pricing (T2), and VaR/ES risk backtesting (T5). On the remaining three tasks — stylized-facts battery (F2), martingale check (F5), and predictive validity (T3) — most of the field is at or beyond the real-vs-real noise floor, meaning score differences among the top models are not resolved by the metric. See per-task boards below; the aggregate summary is at the end.

---

# Per-task boards

Each task uses its own metric with a pinned implementation (`benchmark/scorers/`). CI = 95% t-interval over seeds; `≈#1` = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field).

## F1 — Synthetic-data quality (finval 0.6.1, gate-penalized)

**What it measures.** Aggregate finance-aware quality across ~14 checks (heavy tails, volatility clustering, tail dependence, memorization, drawdown shape, regime calibration). If a model flunks a hard gate — tails, memorization, drawdown — it cannot rank as if it passed.

Scored with [finval](https://github.com/sablier-ai/finval) v0.6.1 (the finance-aware path-quality suite).

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | Sablier-Flow-Next | 5 | 0.875 ± 0.008 | ±0.009 | #1 |
| 2 | Sablier-Flow | 3 | 0.870 ± 0.013 | ±0.031 | ≈#1 |
| 3 | ImagenTime | 5 | 0.666 ± 0.108 | ±0.134 | ≈#1 |
| 4 | Block-Bootstrap | 5 | 0.503 ± 0.026 | ±0.032 |  |
| 5 | t-Copula | 5 | 0.468 ± 0.010 | ±0.013 |  |
| 6 | Historical-Sim | 5 | 0.463 ± 0.008 | ±0.010 |  |
| 7 | Gaussian-iid | 5 | 0.359 ± 0.005 | ±0.006 |  |
| 8 | Sablier-Flow-Old | 5 | 0.344 ± 0.080 | ±0.100 |  |
| 9 | FHS | 5 | 0.239 ± 0.018 | ±0.022 |  |
| 10 | DCC-t | 5 | 0.193 ± 0.016 | ±0.020 |  |
| 11 | QuantGAN | 1 | 0.183 | n=1 provisional |  |
| 12 | GARCH-t | 5 | 0.173 ± 0.011 | ±0.013 |  |
| 13 | FM-TS | 5 | 0.152 ± 0.006 | ±0.008 |  |
| 14 | Diffusion-TS | 5 | 0.141 ± 0.009 | ±0.011 |  |
| 15 | KoVAE | 5 | 0.115 ± 0.006 | ±0.008 |  |
| 16 | TimeGAN-600 | 5 | 0.082 ± 0.032 | ±0.040 |  |
| 17 | TimeGAN | 5 | 0.055 ± 0.013 | ±0.016 |  |
| 18 | TimeVAE | 5 | 0.008 ± 0.000 | ±0.000 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 10 reps)_ | — | 0.598 ± 0.064 | — | — |

**Field vs floor:** clear headroom to the real-data reference — the task discriminates the field well

## F4 — Distributional distance (W1/MMD/SigW1)

**What it measures.** How close the synth distribution is to real across three lenses: each asset's distribution (Wasserstein-1), the joint distribution across assets (MMD), and the shape of the paths through time (signature-MMD). Shape and scale are scored separately so a model that just shrinks variance can't game it.

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | Sablier-Flow-Next | 5 | 0.819 ± 0.011 | ±0.014 | #1 |
| 2 | ImagenTime | 5 | 0.810 ± 0.008 | ±0.010 | ≈#1 |
| 3 | Sablier-Flow | 3 | 0.808 ± 0.011 | ±0.027 | ≈#1 |
| 4 | Block-Bootstrap | 5 | 0.803 ± 0.009 | ±0.011 | ≈#1 |
| 5 | Historical-Sim | 5 | 0.782 ± 0.005 | ±0.006 |  |
| 6 | t-Copula | 5 | 0.781 ± 0.003 | ±0.004 |  |
| 7 | FHS | 5 | 0.767 ± 0.003 | ±0.004 |  |
| 8 | Gaussian-iid | 5 | 0.761 ± 0.002 | ±0.003 |  |
| 9 | DCC-t | 5 | 0.751 ± 0.008 | ±0.010 |  |
| 10 | FM-TS | 5 | 0.751 ± 0.005 | ±0.006 |  |
| 11 | Sablier-Flow-Old | 5 | 0.743 ± 0.013 | ±0.016 |  |
| 12 | GARCH-t | 5 | 0.736 ± 0.005 | ±0.006 |  |
| 13 | QuantGAN | 1 | 0.731 | n=1 provisional |  |
| 14 | Diffusion-TS | 5 | 0.730 ± 0.021 | ±0.027 |  |
| 15 | KoVAE | 5 | 0.724 ± 0.013 | ±0.016 |  |
| 16 | TimeGAN-600 | 5 | 0.636 ± 0.040 | ±0.049 |  |
| 17 | TimeGAN | 5 | 0.610 ± 0.015 | ±0.019 |  |
| 18 | TimeVAE | 5 | 0.363 ± 0.011 | ±0.014 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 12 reps)_ | — | 0.684 ± 0.048 | — | — |

**Field vs floor:** most of the field scores at or beyond the real-data reference; the task saturates on this panel, so differences beyond the floor are within measurement resolution

## T2 — Options pricing / IV smile

**What it measures.** Monte-Carlo repricing of vanilla options on the synth vs the real market's Black-Scholes implied-vol smile, across a moneyness × maturity grid, in basis points. If a model can't price options, it can't support any volatility-strategy backtest.

Metric: bps (lower better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field; untestable at n<2).

| Rank | Competitor | n | bps | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | Sablier-Flow | 3 | 277.000 ± 5.524 | ±13.722 | #1 |
| 2 | Sablier-Flow-Next | 5 | 405.552 ± 253.318 | ±314.536 | ≈#1 |
| 3 | Historical-Sim | 5 | 810.779 ± 78.492 | ±97.461 |  |
| 4 | ImagenTime | 5 | 924.804 ± 94.728 | ±117.621 |  |
| 5 | Gaussian-iid | 5 | 936.138 ± 100.310 | ±124.552 |  |
| 6 | t-Copula | 5 | 952.339 ± 77.397 | ±96.102 |  |
| 7 | Block-Bootstrap | 5 | 999.619 ± 131.192 | ±162.896 |  |
| 8 | GARCH-t | 5 | 1104.210 ± 154.038 | ±191.263 |  |
| 9 | DCC-t | 5 | 1106.196 ± 147.231 | ±182.811 |  |
| 10 | FHS | 5 | 1151.963 ± 69.785 | ±86.650 |  |
| 11 | Sablier-Flow-Old | 5 | 1337.329 ± 117.330 | ±145.684 |  |
| 12 | QuantGAN | 1 | 1537.973 | n=1 provisional |  |
| 13 | FM-TS | 5 | 1562.532 ± 97.082 | ±120.543 |  |
| 14 | Diffusion-TS | 5 | 1690.953 ± 72.220 | ±89.673 |  |
| 15 | KoVAE | 5 | 1701.567 ± 58.478 | ±72.610 |  |
| 16 | TimeGAN-600 | 5 | 1958.107 ± 64.836 | ±80.504 |  |
| 17 | TimeVAE | 5 | 2000.877 ± 9.293 | ±11.539 |  |
| 18 | TimeGAN | 5 | 2006.209 ± 6.241 | ±7.749 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 1384.348 ± 470.844 | — | — |

**Field vs floor:** roughly half the field is within ±1 sd of the real-data reference — read fine rank differences here with the floor in mind

## T5 — VaR/ES risk backtesting

**What it measures.** Fit VaR / Expected-Shortfall on the synth, backtest on real. Four regulator-standard tests combined (Kupiec + Christoffersen + Acerbi-Székely + Basel traffic-light). The most basic professional risk-model check.

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | Sablier-Flow | 3 | 0.796 ± 0.043 | ±0.107 | #1 |
| 2 | Sablier-Flow-Next | 5 | 0.669 ± 0.097 | ±0.120 | ≈#1 |
| 3 | ImagenTime | 5 | 0.308 ± 0.030 | ±0.037 |  |
| 4 | Block-Bootstrap | 5 | 0.256 ± 0.019 | ±0.023 |  |
| 5 | Historical-Sim | 5 | 0.239 ± 0.014 | ±0.017 |  |
| 6 | t-Copula | 5 | 0.236 ± 0.015 | ±0.019 |  |
| 7 | Sablier-Flow-Old | 5 | 0.214 ± 0.065 | ±0.081 |  |
| 8 | Gaussian-iid | 5 | 0.153 ± 0.000 | ±0.000 |  |
| 9 | QuantGAN | 1 | 0.125 | n=1 provisional |  |
| 10 | Diffusion-TS | 5 | 0.078 ± 0.007 | ±0.008 |  |
| 11 | FM-TS | 5 | 0.064 ± 0.027 | ±0.034 |  |
| 12 | TimeGAN-600 | 5 | 0.036 ± 0.031 | ±0.039 |  |
| 13 | FHS | 5 | 0.008 ± 0.011 | ±0.014 |  |
| 14 | GARCH-t | 5 | 0.003 ± 0.006 | ±0.007 |  |
| 15 | DCC-t | 5 | 0.000 ± 0.000 | ±0.000 |  |
| 16 | KoVAE | 5 | 0.000 ± 0.000 | ±0.000 |  |
| 17 | TimeGAN | 5 | 0.000 ± 0.000 | ±0.000 |  |
| 18 | TimeVAE | 5 | 0.000 ± 0.000 | ±0.000 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 0.420 ± 0.098 | — | — |

**Field vs floor:** clear headroom to the real-data reference — the task discriminates the field well

## F2 — Stylized-facts battery (Cont 2001)

**What it measures.** Distance to the 11 canonical Cont-2001 stylized facts of asset returns (heavy tails, volatility clustering, leverage effect, gain/loss asymmetry, timescale asymmetry).

Metric: dist (lower better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field; untestable at n<2).

| Rank | Competitor | n | dist | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | ImagenTime | 5 | 0.379 ± 0.004 | ±0.005 | #1 |
| 2 | FM-TS | 5 | 0.379 ± 0.008 | ±0.010 | ≈#1 |
| 3 | Block-Bootstrap | 5 | 0.381 ± 0.007 | ±0.009 | ≈#1 |
| 4 | FHS | 5 | 0.387 ± 0.004 | ±0.006 | ≈#1 |
| 5 | Sablier-Flow-Next | 5 | 0.389 ± 0.007 | ±0.009 | ≈#1 |
| 6 | Sablier-Flow | 3 | 0.395 ± 0.007 | ±0.017 | ≈#1 |
| 7 | Diffusion-TS | 5 | 0.399 ± 0.010 | ±0.012 | ≈#1 |
| 8 | KoVAE | 5 | 0.407 ± 0.005 | ±0.007 |  |
| 9 | Sablier-Flow-Old | 5 | 0.416 ± 0.006 | ±0.008 |  |
| 10 | DCC-t | 5 | 0.444 ± 0.011 | ±0.014 |  |
| 11 | Historical-Sim | 5 | 0.453 ± 0.007 | ±0.009 |  |
| 12 | t-Copula | 5 | 0.460 ± 0.003 | ±0.004 |  |
| 13 | Gaussian-iid | 5 | 0.462 ± 0.001 | ±0.001 |  |
| 14 | GARCH-t | 5 | 0.466 ± 0.046 | ±0.057 | ≈#1 |
| 15 | QuantGAN | 1 | 0.491 | n=1 provisional |  |
| 16 | TimeGAN-600 | 5 | 0.603 ± 0.045 | ±0.055 |  |
| 17 | TimeVAE | 5 | 0.629 ± 0.011 | ±0.014 |  |
| 18 | TimeGAN | 5 | 0.644 ± 0.020 | ±0.024 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 0.604 ± 0.038 | — | — |

**Field vs floor:** most of the field scores at or beyond the real-data reference; the task saturates on this panel, so differences beyond the floor are within measurement resolution

## F5 — Martingale / no-drift check

**What it measures.** Does the synth reproduce the near-random-walk structure of real returns without manufacturing predictable patterns? Both too-predictable and too-anti-predictable synths are penalized.

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | QuantGAN | 1 | 0.574 | n=1 provisional | #1 |
| 2 | ImagenTime | 5 | 0.572 ± 0.018 | ±0.022 |  |
| 3 | Block-Bootstrap | 5 | 0.565 ± 0.007 | ±0.008 |  |
| 4 | FM-TS | 5 | 0.562 ± 0.007 | ±0.009 |  |
| 5 | KoVAE | 5 | 0.551 ± 0.013 | ±0.017 |  |
| 6 | Diffusion-TS | 5 | 0.549 ± 0.011 | ±0.013 |  |
| 7 | Sablier-Flow-Next | 5 | 0.548 ± 0.027 | ±0.033 |  |
| 8 | Sablier-Flow-Old | 5 | 0.540 ± 0.009 | ±0.011 |  |
| 9 | Sablier-Flow | 3 | 0.530 ± 0.029 | ±0.073 |  |
| 10 | DCC-t | 5 | 0.521 ± 0.018 | ±0.022 |  |
| 11 | GARCH-t | 5 | 0.513 ± 0.008 | ±0.010 |  |
| 12 | t-Copula | 5 | 0.494 ± 0.020 | ±0.024 |  |
| 13 | FHS | 5 | 0.494 ± 0.011 | ±0.014 |  |
| 14 | Historical-Sim | 5 | 0.491 ± 0.009 | ±0.011 |  |
| 15 | Gaussian-iid | 5 | 0.489 ± 0.003 | ±0.003 |  |
| 16 | TimeGAN-600 | 5 | 0.265 ± 0.174 | ±0.215 |  |
| 17 | TimeGAN | 5 | 0.129 ± 0.040 | ±0.050 |  |
| 18 | TimeVAE | 5 | 0.044 ± 0.018 | ±0.022 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 0.348 ± 0.082 | — | — |

**Field vs floor:** most of the field scores at or beyond the real-data reference; the task saturates on this panel, so differences beyond the floor are within measurement resolution

## T3 — Predictive validity (TSTR, multi-family)

**What it measures.** Train strategies on the synth, run them on real data, compare Sharpe rankings via Spearman ρ. The direct test of whether a synth picks winners on real markets. The real target spans many 2020-2023 regimes; a generator that reproduces only one regime is structurally capped below the multi-regime ceiling — the T3 diagnostic reports a regime-locked reference (~0.55) alongside the perfect multi-regime ceiling (~0.95) so a reader can tell whether a low rho is a task property or a model defect.

Metric: rho (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, Holm-corrected across the field; untestable at n<2).

| Rank | Competitor | n | rho | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | ImagenTime | 5 | 0.631 ± 0.025 | ±0.031 | #1 |
| 2 | Block-Bootstrap | 5 | 0.630 ± 0.031 | ±0.038 | ≈#1 |
| 3 | Sablier-Flow-Next | 5 | 0.575 ± 0.129 | ±0.161 | ≈#1 |
| 4 | TimeGAN-600 | 5 | 0.575 ± 0.112 | ±0.139 | ≈#1 |
| 5 | FM-TS | 5 | 0.549 ± 0.134 | ±0.166 | ≈#1 |
| 6 | TimeGAN | 5 | 0.539 ± 0.057 | ±0.071 | ≈#1 |
| 7 | KoVAE | 5 | 0.508 ± 0.179 | ±0.222 | ≈#1 |
| 8 | Sablier-Flow | 3 | 0.507 ± 0.086 | ±0.213 | ≈#1 |
| 9 | QuantGAN | 1 | 0.458 | n=1 provisional |  |
| 10 | t-Copula | 5 | 0.395 ± 0.273 | ±0.339 | ≈#1 |
| 11 | Diffusion-TS | 5 | 0.385 ± 0.138 | ±0.171 | ≈#1 |
| 12 | Gaussian-iid | 5 | 0.279 ± 0.164 | ±0.204 | ≈#1 |
| 13 | Historical-Sim | 5 | 0.159 ± 0.118 | ±0.147 |  |
| 14 | FHS | 5 | 0.059 ± 0.192 | ±0.239 |  |
| 15 | Sablier-Flow-Old | 5 | 0.022 ± 0.096 | ±0.119 |  |
| 16 | DCC-t | 5 | -0.003 ± 0.134 | ±0.167 |  |
| 17 | GARCH-t | 5 | -0.123 ± 0.101 | ±0.126 |  |
| 18 | TimeVAE | 5 | -0.304 ± 0.075 | ±0.093 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | -0.170 ± 0.194 | — | — |

**Field vs floor:** most of the field scores at or beyond the real-data reference; the task saturates on this panel, so differences beyond the floor are within measurement resolution

---

# Aggregate summary

A rank-based aggregate combining all 7 tasks. Read it as a **summary**, not the primary result — mean-rank penalizes strong specialists on saturated tasks and rewards even-across-the-board consistency, so it can rank a model with two decisive wins and one saturated loss below a model with no wins and no losses. Per-task boards above are the primary evidence.

## Overall

| # | Model | Family | Mean rank | Coverage |
|--:|--|--|--:|--:|
| 1 | ImagenTime | neural | 2.29 | 7/7 |
| 2 | Sablier-Flow-Next | flow | 3.00 | 7/7 |
| 3 | Block-Bootstrap | classical | 3.86 | 7/7 |
| 4 | Sablier-Flow | flow | 4.29 | 7/7 |
| 5= | t-Copula | classical | 8.14 | 7/7 |
| 5= | Historical-Sim | classical | 8.14 | 7/7 |
| 7 | FM-TS | neural | 8.29 | 7/7 |
| 8 | Gaussian-iid | classical | 9.71 | 7/7 |
| 9 | Sablier-Flow-Old | flow | 9.86 | 7/7 |
| 10= | FHS | classical | 10.00 | 7/7 |
| 10= | QuantGAN | neural | 10.00 | 7/7 |
| 12 | Diffusion-TS | neural | 10.86 | 7/7 |
| 13 | DCC-t | classical | 11.29 | 7/7 |
| 14 | KoVAE | neural | 11.57 | 7/7 |
| 15 | GARCH-t | classical | 12.57 | 7/7 |
| 16 | TimeGAN-600 | neural | 13.71 | 7/7 |
| 17 | TimeGAN | neural | 15.71 | 7/7 |
| 18 | TimeVAE | neural | 17.71 | 7/7 |

## Per-task rank matrix

Rank of each model on every task (**1** = best in column). Detailed score tables with confidence intervals and noise floors are in the [per-task boards](#per-task-boards) above.

| Model | F1 | F4 | T2 | T5 | F2 | F5 | T3 | Mean |
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|--:|
| ImagenTime | 3 | 2 | 4 | 3 | **1** | 2 | **1** | 2.29 |
| Sablier-Flow-Next | **1** | **1** | 2 | 2 | 5 | 7 | 3 | 3.00 |
| Block-Bootstrap | 4 | 4 | 7 | 4 | 3 | 3 | 2 | 3.86 |
| Sablier-Flow | 2 | 3 | **1** | **1** | 6 | 9 | 8 | 4.29 |
| t-Copula | 5 | 6 | 6 | 6 | 12 | 12 | 10 | 8.14 |
| Historical-Sim | 6 | 5 | 3 | 5 | 11 | 14 | 13 | 8.14 |
| FM-TS | 13 | 10 | 13 | 11 | 2 | 4 | 5 | 8.29 |
| Gaussian-iid | 7 | 8 | 5 | 8 | 13 | 15 | 12 | 9.71 |
| Sablier-Flow-Old | 8 | 11 | 11 | 7 | 9 | 8 | 15 | 9.86 |
| FHS | 9 | 7 | 10 | 13 | 4 | 13 | 14 | 10.00 |
| QuantGAN | 11 | 13 | 12 | 9 | 15 | **1** | 9 | 10.00 |
| Diffusion-TS | 14 | 14 | 14 | 10 | 7 | 6 | 11 | 10.86 |
| DCC-t | 10 | 9 | 9 | 15 | 10 | 10 | 16 | 11.29 |
| KoVAE | 15 | 15 | 15 | 16 | 8 | 5 | 7 | 11.57 |
| GARCH-t | 12 | 12 | 8 | 14 | 14 | 11 | 17 | 12.57 |
| TimeGAN-600 | 16 | 16 | 16 | 12 | 16 | 16 | 4 | 13.71 |
| TimeGAN | 17 | 17 | 18 | 17 | 18 | 17 | 6 | 15.71 |
| TimeVAE | 18 | 18 | 17 | 18 | 17 | 18 | 18 | 17.71 |

Tasks: **F1** Synthetic-data quality (finval 0.6.1, gate-penalized) · **F4** Distributional distance (W1/MMD/SigW1) · **T2** Options pricing / IV smile · **T5** VaR/ES risk backtesting · **F2** Stylized-facts battery (Cont 2001) · **F5** Martingale / no-drift check · **T3** Predictive validity (TSTR, multi-family)

**Provenance.** Sablier-Flow, Sablier-Flow-Next, and Sablier-Flow-Old are Sablier's three published entries (production, top research candidate, previous production). External baselines run their published defaults. Replay-resampling rows (Historical-Sim, Block-Bootstrap, FHS) resample real training data by construction; memorization-guard flags for those rows are expected.

**Scope.** v1 covers one frozen panel (`us_equities_macro`, 7 features), horizon H=60, one out-of-sample window. Current entries are maintainer-generated; open external submissions are planned for a future edition. Models within a significance group should be read as tied.
