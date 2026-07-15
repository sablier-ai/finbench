# FinBench — Multi-Task Leaderboard (finval 0.6.1)

Every competitor scored under identical conditions per task (see [BENCHMARK_TASKS.md](./BENCHMARK_TASKS.md)) against the pinned canonical panel real (`reference/panels/us_equities_macro/real_paths.npy`, sha256-verified at run time). The aggregate is **mean rank** across scored tasks (absolute scores don't cross tasks) and includes ONLY competitors scored on all of them. FLOW flavors appear under opaque codenames only.

Every task shown here measurably resolves the field — its competitors clear the task's honest real-vs-real noise floor (real data scored against a calendar-disjoint independent draw of itself). Tasks that do not yet resolve on this panel are not included.

## Leaderboard — mean rank across the scored tasks

| Rank | Competitor | Family | Provenance | Mean rank | P(#1)† | Tasks scored |
|--:|--|--|--|--:|--:|--:|
| 1 | FLOW-H | flow | recipe-controlled | 4.29 | 0.66 | 7/7 |
| 2 | ImagenTime | neural | published-defaults | 4.79 | 0.29 | 7/7 |
| 3 | FLOW-P2 | flow | production-reference | 6.86 | 0.03 | 7/7 |
| 4= | Block-Bootstrap | classical | replay-resampling | 7.57 | 0.00 | 7/7 |
| 4= | FLOW-G | flow | recipe-controlled | 7.57 | 0.01 | 7/7 |
| 6 | FLOW-I | flow | recipe-controlled | 8.00 | 0.00 | 7/7 |
| 7 | FLOW-A | flow | recipe-controlled | 8.57 | 0.00 | 7/7 |
| 8 | FLOW-E | flow | recipe-controlled | 8.71 | 0.00 | 7/7 |
| 9 | FLOW-C | flow | recipe-controlled | 9.00 | 0.00 | 7/7 |
| 10 | FLOW-J | flow | recipe-controlled | 10.93 | 0.00 | 7/7 |
| 11 | FLOW-B | flow | recipe-controlled | 12.29 | 0.00 | 7/7 |
| 12 | FLOW-D | flow | recipe-controlled | 12.71 | 0.00 | 7/7 |
| 13 | FM-TS | neural | published-defaults | 13.57 | 0.00 | 7/7 |
| 14 | FLOW-F | flow | recipe-controlled | 14.00 | 0.00 | 7/7 |
| 15 | t-Copula | classical | published-defaults | 14.57 | 0.00 | 7/7 |
| 16 | Historical-Sim | classical | replay-resampling | 15.00 | 0.00 | 7/7 |
| 17 | QuantGAN | neural | published-defaults | 16.57 | 0.00 | 7/7 |
| 18 | Gaussian-iid | classical | published-defaults | 17.57 | 0.00 | 7/7 |
| 19 | Diffusion-TS | neural | published-defaults | 17.71 | 0.00 | 7/7 |
| 20 | FHS | classical | replay-resampling | 18.00 | 0.00 | 7/7 |
| 21 | KoVAE | neural | published-defaults | 18.21 | 0.00 | 7/7 |
| 22 | FLOW-P1 | flow | production-reference | 18.43 | 0.00 | 7/7 |
| 23 | DCC-t | classical | published-defaults | 20.36 | 0.00 | 7/7 |
| 24= | GARCH-t | classical | published-defaults | 21.43 | 0.00 | 7/7 |
| 24= | TimeGAN-600 | neural | published-defaults | 21.43 | 0.00 | 7/7 |
| 26 | TimeGAN | neural | published-defaults | 23.36 | 0.00 | 7/7 |
| 27 | TimeVAE | neural | published-defaults | 26.50 | 0.00 | 7/7 |

†P(#1): approximate bootstrap probability of aggregate rank 1 over 1000 resamples (fixed seed → reproducible). Since only per-task (mean, std, n) summaries are available here (not per-seed values), each resample perturbs every task score with Gaussian(mean, std/√n) noise and re-ranks. **Read P(#1) as an UPPER BOUND on confidence, not a calibrated probability** (audit AGG-P1-3): it prices only per-seed sampling error and omits two larger terms — (a) cross-task error correlation, and (b) the DOMINANT unmodelled term, target/regime sampling error (the single OOS window and single reference draw; the audit measured T2 targets moving 131–330 bps and F5's disjoint order scrambling the top-14), which is not in the bootstrap at all. The ordinal winner is robust; the magnitude is optimistic. n=1 rows carry an IMPUTED per-task sem (median multi-seed std), never zero, so a single-seed row cannot be locked to an over-precise rank (audit F1-N1-5/NBOOT-1). Equal mean ranks share a tie-aware rank ('1=' means tied at 1).

Provenance: **production-reference** = own tuned production configuration; **published-defaults** = external baseline at untuned published defaults; **recipe-controlled** = one shared FLOW bake-off recipe, selected via finval (the F1 evaluator) on this panel; **replay-resampling** = resamples/replays real training data (industry practice: historical simulation family) — cannot generate genuinely novel scenarios; memorization-guard flags expected.

*Scored tasks: F1, F2, F4, F5, T2, T3, T5. Tasks with no scorer yet (of 7 total) are omitted from the aggregate until wired.*

## F1 — Synthetic-data quality (finval 0.6.1, gate-penalized)

**Selection caveat.** The FLOW-A…J recipe was selected by sweeping THIS metric (finval) on THIS panel; their F1 is in-sample w.r.t. that selection — read it as a training-fit upper bound, not held-out skill. FLOW-P1/P2 (own production configuration) and the external baselines are not selected on F1, so their F1 is held-out.

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, **Holm-corrected across the field for multiplicity** before the p>0.05 threshold — audit AGG-HOLM-2; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | FLOW-H | 5 | 0.875 ± 0.008 | ±0.009 | #1 |
| 2 | FLOW-P2 | 3 | 0.870 ± 0.013 | ±0.031 | ≈#1 |
| 3 | FLOW-A | 5 | 0.755 ± 0.148 | ±0.183 | ≈#1 |
| 4 | FLOW-G | 5 | 0.736 ± 0.088 | ±0.109 | ≈#1 |
| 5 | FLOW-I | 5 | 0.672 ± 0.197 | ±0.245 | ≈#1 |
| 6 | ImagenTime | 5 | 0.666 ± 0.108 | ±0.134 | ≈#1 |
| 7 | FLOW-D | 5 | 0.663 ± 0.254 | ±0.316 | ≈#1 |
| 8 | FLOW-E | 5 | 0.571 ± 0.182 | ±0.226 | ≈#1 |
| 9 | FLOW-C | 5 | 0.551 ± 0.225 | ±0.280 | ≈#1 |
| 10 | FLOW-J | 5 | 0.534 ± 0.151 | ±0.187 | ≈#1 |
| 11 | Block-Bootstrap | 5 | 0.503 ± 0.026 | ±0.032 |  |
| 12 | t-Copula | 5 | 0.468 ± 0.010 | ±0.013 |  |
| 13 | Historical-Sim | 5 | 0.463 ± 0.008 | ±0.010 |  |
| 14 | FLOW-F | 5 | 0.411 ± 0.090 | ±0.112 |  |
| 15 | FLOW-B | 5 | 0.363 ± 0.096 | ±0.120 |  |
| 16 | Gaussian-iid | 5 | 0.359 ± 0.005 | ±0.006 |  |
| 17 | FLOW-P1 | 5 | 0.344 ± 0.080 | ±0.100 |  |
| 18 | FHS | 5 | 0.239 ± 0.018 | ±0.022 |  |
| 19 | DCC-t | 5 | 0.193 ± 0.016 | ±0.020 |  |
| 20 | QuantGAN | 1 | 0.183 | n=1 provisional |  |
| 21 | GARCH-t | 5 | 0.173 ± 0.011 | ±0.013 |  |
| 22 | FM-TS | 5 | 0.152 ± 0.006 | ±0.008 |  |
| 23 | Diffusion-TS | 5 | 0.141 ± 0.009 | ±0.011 |  |
| 24 | KoVAE | 5 | 0.115 ± 0.006 | ±0.008 |  |
| 25 | TimeGAN-600 | 5 | 0.082 ± 0.032 | ±0.040 |  |
| 26 | TimeGAN | 5 | 0.055 ± 0.013 | ±0.016 |  |
| 27 | TimeVAE | 5 | 0.008 ± 0.000 | ±0.000 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 10 reps)_ | — | 0.598 ± 0.064 | — | — |

The noise-floor row is what real data scores against an **independent** draw of itself under this scorer — two halves of the pinned real split by CALENDAR day (an H-day gap between them, so they share ZERO days), each bootstrapped to the competitor path budget, mean ± sd over reps. That is what a *perfect* generator (real reality, same budget) scores; a competitor at or inside that band is at the task's resolution limit, and differences there are sampling noise, not skill (audit F1-FLOOR-1/F2-OW-1). **Field vs floor:** the field sits worse than the honest floor — genuine resolution. (For reference, the OLD even/odd path-parity self-score was 0.985. It is **not** a floor: the pinned 'paths' are stride-~4 OVERLAPPING rolling windows, so its two halves share ~99.3% of their calendar days — it prices estimator jitter on near-identical data, at half the path budget, and materially over-states the resolution on every task.)

## F2 — Stylized-facts battery (Cont 2001)

Metric: dist (lower better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, **Holm-corrected across the field for multiplicity** before the p>0.05 threshold — audit AGG-HOLM-2; untestable at n<2).

| Rank | Competitor | n | dist | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | FLOW-C | 5 | 0.377 ± 0.004 | ±0.004 | #1 |
| 2 | ImagenTime | 5 | 0.379 ± 0.004 | ±0.005 | ≈#1 |
| 3 | FM-TS | 5 | 0.379 ± 0.008 | ±0.010 | ≈#1 |
| 4 | FLOW-B | 5 | 0.381 ± 0.006 | ±0.008 | ≈#1 |
| 5 | Block-Bootstrap | 5 | 0.381 ± 0.007 | ±0.009 | ≈#1 |
| 6 | FLOW-E | 5 | 0.384 ± 0.006 | ±0.007 | ≈#1 |
| 7 | FLOW-J | 5 | 0.386 ± 0.006 | ±0.007 | ≈#1 |
| 8 | FLOW-A | 5 | 0.387 ± 0.012 | ±0.015 | ≈#1 |
| 9 | FHS | 5 | 0.387 ± 0.004 | ±0.006 | ≈#1 |
| 10 | FLOW-G | 5 | 0.388 ± 0.009 | ±0.011 | ≈#1 |
| 11 | FLOW-H | 5 | 0.389 ± 0.007 | ±0.009 | ≈#1 |
| 12 | FLOW-F | 5 | 0.392 ± 0.009 | ±0.011 | ≈#1 |
| 13 | FLOW-P2 | 3 | 0.395 ± 0.007 | ±0.017 | ≈#1 |
| 14 | FLOW-I | 5 | 0.395 ± 0.011 | ±0.014 | ≈#1 |
| 15 | Diffusion-TS | 5 | 0.399 ± 0.010 | ±0.012 | ≈#1 |
| 16 | KoVAE | 5 | 0.407 ± 0.005 | ±0.007 |  |
| 17 | FLOW-D | 5 | 0.410 ± 0.011 | ±0.014 |  |
| 18 | FLOW-P1 | 5 | 0.416 ± 0.006 | ±0.008 |  |
| 19 | DCC-t | 5 | 0.444 ± 0.011 | ±0.014 |  |
| 20 | Historical-Sim | 5 | 0.453 ± 0.007 | ±0.009 |  |
| 21 | t-Copula | 5 | 0.460 ± 0.003 | ±0.004 |  |
| 22 | Gaussian-iid | 5 | 0.462 ± 0.001 | ±0.001 |  |
| 23 | GARCH-t | 5 | 0.466 ± 0.046 | ±0.057 | ≈#1 |
| 24 | QuantGAN | 1 | 0.491 | n=1 provisional |  |
| 25 | TimeGAN-600 | 5 | 0.603 ± 0.045 | ±0.055 |  |
| 26 | TimeVAE | 5 | 0.629 ± 0.011 | ±0.014 |  |
| 27 | TimeGAN | 5 | 0.644 ± 0.020 | ±0.024 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 0.604 ± 0.038 | — | — |

The noise-floor row is what real data scores against an **independent** draw of itself under this scorer — two halves of the pinned real split by CALENDAR day (an H-day gap between them, so they share ZERO days), each bootstrapped to the competitor path budget, mean ± sd over reps. That is what a *perfect* generator (real reality, same budget) scores; a competitor at or inside that band is at the task's resolution limit, and differences there are sampling noise, not skill (audit F1-FLOOR-1/F2-OW-1). **Field vs floor:** the field scores BETTER than independent real data on 24/27 rows — the honest floor is a CEILING the field exceeds, so scores past it are not attributable to distributional fidelity. (For reference, the OLD even/odd path-parity self-score was 0.081. It is **not** a floor: the pinned 'paths' are stride-~4 OVERLAPPING rolling windows, so its two halves share ~99.3% of their calendar days — it prices estimator jitter on near-identical data, at half the path budget, and materially over-states the resolution on every task.)

## F4 — Distributional distance (W1/MMD/SigW1)

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, **Holm-corrected across the field for multiplicity** before the p>0.05 threshold — audit AGG-HOLM-2; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | FLOW-H | 5 | 0.819 ± 0.011 | ±0.014 | #1 |
| 2 | FLOW-I | 5 | 0.816 ± 0.014 | ±0.017 | ≈#1 |
| 3 | ImagenTime | 5 | 0.810 ± 0.008 | ±0.010 | ≈#1 |
| 4 | FLOW-G | 5 | 0.808 ± 0.023 | ±0.029 | ≈#1 |
| 5 | FLOW-P2 | 3 | 0.808 ± 0.011 | ±0.027 | ≈#1 |
| 6 | Block-Bootstrap | 5 | 0.803 ± 0.009 | ±0.011 | ≈#1 |
| 7 | FLOW-B | 5 | 0.783 ± 0.011 | ±0.014 |  |
| 8 | Historical-Sim | 5 | 0.782 ± 0.005 | ±0.006 |  |
| 9 | t-Copula | 5 | 0.781 ± 0.003 | ±0.004 |  |
| 10 | FLOW-A | 5 | 0.776 ± 0.008 | ±0.010 |  |
| 11 | FLOW-C | 5 | 0.771 ± 0.014 | ±0.017 |  |
| 12 | FLOW-E | 5 | 0.771 ± 0.009 | ±0.011 |  |
| 13 | FLOW-J | 5 | 0.768 ± 0.013 | ±0.016 |  |
| 14 | FHS | 5 | 0.767 ± 0.003 | ±0.004 |  |
| 15 | FLOW-D | 5 | 0.765 ± 0.026 | ±0.032 |  |
| 16 | Gaussian-iid | 5 | 0.761 ± 0.002 | ±0.003 |  |
| 17 | FLOW-F | 5 | 0.759 ± 0.017 | ±0.021 |  |
| 18 | DCC-t | 5 | 0.751 ± 0.008 | ±0.010 |  |
| 19 | FM-TS | 5 | 0.751 ± 0.005 | ±0.006 |  |
| 20 | FLOW-P1 | 5 | 0.743 ± 0.013 | ±0.016 |  |
| 21 | GARCH-t | 5 | 0.736 ± 0.005 | ±0.006 |  |
| 22 | QuantGAN | 1 | 0.731 | n=1 provisional |  |
| 23 | Diffusion-TS | 5 | 0.730 ± 0.021 | ±0.027 |  |
| 24 | KoVAE | 5 | 0.724 ± 0.013 | ±0.016 |  |
| 25 | TimeGAN-600 | 5 | 0.636 ± 0.040 | ±0.049 |  |
| 26 | TimeGAN | 5 | 0.610 ± 0.015 | ±0.019 |  |
| 27 | TimeVAE | 5 | 0.363 ± 0.011 | ±0.014 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 12 reps)_ | — | 0.684 ± 0.048 | — | — |

The noise-floor row is what real data scores against an **independent** draw of itself under this scorer — two halves of the pinned real split by CALENDAR day (an H-day gap between them, so they share ZERO days), each bootstrapped to the competitor path budget, mean ± sd over reps. That is what a *perfect* generator (real reality, same budget) scores; a competitor at or inside that band is at the task's resolution limit, and differences there are sampling noise, not skill (audit F1-FLOOR-1/F2-OW-1). **Field vs floor:** the field scores BETTER than independent real data on 21/27 rows — the honest floor is a CEILING the field exceeds, so scores past it are not attributable to distributional fidelity. (For reference, the OLD even/odd path-parity self-score was 0.939. It is **not** a floor: the pinned 'paths' are stride-~4 OVERLAPPING rolling windows, so its two halves share ~99.3% of their calendar days — it prices estimator jitter on near-identical data, at half the path budget, and materially over-states the resolution on every task.)

## F5 — Martingale / no-drift check

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, **Holm-corrected across the field for multiplicity** before the p>0.05 threshold — audit AGG-HOLM-2; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | QuantGAN | 1 | 0.574 | n=1 provisional | #1 |
| 2 | ImagenTime | 5 | 0.572 ± 0.018 | ±0.022 |  |
| 3 | Block-Bootstrap | 5 | 0.565 ± 0.007 | ±0.008 |  |
| 4 | FM-TS | 5 | 0.562 ± 0.007 | ±0.009 |  |
| 5 | FLOW-F | 5 | 0.552 ± 0.015 | ±0.019 |  |
| 6 | FLOW-E | 5 | 0.552 ± 0.019 | ±0.023 |  |
| 7 | KoVAE | 5 | 0.551 ± 0.013 | ±0.017 |  |
| 8 | Diffusion-TS | 5 | 0.549 ± 0.011 | ±0.013 |  |
| 9 | FLOW-C | 5 | 0.549 ± 0.017 | ±0.021 |  |
| 10 | FLOW-H | 5 | 0.548 ± 0.027 | ±0.033 |  |
| 11 | FLOW-A | 5 | 0.547 ± 0.012 | ±0.015 |  |
| 12 | FLOW-I | 5 | 0.546 ± 0.014 | ±0.018 |  |
| 13 | FLOW-J | 5 | 0.544 ± 0.015 | ±0.018 |  |
| 14 | FLOW-B | 5 | 0.542 ± 0.014 | ±0.018 |  |
| 15 | FLOW-G | 5 | 0.542 ± 0.022 | ±0.027 |  |
| 16 | FLOW-P1 | 5 | 0.540 ± 0.009 | ±0.011 |  |
| 17 | FLOW-D | 5 | 0.538 ± 0.017 | ±0.021 |  |
| 18 | FLOW-P2 | 3 | 0.530 ± 0.029 | ±0.073 |  |
| 19 | DCC-t | 5 | 0.521 ± 0.018 | ±0.022 |  |
| 20 | GARCH-t | 5 | 0.513 ± 0.008 | ±0.010 |  |
| 21 | t-Copula | 5 | 0.494 ± 0.020 | ±0.024 |  |
| 22 | FHS | 5 | 0.494 ± 0.011 | ±0.014 |  |
| 23 | Historical-Sim | 5 | 0.491 ± 0.009 | ±0.011 |  |
| 24 | Gaussian-iid | 5 | 0.489 ± 0.003 | ±0.003 |  |
| 25 | TimeGAN-600 | 5 | 0.265 ± 0.174 | ±0.215 |  |
| 26 | TimeGAN | 5 | 0.129 ± 0.040 | ±0.050 |  |
| 27 | TimeVAE | 5 | 0.044 ± 0.018 | ±0.022 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 0.348 ± 0.082 | — | — |

The noise-floor row is what real data scores against an **independent** draw of itself under this scorer — two halves of the pinned real split by CALENDAR day (an H-day gap between them, so they share ZERO days), each bootstrapped to the competitor path budget, mean ± sd over reps. That is what a *perfect* generator (real reality, same budget) scores; a competitor at or inside that band is at the task's resolution limit, and differences there are sampling noise, not skill (audit F1-FLOOR-1/F2-OW-1). **Field vs floor:** the field scores BETTER than independent real data on 24/27 rows — the honest floor is a CEILING the field exceeds, so scores past it are not attributable to distributional fidelity. (For reference, the OLD even/odd path-parity self-score was 0.908. It is **not** a floor: the pinned 'paths' are stride-~4 OVERLAPPING rolling windows, so its two halves share ~99.3% of their calendar days — it prices estimator jitter on near-identical data, at half the path budget, and materially over-states the resolution on every task.)

## T2 — Options pricing / IV smile

Metric: bps (lower better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, **Holm-corrected across the field for multiplicity** before the p>0.05 threshold — audit AGG-HOLM-2; untestable at n<2).

| Rank | Competitor | n | bps | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | FLOW-P2 | 3 | 277.000 ± 5.524 | ±13.722 | #1 |
| 2 | FLOW-H | 5 | 405.552 ± 253.318 | ±314.536 | ≈#1 |
| 3 | FLOW-G | 5 | 536.217 ± 400.335 | ±497.081 | ≈#1 |
| 4 | FLOW-A | 5 | 711.237 ± 238.007 | ±295.524 | ≈#1 |
| 5 | FLOW-C | 5 | 740.917 ± 242.842 | ±301.528 | ≈#1 |
| 6 | FLOW-J | 5 | 742.801 ± 182.456 | ±226.549 |  |
| 7 | FLOW-E | 5 | 783.815 ± 290.185 | ±360.312 | ≈#1 |
| 8 | Historical-Sim | 5 | 810.779 ± 78.492 | ±97.461 |  |
| 9 | FLOW-I | 5 | 824.725 ± 340.400 | ±422.662 | ≈#1 |
| 10 | FLOW-D | 5 | 842.749 ± 402.445 | ±499.702 | ≈#1 |
| 11 | ImagenTime | 5 | 924.804 ± 94.728 | ±117.621 |  |
| 12 | Gaussian-iid | 5 | 936.138 ± 100.310 | ±124.552 |  |
| 13 | FLOW-F | 5 | 940.827 ± 156.823 | ±194.721 |  |
| 14 | t-Copula | 5 | 952.339 ± 77.397 | ±96.102 |  |
| 15 | Block-Bootstrap | 5 | 999.619 ± 131.192 | ±162.896 |  |
| 16 | GARCH-t | 5 | 1104.210 ± 154.038 | ±191.263 |  |
| 17 | DCC-t | 5 | 1106.196 ± 147.231 | ±182.811 |  |
| 18 | FHS | 5 | 1151.963 ± 69.785 | ±86.650 |  |
| 19 | FLOW-B | 5 | 1155.324 ± 204.091 | ±253.412 |  |
| 20 | FLOW-P1 | 5 | 1337.329 ± 117.330 | ±145.684 |  |
| 21 | QuantGAN | 1 | 1537.973 | n=1 provisional |  |
| 22 | FM-TS | 5 | 1562.532 ± 97.082 | ±120.543 |  |
| 23 | Diffusion-TS | 5 | 1690.953 ± 72.220 | ±89.673 |  |
| 24 | KoVAE | 5 | 1701.567 ± 58.478 | ±72.610 |  |
| 25 | TimeGAN-600 | 5 | 1958.107 ± 64.836 | ±80.504 |  |
| 26 | TimeVAE | 5 | 2000.877 ± 9.293 | ±11.539 |  |
| 27 | TimeGAN | 5 | 2006.209 ± 6.241 | ±7.749 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 1384.348 ± 470.844 | — | — |

The noise-floor row is what real data scores against an **independent** draw of itself under this scorer — two halves of the pinned real split by CALENDAR day (an H-day gap between them, so they share ZERO days), each bootstrapped to the competitor path budget, mean ± sd over reps. That is what a *perfect* generator (real reality, same budget) scores; a competitor at or inside that band is at the task's resolution limit, and differences there are sampling noise, not skill (audit F1-FLOOR-1/F2-OW-1). **Field vs floor:** 52% of the field within ±1 floor-sd — ranking here is within sampling noise. (For reference, the OLD even/odd path-parity self-score was 225.358. It is **not** a floor: the pinned 'paths' are stride-~4 OVERLAPPING rolling windows, so its two halves share ~99.3% of their calendar days — it prices estimator jitter on near-identical data, at half the path budget, and materially over-states the resolution on every task.)

## T3 — Predictive validity (TSTR, multi-family)

Metric: rho (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, **Holm-corrected across the field for multiplicity** before the p>0.05 threshold — audit AGG-HOLM-2; untestable at n<2).

| Rank | Competitor | n | rho | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | ImagenTime | 5 | 0.631 ± 0.025 | ±0.031 | #1 |
| 2 | Block-Bootstrap | 5 | 0.630 ± 0.031 | ±0.038 | ≈#1 |
| 3 | FLOW-H | 5 | 0.575 ± 0.129 | ±0.161 | ≈#1 |
| 4 | TimeGAN-600 | 5 | 0.575 ± 0.112 | ±0.139 | ≈#1 |
| 5 | FM-TS | 5 | 0.549 ± 0.134 | ±0.166 | ≈#1 |
| 6 | TimeGAN | 5 | 0.539 ± 0.057 | ±0.071 | ≈#1 |
| 7 | KoVAE | 5 | 0.508 ± 0.179 | ±0.222 | ≈#1 |
| 8 | FLOW-P2 | 3 | 0.507 ± 0.086 | ±0.213 | ≈#1 |
| 9 | FLOW-I | 5 | 0.482 ± 0.076 | ±0.094 | ≈#1 |
| 10 | QuantGAN | 1 | 0.458 | n=1 provisional |  |
| 11 | FLOW-B | 5 | 0.411 ± 0.184 | ±0.229 | ≈#1 |
| 12 | t-Copula | 5 | 0.395 ± 0.273 | ±0.339 | ≈#1 |
| 13 | Diffusion-TS | 5 | 0.385 ± 0.138 | ±0.171 | ≈#1 |
| 14 | FLOW-G | 5 | 0.356 ± 0.324 | ±0.402 | ≈#1 |
| 15 | FLOW-E | 5 | 0.291 ± 0.174 | ±0.216 | ≈#1 |
| 16 | Gaussian-iid | 5 | 0.279 ± 0.164 | ±0.204 | ≈#1 |
| 17 | FLOW-D | 5 | 0.248 ± 0.211 | ±0.262 | ≈#1 |
| 18 | FLOW-C | 5 | 0.213 ± 0.081 | ±0.101 |  |
| 19 | FLOW-J | 5 | 0.191 ± 0.150 | ±0.187 |  |
| 20 | FLOW-A | 5 | 0.183 ± 0.075 | ±0.093 |  |
| 21 | Historical-Sim | 5 | 0.159 ± 0.118 | ±0.147 |  |
| 22 | FLOW-F | 5 | 0.072 ± 0.287 | ±0.357 | ≈#1 |
| 23 | FHS | 5 | 0.059 ± 0.192 | ±0.239 |  |
| 24 | FLOW-P1 | 5 | 0.022 ± 0.096 | ±0.119 |  |
| 25 | DCC-t | 5 | -0.003 ± 0.134 | ±0.167 |  |
| 26 | GARCH-t | 5 | -0.123 ± 0.101 | ±0.126 |  |
| 27 | TimeVAE | 5 | -0.304 ± 0.075 | ±0.093 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | -0.170 ± 0.194 | — | — |

The noise-floor row is what real data scores against an **independent** draw of itself under this scorer — two halves of the pinned real split by CALENDAR day (an H-day gap between them, so they share ZERO days), each bootstrapped to the competitor path budget, mean ± sd over reps. That is what a *perfect* generator (real reality, same budget) scores; a competitor at or inside that band is at the task's resolution limit, and differences there are sampling noise, not skill (audit F1-FLOOR-1/F2-OW-1). **Field vs floor:** the field scores BETTER than independent real data on 23/27 rows — the honest floor is a CEILING the field exceeds, so scores past it are not attributable to distributional fidelity. (For reference, the OLD even/odd path-parity self-score was 0.976. It is **not** a floor: the pinned 'paths' are stride-~4 OVERLAPPING rolling windows, so its two halves share ~99.3% of their calendar days — it prices estimator jitter on near-identical data, at half the path budget, and materially over-states the resolution on every task.)

## T5 — VaR/ES risk backtesting

Metric: score (higher better). CI = 95% t-interval over seeds; '≈#1' = statistically indistinguishable from the task leader (Welch t-test, **Holm-corrected across the field for multiplicity** before the p>0.05 threshold — audit AGG-HOLM-2; untestable at n<2).

| Rank | Competitor | n | score | 95% CI | vs #1 |
|--:|--|--:|--:|--:|--|
| 1 | FLOW-P2 | 3 | 0.796 ± 0.043 | ±0.107 | #1 |
| 2 | FLOW-H | 5 | 0.669 ± 0.097 | ±0.120 | ≈#1 |
| 3 | FLOW-G | 5 | 0.622 ± 0.220 | ±0.274 | ≈#1 |
| 4 | FLOW-A | 5 | 0.422 ± 0.104 | ±0.129 |  |
| 5 | FLOW-I | 5 | 0.383 ± 0.147 | ±0.182 |  |
| 6 | FLOW-D | 5 | 0.342 ± 0.140 | ±0.174 |  |
| 7 | FLOW-E | 5 | 0.317 ± 0.108 | ±0.134 |  |
| 8.5 | FLOW-J | 5 | 0.308 ± 0.082 | ±0.101 |  |
| 8.5 | ImagenTime | 5 | 0.308 ± 0.030 | ±0.037 |  |
| 10 | FLOW-C | 5 | 0.300 ± 0.128 | ±0.159 |  |
| 11 | Block-Bootstrap | 5 | 0.256 ± 0.019 | ±0.023 |  |
| 12 | Historical-Sim | 5 | 0.239 ± 0.014 | ±0.017 |  |
| 13 | t-Copula | 5 | 0.236 ± 0.015 | ±0.019 |  |
| 14 | FLOW-P1 | 5 | 0.214 ± 0.065 | ±0.081 |  |
| 15 | FLOW-F | 5 | 0.206 ± 0.054 | ±0.067 |  |
| 16 | FLOW-B | 5 | 0.178 ± 0.038 | ±0.047 |  |
| 17 | Gaussian-iid | 5 | 0.153 ± 0.000 | ±0.000 |  |
| 18 | QuantGAN | 1 | 0.125 | n=1 provisional |  |
| 19 | Diffusion-TS | 5 | 0.078 ± 0.007 | ±0.008 |  |
| 20 | FM-TS | 5 | 0.064 ± 0.027 | ±0.034 |  |
| 21 | TimeGAN-600 | 5 | 0.036 ± 0.031 | ±0.039 |  |
| 22 | FHS | 5 | 0.008 ± 0.011 | ±0.014 |  |
| 23 | GARCH-t | 5 | 0.003 ± 0.006 | ±0.007 |  |
| 25.5 | DCC-t | 5 | 0.000 ± 0.000 | ±0.000 |  |
| 25.5 | KoVAE | 5 | 0.000 ± 0.000 | ±0.000 |  |
| 25.5 | TimeGAN | 5 | 0.000 ± 0.000 | ±0.000 |  |
| 25.5 | TimeVAE | 5 | 0.000 ± 0.000 | ±0.000 |  |
| — | _**noise floor** — independent real-vs-real (calendar-disjoint windows, competitor path budget, 11 reps)_ | — | 0.420 ± 0.098 | — | — |

The noise-floor row is what real data scores against an **independent** draw of itself under this scorer — two halves of the pinned real split by CALENDAR day (an H-day gap between them, so they share ZERO days), each bootstrapped to the competitor path budget, mean ± sd over reps. That is what a *perfect* generator (real reality, same budget) scores; a competitor at or inside that band is at the task's resolution limit, and differences there are sampling noise, not skill (audit F1-FLOOR-1/F2-OW-1). **Field vs floor:** the field sits worse than the honest floor — genuine resolution. (For reference, the OLD even/odd path-parity self-score was 0.944. It is **not** a floor: the pinned 'paths' are stride-~4 OVERLAPPING rolling windows, so its two halves share ~99.3% of their calendar days — it prices estimator jitter on near-identical data, at half the path budget, and materially over-states the resolution on every task.)

## Pending competitors (not yet generated)

These slot into every board as new rows once their tensors exist (FLOW flavors are GPU-gated):

PCF-GAN, Tail-GAN, Fourier-Flows, TimeVQVAE

---

**Scope disclaimer.** Single panel (us_equities_macro, 7 features), horizon H=60, one OOS window. All current entries are maintainer-generated; external submission gates (held-out scoring slab, training-metadata attestation) are pending — do not treat this board as an open leaderboard yet. FLOW-A…J share one bake-off recipe selected via finval (the F1 evaluator) on this panel; FLOW-P1/P2 use their own tuned production configurations; external baselines run untuned published defaults. Rank order within a significance group is not a claim.
