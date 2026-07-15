# TimeGAN-600 — robustness submission

Tests whether TimeGAN's poor aggregate on FinBench v1 is an
under-training artefact. Same architecture and hyperparameters as
the standard `timegan/` submission, with the per-phase epoch count
tripled (`emb_epochs = sup_epochs = gan_epochs = 600`).

Result: aggregate `overall_score = 0.389 ± 0.095`, `7.8 / 14` pass —
within the standard-deviation band of the canonical TimeGAN row
(`0.388 ± 0.110`, `9 / 14` pass). Tripling the training budget
does not lift TimeGAN out of the bottom of the leaderboard. The
poor result is structural to the architecture on this panel, not
an under-training artefact.

This row is reported as a robustness footnote alongside the
standard `timegan/` row; it does not replace the canonical
submission so that the leaderboard remains comparable to the
field-standard `300 / 300 / 300`-epoch reporting convention used
by Yoon et al. (NeurIPS 2019).
