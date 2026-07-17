# TimeGAN-600 — robustness submission

Tests whether TimeGAN's low position on FinBench v1 is an under-training
artefact. Same architecture and hyperparameters as the standard `timegan/`
submission, with the per-phase epoch count tripled
(`emb_epochs = sup_epochs = gan_epochs = 600`).

Result: TimeGAN-600 and the canonical TimeGAN land within a rank or two of
each other on every task. Tripling the training budget does not lift TimeGAN
off the bottom of the leaderboard, so the outcome is a property of the
architecture on this panel, not an under-training artefact.

Reported as a robustness footnote alongside the standard `timegan/` row; the
canonical row is kept so the leaderboard stays comparable to the
`300 / 300 / 300`-epoch reporting convention used by Yoon et al.
(NeurIPS 2019).
