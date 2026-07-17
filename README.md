# FinBench

**A public benchmark for multivariate financial time-series generation.**

FinBench evaluates synthetic financial time-series generators on the properties
that actually matter for financial backtesting — fat tails, volatility
clustering, leverage effect, tail dependence, drawdown distribution, risk-model
calibration, strategy-rank transfer — not on the distributional-matching proxies
(DS / PS / Context-FID) that dominate general-purpose TS-gen benchmarks.

> **Why this exists.** Every popular TS-gen benchmark (TimeGAN protocol,
> TSGBench, GenTS) measures *can a classifier distinguish synth from real?* That
> metric **rewards mode collapse** and ignores everything a quant cares about. A
> model that produces smooth, low-variance synth can ace DS/PS while violating
> vol clustering, leverage effect, and heavy-tail behavior — exactly the failure
> modes that destroy backtests.
>
> FinBench replaces that beauty contest with a suite of **financial stylized-fact
> and economic-utility tasks** grounded in the empirical-finance literature (Cont
> 2001, Black 1976, Joe 1997, Bailey & López de Prado 2014). If your model passes
> FinBench, your synth is good enough to fit strategies on.

## Quick links

- [Task suite](./BENCHMARK_TASKS.md) — the seven tasks, metrics, and fairness contract
- [Panel & submission spec](./BENCHMARK.md) — the frozen panel and protocol
- [Leaderboard](./MULTITASK_LEADERBOARD.md) — overall ranking, per-task matrix, and detailed score tables
- [Getting started](./GETTING_STARTED.md) — submit your model
- [finval](https://github.com/sablier-ai/finval) — the scoring library FinBench uses

## The suite

Seven tasks on the `us_equities_macro` panel (`D=7`, `N=200`, `H=60`, OOS). A
model is generated **once** per panel and scored by every task; per-task boards
report the task's native metric with a real-vs-real noise floor, and a rank-based
aggregate combines them.

Sablier publishes three rows on the board: **Sablier-Flow** (currently
shipping in the [Sablier SDK](https://pypi.org/project/sablier-flow/)),
**Sablier-Flow-Next** (top research candidate for the next production
release), and **Sablier-Flow-Old** (previous production model, kept for
progression comparison). Between them, Sablier takes the top spot on
4 of the 7 tasks — overall synthetic-data quality (F1), distributional
distance (F4), options pricing (T2), and VaR/ES risk backtesting (T5).
External baselines run their published defaults.

**Fidelity** — does the synth reproduce the real panel's distribution?

- **F1** — Synthetic-data quality (finval 0.6.1 aggregate, gate-penalized): the
  finance-aware path suite covering Cont (2001) stylized facts plus tail
  dependence, coskew, and regime calibration.
- **F2** — Stylized-facts battery: Cont (2001) 11-fact synth-vs-real distances.
- **F4** — Distributional distance: Wasserstein-1 / MMD / signature-MMD, with
  shape and scale scored separately so shrinking earns no bonus.
- **F5** — Martingale / no-free-alpha: does the synth reproduce the real panel's
  near-random-walk structure, without manufacturing tradeable predictability?

**Utility** — economic downstream evidence a distributional number can't provide.

- **T2** — Options pricing / implied-vol smile: Black-Scholes IV RMSE vs the real
  smile, in basis points.
- **T3** — Predictive validity (TSTR): Spearman ρ between real and synthetic
  strategy-Sharpe rankings over a frozen multi-family strategy book.
- **T5** — VaR / ES risk backtesting: Kupiec + Christoffersen + Acerbi–Székely +
  Basel traffic-light, with an ES-over-fatness gate.

Tasks are orthogonal: a model can win fidelity and still misrank strategies — no
single number hides a failure mode. See [BENCHMARK_TASKS.md](./BENCHMARK_TASKS.md)
for metrics, directions, and citations, and
[MULTITASK_LEADERBOARD.md](./MULTITASK_LEADERBOARD.md) for current standings.

## Submit your model

1. Generate `(200, 60, 7)` synthetic returns for the FinBench panel
   (`pip install sablier-flow` for the panel data).
2. Archive your seeds under `reference/<your_method>/seed_*/` (the
   [`examples/submit.py`](./examples/submit.py) template builds the directory).
3. Run `python -m benchmark.run` to score every task and open a PR adding your
   `reference/<your_method>/` directory.

The protocol does not require you to share your model code or weights — only the
synthetic outputs are needed to score. See
[GETTING_STARTED.md](./GETTING_STARTED.md) for the full walkthrough.

## Versioning

Each FinBench edition is frozen at a git tag for reproducibility, with a pinned
finval version. Scores are only comparable within a single finval version (the
metric implementations shift across versions), so every result records the finval
that produced it. Once a number is committed under a tag, the protocol it was
scored against will not change.

## License

Code: MIT. Reference results: CC-BY 4.0.
