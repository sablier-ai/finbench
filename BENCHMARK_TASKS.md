# FinBench — Multi-Task Benchmark

FinBench is a **suite of independent tasks**, each a fair competition under
identical conditions, with a **per-task leaderboard** and a **rank-based
aggregate**.

Fix the task, fix the data, fix the protocol, then let *every* model — every
FLOW flavor and every external baseline — compete under exactly the same
conditions. No model gets a private advantage; the only thing that varies is the
model.

## 1. Tasks

A model is generated **once** per panel and then scored by every applicable task,
so a model cannot be tuned to a task. Every task scores the same archived path
tensors (`reference/<competitor>/…`) against the pinned canonical panel real.

The suite has two layers. The **fidelity layer** measures whether the synthetic
data reproduces the real panel's distributional and stylized-fact signature. The
**utility layer** measures economic downstream behavior a distributional number
can't capture. Each task follows an established protocol with a pinned metric
implementation and citation.

**Fidelity layer**

| ID | Task | Metric | Dir | Source |
|----|------|--------|-----|--------|
| **F1** | Synthetic-data quality (finance-aware) | finval 0.6.1 aggregate (path-scored suite), gate-penalized | ↑ | our suite; covers Cont (2001) stylized facts + tail-dependence / coskew / regime calibration. A model that fails a hard gate (tails, memorization, drawdown) cannot rank as if it passed. |
| **F2** | Stylized-facts battery | Cont (2001) 11-fact synth-vs-real distances (heavy tails, volatility clustering, leverage, gain/loss asymmetry, timescale asymmetry) | ↓ | Cont (2001), *Quantitative Finance* 1(2) |
| **F4** | Distributional distance | Wasserstein-1 / RBF-MMD / signature-MMD, shape and scale scored separately | ↑ | Villani (2009); Gretton et al. (2012); Ni et al. (2021) |
| **F5** | Martingale / no-free-alpha check | reproduction of the real panel's near-random-walk structure; manufactured predictability penalized in either direction | ↑ | Wiese et al. (2021); Lo & MacKinlay (1988) variance-ratio |

**Utility layer**

| ID | Task | Metric | Dir | Source |
|----|------|--------|-----|--------|
| **T2** | Options pricing / IV smile | Black-Scholes implied-vol RMSE vs the real smile across a moneyness × maturity grid, in basis points | ↓ | Monte-Carlo repricing + BS inversion |
| **T3** | Predictive validity (TSTR) | Spearman ρ between real and synthetic mean-Sharpe rankings over a frozen multi-family strategy book (marginal + cross-asset/joint) | ↑ | Esteban et al. (2017) train-on-synth/test-on-real |
| **T5** | VaR / ES risk backtesting | Kupiec POF + Christoffersen (independence + conditional coverage) + Acerbi–Székely ES test + Basel traffic-light, with an ES-over-fatness gate | ↑ | Kupiec (1995); Christoffersen (1998); Acerbi–Székely (2014); Basel (1996) |

Tasks are **orthogonal axes**. A model can win fidelity (F1) and still misrank
strategies (T3) — that is the point of a multi-task suite: no single number hides
a failure mode. Every task shown resolves the field: its competitors clear the
task's honest real-vs-real noise floor.

A **memorization/copy guard** (`benchmark/guards.py`) runs before ranking, not as
a scored task: MEMORISATION / COPY verdicts are disqualified (excluded from every
board, listed with verdict + detail); SUSPICIOUS rows are ranked but
asterisk-flagged. If the guard module is missing the runner prints a loud warning
and the board must not be published.

## 2. Competitors

Three families, all scored identically:

- **Sablier-Flow rows** — three entries under transparent labels:
  - **Sablier-Flow** — currently shipping in the [Sablier SDK](https://pypi.org/project/sablier-flow/).
  - **Sablier-Flow-Next** — top research candidate for the next production release.
  - **Sablier-Flow-Old** — previous production model, kept for progression comparison.
- **External neural baselines** — KoVAE, Diffusion-TS, TimeVAE, TimeGAN,
  QuantGAN, ImagenTime, FM-TS. Published under their real names, at their
  authors' published defaults.
- **Classical & replay methods** — whatever a practitioner would actually use
  *instead of* a deep generator competes on the boards for the tasks it's used
  for: GARCH-t, DCC-t, Gaussian-iid, t-Copula (parametric), and the replay family
  Historical-Sim, Block-Bootstrap, FHS (filtered historical simulation).
  Replay-based rows are **ranked and labeled `replay-resampling`**, with
  memorization-guard flags shown rather than treated as disqualifying — they
  replay real data by construction. "Beats historical simulation" is exactly the
  claim a practitioner needs tested; a board without these rows can't support it.

Every row is **provenance-marked** on the board (`benchmark/registry.py`):

- `production` — Sablier-Flow, currently shipping in the SDK.
- `research` — Sablier-Flow-Next, top research candidate for the next production
  version.
- `production-legacy` — Sablier-Flow-Old, previous production model.
- `published-defaults` — external baselines at their own untuned published
  defaults, no per-panel tuning.
- `replay-resampling` — resamples/replays real training data (the historical
  simulation family); scenarios are drawn from history rather than generated,
  so memorization-guard flags are expected by construction.

Every competitor is generated once per panel under the frozen protocol
(`BENCHMARK.md` §1–2) and archived to `reference/<competitor>/`. All competitors
are scored against the **pinned canonical panel real**
(`reference/panels/us_equities_macro/real_paths.npy`, sha256-verified at run
time); a `real*.npy` inside a competitor archive is rejected unless
byte-identical — a submitter never supplies their own ground truth.

## 3. Leaderboards

- **Per-task board** — absolute scores for every competitor on that task, with
  the task's native metric, a seed count `n`, a 95% t-CI over seeds, a `≈#1`
  marker for entries statistically indistinguishable from the task leader (Welch
  test), and a real-vs-real noise-floor row.
- **Aggregate board** — **mean rank across tasks.** Absolute scores do *not*
  transfer across tasks (an RMSE in bps and a Spearman ρ are not comparable), so
  the aggregate is rank-based. It includes **only competitors scored on all
  scored tasks** — a missing cell (scorer error or coverage gap) moves the row to
  a separate partial-coverage table, so crashing a bad task can never improve a
  rank. Aggregate ranks are tie-aware (`1=` for exact ties) with an approximate
  bootstrap P(#1) column; coverage is reported (no silent gaps).

## 4. Fairness contract

For a result to enter a board, it must satisfy:

1. **Same panel + split** — the task's frozen panel and train/OOS boundary, for
   every competitor.
2. **Same generation protocol** — same `N` paths, same horizon `H`, same OOS
   anchor. Generated once; not re-rolled per task.
3. **Same evaluator** — one pinned metric implementation per task (e.g. finval
   0.6.1 for F1), applied identically to all competitors.
4. **No per-task tuning** — a competitor's model/config is fixed before it sees
   any task. Hyperparameters are the published/production defaults.
5. **Seeds reported** — mean ± std across the task's seed set; no cherry-picking.

Any deviation (a baseline that can't run at a panel's `D`, a task that needs
inputs a model can't produce) is **logged as a coverage gap**, never silently
dropped. Replay methods are **ranked with disclosure** (`replay-resampling`
provenance + guard flags shown), not excluded — exclusion would hide the
most-used practitioner alternative from exactly the comparison that matters.
