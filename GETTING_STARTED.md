# Getting Started — submit your method to FinBench

The **shortest path** from "I have a multivariate-TS-gen model" to "my numbers
are on the leaderboard." For the panel and protocol, see
[BENCHMARK.md](./BENCHMARK.md); for the tasks, see
[BENCHMARK_TASKS.md](./BENCHMARK_TASKS.md).

## Setup (one time)

```bash
git clone https://github.com/sablier-ai/finbench
cd finbench
pip install sablier-flow                                 # for the panel
pip install finval                                       # for scoring
```

## 1. Load the panel

```python
import sablier_flow
df = sablier_flow.demo_data("us_equities_macro_2010_2023")
# 3522 rows × 7 features (IWM, QQQ, SPY, TLT, VIX, TNX, DXY), daily 2010–2023.
```

## 2. Train on 2010-2019; generate (200, 60, 7) returns

```python
train_df = df.loc[df.index < "2020-01-01"]

# YOUR MODEL HERE — any architecture, any framework.
synth = your_model.train_and_sample(
    train_df,
    horizon=60,      # 3 trading months
    n_paths=200,     # FinBench panel standard
)
# synth.shape must be exactly (200, 60, 7), float32 returns.
```

Per-feature transform conventions (so `synth` is in the right space):

| Feature       | Transform           |
|---------------|---------------------|
| IWM/QQQ/SPY/TLT/DXY | log-returns    |
| VIX/TNX       | first-differences   |

Feature order is `[IWM, QQQ, SPY, TLT, VIX, TNX, DXY]`.

## 3. Archive each seed

Generate one tensor per seed and build the seed directory with
`examples/submit.py`:

```bash
python examples/submit.py \
    --synth your_synth_seed0.npy \
    --name your_method \
    --seed 0 \
    --paper https://arxiv.org/abs/YOUR_PAPER
```

This writes:

```
reference/your_method/seed_0/
    synth_paths.npy
    real_paths.npy
    meta.json
    finval_scores.json
```

Repeat for each seed. The benchmark always scores your synth against the **pinned
canonical panel real** — you never supply your own ground truth.

## 4. Score every task

From the repo root, run the multi-task board:

```bash
python -m benchmark.run
```

This loads every archived competitor, runs the memorization/copy guard, scores
all seven tasks (F1, F2, F4, F5, T2, T3, T5) under identical conditions, and
writes `MULTITASK_LEADERBOARD.md` — per-task boards plus the rank-based aggregate.

To score a subset while iterating:

```bash
FINBENCH_TASKS=F1,T3 python -m benchmark.run
```

## 5. Open a PR

```bash
git checkout -b submit-your_method
git add reference/your_method/
git commit -m "Submit your_method to FinBench"
git push origin submit-your_method
gh pr create
```

We review for **protocol compliance**, not for whether your numbers are "good
enough." Every reproducible submission is accepted.

## FAQ

**Do I need to share my model code?** No. Only the synthetic outputs.

**Which shape does my synth need?** Exactly `(200, 60, 7)`, float32 returns, in
native return space (log-returns for IWM/QQQ/SPY/TLT/DXY, first-differences for
VIX/TNX), feature order `[IWM, QQQ, SPY, TLT, VIX, TNX, DXY]`.

**What if my model needs per-dataset tuning?** Submissions with tuned
hyperparameters are accepted but marked on the leaderboard. Document the tuning
protocol in your `meta.json`.

**Can I rerun the reference competitors myself?** Yes. The `reference/` directory
ships the seeds + raw scores; you can reproduce or verify any number.
