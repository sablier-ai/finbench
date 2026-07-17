# FinBench v1 — Panel & Submission Protocol

**Status: frozen.** This document defines the v1 panel, split, generation
requirements, and submission format. The **tasks and metrics** are defined in
[BENCHMARK_TASKS.md](./BENCHMARK_TASKS.md); the current standings are in
[MULTITASK_LEADERBOARD.md](./MULTITASK_LEADERBOARD.md). Any change to the panel,
split, or task set requires a new version (v2) on a new branch — v1 numbers must
remain comparable forever.

## 1. Panel

The v1 panel is the bundled `us_equities_macro_2010_2023` dataset shipped in
[sablier-flow](https://pypi.org/project/sablier-flow/). **7 features, daily,
2010-01-04 to 2023-12-28 (3,522 rows).**

Feature order (canonical, frozen for v1):
``[IWM, QQQ, SPY, TLT, VIX, TNX, DXY]``.

| Symbol | Type        | Notes                          | data_type   |
|--------|-------------|--------------------------------|-------------|
| IWM    | Equity ETF  | Russell 2000 small-cap         | `price`     |
| QQQ    | Equity ETF  | Nasdaq-100                     | `price`     |
| SPY    | Equity ETF  | S&P 500                        | `price`     |
| TLT    | Bond ETF    | 20+ year Treasury              | `price`     |
| VIX    | Vol index   | Implied vol of S&P 500         | `level`     |
| TNX    | Rate        | 10-year Treasury yield         | `level`     |
| DXY    | FX index    | US Dollar Index                | `price`     |

Load via:

```python
import sablier_flow
df = sablier_flow.demo_data("us_equities_macro_2010_2023")
# Returns a (3522, 7) DataFrame with DatetimeIndex.
```

The panel is **fully public** — no licensing required for any use.

## 2. Train / OOS split

- **Train**: `2010-01-04` through `2019-12-31` (inclusive)
- **OOS**:   `2020-01-02` through `2023-12-28` (inclusive)

Models are fit on train only. The OOS slice is the real reference every
submission is scored against — it is never seen during generation.

## 3. Generation requirements

Submit `(n_paths, horizon, n_features) = (200, 60, 7)` synthetic windows,
matching the shape of the FinBench real reference:

- **Horizon**: 60 trading days (~3 months)
- **n_paths**: 200
- **n_seeds**: 5 (report mean ± std across seeds 0–4)
- **Output space**: native return space — log-return for `price` features,
  first-difference for `level` features (see the `data_type` column above).

The real reference (200 sliding windows from the OOS slice, evenly sampled) is
shared and its sha256 is asserted at load time, so scoring is deterministic. It
lives at `reference/panels/us_equities_macro/real_paths.npy`.

## 4. Tasks & metrics

The board scores each submission on the tasks defined in
[BENCHMARK_TASKS.md](./BENCHMARK_TASKS.md) — fidelity (distributional quality,
stylized facts, distributional distance, martingale behaviour) and utility
(options pricing, predictive validity, VaR/ES risk). Quality metrics come from
[finval](https://github.com/sablier-ai/finval) (version pinned per leaderboard
generation). You submit paths; the board computes every task score from them.

Each task is reported against an honest **real-vs-real noise floor** — the real
panel scored against an independent, calendar-disjoint draw of itself — so a
task only separates the field when models differ by more than that floor.

## 5. Hyperparameters

Each method uses its own **published defaults** — no per-panel tuning. This is
standard practice in the time-series-generation literature; the goal is to
compare architectures, not hyperparameter sweeps. If a method needs tuning to
converge, document it in the submission's `meta.json`.

## 6. Reproducibility

- Random seeds: 0, 1, 2, 3, 4.
- Software versions: pin in your `meta.json` (Python, framework, finval version).
- Hardware: report the GPU in `meta.json`. Submissions are scored on output
  quality only, never on speed, so they are not GPU-comparable.

## 7. Submission format

A submission is a directory under `reference/<method_name>/`:

```
reference/<method_name>/
  seed_0/
    synth_paths.npy        # (200, 60, 7) float32, native return space
    meta.json              # per-seed metadata (schema below)
  seed_1/ ... seed_4/
  README.md                # one-paragraph description of the method + citation
```

Per-seed `meta.json`:

```json
{
  "method": "your_method_name",
  "version": "1.0.0",
  "code": "https://github.com/your-org/your-repo",
  "paper": "https://arxiv.org/abs/...",
  "seed": 0,
  "framework": "pytorch 2.7.0",
  "gpu": "NVIDIA A100 SXM4 40GB"
}
```

Workflow: `python examples/submit.py` scores one seed and writes its `meta.json`;
`python examples/score.py --name <method>` aggregates across seeds; and
`python -m benchmark.run` regenerates the multi-task leaderboard.

## 8. What FinBench does NOT require

- Your model code (encouraged, not required)
- Your weights (irrelevant — only outputs matter)
- Any specific framework (PyTorch, TF, JAX all work)
- Any specific architecture (GAN, VAE, diffusion, flow, autoregressive — all welcome)

We score outputs, not internals. Same protocol everyone runs against.

## 9. Versioning

- Scorer upgrades that don't change metric semantics are picked up by a pure
  recompute on the stored paths (`python -m benchmark.run`) — no model re-runs
  — and produce a new versioned leaderboard. Prior versions stay frozen.
- Changes to the panel, split, or task set require a new FinBench version
  (v2, v3, …). v1 numbers stay valid forever.
