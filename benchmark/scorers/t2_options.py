"""T2 — Options pricing / IV-smile fidelity scorer (LOWER = BETTER).

Adapts examples/option_pricing.py to the benchmark scorer interface, keeping its
protocol EXACTLY:

  1. Take the SPY column (the underlying with the most liquid options) of each
     path tensor of native daily log-returns.
  2. Build the terminal price S_T = exp(sum of log-returns to maturity), S0 = 1,
     and martingale-correct it: S_T <- S_T / E[S_T], so E[S_T] = 1 under r = 0.
     The smile is determined by the SHAPE of the terminal-return distribution
     (vol, skew, kurtosis, vol-of-vol), not the drift — de-drifting both synth
     and real to r = 0 makes the comparison measure-consistent and isolates
     distributional fidelity.
  3. MC-price undiscounted European calls on the moneyness grid
     K in {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15} at maturities
     {21, 42, 60} trading days (T in years with 252 trading days).
  4. Invert Black-Scholes (Brent root-find) to a model-implied IV smile.
  5. Score = RMSE between the model smile and the REAL data's empirical smile
     (same construction on the paired real tensor), averaged over maturities.
     Reported in BASIS POINTS of implied vol (1 vol-pt = 100 bps, i.e.
     IV 0.20 = 2000 bps).

NaN-strike policy (audit T2-3) — the strike grid is FIXED for everyone:
  * The scoring grid at each maturity is the set of strikes where the REAL
    smile is invertible (real-side NaN strikes are an assessability gap and
    are excluded SYMMETRICALLY — the same grid applies to every competitor;
    on the archived panel the real smile is finite at all 21 strikes).
  * A strike on that grid where the SYNTH smile fails to invert is NEVER
    dropped from the RMSE. It is scored at a worst-case penalty IV:
    MC price <= intrinsic (under-dispersed tails) -> IV = 0.0; MC price >=
    the sigma = 5.0 Black-Scholes bound (blow-up) -> IV = 5.0. This removes
    the per-competitor variable-denominator bug where nanmean silently
    scored under-dispersed generators on the easy ATM strikes only, and it
    maps an all-NaN synth smile to a large finite worst-case RMSE instead
    of NaN (which used to poison the board ranks, F-03).
  * Penalized strike counts are reported in ``detail["nan_strikes"]``.

Drift is NOT scored here — T2 is a PURE smile-SHAPE RMSE.
  A prior revision multiplied the smile RMSE by a spurious-drift gate,
  rmse * (1 + DRIFT_LAMBDA * excess_drift), where
      excess_drift = mean_m max(0, |E[S_T]_synth - 1| - |E[S_T]_real - 1|)
  is measured on the raw (pre-correction) terminal prices. That gate was
  removed on two grounds:

  1. It charged noise. The drift allowance is the real panel's own deviation
     from the martingale, estimated on 200 stride-1 OVERLAPPING 60d windows
     from a single ~4-year regime (~16 independent blocks). At that effective
     sample size the real forward's excess over 1.0 is indistinguishable from
     zero and flips sign with the window, so the gate penalized most of the
     honest field for out-deviating a reference that is itself noise. The smile
     RMSE is already de-drifted (martingale-corrected) on both sides, so it
     measures shape, not the forward — re-charging the forward double-counts
     what F5 already scores.

  2. It was gameable without changing the generator. excess_drift is a property
     of the SUBMITTED 200 paths; the panel pins the path COUNT, not which 200,
     so a submitter able to generate more than 200 paths could simply select
     the 200 whose sample forward sits inside the allowance and pay a zero
     charge. Closing that non-gameably needs a protocol-level committed pool,
     out of scope for a scorer that only sees the submitted subset.

  So T2 measures smile SHAPE only, and drift / martingale behaviour is scored by
  F5. The measured forwards are still reported as a NON-SCORING diagnostic in
  ``detail["drift_diagnostic"]`` (they cannot move a rank). A degenerate synth
  whose smile fails to invert (all-NaN, exp-overflow) is still penalized in the
  SCORE via the worst-case penalty IV on every strike (NaN-strike policy above).

Direction: the tasks registry marks T2 lower-better ("rmse") — ``score()["mean"]``
is the mean IV-smile RMSE in bps across a competitor's (synth, real) pairs/seeds
(SMILE SHAPE ONLY, no drift charge); smaller means the generator's option prices
are closer to the real market's.

Interface:
    score(loaded, feature_names=None) -> {"mean", "std", "n", "detail"}
where ``loaded`` is the list of (synth, real) numpy pairs, each (n_paths, 60, 7)
of native daily increments (log-returns for price-type features).
``detail`` carries per-maturity RMSE (bps, averaged over seeds), per-seed scores,
and a NON-SCORING ``drift_diagnostic`` (measured forwards, never charged).

Protocol references: VolGAN (PMC13060012) and Cohen-Reisinger-Wang
(arXiv 2202.07148) pin the fuller IV-surface task (static no-arbitrage counts,
term structure); this scorer is the smile-RMSE component that
examples/option_pricing.py implements — extensions are tracked in
BENCHMARK_TASKS.md, not silently added here.

Run a ranked smoke test from the finbench repo root:
    python -m benchmark.scorers.t2_options
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

# v1 panel feature order (see benchmark/registry.py + BENCHMARK_TASKS.md).
FEATURE_ORDER = ["IWM", "QQQ", "SPY", "TLT", "VIX", "TNX", "DXY"]

MONEYNESS = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
MATURITIES = [21, 42, 60]  # trading days (paths are 60d)
VOL_TO_BPS = 1e4           # IV is a decimal vol; 1.0 vol = 100 vol-pts = 10_000 bps

# Worst-case penalty IVs for synth strikes that fail to invert (audit T2-3).
IV_FLOOR = 0.0   # MC price <= intrinsic: the model's IV is effectively 0
IV_CAP = 5.0     # MC price >= BS(sigma=5): the model's IV is beyond the bracket

# The pre-correction spurious-drift GATE has been REMOVED (
# the drift-gate review): it charged window-dependent noise and was driven to
# a zero charge by pure path selection. T2 now scores smile SHAPE only; drift is
# scored by F5. The measured forward is kept as a NON-SCORING diagnostic below.


# ---------------------------------------------------------------------------
# Black-Scholes machinery — adapted from examples/option_pricing.py
# ---------------------------------------------------------------------------

def bs_call(S, K, T, sig):
    if sig <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    return S * norm.cdf(d1) - K * norm.cdf(d1 - sig * np.sqrt(T))


def implied_vol(price, K, T):
    intrinsic = max(1.0 - K, 0.0)
    if price <= intrinsic + 1e-9:
        return np.nan
    try:
        return brentq(lambda s: bs_call(1.0, K, T, s) - price, 1e-4, IV_CAP, xtol=1e-6)
    except ValueError:
        return np.nan


def penalty_iv(price, K, T):
    """Worst-case IV for a non-invertible MC price (synth side only, T2-3).

    price <= intrinsic  -> IV_FLOOR (under-dispersed: the model prices the
                           option at/below its intrinsic value, IV -> 0);
    otherwise the brentq bracket failed high -> IV_CAP.
    """
    intrinsic = max(1.0 - K, 0.0)
    if price <= intrinsic + 1e-9:
        return IV_FLOOR
    return IV_CAP


def model_smile(returns, maturity_days, spy_idx):
    """returns: (n_paths, 60, 7). Build SPY terminal price at `maturity_days`,
    martingale-correct, MC-price the moneyness grid, invert to IV. T in years (252).

    Returns (ivs_raw, ivs_penalized, drift_dev):
      ivs_raw       — IV per strike, NaN where non-invertible;
      ivs_penalized — same but non-invertible strikes carry the worst-case
                      penalty IV (IV_FLOOR / IV_CAP) instead of NaN;
      drift_dev     — |E[S_T] - 1| of the RAW (pre-correction) terminal prices.
                      NON-SCORING diagnostic only (the drift gate was removed,
                      the drift-gate review); reported, not charged.
    """
    r = returns[:, :maturity_days, spy_idx]              # SPY log-returns to maturity
    ST = np.exp(r.sum(axis=1))                           # terminal price, S0=1
    drift_dev = float(abs(ST.mean() - 1.0))              # BEFORE martingale correction
    ST = ST / ST.mean()                                  # martingale: E[S_T]=1 (r=0)
    T = maturity_days / 252.0
    ivs_raw, ivs_pen = [], []
    for k in MONEYNESS:
        c = np.mean(np.maximum(ST - k, 0.0))             # undiscounted (r=0)
        iv = implied_vol(c, k, T)
        ivs_raw.append(iv)
        ivs_pen.append(iv if np.isfinite(iv) else penalty_iv(c, k, T))
    return np.array(ivs_raw), np.array(ivs_pen), drift_dev


# ---------------------------------------------------------------------------
# Scorer interface
# ---------------------------------------------------------------------------

def score(loaded, feature_names=None) -> dict:
    """Mean IV-smile RMSE (bps, LOWER = BETTER) across a competitor's seeds.

    loaded: list of (synth, real) numpy pairs, each (n_paths, 60, 7).

    Per maturity, the scoring strike grid is the real-invertible strikes
    (real-side NaN = assessability gap, excluded symmetrically for everyone);
    on that grid a synth-NaN strike scores the worst-case penalty IV, never
    dropped (T2-3). The score is SMILE SHAPE ONLY: the spurious-drift GATE was
    REMOVED (the drift-gate review — it charged noise and was
    selection-gameable). The measured forward is reported NON-SCORING in
    detail["drift_diagnostic"]; drift is scored by F5. See the module docstring.
    """
    names = list(feature_names) if feature_names is not None else FEATURE_ORDER
    spy_idx = names.index("SPY")

    per_seed = []                                  # smile RMSE (bps) per seed = the score
    per_seed_excess_drift = []                     # NON-SCORING diagnostic per seed
    per_mat = {T: [] for T in MATURITIES}          # per-maturity smile RMSE (bps)
    nan_strikes = {T: [] for T in MATURITIES}      # penalized synth strikes per seed
    for synth, real in loaded:
        synth = np.asarray(synth, dtype=np.float64)
        real = np.asarray(real, dtype=np.float64)
        errs, ex_drifts = [], []
        for T in MATURITIES:
            syn_raw, syn_pen, d_syn = model_smile(synth, T, spy_idx)
            real_raw, _, d_real = model_smile(real, T, spy_idx)
            grid = np.isfinite(real_raw)           # fixed real-finite strike grid
            ex_drifts.append(max(0.0, d_syn - d_real))   # diagnostic only (not charged)
            if not grid.any():                     # real-side unassessable maturity
                per_mat[T].append(float("nan"))    # (symmetric for every competitor)
                nan_strikes[T].append(0)
                continue
            # synth-NaN strikes on the grid -> worst-case penalty IV (T2-3)
            rmse = float(np.sqrt(np.mean((syn_pen[grid] - real_raw[grid]) ** 2))) * VOL_TO_BPS
            per_mat[T].append(rmse)
            nan_strikes[T].append(int((grid & ~np.isfinite(syn_raw)).sum()))
            errs.append(rmse)
        if not errs:
            continue                               # no assessable maturity (real-side)
        base = float(np.mean(errs))
        per_seed_excess_drift.append(float(np.mean(ex_drifts)))
        per_seed.append(base)                      # SMILE SHAPE ONLY (no drift charge)

    if not per_seed:
        raise ValueError("no (synth, real) pairs loaded / no assessable maturity")

    def _mat_mean(vals):
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    return {
        "mean": float(np.mean(per_seed)),          # bps of IV, lower = better (smile only)
        "std": float(np.std(per_seed)),
        "n": len(per_seed),
        "detail": {
            "unit": "bps of implied vol (1 vol-pt = 100 bps); lower = better",
            "score_formula": "mean_m smile_RMSE_m (bps of IV); no drift charge",
            "per_maturity_rmse_bps": {              # smile RMSE
                f"{T}d": _mat_mean(per_mat[T]) for T in MATURITIES
            },
            "per_seed_rmse_bps": per_seed,          # smile RMSE = the score
            "drift_diagnostic": {                   # NON-SCORING: reported, never charged
                "scored": False,
                "note": ("drift statistic is NON-SCORING: an earlier drift gate was "
                         "found to charge window-dependent noise and could be driven to a "
                         "zero charge by path selection, so it was removed. This statistic "
                         "does NOT enter the score and cannot move a rank; drift/martingale "
                         "is scored by F5."),
                "statistic": "mean over maturities of max(0, |E[S_T_raw]-1| synth - |E[S_T_raw]-1| real)",
                "per_seed_excess_drift": per_seed_excess_drift,
            },
            "nan_strikes": {                        # synth strikes scored at penalty IV
                f"{T}d": nan_strikes[T] for T in MATURITIES
            },
            "penalty_iv": {"floor": IV_FLOOR, "cap": IV_CAP},
            "moneyness_grid": MONEYNESS.tolist(),
            "maturities_days": list(MATURITIES),
        },
    }


if __name__ == "__main__":
    from benchmark.registry import available_competitors

    rows = []
    for comp in available_competitors():
        res = score(comp.load())
        rows.append((comp.name, res))
        pm = res["detail"]["per_maturity_rmse_bps"]
        print(f"  scored {comp.name:16s} n={res['n']}  "
              + "  ".join(f"{k}={v:8.1f}" for k, v in pm.items()))

    rows.sort(key=lambda x: x[1]["mean"])  # lower = better
    print("\n=== T2 — SPY IV-smile RMSE vs real (bps, LOWER = BETTER) — SMILE SHAPE ONLY ===")
    print(f"{'rank':>4}  {'competitor':16s} {'mean':>9} {'std':>9} {'n':>3}   "
          f"{'21d':>8} {'42d':>8} {'60d':>8}   {'excess_drift(diag)':>18}")
    for i, (name, res) in enumerate(rows, 1):
        pm = res["detail"]["per_maturity_rmse_bps"]
        ex = float(np.mean(res["detail"]["drift_diagnostic"]["per_seed_excess_drift"]))
        print(f"{i:>4}  {name:16s} {res['mean']:9.1f} {res['std']:9.1f} {res['n']:>3}   "
              f"{pm['21d']:8.1f} {pm['42d']:8.1f} {pm['60d']:8.1f}   {ex:18.4f}")
    print("  (excess_drift is a NON-SCORING diagnostic — it does not enter 'mean' and cannot move a rank.)")

    # ---- T2-DRIFT-REF control: a spurious risk-free drift is NO LONGER charged by T2 ----
    # (it is F5's job now). The smile RMSE is drift-invariant by construction, so a copy
    # of real + constant drift scores ~0 on T2 — the honest consequence of removing the
    # noise/selection-gameable gate.
    real0 = np.asarray(next(iter(available_competitors())).load()[0][1], dtype=np.float64)
    print("\n  --- controls  ---")
    for nm, drift in [("copy + 30bps/day", 0.0030), ("copy - 30bps/day", -0.0030), ("exact copy", 0.0)]:
        r = score([(real0 + drift, real0)])
        d = r["detail"]["drift_diagnostic"]
        print(f"  {nm:20s} T2 score {r['mean']:8.1f} bps  "
              f"(excess_drift diag {d['per_seed_excess_drift'][0]:.4f}, NOT charged)")
    print("  => T2 no longer charges drift (smile RMSE is drift-invariant); F5 scores drift/martingale.")

    # ---- T2-SELECT control: the selection-gaming vector is CLOSED ----
    # With the gate removed there is no multiplier to zero, so choosing WHICH 200 paths to
    # submit cannot change any drift charge (there is none). Demonstrate: an honest 200 and a
    # 200-subset hand-picked to drive excess_drift -> 0 both score their pure smile RMSE; the
    # drift dimension confers ZERO score advantage either way.
    c = {x.name: x for x in available_competitors()}["FLOW-A"]
    pr = c.load()
    poolA = np.concatenate([np.asarray(s, dtype=np.float64) for s, _ in pr], axis=0)
    realA = np.asarray(pr[0][1], dtype=np.float64)
    spy = FEATURE_ORDER.index("SPY")
    honest = np.asarray(pr[0][0], dtype=np.float64)
    ST60 = np.exp(poolA[:, :60, spy].sum(axis=1))
    order = np.argsort(ST60)
    lo, hi, picked, s = 0, len(order) - 1, [], 0.0
    while len(picked) < 200:                       # two-pointer subset with E[S_T] ~ 1
        if 1.0 - (s / max(len(picked), 1)) >= 0 and hi > lo:
            picked.append(order[hi]); s += ST60[order[hi]]; hi -= 1
        else:
            picked.append(order[lo]); s += ST60[order[lo]]; lo += 1
    sel = poolA[np.array(picked)]
    rh, rs = score([(honest, realA)]), score([(sel, realA)])
    print("\n  --- T2-SELECT: subset-selection can no longer buy a drift discount ---")
    print(f"  honest 200        excess_drift(diag) {rh['detail']['drift_diagnostic']['per_seed_excess_drift'][0]:.4f}  "
          f"T2 score {rh['mean']:8.1f}")
    print(f"  drift-zeroed 200  excess_drift(diag) {rs['detail']['drift_diagnostic']['per_seed_excess_drift'][0]:.4f}  "
          f"T2 score {rs['mean']:8.1f}")
    print("  => selecting paths to zero excess_drift changes NO drift charge (there is none): the "
          "\n     drift-gaming vector is closed. Each subset scores only its own smile RMSE, which is "
          "\n     an inherent property of any sample-based scorer (a submission-protocol matter, not a gate).")
