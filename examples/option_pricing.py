"""Physical-measure option-pricing harness for FinBench generators.

The smile is determined by the SHAPE of the terminal-return distribution (vol, skew, kurtosis,
vol-of-vol) — not the drift. So we martingale-correct each generator's terminal price (E[S_T]=1
under r=0), MC-price a strike grid, invert Black-Scholes to a model-implied IV smile, and compare
each model's smile to the REAL data's smile (same construction). Measure-consistent: everything is
de-drifted to r=0, so this isolates distributional fidelity. The model whose implied smile best
matches the empirical (real-data) smile prices options best.

  python examples/option_pricing.py
"""
from __future__ import annotations
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
sys.path.insert(0, "examples"); import submit
submit.PANEL_NAME = "us_equities_macro_2010_2023"
from submit import load_real_reference, FEATURE_ORDER

SPY = FEATURE_ORDER.index("SPY")            # underlying with the most liquid options
MONEYNESS = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
MATURITIES = [21, 42, 60]                    # trading days (paths are 60d)


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
        return brentq(lambda s: bs_call(1.0, K, T, s) - price, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan


def model_smile(returns, maturity_days):
    """returns: (n_paths, 60, 7). Build SPY terminal price at `maturity_days`, martingale-correct,
    MC-price the moneyness grid, invert to IV. T in years (252)."""
    r = returns[:, :maturity_days, SPY]                 # SPY log-returns to maturity
    ST = np.exp(r.sum(axis=1))                           # terminal price, S0=1
    ST = ST / ST.mean()                                  # martingale: E[S_T]=1 (r=0)
    T = maturity_days / 252.0
    ivs = []
    for k in MONEYNESS:
        c = np.mean(np.maximum(ST - k, 0.0))             # undiscounted (r=0)
        ivs.append(implied_vol(c, k, T))
    return np.array(ivs)


def main():
    real = load_real_reference()
    paths = {
        "REAL (truth)": real,
        "FLOW": np.load("reference/sablier_flow/seed_0/synth_paths.npy").astype(np.float32),
        "DCC-GARCH-t": np.load("/tmp/dcc_t_synth.npy").astype(np.float32),
        "GARCH-t indep": np.load("/tmp/garch_t_synth.npy").astype(np.float32),
    }
    for T in MATURITIES:
        smiles = {n: model_smile(p, T) for n, p in paths.items()}
        real_s = smiles["REAL (truth)"]
        print(f"\n=== {T}d implied-vol smile (annualized %)  —  S&P 500 ===")
        print("  moneyness:   " + "  ".join(f"{m:5.2f}" for m in MONEYNESS) + "   |  ATM   skew(95-105)  RMSE-vs-REAL")
        for n, s in smiles.items():
            atm = s[MONEYNESS == 1.00][0]
            skew = s[MONEYNESS == 0.95][0] - s[MONEYNESS == 1.05][0]   # put-call IV skew
            rmse = np.sqrt(np.nanmean((s - real_s) ** 2)) if n != "REAL (truth)" else 0.0
            grid = "  ".join(f"{v*100:5.1f}" if np.isfinite(v) else "  nan" for v in s)
            tag = "  <- TRUTH" if n == "REAL (truth)" else f"  {rmse*100:5.2f}"
            print(f"  {n:14s} {grid}  | {atm*100:5.1f}  {skew*100:+6.2f}     {tag}")
    # headline: which model's smile is closest to real, averaged over maturities
    print("\n=== smile fidelity (avg IV-RMSE vs REAL across maturities, lower=better) ===")
    scores = {}
    for n, p in paths.items():
        if n == "REAL (truth)":
            continue
        errs = [np.sqrt(np.nanmean((model_smile(p, T) - model_smile(real, T)) ** 2)) for T in MATURITIES]
        scores[n] = np.nanmean(errs)
    for n, e in sorted(scores.items(), key=lambda x: x[1]):
        print(f"  {n:14s} {e*100:5.2f} vol-pts")


if __name__ == "__main__":
    main()
