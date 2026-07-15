"""F5 — No-manufactured-predictability / martingale-structure check (BAND metric).

Motivation (Wiese et al. 2021, "Multi-Asset Spot and Option Market Simulation",
arXiv:2112.06823): a market simulator that leaks TRADEABLE structure — return
predictability the real market does not have — flatters every downstream task
(TSTR, hedging, portfolio-OOS). "Manufactured alpha" in the tradeable sense IS
predictability: the ability to forecast returns from their own past. F5 scores
whether the generator reproduces the real panel's (near-random-walk)
autocorrelation structure; deviation in EITHER direction (too predictable OR too
anti-predictable) is penalised.

Scope: only the 5 price-type features of the v1 us_equities_macro panel
(IWM, QQQ, SPY, TLT, DXY — native daily LOG-RETURNS). VIX and TNX are level
DIFFS, not prices, so a martingale test is not meaningful for them and they are
excluded.

FOUR scored tests, each computed on synth and on real, per feature. Every one is
a SCALE-INVARIANT (indeed AFFINE-invariant) statistic, so no rescale or spread
of a return column can move it:

1. SPURIOUS PREDICTABILITY — AR(5) fit by pooled OLS across paths (lags never
   cross path boundaries; intercept included): in-sample R^2, plus pooled
   per-path-demeaned lag-1 autocorrelation. Compared synth vs real. A synth
   panel with materially higher AR R^2 / |lag-1 ac| than real is flagged as
   "more predictable than real" (manufactured alpha).

2. VARIANCE-RATIO test — Lo & MacKinlay (1988), "Stock Market Prices Do Not
   Follow Random Walks" (Rev. Fin. Studies 1(1)): overlapping, bias-corrected
   VR(q) for q = 5, 20, computed per path and averaged across paths. VR far
   from its real-data counterpart in EITHER direction (mean-reversion or
   momentum manufactured by the generator) is penalized.

======================================================================
RETIRED — the DRIFT / MARTINGALE block is NO LONGER SCORED (removed 2026-07).
======================================================================
F5 previously carried a fifth subscore on unconditional DRIFT. Two successive
formulations were BOTH provably gameable by VARIANCE MANIPULATION, in OPPOSITE
directions, so the block was removed entirely:

  * raw drift in bps/day (mu, or the convexity-corrected martingale ratio
    E[exp(r)]-1): SCALE-DEPENDENT. A generator DEFLATES its variance (returns
    x k<1) to shrink its charged drift toward 0 and games the band UP
    (audit F5-OW-1, the original defect).
  * risk-adjusted drift, Sharpe = mu/sigma_synth: invariant to a pure rescale,
    but the band is centred on the REAL Sharpe, which is ~0 (real drift is
    statistically indistinguishable from 0 — see below). Honest generators'
    Sharpe OVERSHOOTS that ~0 centre, so a mean-preserving variance INFLATION
    (mu fixed, sigma x k>1) dilutes Sharpe back toward the centre and games the
    band UP — the exact analog of the audit's CRITICAL T6-GAME-1. Measured:
    mean-preserving synth x5 lifted FLOW-A 0.538->0.604 (rank 13->2) and DCC-t
    0.514->0.584 (rank 23->4). The scale-invariant Sharpe fix flipped the
    exploit's SIGN (collapse->inflate) rather than removing it.

These are not two bugs but ONE structural impossibility. A statistic neutral to
variance-only inflation must be independent of sigma_synth, hence a function of
mu alone — which is then rescale-gameable; a statistic invariant to a pure
rescale is Sharpe-like — which is inflation-gameable. There is NO pure-drift
statistic immune to BOTH, so NO drift subscore can satisfy the benchmark's
no-single-task-gameable rule.

The drift dimension is ALSO below its own noise floor (audit F5-OW-2/OW-3). The
real panel's per-window means are serially dependent (rolling OOS windows, ~16
independent 60d blocks -> N_eff ~ 10-19), and the real drift-vs-zero t is ~1:
real drift is indistinguishable from 0. A band centred on a reference that is
itself pure noise cannot do honest discriminative work; it only ever supplied a
gaming surface. Retiring it costs F5 nothing it could legitimately measure.

Crucially, tradeable "manufactured alpha" IS predictability, and predictability
is measured by the FOUR retained, scale-invariant subscores (AR R^2, lag-1 AC,
VR5, VR20). An UNconditional constant drift is not tradeable arbitrage (it
cannot be timed) — it is a marginal-mean error, which the marginal / distance /
VaR-ES tasks (F2, F4, T5) penalise directly and where it CANNOT be hidden by
rescaling. F5 therefore scores exactly the drift-free, scale-invariant structure
that is its robust charter, and nothing gameable.

For transparency the retired quantities are STILL COMPUTED and reported in
`detail` as DIAGNOSTICS ONLY (never scored): sharpe_*, drift_bps_*,
mart_dev_bps_*, and the overlap-corrected drift-vs-zero t (t_drift_zero_*, with
n_eff_* and the naive twins) — so a reader can verify real drift sits below its
noise floor. The 200 real tensors are ROLLING (overlapping) OOS windows, so real
per-window means are serially dependent; the naive t (dividing by sqrt(200))
overstates significance ~3.5x (audit F5-OW-5), so t_drift_zero_* uses an
OVERLAP-CORRECTED effective sample size N_eff (Geyer initial-positive
integrated-autocorrelation of the per-window means): for independent synth paths
N_eff ~ N; for overlapping real windows N_eff ~ 10-19.

Scoring (BAND: deviation from REAL behaviour in either direction is bad).
Each of the FOUR components c yields a per-feature subscore

    subscore_c = exp(-|delta_c| / s_c)   in (0, 1],  1 = matches real exactly,

with delta_c = synth statistic - real statistic and fixed scales s_c:

    AR(5) R^2      s = 0.02                          (|R2_s - R2_r|)
    lag-1 autocorr s = 0.05                          (|ac1_s - ac1_r|)
    VR(q), q=5,20  s = sqrt(2(2q-1)(q-1) / (3 q T))  (|VR_s - VR_r|; the
                    Lo-MacKinlay asymptotic std of VR(q) under i.i.d. for one
                    window of length T=60: ~0.283 at q=5, ~0.642 at q=20)

There are now FOUR subscores (ar5_r2, lag1_ac, vr5, vr20). The scales are
calibration constants of the band (chosen at typical daily-equity magnitudes),
not significance thresholds; they are shared by all competitors, so the RANKING
is scale-robust for monotone deviations.

Per-feature score = arithmetic mean of the 4 subscores; per-pair score = mean
over the 5 price features; score()["mean"] = mean over (synth, real) pairs
(seeds), "std" = std over pairs, "n" = number of pairs. 1.0 = synth reproduces
real-data martingale STRUCTURE; scores decay exponentially with deviation.

Because every scored statistic is affine-invariant, F5 is PROVABLY IMMUNE to
variance manipulation in BOTH directions: multiplying a return column by any k,
or applying a mean-preserving spread by any k, leaves all four subscores EXACTLY
unchanged. (Verified: the real x k and honest-generator x k F5 ladders are flat
for both inflation and deflation; the FLOW-A / DCC-t inflation vault above is
CLOSED, and the original deflate exploit stays closed.)

Non-finite subscore policy (audit F5-1) — the two NaN cases are DISTINCT:
  * REAL-side statistic non-finite -> the component is not assessable on this
    panel; it is excluded from the per-feature mean SYMMETRICALLY (the same
    real reference is shared by every competitor, so the exclusion applies
    to everyone identically). On the archived v1 panel all real-side
    statistics are finite, so nothing is excluded in practice.
  * REAL-side finite but SYNTH-side statistic non-finite (degenerate synth:
    zero-variance panels make AR-R^2, lag-1 AC and VR undefined) -> the
    subscore is 0.0, the WORST possible value. It is NEVER dropped from the
    mean: waiving it used to let an all-zero generator score 0.740 (#1 on
    the live board) by having its 4 hardest subscores silently excluded.
    Affected features are reported in detail["synth_degenerate_features"].

Flag: detail["more_predictable_than_real"] is True (and the offending features
listed in detail["flagged_features"]) when, averaged over pairs, either
R2_synth - R2_real > 0.01 or |ac1_synth| - |ac1_real| > 0.05 on any price
feature.

Run from the finbench/ repo root:
    <venv-python> -m benchmark.scorers.f5_martingale
"""
from __future__ import annotations

import numpy as np

# v1 us_equities_macro panel layout
DEFAULT_FEATURE_NAMES = ["IWM", "QQQ", "SPY", "TLT", "VIX", "TNX", "DXY"]
PRICE_FEATURES = ["IWM", "QQQ", "SPY", "TLT", "DXY"]  # log-return columns

AR_LAGS = 5
VR_QS = (5, 20)

# Band scales (see module docstring). NB: there is deliberately NO drift/Sharpe
# scale here — the drift/martingale block was RETIRED (removed 2026-07): every
# pure-drift statistic is gameable by variance manipulation in one direction or
# the other (audit F5-OW-1 and its inflation analog), and real drift sits below
# its own noise floor (F5-OW-2). The four retained subscores are all
# affine-invariant, so F5 cannot be gamed by rescaling.
S_R2 = 0.02
S_AC1 = 0.05

# Flag thresholds for "materially more predictable than real"
FLAG_DR2 = 0.01
FLAG_DAC1 = 0.05


# ---------------------------------------------------------------------------
# per-panel statistics (panel = (n_paths, T) array of daily log-returns)
# ---------------------------------------------------------------------------

def _eff_n(x: np.ndarray) -> float:
    """Overlap-robust effective sample size of a 1-D series (audit F5-OW-5).

    N_eff = N / tau_int with tau_int = 1 + 2*sum rho_k, the sum truncated by the
    Geyer initial-positive-sequence rule (adjacent autocorrelation pairs summed
    until a pair turns non-positive). Parameter-free and self-contained (needs no
    day-index metadata): for INDEPENDENT draws (synth per-path means) the
    autocorrelations vanish -> tau ~ 1 -> N_eff ~ N (no-op); for the ROLLING,
    overlapping real windows the strong serial dependence (lag-1 ac ~ 0.9)
    shrinks N_eff to the ~16 genuinely-independent 60d blocks."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return float(n)
    d = x - x.mean()
    g0 = float((d * d).sum() / n)
    if g0 <= 0:
        return float(n)
    K = n // 4
    tau = 1.0
    k = 1
    while k + 1 <= K:
        r1 = float((d[k:] * d[:-k]).sum() / n) / g0
        r2 = float((d[k + 1:] * d[:-(k + 1)]).sum() / n) / g0
        if r1 + r2 <= 0.0:                       # initial-positive-sequence stop
            break
        tau += 2.0 * (r1 + r2)
        k += 2
    neff = n / tau if tau > 0 else float(n)
    return float(min(max(neff, 1.0), n))


def _drift_stats(panel: np.ndarray):
    """Risk-adjusted drift (scale-invariant) + diagnostics.

    Returns (mu_bps, sharpe, t_eff, t_naive, n_eff) where:
      mu_bps  : raw mean daily log-return, bps/day  (DIAGNOSTIC only)
      sharpe  : mu / sigma of pooled daily log-returns — the SCORED statistic
                (audit F5-OW-1: invariant to rescaling the column; inf/nan for a
                variance-collapsed panel -> WORST subscore via _sub)
      t_eff   : one-sample t of drift vs zero, overlap-corrected via _eff_n
                (audit F5-OW-5; the honest, non-inflated diagnostic)
      t_naive : the old sqrt(N) t (kept for transparency; overstated for real)
      n_eff   : effective #independent windows used for t_eff
    """
    p = np.asarray(panel, dtype=np.float64)
    pm = p.mean(axis=1)                          # (n_paths,) per-window mean
    n = pm.size
    mu = float(p.mean())
    sd_all = float(p.std(ddof=1))
    if np.isfinite(sd_all) and sd_all > 0:
        sharpe = mu / sd_all
    else:                                        # variance-collapsed panel
        sharpe = np.inf if mu != 0 else np.nan
    sd_pm = float(pm.std(ddof=1)) if n > 1 else np.nan
    have_se = bool(sd_pm and np.isfinite(sd_pm) and sd_pm > 0)
    t_naive = mu / (sd_pm / np.sqrt(n)) if have_se else np.nan
    neff = _eff_n(pm)
    t_eff = mu / (sd_pm / np.sqrt(neff)) if (have_se and neff > 0) else np.nan
    return mu * 1e4, float(sharpe), float(t_eff), float(t_naive), float(neff)


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch two-sample t of mean(a) - mean(b)."""
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    denom = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else np.nan


def _martingale_dev_bps(panel: np.ndarray) -> float:
    """(E[exp(r)] - 1) * 1e4 : per-step gross-return deviation from 1, bps/day."""
    return float((np.exp(np.float64(panel)).mean() - 1.0) * 1e4)


def _ar_r2_pooled(panel: np.ndarray, p: int = AR_LAGS) -> float:
    """In-sample R^2 of AR(p) fit by pooled OLS across paths (lags stay within
    each path; intercept included)."""
    n_paths, T = panel.shape
    if T <= p + 1:
        return np.nan
    if not np.isfinite(panel).all():
        return np.nan  # degenerate synth input: nan here -> WORST subscore in _sub,
                       # never a scorer crash (crash = coverage gap, audit F-04)
    # design: for each path, rows t = p..T-1, cols = r_{t-1..t-p}
    y = panel[:, p:].reshape(-1)                                    # (n*(T-p),)
    X = np.stack([panel[:, p - k:T - k].reshape(-1)
                  for k in range(1, p + 1)], axis=1)                # lags 1..p
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - resid @ resid / tss) if tss > 0 else np.nan


def _lag1_autocorr_pooled(panel: np.ndarray) -> float:
    """Lag-1 autocorrelation, per-path demeaned, pooled across paths."""
    d = panel - panel.mean(axis=1, keepdims=True)
    num = float((d[:, 1:] * d[:, :-1]).sum())
    den = float((d ** 2).sum())
    return num / den if den > 0 else np.nan


def _variance_ratio(panel: np.ndarray, q: int) -> float:
    """Lo-MacKinlay (1988) overlapping, bias-corrected VR(q), computed per path
    then averaged across paths (paths are independent windows)."""
    n_paths, T = panel.shape
    if T <= q:
        return np.nan
    mu = panel.mean(axis=1, keepdims=True)                          # (n,1)
    sig1 = ((panel - mu) ** 2).sum(axis=1) / (T - 1)                # (n,)
    # overlapping q-period sums: rows t = q..T
    csum = np.cumsum(panel, axis=1)
    qsum = np.concatenate([csum[:, q - 1:q], csum[:, q:] - csum[:, :-q]], axis=1)
    m = q * (T - q + 1) * (1.0 - q / T)                             # LM bias corr.
    sigq = ((qsum - q * mu) ** 2).sum(axis=1) / m                   # (n,)
    ok = sig1 > 0
    if not ok.any():
        return np.nan
    return float(np.mean(sigq[ok] / sig1[ok]))


def _vr_scale(q: int, T: int) -> float:
    """Asymptotic std of VR(q) under iid for one window of length T (LM 1988)."""
    return float(np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * T)))


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

def _sub(syn_stat: float, real_stat: float, scale: float) -> float:
    """Band subscore with asymmetric non-finite semantics (audit F5-1).

    real_stat non-finite -> nan  (component not assessable on this panel;
                                  excluded symmetrically for every competitor)
    syn_stat  non-finite -> 0.0  (degenerate SYNTH panel: worst possible,
                                  never waived from the mean)
    both finite          -> exp(-|syn - real| / scale)
    """
    if not np.isfinite(real_stat):
        return np.nan
    if not np.isfinite(syn_stat):
        return 0.0
    return float(np.exp(-abs(syn_stat - real_stat) / scale))


def _nm(vals) -> float:
    """Quiet nanmean: mean of the finite entries, nan if none (no warnings)."""
    a = np.asarray(list(vals), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _score_pair(synth: np.ndarray, real: np.ndarray, feature_names):
    """Score one (synth, real) pair. Returns (pair_score, per_feature_detail)."""
    T = synth.shape[1]
    per_feat = {}
    feat_scores = []
    for fname in PRICE_FEATURES:
        if fname not in feature_names:
            continue
        j = feature_names.index(fname)
        s = np.float64(synth[:, :, j])
        r = np.float64(real[:, :, j])

        # DRIFT / MARTINGALE — computed for DIAGNOSTICS ONLY (never scored; the
        # block was retired — every pure-drift statistic is variance-gameable,
        # and real drift is below its noise floor. See module docstring).
        mu_s_bps, sh_s, teff_s, tnv_s, neff_s = _drift_stats(s)
        mu_r_bps, sh_r, teff_r, tnv_r, neff_r = _drift_stats(r)
        pm_s, pm_r = s.mean(axis=1), r.mean(axis=1)
        t_sr = _welch_t(pm_s, pm_r)  # diagnostic only (real windows overlap)
        md_s = _martingale_dev_bps(s)  # diagnostic only
        md_r = _martingale_dev_bps(r)  # diagnostic only

        # (1) spurious predictability                     [SCORED]
        r2_s, r2_r = _ar_r2_pooled(s), _ar_r2_pooled(r)
        ac_s, ac_r = _lag1_autocorr_pooled(s), _lag1_autocorr_pooled(r)

        # (2) variance ratios                             [SCORED]
        vr_s = {q: _variance_ratio(s, q) for q in VR_QS}
        vr_r = {q: _variance_ratio(r, q) for q in VR_QS}

        # Subscores (FOUR, all AFFINE-INVARIANT so no rescale/spread can move
        # them — F5 is provably immune to variance manipulation in either
        # direction). Synth-side non-finite stats score 0.0 (WORST, never
        # waived); only real-side non-finite stats are excluded — and then
        # symmetrically, since the real reference is shared (audit F5-1).
        subs = {
            "ar5_r2":  _sub(r2_s, r2_r, S_R2),
            "lag1_ac": _sub(ac_s, ac_r, S_AC1),
        }
        for q in VR_QS:
            subs[f"vr{q}"] = _sub(vr_s[q], vr_r[q], _vr_scale(q, T))

        syn_stats = {"ar5_r2": r2_s, "lag1_ac": ac_s,
                     **{f"vr{q}": vr_s[q] for q in VR_QS}}
        real_stats = {"ar5_r2": r2_r, "lag1_ac": ac_r,
                      **{f"vr{q}": vr_r[q] for q in VR_QS}}
        n_degenerate = sum(1 for k in subs
                           if np.isfinite(real_stats[k]) and not np.isfinite(syn_stats[k]))

        vals = [v for v in subs.values() if np.isfinite(v)]  # nan = real-side only
        fscore = float(np.mean(vals)) if vals else np.nan
        feat_scores.append(fscore)

        per_feat[fname] = {
            "n_synth_degenerate_subscores": n_degenerate,
            "sharpe_synth": sh_s, "sharpe_real": sh_r,
            "drift_bps_synth": mu_s_bps, "drift_bps_real": mu_r_bps,   # diagnostic
            # overlap-corrected drift-vs-zero t (audit F5-OW-5); naive twins kept
            "t_drift_zero_synth": teff_s, "t_drift_zero_real": teff_r,
            "t_drift_zero_synth_naive": tnv_s, "t_drift_zero_real_naive": tnv_r,
            "n_eff_synth": neff_s, "n_eff_real": neff_r,
            "t_drift_synth_vs_real": t_sr,
            "mart_dev_bps_synth": md_s, "mart_dev_bps_real": md_r,     # diagnostic
            "ar5_r2_synth": r2_s, "ar5_r2_real": r2_r,
            "lag1_ac_synth": ac_s, "lag1_ac_real": ac_r,
            **{f"vr{q}_synth": vr_s[q] for q in VR_QS},
            **{f"vr{q}_real": vr_r[q] for q in VR_QS},
            "subscores": subs,
            "score": fscore,
        }
    pair_score = _nm(feat_scores) if feat_scores else np.nan
    return pair_score, per_feat


def score(loaded, feature_names=None) -> dict:
    """Score ONE competitor from its list of (synth, real) numpy pairs.

    Returns {"mean", "std", "n", "detail"}; mean in [0, 1], 1 = synth matches
    real-data martingale behavior (see module docstring for the band mapping).
    """
    feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)
    pair_scores, pair_details = [], []
    for synth, real in loaded:
        ps, pf = _score_pair(np.asarray(synth), np.asarray(real), feature_names)
        if np.isfinite(ps):
            pair_scores.append(ps)
            pair_details.append(pf)
    if not pair_scores:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "detail": {}}

    # aggregate per-feature detail across pairs (mean over seeds)
    feats = {}
    flagged = []
    degenerate = []
    for fname in PRICE_FEATURES:
        rows = [d[fname] for d in pair_details if fname in d]
        if not rows:
            continue
        agg = {}
        for k in rows[0]:
            if k == "subscores":
                agg[k] = {sk: _nm([r[k][sk] for r in rows]) for sk in rows[0][k]}
            else:
                agg[k] = _nm([r[k] for r in rows])
        feats[fname] = agg
        if agg.get("n_synth_degenerate_subscores", 0) > 0:
            degenerate.append(fname)
        d_r2 = agg["ar5_r2_synth"] - agg["ar5_r2_real"]
        d_ac = abs(agg["lag1_ac_synth"]) - abs(agg["lag1_ac_real"])
        if d_r2 > FLAG_DR2 or d_ac > FLAG_DAC1:
            flagged.append(fname)

    detail = {
        "features": feats,
        "more_predictable_than_real": bool(flagged),
        "flagged_features": flagged,
        "synth_degenerate_features": degenerate,  # scored WORST (0.0), not waived
        "price_features": [f for f in PRICE_FEATURES if f in feature_names],
        "band_scales": {"ar5_r2": S_R2, "lag1_ac": S_AC1,
                        **{f"vr{q}": _vr_scale(q, int(loaded[0][0].shape[1]))
                           for q in VR_QS}},
        "scored_subscores": ["ar5_r2", "lag1_ac", *(f"vr{q}" for q in VR_QS)],
        "drift_block_retired": True,
        "note": ("band metric: 1 = matches real martingale STRUCTURE. FOUR "
                 "scored subscores (ar5_r2, lag1_ac, vr5, vr20), ALL "
                 "affine-invariant -> F5 is provably immune to variance "
                 "manipulation in both directions (rescale AND mean-preserving "
                 "spread leave every subscore unchanged). The DRIFT/MARTINGALE "
                 "block is RETIRED (not scored): every pure-drift statistic is "
                 "variance-gameable in one direction or the other (audit "
                 "F5-OW-1 and its inflation analog T6-GAME-1) and real drift is "
                 "below its noise floor (F5-OW-2). sharpe_*, drift_bps_*, "
                 "mart_dev_bps_*, t_drift_zero_* (overlap-corrected via n_eff, "
                 "F5-OW-5, with _naive twins) remain in detail as DIAGNOSTICS "
                 "ONLY; t_drift_synth_vs_real is diagnostic only too"),
    }
    return {
        "mean": float(np.mean(pair_scores)),
        "std": float(np.std(pair_scores)),
        "n": len(pair_scores),
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# ranked table over all available competitors
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from benchmark.registry import available_competitors

    rows = []
    for comp in available_competitors():
        try:
            res = score(comp.load())
        except Exception as e:  # coverage gap, never silently dropped
            print(f"  [F5] {comp.name}: N/A ({type(e).__name__}: {e})")
            continue
        rows.append((comp.name, res))

    rows.sort(key=lambda r: -(r[1]["mean"] if np.isfinite(r[1]["mean"]) else -1))

    print("\nF5 — Martingale STRUCTURE / no-manufactured-predictability (band; "
          "1 = matches real). SCORE = 4 affine-invariant subscores "
          "(ar5_r2,lag1_ac,vr5,vr20); shrp|Δ| & mart|Δ| are DIAGNOSTIC (retired, "
          "not scored).")
    print(f"{'#':>2}  {'competitor':<18} {'score':>6} {'std':>6} {'n':>2}  "
          f"{'shrp|Δ|d':>8} {'mart|Δ|d':>9} {'AR5 R2 s/r':>12} "
          f"{'VR5 s/r':>12} {'flag':>18}")
    for i, (name, res) in enumerate(rows, 1):
        f = res["detail"].get("features", {})
        if f:
            ds = np.mean([abs(v["sharpe_synth"] - v["sharpe_real"])
                          for v in f.values() if np.isfinite(v["sharpe_synth"])
                          and np.isfinite(v["sharpe_real"])])
            dm = np.mean([abs(v["mart_dev_bps_synth"] - v["mart_dev_bps_real"])
                          for v in f.values()])
            r2s = np.mean([v["ar5_r2_synth"] for v in f.values()])
            r2r = np.mean([v["ar5_r2_real"] for v in f.values()])
            v5s = np.mean([v["vr5_synth"] for v in f.values()])
            v5r = np.mean([v["vr5_real"] for v in f.values()])
            flag = ("MORE-PREDICTABLE" if res["detail"]["more_predictable_than_real"]
                    else "-")
            print(f"{i:>2}  {name:<18} {res['mean']:6.3f} {res['std']:6.3f} "
                  f"{res['n']:>2}  {ds:8.4f} {dm:10.1f} "
                  f"{r2s:5.3f}/{r2r:5.3f} {v5s:5.2f}/{v5r:5.2f} {flag:>18}")
        else:
            print(f"{i:>2}  {name:<18} {'nan':>6}")
    flagged = [n for n, r in rows if r["detail"].get("more_predictable_than_real")]
    print(f"\nFlagged MORE predictable than real (manufactured alpha): "
          f"{', '.join(flagged) if flagged else 'none'}")
