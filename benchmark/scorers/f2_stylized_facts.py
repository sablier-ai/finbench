"""F2 — Cont (2001) 11-stylized-facts battery, as synth-vs-real distances.

Reference: R. Cont, "Empirical properties of asset returns: stylized facts and
statistical issues", *Quantitative Finance* 1(2), 2001, pp. 223-236
(DOI 10.1088/1469-7688/1/2/304). Fact numbering below follows Cont's list.
Secondary references for individual estimators:
  - Hill (1975), Ann. Statist. 3(5) — tail-index estimator (fact 2, diagnostic).
  - Wiese et al., "Quant GANs" (Quant. Finance 2020) — ACF / leverage
    score style (facts 1, 6, 8, 9 are profile-distance scores in that spirit).
  - Zumbach & Lynch (2001) / Mueller et al. (1997) — coarse-fine volatility
    correlation asymmetry (fact 11).
  - RiskMetrics (1996) EWMA (lambda=0.94) — volatility filter for fact 7.

Every fact is scored as a DISTANCE between the synthetic panel and the real
OOS reference panel: we do not test whether the synthetic data exhibits the
fact in the abstract; we test whether it reproduces the *real panel's*
signature of the fact (including the magnitude of the fact STATISTIC — e.g.
the kurtosis level, the ACF decay — NOT the magnitude of the return scale
itself). Lower = better.

SCALE-BLINDNESS (a design property, stated up front so no reader over-reads a
rank: F2 is scale-blind BY CONSTRUCTION. Cont's stylized facts are scale-free
properties (autocorrelation shapes, kurtosis, clustering, asymmetry indices),
and several fact statistics here are additionally relative or bounded
(e.g. facts 2/4/7 divide the kurtosis error by (1+|k_r|); fact 3 is a bounded
index). A generator that reproduces every fact but multiplies a return column
by a constant (a 2x volatility-magnitude error) is NOT penalized by F2 —
several facts do not even see it. Volatility/return MAGNITUDE fidelity is
owned by other tasks (F4/T5), not F2. Do not read an F2 rank as evidence of a
realistic return scale.

The 11 facts and how each is computed
-------------------------------------
 1. Absence of linear autocorrelation      -> REUSED finval.metrics.compute_acf_returns
                                              (mean |ACF_synth - ACF_real| at lags 1,5,10,20).
 2. Heavy tails                            -> implemented here: per-feature RELATIVE excess-
                                              kurtosis error on pooled returns,
                                              mean_f |k_s(f) - k_r(f)| / (1 + |k_r(f)|)
                                              (the SAME relative form as facts 4 and 7).
                                              The raw finval |delta-kurtosis| per feature and a
                                              pooled two-sided Hill tail-index distance are
                                              reported in `detail` as supplementary diagnostics
                                              (not part of the fact score). NB:
                                              a RAW mean of |delta-kurtosis| across features
                                              whose real excess kurtosis spans ~1.6 (DXY) to
                                              ~28 (VIX level-diffs) is a units/scale-mixing
                                              average dominated by the single highest-kurtosis
                                              feature (VIX carried ~48-94% of the fact); the
                                              per-feature (1+|k_r|) normalization removes that
                                              domination.
 3. Gain/loss asymmetry                    -> implemented here: bounded tail-asymmetry index
                                              A = (L - G) / (L + G), where L = E[-r | r <= q05]
                                              and G = E[r | r >= q95] on pooled returns;
                                              distance = |A_s - A_r|. Skewness distance is
                                              reported as supplementary detail.
 4. Aggregational Gaussianity              -> implemented here: pooled excess kurtosis of
                                              non-overlapping aggregated returns at
                                              tau = 1, 5, 20 days; distance =
                                              mean_tau |k_s(tau) - k_r(tau)| / (1 + |k_r(tau)|).
 5. Intermittency                          -> implemented here: coefficient of variation of
                                              5-day block realized vol within each path
                                              (burst-iness of volatility), averaged over
                                              paths; distance = |CV_s - CV_r|.
 6. Volatility clustering                  -> REUSED finval.metrics.compute_volatility_clustering
                                              (ACF of r^2 at lags 1..5, synth-vs-real error).
 7. Conditional heavy tails                -> implemented here: returns standardized by a
                                              per-path RiskMetrics EWMA vol (lambda=0.94,
                                              10-step burn-in); pooled excess kurtosis of
                                              the standardized residuals;
                                              distance = |k_s - k_r| / (1 + |k_r|).
                                              DEVIATION (documented): a per-path GARCH(1,1)
                                              MLE is not statistically meaningful at 60
                                              observations per path, and pooling paths
                                              through one GARCH fit is invalid across path
                                              boundaries, so the EWMA filter (the option the
                                              task spec explicitly allows) is used instead.
 8. Slow decay of |r| autocorrelation      -> implemented here: per-path ACF of |r| at lags
                                              (1,2,3,5,7,10,15,20) averaged over paths;
                                              distance = mean |ACF_s - ACF_r| over the lag
                                              profile. DEVIATION (documented): a power-law /
                                              Hurst exponent fit is unreliable at 60-step
                                              horizons, so the decay PROFILE is compared
                                              directly instead of its fitted exponent.
 9. Leverage effect                        -> REUSED finval.metrics.compute_leverage_effect
                                              (corr(r_t, |r_{t+k}|) profile error, lags
                                              1,2,5,10), on price-type features only.
10. Volume/volatility correlation          -> **N/A on this panel** (us_equities_macro has
                                              no volume series). Reported explicitly as
                                              assessable=False and EXCLUDED from the mean;
                                              it is never faked.
11. Coarse-fine timescale asymmetry        -> implemented here (Zumbach effect):
    (Zumbach)                                 v_coarse(t) = (sum of r over the trailing 5d
                                              window)^2, v_fine(t) = r_t^2;
                                              A(l) = corr(v_c(t), v_f(t+l)) -
                                                     corr(v_f(t), v_c(t+l)), l = 1..5,
                                              per path, averaged over paths;
                                              distance = mean_l |A_s(l) - A_r(l)|.

Feature scope (documented per the task spec)
--------------------------------------------
The v1 panel mixes price-type LOG-RETURN features (IWM, QQQ, SPY, TLT, DXY)
with level-DIFF features (VIX, TNX). Facts whose *definition* is specific to
asset returns — #3 (gain/loss asymmetry of prices), #4 (aggregational
Gaussianity of returns under the CLT narrative), #9 (equity leverage effect)
— are computed on price-type features ONLY. All other assessable facts
(1, 2, 5, 6, 7, 8, 11) are well-defined synth-vs-real distances for any
stationary daily-increment series and are computed on ALL 7 features.
Each fact is computed per-feature and then averaged across its feature scope.

Normalization (documented)
--------------------------
Per-fact raw distances live on heterogeneous scales (ACF units vs kurtosis
units), so each is mapped to [0, 1) with the soft saturating transform
    norm(d) = d / (d + s_f)
where s_f is a per-fact scale constant (the distance at which the normalized
score hits 0.5; constants in ``FACT_SCALES``, chosen at the order of magnitude
of a "clearly wrong" mismatch for that statistic, e.g. 0.05 ACF units,
0.5 relative-kurtosis-error units for facts 2/4/7). The overall score is the unweighted mean of the
normalized distances of the 10 ASSESSABLE facts. Direction: LOWER = BETTER
(0 = perfect reproduction of every fact), matching the "down" direction for
F2 in BENCHMARK_TASKS.md.

Non-finite-fact policy — two NaN cases, handled DIFFERENTLY:
  * REAL-side / panel-level non-computability (fact 10's missing volume
    series; a real-side statistic that cannot be estimated on this panel)
    -> the fact is excluded from the mean SYMMETRICALLY: the real reference
    is shared, so the same exclusion applies to every competitor.
  * SYNTH-induced non-finiteness (degenerate submission tensors: constant /
    zero-variance panels make kurtosis, tail indices, ACFs undefined)
    -> the fact scores the WORST-POSSIBLE normalized distance 1.0. It is
    NEVER dropped: waiving these facts used to let an all-zeros tensor keep
    only its 4 easiest facts and rank 15/19.
  A per-feature variance floor backstops the split: a synth feature whose
  variance is < 1e-6 x the real feature's variance is treated as degenerate
  for every fact in whose scope it falls (this also catches constant panels
  for which an estimator returns a misleading FINITE value, e.g. ACF = 0 on
  a constant series being scored as "reproduces absence of autocorrelation").
  Degenerate facts are flagged ``synth_degenerate`` in the detail.

Interface: ``score(loaded, feature_names=None) -> {"mean","std","n","detail"}``
where ``loaded`` is the registry's list of (synth, real) pairs for ONE
competitor; mean/std are over pairs (seeds).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from finval.metrics import (
    compute_acf_returns,
    compute_leverage_effect,
    compute_volatility_clustering,
)

# ---------------------------------------------------------------------------
# Panel semantics (v1 us_equities_macro)
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = ["IWM", "QQQ", "SPY", "TLT", "VIX", "TNX", "DXY"]
LEVEL_DIFF_FEATURES = {"VIX", "TNX"}  # simple diffs of levels, not log-returns

FACT_NAMES = {
    1: "linear_autocorrelation",
    2: "heavy_tails",
    3: "gain_loss_asymmetry",
    4: "aggregational_gaussianity",
    5: "intermittency",
    6: "volatility_clustering",
    7: "conditional_heavy_tails",
    8: "slow_decay_abs_acf",
    9: "leverage_effect",
    10: "volume_volatility_correlation",
    11: "coarse_fine_asymmetry",
}

# Per-fact soft-normalization scales: norm = d / (d + s). s is the raw distance
# at which the normalized score reaches 0.5 (see module docstring).
FACT_SCALES = {
    1: 0.05,   # ACF units
    2: 0.50,   # relative kurtosis error (dimensionless; matches facts 4 & 7)
    3: 0.20,   # bounded asymmetry-index units (index lives in [-1, 1])
    4: 0.50,   # relative kurtosis error (dimensionless)
    5: 0.10,   # CV units
    6: 0.05,   # ACF units
    7: 0.50,   # relative kurtosis error (dimensionless)
    8: 0.05,   # ACF units
    9: 0.05,   # correlation units
    11: 0.05,  # correlation-difference units
}

ABS_ACF_LAGS = (1, 2, 3, 5, 7, 10, 15, 20)
AGG_SCALES = (1, 5, 20)
ZUMBACH_LAGS = (1, 2, 3, 4, 5)
INTERMITTENCY_BLOCK = 5
EWMA_LAMBDA = 0.94
EWMA_BURNIN = 10
TAIL_Q = 0.05
HILL_FRAC = 0.05

# Variance floor: a synth feature with variance below this
# fraction of the real feature's variance is a degenerate submission slab —
# every fact in whose scope it falls scores WORST (normalized 1.0).
SYNTH_VAR_FLOOR_REL = 1e-6


# ---------------------------------------------------------------------------
# Low-level helpers (all nan-aware; paths axis = 0, time axis = 1)
# ---------------------------------------------------------------------------
def _pooled(x2d: np.ndarray) -> np.ndarray:
    """Flatten a (paths, T) slab to a clean 1-D pooled sample."""
    v = np.asarray(x2d, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


def _excess_kurtosis(v: np.ndarray) -> float:
    if v.size < 30:
        return np.nan
    return float(stats.kurtosis(v, fisher=True, bias=True))


def _hill_index(v: np.ndarray, frac: float = HILL_FRAC) -> float:
    """Two-sided pooled Hill tail-index estimate (mean of left/right tails).

    Hill (1975): xi_hat = (1/k) * sum_{i=1..k} log(X_(i) / X_(k+1)) on the k
    largest order statistics. Larger xi = heavier tail.
    """
    out = []
    for tail in (v[v > 0], -v[v < 0]):
        t = np.sort(tail)[::-1]
        k = int(np.floor(frac * v.size))
        if k < 10 or t.size <= k:
            continue
        top, thresh = t[:k], t[k]
        if thresh <= 0:
            continue
        out.append(float(np.mean(np.log(top / thresh))))
    return float(np.mean(out)) if out else np.nan


def _gain_loss_asym(v: np.ndarray, q: float = TAIL_Q) -> float:
    """Bounded tail-asymmetry index A = (L - G)/(L + G) in [-1, 1].

    L = mean loss magnitude beyond the q-quantile, G = mean gain beyond the
    (1-q)-quantile. A > 0 means large drawdowns exceed large gains (Cont #3).
    """
    if v.size < 100:
        return np.nan
    lo, hi = np.quantile(v, [q, 1.0 - q])
    L = float(np.mean(-v[v <= lo])) if np.any(v <= lo) else np.nan
    G = float(np.mean(v[v >= hi])) if np.any(v >= hi) else np.nan
    if not (np.isfinite(L) and np.isfinite(G)) or (L + G) <= 0:
        return np.nan
    return (L - G) / (L + G)


def _agg_kurtosis_profile(x2d: np.ndarray, scales=AGG_SCALES) -> dict[int, float]:
    """Pooled excess kurtosis of non-overlapping tau-day aggregated returns."""
    p, t = x2d.shape
    prof = {}
    for tau in scales:
        nb = t // tau
        if nb < 2 and tau > 1:
            prof[tau] = np.nan
            continue
        agg = x2d[:, : nb * tau].reshape(p, nb, tau).sum(axis=2)
        prof[tau] = _excess_kurtosis(_pooled(agg))
    return prof


def _block_vol_cv(x2d: np.ndarray, block: int = INTERMITTENCY_BLOCK) -> float:
    """Mean over paths of CV(realized vol of non-overlapping `block`-day windows)."""
    p, t = x2d.shape
    nb = t // block
    if nb < 4:
        return np.nan
    r2 = np.square(x2d[:, : nb * block]).reshape(p, nb, block)
    v = np.sqrt(np.nanmean(r2, axis=2))          # (paths, nb) block vols
    mu = np.nanmean(v, axis=1)
    sd = np.nanstd(v, axis=1)
    cv = np.where(mu > 0, sd / mu, np.nan)
    return float(np.nanmean(cv))


def _ewma_std_residual_kurtosis(x2d: np.ndarray, lam: float = EWMA_LAMBDA,
                                burnin: int = EWMA_BURNIN) -> float:
    """Pooled excess kurtosis of EWMA-vol-standardized residuals (fact 7)."""
    x = np.asarray(x2d, dtype=np.float64)
    p, t = x.shape
    if t <= burnin + 20:
        return np.nan
    x0 = np.nan_to_num(x, nan=0.0)
    var0 = np.nanvar(x[:, :burnin], axis=1)
    var0 = np.where(np.isfinite(var0) & (var0 > 0), var0, np.nanvar(x0) + 1e-12)
    sig2 = np.empty((p, t))
    sig2[:, 0] = var0
    for i in range(1, t):
        sig2[:, i] = lam * sig2[:, i - 1] + (1.0 - lam) * np.square(x0[:, i - 1])
    z = x[:, burnin:] / np.sqrt(sig2[:, burnin:] + 1e-18)
    return _excess_kurtosis(_pooled(z))


def _mean_path_acf(x2d: np.ndarray, lags) -> np.ndarray:
    """ACF of each row at the given lags, averaged over rows (nan-aware)."""
    x = np.asarray(x2d, dtype=np.float64)
    xc = x - np.nanmean(x, axis=1, keepdims=True)
    var = np.nanmean(np.square(xc), axis=1)  # (paths,)
    out = np.full(len(lags), np.nan)
    for j, lag in enumerate(lags):
        if lag >= x.shape[1] - 2:
            continue
        num = np.nanmean(xc[:, :-lag] * xc[:, lag:], axis=1)
        acf = np.where(var > 1e-18, num / var, np.nan)
        out[j] = np.nanmean(acf)
    return out


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson corr of each row of a with the matching row of b (nan-aware)."""
    ac = a - np.nanmean(a, axis=1, keepdims=True)
    bc = b - np.nanmean(b, axis=1, keepdims=True)
    num = np.nanmean(ac * bc, axis=1)
    den = np.sqrt(np.nanmean(ac * ac, axis=1) * np.nanmean(bc * bc, axis=1))
    return np.where(den > 1e-18, num / den, np.nan)


def _zumbach_profile(x2d: np.ndarray, window: int = 5, lags=ZUMBACH_LAGS) -> np.ndarray:
    """Mean-over-paths Zumbach asymmetry A(l) = C(+l) - C(-l), l in `lags`.

    v_fine(t) = r_t^2; v_coarse(t) = (sum_{i=t-w+1..t} r_i)^2;
    C(+l) = corr(v_c(t), v_f(t+l)); C(-l) = corr(v_f(t), v_c(t+l)).
    Zumbach effect: coarse vol predicts future fine vol better than the
    reverse, so A(l) > 0 for equities at short lags.
    """
    x = np.nan_to_num(np.asarray(x2d, dtype=np.float64), nan=0.0)
    p, t = x.shape
    cs = np.cumsum(np.concatenate([np.zeros((p, 1)), x], axis=1), axis=1)
    coarse = np.square(cs[:, window:] - cs[:, :-window])  # (p, t-w+1) at times w-1..t-1
    fine = np.square(x[:, window - 1:])                   # aligned to the same times
    out = np.full(len(lags), np.nan)
    for j, lag in enumerate(lags):
        if coarse.shape[1] - lag < 20:
            continue
        c_pos = _rowwise_corr(coarse[:, :-lag], fine[:, lag:])
        c_neg = _rowwise_corr(fine[:, :-lag], coarse[:, lag:])
        out[j] = np.nanmean(c_pos) - np.nanmean(c_neg)
    return out


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if d.size else np.nan


# ---------------------------------------------------------------------------
# Non-finite attribution helpers: per-feature / per-fact distances
# use the convention  None = real-side not computable (excluded symmetrically),
# inf = synth-induced degeneracy (scores WORST), finite float = normal.
# ---------------------------------------------------------------------------
def _degenerate_feature_names(synth: np.ndarray, real: np.ndarray,
                              names: list[str]) -> set[str]:
    """Synth features that are (near-)constant while the real feature is not."""
    deg = set()
    for i, nm in enumerate(names):
        rv = float(np.nanvar(real[:, :, i]))
        sv = float(np.nanvar(synth[:, :, i]))
        if np.isfinite(rv) and rv > 0 and (
                not np.isfinite(sv) or sv < SYNTH_VAR_FLOOR_REL * rv):
            deg.add(nm)
    return deg


def _pf_classify(s_stat: float, r_stat: float):
    """Per-feature distance with side attribution."""
    if not np.isfinite(r_stat):
        return None          # real-side: not computable on this panel
    if not np.isfinite(s_stat):
        return np.inf        # synth-induced: worst possible
    return float(abs(s_stat - r_stat))


def _profile_distance(a_s: np.ndarray, a_r: np.ndarray):
    """Profile (per-lag/per-tau) distance with side attribution: the grid is
    the real-finite entries; a synth non-finite entry ON that grid => inf."""
    a_s = np.asarray(a_s, dtype=np.float64)
    a_r = np.asarray(a_r, dtype=np.float64)
    m = np.isfinite(a_r)
    if not m.any():
        return None
    if not np.isfinite(a_s[m]).all():
        return np.inf
    return float(np.mean(np.abs(a_s[m] - a_r[m])))


def _fact_from_pf(pf: dict):
    """Fact distance from per-feature values (None/inf/float). Any
    synth-degenerate feature in scope makes the whole fact worst (inf)."""
    vals = [v for v in pf.values() if v is not None]
    if not vals:
        return None
    if any(np.isinf(v) for v in vals):
        return float("inf")
    return float(np.mean(vals))


def _attributed_finval(value: float, real_real_fn):
    """Attribute a non-finite finval metric value: if the same metric is
    finite on (real, real), the synth caused it (inf = worst); otherwise the
    fact is not computable on this panel (None, excluded symmetrically)."""
    if np.isfinite(value):
        return float(value)
    try:
        rr = float(real_real_fn().value)
    except Exception:
        rr = float("nan")
    return float("inf") if np.isfinite(rr) else None


# ---------------------------------------------------------------------------
# Per-pair battery
# ---------------------------------------------------------------------------
def _battery(synth: np.ndarray, real: np.ndarray, names: list[str]) -> dict:
    """All 11 fact distances for one (synth, real) pair. Returns
    {fact_id: {"distance": float|None, "assessable": bool, "per_feature": {...}, ...}}.
    """
    synth = np.asarray(synth, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    n_feat = synth.shape[2]
    names = list(names)[:n_feat]
    price_idx = [i for i, n in enumerate(names) if n not in LEVEL_DIFF_FEATURES]
    price_names = [names[i] for i in price_idx]
    facts: dict[int, dict] = {}

    # Degenerate synth features (variance floor): every fact whose
    # scope contains one scores WORST — including finval facts, whose
    # estimators can return misleading finite values on constant series
    # (e.g. ACF = 0 read as "reproduces absence of autocorrelation").
    deg = _degenerate_feature_names(synth, real, names)
    deg_all = bool(deg)                              # any degenerate feature
    deg_price = bool(deg & set(price_names))         # degenerate price feature

    def _finval_dist(res, real_real_fn, degenerate_in_scope) -> float | None:
        if degenerate_in_scope:
            return float("inf")
        return _attributed_finval(float(res.value), real_real_fn)

    # -- Fact 1: absence of linear autocorrelation (finval, all features) ----
    r1 = compute_acf_returns(synth, real, feature_names=names)
    facts[1] = {"distance": _finval_dist(r1, lambda: compute_acf_returns(
                    real, real, feature_names=names), deg_all),
                "per_feature": dict(getattr(r1, "per_feature", {}) or {}),
                "source": "finval.compute_acf_returns"}

    # -- Fact 2: heavy tails (RELATIVE pooled-kurtosis error, all features) ---
    # The fact is the mean over features of the RELATIVE excess-
    # kurtosis error  |k_s(f) - k_r(f)| / (1 + |k_r(f)|)  — the SAME form as
    # facts 4 and 7 of this battery. The former implementation averaged finval's
    # RAW |delta kurtosis| across features whose real excess kurtosis spans
    # ~1.6 (DXY) to ~28 (VIX level-diffs), so the single highest-kurtosis
    # feature carried ~48-94% of the fact-2 distance (a units/scale-mixing
    # average, a unit-mixing hazard). Normalizing per feature by
    # (1 + |k_r(f)|) balances the per-feature contribution shares. The raw
    # finval per-feature errors and a two-sided Hill tail-index distance are
    # retained as `detail` diagnostics only (not part of the fact score).
    pf2, kraw2, hill_d = {}, {}, {}
    for i, nm in enumerate(names):
        hs = _hill_index(_pooled(synth[:, :, i]))
        hr = _hill_index(_pooled(real[:, :, i]))
        hill_d[nm] = abs(hs - hr) if np.isfinite(hs) and np.isfinite(hr) else np.nan
        if nm in deg:
            pf2[nm] = np.inf
            kraw2[nm] = np.nan
            continue
        k_s = _excess_kurtosis(_pooled(synth[:, :, i]))
        k_r = _excess_kurtosis(_pooled(real[:, :, i]))
        kraw2[nm] = abs(k_s - k_r) if np.isfinite(k_s) and np.isfinite(k_r) else np.nan
        if not np.isfinite(k_r):
            pf2[nm] = None
        elif not np.isfinite(k_s):
            pf2[nm] = np.inf
        else:
            pf2[nm] = float(abs(k_s - k_r) / (1.0 + abs(k_r)))
    facts[2] = {"distance": _fact_from_pf(pf2), "per_feature": pf2,
                "source": "here (pooled excess-kurtosis relative error, mean over features)",
                "supplementary_raw_kurtosis_distance": {k: (None if not np.isfinite(v) else round(v, 4))
                                                        for k, v in kraw2.items()},
                "supplementary_hill_index_distance": {k: (None if not np.isfinite(v) else round(v, 4))
                                                      for k, v in hill_d.items()}}

    # -- Fact 3: gain/loss asymmetry (here, price-type only) ------------------
    pf3, skew_d = {}, {}
    for i, nm in zip(price_idx, price_names):
        if nm in deg:
            pf3[nm] = np.inf
            skew_d[nm] = np.nan
            continue
        p_s, p_r = _pooled(synth[:, :, i]), _pooled(real[:, :, i])
        pf3[nm] = _pf_classify(_gain_loss_asym(p_s), _gain_loss_asym(p_r))
        skew_d[nm] = (float(abs(stats.skew(p_s) - stats.skew(p_r)))
                      if p_s.size > 30 and p_r.size > 30 else np.nan)
    facts[3] = {"distance": _fact_from_pf(pf3), "per_feature": pf3,
                "source": "here (tail-asymmetry index, q=0.05)",
                "supplementary_skewness_distance": {k: (None if not np.isfinite(v) else round(v, 4))
                                                    for k, v in skew_d.items()}}

    # -- Fact 4: aggregational Gaussianity (here, price-type only) ------------
    pf4, prof4 = {}, {}
    for i, nm in zip(price_idx, price_names):
        if nm in deg:
            pf4[nm] = np.inf
            continue
        ks = _agg_kurtosis_profile(synth[:, :, i])
        kr = _agg_kurtosis_profile(real[:, :, i])
        taus_r = [t for t in AGG_SCALES if np.isfinite(kr[t])]
        if not taus_r:
            pf4[nm] = None
        elif any(not np.isfinite(ks[t]) for t in taus_r):
            pf4[nm] = np.inf
        else:
            pf4[nm] = float(np.mean([abs(ks[t] - kr[t]) / (1.0 + abs(kr[t]))
                                     for t in taus_r]))
        prof4[nm] = {f"tau_{t}": {"synth": None if not np.isfinite(ks[t]) else round(ks[t], 3),
                                  "real": None if not np.isfinite(kr[t]) else round(kr[t], 3)}
                     for t in AGG_SCALES}
    facts[4] = {"distance": _fact_from_pf(pf4), "per_feature": pf4,
                "source": "here (pooled kurtosis at tau=1/5/20d, relative error)",
                "supplementary_kurtosis_profiles": prof4}

    # -- Fact 5: intermittency (here, all features) ----------------------------
    pf5 = {}
    for i, nm in enumerate(names):
        if nm in deg:
            pf5[nm] = np.inf
            continue
        pf5[nm] = _pf_classify(_block_vol_cv(synth[:, :, i]), _block_vol_cv(real[:, :, i]))
    facts[5] = {"distance": _fact_from_pf(pf5), "per_feature": pf5,
                "source": "here (CV of 5d block realized vol per path)"}

    # -- Fact 6: volatility clustering (finval, all features) -----------------
    r6 = compute_volatility_clustering(synth, real, feature_names=names)
    facts[6] = {"distance": _finval_dist(r6, lambda: compute_volatility_clustering(
                    real, real, feature_names=names), deg_all),
                "per_feature": dict(getattr(r6, "per_feature", {}) or {}),
                "source": "finval.compute_volatility_clustering"}

    # -- Fact 7: conditional heavy tails (here, all features) -----------------
    pf7 = {}
    for i, nm in enumerate(names):
        if nm in deg:
            pf7[nm] = np.inf
            continue
        k_s = _ewma_std_residual_kurtosis(synth[:, :, i])
        k_r = _ewma_std_residual_kurtosis(real[:, :, i])
        if not np.isfinite(k_r):
            pf7[nm] = None
        elif not np.isfinite(k_s):
            pf7[nm] = np.inf
        else:
            pf7[nm] = float(abs(k_s - k_r) / (1.0 + abs(k_r)))
    facts[7] = {"distance": _fact_from_pf(pf7), "per_feature": pf7,
                "source": "here (EWMA lambda=0.94 standardized-residual kurtosis)"}

    # -- Fact 8: slow decay of |r| ACF (here, all features) -------------------
    pf8 = {}
    for i, nm in enumerate(names):
        if nm in deg:
            pf8[nm] = np.inf
            continue
        a_s = _mean_path_acf(np.abs(synth[:, :, i]), ABS_ACF_LAGS)
        a_r = _mean_path_acf(np.abs(real[:, :, i]), ABS_ACF_LAGS)
        pf8[nm] = _profile_distance(a_s, a_r)
    facts[8] = {"distance": _fact_from_pf(pf8), "per_feature": pf8,
                "source": f"here (|r| ACF profile at lags {ABS_ACF_LAGS})"}

    # -- Fact 9: leverage effect (finval, price-type only) --------------------
    r9 = compute_leverage_effect(synth[:, :, price_idx], real[:, :, price_idx],
                                 feature_names=price_names)
    facts[9] = {"distance": _finval_dist(r9, lambda: compute_leverage_effect(
                    real[:, :, price_idx], real[:, :, price_idx],
                    feature_names=price_names), deg_price),
                "per_feature": dict(getattr(r9, "per_feature", {}) or {}),
                "source": "finval.compute_leverage_effect"}

    # -- Fact 10: volume/volatility correlation — NOT ASSESSABLE --------------
    facts[10] = {"distance": None,
                 "reason": "us_equities_macro v1 panel carries no volume series; "
                           "fact cannot be computed from static return tensors."}

    # -- Fact 11: coarse-fine timescale asymmetry (here, all features) --------
    pf11 = {}
    for i, nm in enumerate(names):
        if nm in deg:
            pf11[nm] = np.inf
            continue
        z_s = _zumbach_profile(synth[:, :, i])
        z_r = _zumbach_profile(real[:, :, i])
        pf11[nm] = _profile_distance(z_s, z_r)
    facts[11] = {"distance": _fact_from_pf(pf11), "per_feature": pf11,
                 "source": f"here (Zumbach A(l)=C(+l)-C(-l), 5d coarse window, lags {ZUMBACH_LAGS})"}

    # assessable flags + normalization:
    #   distance None -> not assessable (panel/real-side; excluded symmetrically)
    #   distance inf  -> synth-degenerate: assessable, WORST normalized score 1.0
    #   finite        -> normalized = d / (d + scale)
    for fid, f in facts.items():
        d = f.get("distance")
        if fid == 10 or d is None:
            f["assessable"] = False
            f["synth_degenerate"] = False
            f["normalized"] = None
        elif np.isinf(d) or np.isnan(d):  # nan defensively counts as synth-induced
            f["assessable"] = True
            f["synth_degenerate"] = True
            f["normalized"] = 1.0
        else:
            f["assessable"] = True
            f["synth_degenerate"] = False
            f["normalized"] = float(d / (d + FACT_SCALES[fid]))
        f["degenerate_features"] = sorted(deg) if f["synth_degenerate"] else []
        if "per_feature" in f:
            f["per_feature"] = {k: ("synth_degenerate" if v is not None
                                    and not isinstance(v, str) and np.isinf(v)
                                    else None if v is None or not np.isfinite(v)
                                    else round(float(v), 5))
                                for k, v in f["per_feature"].items()}
    return facts


def _pair_overall(facts: dict) -> float:
    vals = [f["normalized"] for f in facts.values() if f["assessable"]]
    return float(np.mean(vals)) if vals else np.nan


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def score(loaded, feature_names=None) -> dict:
    """F2 stylized-facts battery score for one competitor.

    Args:
        loaded: list of (synth, real) pairs, each (n_paths, horizon, n_features).
        feature_names: optional; defaults to the v1 us_equities_macro order.

    Returns:
        {"mean": float, "std": float, "n": int, "detail": dict} where mean is
        the across-pair mean of the normalized-battery score in [0, 1],
        LOWER = BETTER (0 = perfect reproduction of all assessable facts).
    """
    names = list(feature_names) if feature_names else DEFAULT_FEATURES
    per_pair_overall: list[float] = []
    per_pair_facts: list[dict] = []
    for synth, real in loaded:
        facts = _battery(synth, real, names)
        per_pair_facts.append(facts)
        per_pair_overall.append(_pair_overall(facts))

    ov = np.asarray(per_pair_overall, dtype=np.float64)
    ok = np.isfinite(ov)

    # Aggregate per-fact detail across pairs (mean of distances / normalized).
    fact_detail: dict[str, dict] = {}
    for fid in sorted(FACT_NAMES):
        key = f"fact_{fid:02d}_{FACT_NAMES[fid]}"
        if fid == 10:
            fact_detail[key] = {"assessable": False, "distance": None, "normalized": None,
                                "reason": per_pair_facts[0][10]["reason"] if per_pair_facts
                                else "no volume series on this panel"}
            continue
        # finite distances only for the raw-distance summary (a synth-degenerate
        # pair has distance inf); the NORMALIZED mean keeps every assessable
        # pair — degenerate pairs contribute their worst-possible 1.0.
        ds = [pf[fid]["distance"] for pf in per_pair_facts
              if pf[fid]["assessable"] and np.isfinite(pf[fid]["distance"])]
        ns = [pf[fid]["normalized"] for pf in per_pair_facts if pf[fid]["assessable"]]
        n_deg = sum(1 for pf in per_pair_facts if pf[fid].get("synth_degenerate"))
        entry = {
            "assessable": bool(ns),
            "distance": float(np.mean(ds)) if ds else None,
            "distance_std": float(np.std(ds)) if ds else None,
            "normalized": float(np.mean(ns)) if ns else None,
            "synth_degenerate_pairs": n_deg,
            "scale": FACT_SCALES[fid],
            "source": per_pair_facts[0][fid].get("source") if per_pair_facts else None,
            "feature_scope": ("price-type only" if fid in (3, 4, 9) else "all features"),
            "per_feature_mean": {},
        }
        # per-feature mean distance across pairs (finite entries only;
        # "synth_degenerate" markers are strings)
        if per_pair_facts:
            feat_keys = per_pair_facts[0][fid].get("per_feature", {}).keys()
            for fk in feat_keys:
                vals = [pf[fid]["per_feature"].get(fk) for pf in per_pair_facts]
                vals = [v for v in vals
                        if v is not None and not isinstance(v, str) and np.isfinite(v)]
                entry["per_feature_mean"][fk] = round(float(np.mean(vals)), 5) if vals else None
            for supp in ("supplementary_raw_kurtosis_distance",
                         "supplementary_hill_index_distance", "supplementary_skewness_distance",
                         "supplementary_kurtosis_profiles"):
                if supp in per_pair_facts[0][fid]:
                    entry[supp] = per_pair_facts[0][fid][supp]  # first-seed diagnostic
        fact_detail[key] = entry

    n_assessable = sum(1 for k, v in fact_detail.items() if v["assessable"])
    return {
        "mean": float(np.mean(ov[ok])) if ok.any() else float("nan"),
        "std": float(np.std(ov[ok])) if ok.any() else float("nan"),
        "n": int(ok.sum()),
        "detail": {
            "direction": "lower_is_better",
            "n_facts_assessable": n_assessable,
            "n_facts_total": 11,
            "normalization": "per-fact d/(d+scale) in [0,1); overall = mean over assessable facts",
            "feature_names": names,
            "price_type_features": [n for n in names if n not in LEVEL_DIFF_FEATURES],
            "facts": fact_detail,
            "per_pair_overall": [None if not np.isfinite(v) else round(float(v), 5)
                                 for v in per_pair_overall],
        },
    }


# ---------------------------------------------------------------------------
# CLI: ranked table over all available competitors
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from benchmark.registry import available_competitors

    rows = []
    for comp in available_competitors():
        res = score(comp.load())
        rows.append((comp.name, comp.family, res))
        print(f"  scored {comp.name} ({res['n']} seeds)")

    rows.sort(key=lambda r: r[2]["mean"])  # lower = better
    print("\nF2 — Cont (2001) stylized-facts battery (normalized distance, LOWER = better)")
    print("Fact 10 (volume/volatility) N/A on this panel -> mean over the 10 assessable facts.\n")
    hdr = f"{'rank':>4}  {'competitor':<18} {'family':<9} {'mean':>7} {'std':>7} {'n':>3}   worst fact"
    print(hdr)
    print("-" * len(hdr))
    for rank, (name, family, res) in enumerate(rows, 1):
        facts = res["detail"]["facts"]
        worst = max((v["normalized"], k) for k, v in facts.items() if v["assessable"])
        print(f"{rank:>4}  {name:<18} {family:<9} {res['mean']:>7.4f} {res['std']:>7.4f} "
              f"{res['n']:>3}   {worst[1]} ({worst[0]:.3f})")
