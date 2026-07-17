"""T3 — Predictive validity (TSTR): multi-family strategy-rank transfer.

Headline = Spearman rho between the REAL and SYNTH mean-Sharpe vectors over a
frozen, economically-diverse strategy family (HIGHER = BETTER). "Train on synth,
test on real": a good generator should rank a whole book of trading strategies
the same way the real out-of-sample panel does.

Family design (FAMILY_VERSION ``v3-multifamily-2026-07-15``)
------------------------------------------------------------
A TSTR family is only informative if its ranking cannot be reproduced by a
structure-free null. A family of momentum/reversion variants alone reduces, to
first order, to the sign/size of lag-1 autocorrelation — one stylized fact F2
already scores — and a plain Gaussian AR(1) can rank it well. The v3 family is
therefore built so the ranking depends on facts a marginal-only AR(1) null
CANNOT reproduce (and it contains no mirror-pair variants, which would deflate
rho and break the variant bootstrap):

  MARGINAL families (a well-fit AR(1) can rank these — kept for coverage):
    * cs_mom   cross-sectional momentum   (rank 7 features by past return)
    * ts_mom   time-series momentum       (per-asset trend follow)
    * ou_rev   level mean-reversion       (z-score of the reconstructed price
               to its rolling mean — a REAL OU construction, NOT sign-flipped
               momentum, so no mirror duplicates arise)
    * vol_mom  vol-managed momentum       (trend scaled by 1/recent-vol — its
               EDGE over ts_mom exists only under volatility clustering, which
               an AR(1) lacks, so its rank vs ts_mom is a vol-clustering probe)

  JOINT / cross-asset families (an independent-asset null gets ~0 on these —
  the primary collapse-breakers):
    * xrisk    VIX-conditioned risk-on/off (VIX up -> short equity / long bonds)
    * xmacro   rates-conditioned macro     (TNX up -> short bonds / long USD)
    * ftq      flight-to-quality           (equity drawdown -> long bonds/short eq)
    * leadlag  6 curated cross-asset lead-lag pairs x 3 lookbacks
               (VIX->SPY, TNX->TLT, DXY->SPY, ...)

39 variants total, 27 joint / 12 marginal.

A transaction-cost turnover penalty (``COST``, in z-units) is applied to every
variant. Besides being economically real, it breaks the sign-symmetry the old
family had: a long/short leg and its exact mirror have identical turnover, so a
cost term makes ``net(mom) != -net(rev)`` — the cost dimension that most
re-orders real strategy ranking.

The result: the AR(1) null ranks mid-pack, below every strong real generator
and below the shuffled-time control (see ``__main__``) — evidence the family
measures joint structure, not one autocorrelation number.

Per-feature standardization (KEPT — do not regress)
---------------------------------------------------
Every feature's native increment is divided by its FROZEN train-period scale
(``TRAIN_SCALES``) before any strategy logic runs, so VIX diffs (train std ~1.62
points) no longer dwarf equity log-returns (std ~0.01) and each feature
contributes comparably. This is retained verbatim. Because the
panel mixes log-returns and level-diffs of wildly different native scale, the
turnover cost is likewise applied in z-units (a single native-bps cost across
incommensurate features would re-introduce the very unit-mixing this fix removed).

The Gaussian-AR(1) null (the permanent collapse detector)
---------------------------------------------------------
``gaussian_ar1_null_rho()`` builds the null INSIDE this scorer — per-feature
Gaussian AR(1) with phi and sigma fitted to the TRAIN period (``TRAIN_PHI`` /
``TRAIN_SCALES``, frozen constants; a real submission can only see train), assets
independent, homoskedastic, thin-tailed — and scores it against the pinned
canonical real. Its rho is exposed in every ``score()`` detail as
``null_ar1_rho`` and printed as a permanent board row by ``__main__``. If the
null ever climbs back near the top, the family has re-collapsed.

Ceilings and floors
-------------------
The real target is 200 rolling 60-day windows spanning many 2020-2023 regimes.
A model conditioned on a single anchor date can only reproduce one regime's
strategy ranking, capping its attainable rho below the multi-regime perfect
ceiling. ``diagnostics()`` reports both a multi-regime perfect ceiling
(resample all real windows, rho ~ 0.95) and a single-anchor ceiling estimated
from a perfect generator locked to one 60-day regime, so a reader can tell
whether a low rho is a task property or a model defect.

Estimator / contract
--------------------
``score()["mean"]`` = per-seed-mean rho (rho computed per (synth, real) seed pair
and averaged — the repo-wide mean+-std-across-seeds protocol). ``std`` = std of
per-seed rhos, ``n`` = number of seed pairs. ``detail`` additionally reports the
pooled-over-seeds rho, a bootstrap 95% CI that resamples BOTH the synth AND the
real (target) PATHS — so target-side sampling noise is folded in, not only
synth-seed noise — the per-family rhos, the AR(1)-null rho, and the
per-feature |PnL| shares.

Run the ranked board (with the null, shuffled-time control, floors/ceilings and
flip-probability tie clustering) from the finbench repo root:
    python -m benchmark.scorers.t3_tstr
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

N_FEATURES = 7
FAMILY_VERSION = "v3-multifamily-2026-07-15"

# Canonical feature order of the us_equities_macro_2010_2023 tensors.
FEATURE_ORDER = ("IWM", "QQQ", "SPY", "TLT", "VIX", "TNX", "DXY")

# FROZEN per-feature scales: std (ddof=1) of each feature's NATIVE increment
# (log-returns for IWM/QQQ/SPY/TLT/DXY, first differences for VIX/TNX) over the
# TRAIN period (<= 2019-12-31) of sablier_flow.demo_data. Verified 2026-07-15 to
# reproduce exactly from demo_data. Changing these changes the family — bump
# FAMILY_VERSION instead of editing silently.
TRAIN_SCALES = {
    "IWM": 0.01212684251,
    "QQQ": 0.01088146027,
    "SPY": 0.00926929526,
    "TLT": 0.008659169078,
    "VIX": 1.620361686,
    "TNX": 0.04838038981,
    "DXY": 0.004471543245,
}

# FROZEN per-feature TRAIN-period lag-1 autocorrelation (phi) — the ONLY fact the
# Gaussian-AR(1) null is allowed to know (fit to train, no OOS look-ahead), so
# the null is a fair, reproducible, permanent collapse detector. Computed once on
# 2026-07-15 from demo_data train increments. Note these are much WEAKER than the
# OOS-target lag-1 (e.g. SPY -0.041 train vs -0.179 OOS): the 2020-2023 regime
# reversed harder than train, so even the null's marginal edge is honest, not a
# peek at the answer.
TRAIN_PHI = {
    "IWM": -0.035299,
    "QQQ": -0.023445,
    "SPY": -0.040814,
    "TLT": -0.063073,
    "VIX": -0.099175,
    "TNX": -0.039817,
    "DXY": 0.004796,
}

_SCALES = np.array([TRAIN_SCALES[f] for f in FEATURE_ORDER], dtype=np.float32)

# Turnover cost in z-units (see docstring: native-bps across incommensurate
# features would re-break the unit-mixing fix). Breaks the long/short mirror
# sign-symmetry and penalizes high-turnover short-lookback legs.
COST = 0.02

# Feature indices (canonical order).
_EQ = (0, 1, 2)        # IWM, QQQ, SPY
_TLT, _VIX, _TNX, _DXY = 3, 4, 5, 6

# Curated cross-asset lead-lag pairs (leader_idx, follower_idx, sign, name) with
# fixed macro priors; an independent-asset null scores ~0 on all of them.
LEADLAG_PAIRS = [
    (_VIX, 2, -1, "vix_spy"),   # vol up -> equity down
    (_VIX, _TLT, +1, "vix_tlt"),  # vol up -> bonds up (flight to quality)
    (2, 0, +1, "spy_iwm"),      # large-cap leads small-cap
    (_TNX, _TLT, -1, "tnx_tlt"),  # yields up -> bond price down
    (_TNX, _DXY, +1, "tnx_dxy"),  # yields up -> dollar up
    (_DXY, 2, -1, "dxy_spy"),   # dollar up -> equity down
]

# Family group -> {"marginal","joint"} lens (for the board legend / detail).
FAMILY_KIND = {
    "cs_mom": "marginal", "ts_mom": "marginal", "ou_rev": "marginal",
    "vol_mom": "marginal", "xrisk": "joint", "xmacro": "joint",
    "ftq": "joint", "leadlag": "joint",
}


# ---------------------------------------------------------------------------
# Standardization (KEPT — see docstring)
# ---------------------------------------------------------------------------

def _standardize(R: np.ndarray, feature_names=None) -> np.ndarray:
    """Divide each feature's native increments by its frozen train-period scale."""
    R = np.asarray(R, dtype=np.float32)
    if feature_names is None:
        scales = _SCALES
    else:
        try:
            scales = np.array([TRAIN_SCALES[f] for f in feature_names], dtype=np.float32)
        except KeyError as e:
            raise ValueError(f"unknown feature {e} — scales exist only for {FEATURE_ORDER}")
    if R.shape[-1] != len(scales):
        raise ValueError(f"expected {len(scales)} features, got shape {R.shape}")
    return R / scales


# ---------------------------------------------------------------------------
# Strategy weight generators. Each returns W of shape (n_paths, T, F): the
# weight DECIDED at time t and held over R[t+1]; NaN where no decision (warm-up).
# R is already standardized (z-units). Weights are dollar-scaled; Sharpe (below)
# is scale-free so leverage constants are cosmetic.
# ---------------------------------------------------------------------------

def _blank(n, T, F):
    W = np.empty((n, T, F), dtype=np.float32)
    W[:] = np.nan
    return W


def w_cs_mom(R, lb):
    """Cross-sectional momentum: rank features by past-lb cumret, long top / short bottom."""
    n, T, F = R.shape
    if lb >= T - 1:
        return _blank(n, T, F)
    P = np.cumsum(R, axis=1)
    k = F // 2
    W = _blank(n, T, F)
    for t in range(lb, T - 1):
        past = P[:, t, :] - P[:, t - lb, :]
        ranks = np.argsort(np.argsort(past, axis=1), axis=1)
        W[:, t, :] = np.where(ranks >= F - k, 1.0, np.where(ranks < k, -1.0, 0.0)) / k
    return W


def w_ts_mom(R, lb):
    """Time-series momentum: per-asset sign of past-lb cumret."""
    n, T, F = R.shape
    if lb >= T - 1:
        return _blank(n, T, F)
    P = np.cumsum(R, axis=1)
    W = _blank(n, T, F)
    for t in range(lb, T - 1):
        W[:, t, :] = np.sign(P[:, t, :] - P[:, t - lb, :]) / F
    return W


def w_ou_rev(R, lb, thr=1.5):
    """Level mean-reversion (OU/Bollinger): fade the z-score of price vs its rolling mean.

    A genuine construction, NOT sign-flipped momentum (no mirror duplicates).
    """
    n, T, F = R.shape
    if lb >= T - 1:
        return _blank(n, T, F)
    P = np.cumsum(R, axis=1)
    W = _blank(n, T, F)
    for t in range(lb, T - 1):
        win = P[:, t - lb:t, :]
        z = (P[:, t, :] - win.mean(axis=1)) / (win.std(axis=1) + 1e-9)
        W[:, t, :] = -np.clip(z, -thr, thr) / thr / F
    return W


def w_vol_mom(R, lb):
    """Vol-managed momentum: trend signal scaled by 1/recent-vol.

    Under homoskedasticity (the AR(1) null) 1/vol is ~constant, so this collapses
    to ts_mom; its rank DIFFERENCE from ts_mom exists only under vol clustering.
    """
    n, T, F = R.shape
    if lb >= T - 1:
        return _blank(n, T, F)
    P = np.cumsum(R, axis=1)
    W = _blank(n, T, F)
    for t in range(lb, T - 1):
        rv = R[:, t - lb:t, :].std(axis=1) + 1e-6
        W[:, t, :] = np.sign(P[:, t, :] - P[:, t - lb, :]) / rv / F
    return W


def _sign_position(R, lb, signal_idx, base, flip=False):
    """Position a fixed cross-asset `base` vector by the sign of leader `signal_idx`'s past cumret."""
    n, T, F = R.shape
    P = np.cumsum(R, axis=1)
    W = _blank(n, T, F)
    base = np.asarray(base, dtype=np.float32)
    base = base / (np.sum(np.abs(base)) + 1e-12)
    for t in range(lb, T - 1):
        s = np.sign(P[:, t, signal_idx] - P[:, t - lb, signal_idx])
        if flip:
            s = -s
        W[:, t, :] = s[:, None] * base[None, :]
    return W


def w_xrisk(R, lb):
    """VIX-conditioned risk-on/off: VIX rising -> short equity, long bonds."""
    base = np.zeros(N_FEATURES, dtype=np.float32)
    for i in _EQ:
        base[i] = 1.0
    base[_TLT] = -1.0                       # risk-ON = long equity / short bonds
    return _sign_position(R, lb, _VIX, base, flip=True)  # VIX up -> risk OFF


def w_xmacro(R, lb):
    """Rates-conditioned macro: TNX rising -> short bonds & small-caps, long USD."""
    base = np.zeros(N_FEATURES, dtype=np.float32)
    base[_TLT] = -1.0
    base[_DXY] = 1.0
    base[0] = -1.0                          # IWM (rate-sensitive small caps)
    return _sign_position(R, lb, _TNX, base, flip=False)


def w_ftq(R, lb):
    """Continuous flight-to-quality: position ~ -z(equity drawdown); risk-off -> long bonds."""
    n, T, F = R.shape
    if lb >= T - 1:
        return _blank(n, T, F)
    P = np.cumsum(R, axis=1)
    W = _blank(n, T, F)
    for t in range(lb, T - 1):
        eq = (P[:, t, 0] + P[:, t, 1] + P[:, t, 2]) / 3.0 \
            - (P[:, t - lb, 0] + P[:, t - lb, 1] + P[:, t - lb, 2]) / 3.0
        sd = R[:, max(0, t - lb):t, :3].mean(axis=2).std(axis=1) + 1e-9
        z = np.clip(eq / (sd * np.sqrt(lb)), -2, 2)
        w = np.zeros((n, F), dtype=np.float32)
        for i in _EQ:
            w[:, i] = z / 6.0
        w[:, _TLT] = -z / 2.0
        W[:, t, :] = w
    return W


def w_leadlag(R, lb, leader, follower, sgn):
    """Cross-asset lead-lag: trade `follower` on the sign of `leader`'s past cumret."""
    n, T, F = R.shape
    if lb >= T - 1:
        return _blank(n, T, F)
    P = np.cumsum(R, axis=1)
    W = _blank(n, T, F)
    for t in range(lb, T - 1):
        s = sgn * np.sign(P[:, t, leader] - P[:, t - lb, leader])
        w = np.zeros((n, F), dtype=np.float32)
        w[:, follower] = s
        W[:, t, :] = w
    return W


# ---------------------------------------------------------------------------
# Frozen family (39 variants, 7 mechanisms; 27 joint / 12 marginal)
# ---------------------------------------------------------------------------

def _build_family():
    """Return list of (variant_name, family_group, weight_fn(R))."""
    fam = []
    for lb in (5, 10, 20):
        fam.append((f"cs_mom_lb{lb}", "cs_mom", lambda R, l=lb: w_cs_mom(R, l)))
    for lb in (5, 10, 20):
        fam.append((f"ts_mom_lb{lb}", "ts_mom", lambda R, l=lb: w_ts_mom(R, l)))
    for lb in (10, 20, 40):
        fam.append((f"ou_rev_lb{lb}", "ou_rev", lambda R, l=lb: w_ou_rev(R, l)))
    for lb in (10, 20, 40):
        fam.append((f"vol_mom_lb{lb}", "vol_mom", lambda R, l=lb: w_vol_mom(R, l)))
    for lb in (5, 10, 20):
        fam.append((f"xrisk_lb{lb}", "xrisk", lambda R, l=lb: w_xrisk(R, l)))
    for lb in (5, 10, 20):
        fam.append((f"xmacro_lb{lb}", "xmacro", lambda R, l=lb: w_xmacro(R, l)))
    for lb in (5, 10, 20):
        fam.append((f"ftq_lb{lb}", "ftq", lambda R, l=lb: w_ftq(R, l)))
    for (a, b, sg, nm) in LEADLAG_PAIRS:
        for lb in (5, 10, 20):
            fam.append((f"ll_{nm}_lb{lb}", "leadlag",
                        lambda R, a=a, b=b, sg=sg, l=lb: w_leadlag(R, l, a, b, sg)))
    return fam


_FAMILY = _build_family()
VARIANT_NAMES = [n for (n, g, fn) in _FAMILY]
VARIANT_GROUPS = [g for (n, g, fn) in _FAMILY]


# ---------------------------------------------------------------------------
# PnL / Sharpe
# ---------------------------------------------------------------------------

def _weights_to_perpath_sharpe(W, R, cost):
    """Per-path annualized Sharpe of a weight tensor, net of turnover cost.

    W (n,T,F) decided at t, applied to R[t+1]; turnover cost = cost * sum_f|dW|.
    Robust: a warm-up-masked / near-flat variant returns Sharpe 0 (not inf/NaN).
    """
    n, T, F = R.shape
    Wf = np.nan_to_num(W, nan=0.0)
    valid = ~np.isnan(W).all(axis=2)                       # (n,T) decision present
    gross = np.sum(Wf[:, :-1, :] * R[:, 1:, :], axis=2)    # (n,T-1), t=0..T-2
    Wprev = np.concatenate([np.zeros((n, 1, F), dtype=Wf.dtype), Wf[:, :-1, :]], axis=1)
    turn = np.sum(np.abs(Wf - Wprev), axis=2)              # (n,T)
    net = gross - cost * turn[:, :-1]
    net = np.where(valid[:, :-1], net, np.nan)
    mu = np.nanmean(net, axis=1)
    sd = np.nanstd(net, axis=1)
    sh = np.where(sd > 1e-6, (mu / np.maximum(sd, 1e-6)) * np.sqrt(252), 0.0)
    return np.nan_to_num(sh, nan=0.0, posinf=0.0, neginf=0.0)


def _perpath_sharpe_matrix(R, feature_names=None, cost=COST):
    """(n_paths, n_variants) per-path Sharpe matrix + per-feature |PnL| share.

    Standardizes R first (see docstring). The per-path matrix lets the CI
    resample PATHS (folding target-side noise) without re-running strategy logic.
    """
    Rs = _standardize(R, feature_names)
    n = Rs.shape[0]
    M = np.empty((n, len(_FAMILY)), dtype=np.float64)
    abs_by_feat = np.zeros(Rs.shape[-1], dtype=np.float64)
    for j, (name, grp, fn) in enumerate(_FAMILY):
        W = fn(Rs)
        M[:, j] = _weights_to_perpath_sharpe(W, Rs, cost)
        Wf = np.nan_to_num(W[:, :-1, :], nan=0.0)
        abs_by_feat += np.abs(Wf * Rs[:, 1:, :], dtype=np.float64).sum(axis=(0, 1))
    total = abs_by_feat.sum()
    share = abs_by_feat / total if total > 0 else abs_by_feat
    return M, share


def evaluate_family(R, feature_names=None, cost=COST):
    """Mean-across-paths Sharpe per variant -> dict {variant_name: sharpe}."""
    M, _ = _perpath_sharpe_matrix(R, feature_names, cost)
    vec = np.nanmean(M, axis=0)
    return dict(zip(VARIANT_NAMES, vec.tolist()))


# ---------------------------------------------------------------------------
# Gaussian-AR(1) null (permanent collapse detector)
# ---------------------------------------------------------------------------

def gaussian_ar1_null_tensor(seed, n_paths=200, T=60):
    """Per-feature Gaussian AR(1) fit to TRAIN phi/sigma; assets independent,
    homoskedastic, thin-tailed. Native units (so _standardize applies)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_paths, T, N_FEATURES), dtype=np.float32)
    for i, f in enumerate(FEATURE_ORDER):
        phi = TRAIN_PHI[f]
        sig = TRAIN_SCALES[f]
        esd = sig * np.sqrt(max(1e-6, 1.0 - phi * phi))
        p = np.zeros((n_paths, T), dtype=np.float64)
        p[:, 0] = rng.normal(0.0, sig, n_paths)
        for t in range(1, T):
            p[:, t] = phi * p[:, t - 1] + rng.normal(0.0, esd, n_paths)
        out[:, :, i] = p.astype(np.float32)
    return out


_NULL_CACHE = {}


def gaussian_ar1_null_rho(real, n_seeds=5, cost=COST):
    """Mean rho of the train-fit Gaussian-AR(1) null vs `real` (cached)."""
    key = (int(real.shape[0]), int(real.shape[1]), n_seeds, cost)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    real_vec = np.nanmean(_perpath_sharpe_matrix(real, cost=cost)[0], axis=0)
    rhos = []
    for s in range(n_seeds):
        nt = gaussian_ar1_null_tensor(s, n_paths=real.shape[0], T=real.shape[1])
        sv = np.nanmean(_perpath_sharpe_matrix(nt, cost=cost)[0], axis=0)
        rhos.append(float(spearmanr(real_vec, sv)[0]))
    res = {"mean": float(np.mean(rhos)), "std": float(np.std(rhos)), "per_seed": rhos}
    _NULL_CACHE[key] = res
    return res


# ---------------------------------------------------------------------------
# Bootstrap CI that folds in TARGET-side sampling noise
# ---------------------------------------------------------------------------

def _bootstrap_rho_ci(synth_M, real_M, n_boot=1000, seed=0):
    """95% CI on rho by resampling BOTH synth and real PATHS with replacement.

    Unlike a variant-index bootstrap (which sees only synth), this prices the
    real (target) Sharpe-vector estimation noise too.
    Returns (lo, hi, se, draws).
    """
    rng = np.random.default_rng(seed)
    ns, nr = synth_M.shape[0], real_M.shape[0]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sv = np.nanmean(synth_M[rng.integers(0, ns, ns)], axis=0)
        rv = np.nanmean(real_M[rng.integers(0, nr, nr)], axis=0)
        draws[b] = spearmanr(rv, sv)[0]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi), float(np.nanstd(draws)), draws


# ---------------------------------------------------------------------------
# Scorer interface
# ---------------------------------------------------------------------------

def score(loaded, feature_names=None, cost=COST):
    """TSTR strategy-rank transfer: Spearman rho (HIGHER = BETTER).

    loaded: list of (synth, real) numpy pairs, each (n_paths, 60, 7) in
    FEATURE_ORDER (pass feature_names if ordered differently — standardization is
    feature-order dependent). `real` is the pinned canonical panel (identical
    across seeds via the registry).

    Headline "mean" = per-seed-mean rho. detail carries the pooled rho, a
    target-noise-folded bootstrap CI, the per-family rhos, the AR(1)-null rho,
    and per-feature |PnL| shares.
    """
    if not loaded:
        raise ValueError("no (synth, real) pairs loaded")

    real0 = np.asarray(loaded[0][1], dtype=np.float32)
    real_M, real_share = _perpath_sharpe_matrix(real0, feature_names, cost)
    real_vec = np.nanmean(real_M, axis=0)

    # Per-seed rho (headline estimator).
    per_seed_rho = []
    synth_Ms = []
    for s, _r in loaded:
        Ms, _ = _perpath_sharpe_matrix(np.asarray(s, dtype=np.float32), feature_names, cost)
        synth_Ms.append(Ms)
        per_seed_rho.append(float(spearmanr(real_vec, np.nanmean(Ms, axis=0))[0]))

    # Pooled-over-seeds synth Sharpe vector + share.
    synth_stack = np.concatenate(synth_Ms, axis=0)
    synth_vec = np.nanmean(synth_stack, axis=0)
    _, synth_share = _perpath_sharpe_matrix(
        np.concatenate([np.asarray(s, dtype=np.float32) for s, _ in loaded], axis=0),
        feature_names, cost)
    rho_pooled, p_val = spearmanr(real_vec, synth_vec)

    # CI folds in target-side path noise (resamples real AND synth paths).
    ci_lo, ci_hi, ci_se, _ = _bootstrap_rho_ci(synth_stack, real_M)

    # Per-family rhos (diagnostic — which mechanisms transfer).
    per_family = {}
    for grp in dict.fromkeys(VARIANT_GROUPS):
        idx = [i for i, g in enumerate(VARIANT_GROUPS) if g == grp]
        if len(idx) >= 2:
            per_family[grp] = float(spearmanr([real_vec[i] for i in idx],
                                              [synth_vec[i] for i in idx])[0])

    null = gaussian_ar1_null_rho(real0, cost=cost)
    feats = list(feature_names) if feature_names is not None else list(FEATURE_ORDER)

    return {
        "mean": float(np.mean(per_seed_rho)),          # per-seed-mean rho, higher = better
        "std": float(np.std(per_seed_rho)),
        "n": len(loaded),
        "detail": {
            "unit": "Spearman rho (higher = better)",
            "strategy_family_version": FAMILY_VERSION,
            "headline_estimator": "per-seed-mean rho",
            "n_strategy_variants": len(VARIANT_NAMES),
            "n_joint_variants": sum(1 for g in VARIANT_GROUPS if FAMILY_KIND[g] == "joint"),
            "n_marginal_variants": sum(1 for g in VARIANT_GROUPS if FAMILY_KIND[g] == "marginal"),
            "turnover_cost_zunits": cost,
            "per_seed_rho": per_seed_rho,
            "rho_pooled": float(rho_pooled),
            "p_value_pooled": float(p_val),
            "rho_ci95_target_and_synth_noise": [ci_lo, ci_hi],
            "rho_ci95_se": ci_se,
            "per_family_rho": per_family,
            "null_ar1_rho": null["mean"],              # THE collapse detector
            "null_ar1_rho_std": null["std"],
            "mean_abs_sharpe_gap": float(np.mean(np.abs(real_vec - synth_vec))),
            "feature_abs_pnl_share_real": dict(zip(feats, np.round(real_share, 4).tolist())),
            "feature_abs_pnl_share_synth": dict(zip(feats, np.round(synth_share, 4).tolist())),
            "real_sharpes": dict(zip(VARIANT_NAMES, real_vec.tolist())),
            "synth_sharpes": dict(zip(VARIANT_NAMES, synth_vec.tolist())),
        },
    }


# ---------------------------------------------------------------------------
# Field-level diagnostics: floors, ceilings, controls (for the board + honesty)
# ---------------------------------------------------------------------------

def diagnostics(real, cost=COST):
    """Floors/ceilings/controls that frame the honest reading of the board.

    * self_consistency: even/odd path split of the real target — a MEMORIZATION
      reference (near-duplicate overlapping windows share ~92% of days),
      NOT an attainable target.
    * multi_regime_ceiling: 200 windows resampled from ALL real regimes — what a
      PERFECT multi-anchor generator scores.
    * single_anchor_ceiling: each of several contiguous ~one-regime sub-blocks of
      the (start-sorted) real windows scored vs the full target — the BEST a
      single-anchor generator can do. The gap multi_regime - single_anchor is
      the anchor-mismatch TASK PENALTY.
    * ar1_null / shuffled_real: known-bad controls that must rank below real
      generators.
    """
    real = np.asarray(real, dtype=np.float32)
    n = real.shape[0]
    real_vec = np.nanmean(_perpath_sharpe_matrix(real, cost=cost)[0], axis=0)

    def vec(x):
        return np.nanmean(_perpath_sharpe_matrix(np.asarray(x, np.float32), cost=cost)[0], axis=0)

    ev = vec(real[0::2]); od = vec(real[1::2])
    self_consistency = float(spearmanr(ev, od)[0])

    rng = np.random.default_rng(0)
    mr = [float(spearmanr(real_vec, vec(real[rng.integers(0, n, n)]))[0]) for _ in range(10)]

    blocks = [(0, 60), (int(0.25 * n), int(0.25 * n) + 60),
              (int(0.45 * n), int(0.45 * n) + 60), (max(0, n - 60), n)]
    sa = [float(spearmanr(real_vec, vec(real[a:b]))[0]) for a, b in blocks if b - a > 40]

    rng2 = np.random.default_rng(1)
    sh = []
    for _ in range(3):
        Rs = real.copy()
        for i in range(n):
            rng2.shuffle(Rs[i])            # shuffle time axis within each path
        sh.append(float(spearmanr(real_vec, vec(Rs))[0]))

    null = gaussian_ar1_null_rho(real, cost=cost)
    return {
        "self_consistency_even_odd": self_consistency,
        "multi_regime_ceiling_mean": float(np.mean(mr)),
        "multi_regime_ceiling_std": float(np.std(mr)),
        "single_anchor_ceiling_mean": float(np.mean(sa)),
        "single_anchor_ceiling_max": float(np.max(sa)),
        "single_anchor_ceiling_min": float(np.min(sa)),
        "anchor_mismatch_penalty": float(np.mean(mr) - np.mean(sa)),
        "ar1_null_rho": null["mean"],
        "shuffled_real_rho": float(np.mean(sh)),
    }


# ---------------------------------------------------------------------------
# Board (with null row, controls, floors/ceilings, flip-probability tie tiers)
# ---------------------------------------------------------------------------

def _board():
    from benchmark.registry import available_competitors, load_canon_real

    canon = load_canon_real().astype(np.float32)
    real_M = _perpath_sharpe_matrix(canon, cost=COST)[0]
    real_vec = np.nanmean(real_M, axis=0)
    nr = real_M.shape[0]

    # Shared real-path bootstrap indices so cross-competitor comparisons are paired.
    B = 500
    rng = np.random.default_rng(12345)
    real_idx = [rng.integers(0, nr, nr) for _ in range(B)]
    real_boot = np.stack([np.nanmean(real_M[ix], axis=0) for ix in real_idx])  # (B, V)

    rows = []  # (name, kind, point_rho, draws[B])
    for comp in available_competitors():
        Ms = [_perpath_sharpe_matrix(np.asarray(s, np.float32), cost=COST)[0] for s, _ in comp.load()]
        stack = np.concatenate(Ms, axis=0)
        # Point = per-seed-mean rho (the score() headline; avoids the seed-count
        # advantage a pooled rho gives).
        point = float(np.mean([spearmanr(real_vec, np.nanmean(m, axis=0))[0] for m in Ms]))
        rboot = np.random.default_rng(hash(comp.name) & 0xFFFF)
        draws = np.array([spearmanr(real_boot[b],
                          np.nanmean(stack[rboot.integers(0, stack.shape[0], stack.shape[0])], axis=0))[0]
                          for b in range(B)])
        rows.append((comp.name, comp.family, point, draws))

    # Permanent AR(1) null row + shuffled-time control, bootstrapped the same way.
    null_stack = np.concatenate(
        [_perpath_sharpe_matrix(gaussian_ar1_null_tensor(s, n_paths=nr, T=canon.shape[1]),
                                cost=COST)[0] for s in range(5)], axis=0)
    nboot = np.random.default_rng(777)
    ndraws = np.array([spearmanr(real_boot[b],
                       np.nanmean(null_stack[nboot.integers(0, null_stack.shape[0], null_stack.shape[0])], axis=0))[0]
                       for b in range(B)])
    rows.append(("AR1-NULL(control)", "control", gaussian_ar1_null_rho(canon)["mean"], ndraws))

    sh = canon.copy()
    srng = np.random.default_rng(1)
    for i in range(sh.shape[0]):
        srng.shuffle(sh[i])
    sh_M = _perpath_sharpe_matrix(sh, cost=COST)[0]
    sboot = np.random.default_rng(888)
    sdraws = np.array([spearmanr(real_boot[b],
                       np.nanmean(sh_M[sboot.integers(0, sh_M.shape[0], sh_M.shape[0])], axis=0))[0]
                       for b in range(B)])
    rows.append(("SHUFFLED-real(control)", "control", float(spearmanr(real_vec, np.nanmean(sh_M, axis=0))[0]), sdraws))

    rows.sort(key=lambda x: x[2], reverse=True)

    # Flip-probability tie tiers: adjacent rows cluster when P(order flips) > 0.10.
    tiers = [0] * len(rows)
    for i in range(1, len(rows)):
        p_flip = float(np.mean(rows[i][3] >= rows[i - 1][3]))
        tiers[i] = tiers[i - 1] if p_flip > 0.10 else tiers[i - 1] + 1

    diag = diagnostics(canon)
    print(f"\n=== T3 — TSTR strategy-rank transfer (Spearman rho, HIGHER = BETTER) "
          f"[family {FAMILY_VERSION}] ===")
    print(f"{len(VARIANT_NAMES)} variants, 7 mechanisms "
          f"({sum(1 for g in VARIANT_GROUPS if FAMILY_KIND[g]=='joint')} joint / "
          f"{sum(1 for g in VARIANT_GROUPS if FAMILY_KIND[g]=='marginal')} marginal), "
          f"turnover cost {COST} z-units")
    print("\nFloors / ceilings (honest framing):")
    print(f"  self-consistency (even/odd real)   = {diag['self_consistency_even_odd']:+.3f}  "
          f"(MEMORIZATION reference, not attainable)")
    print(f"  multi-regime PERFECT ceiling       = {diag['multi_regime_ceiling_mean']:+.3f} "
          f"+- {diag['multi_regime_ceiling_std']:.3f}  (perfect multi-anchor generator)")
    print(f"  single-anchor ceiling              = {diag['single_anchor_ceiling_mean']:+.3f} "
          f"[{diag['single_anchor_ceiling_min']:+.3f}, {diag['single_anchor_ceiling_max']:+.3f}]  "
          f"(BEST a single-anchor generator can reach)")
    print(f"  anchor-mismatch task penalty       = {diag['anchor_mismatch_penalty']:.3f}  "
          f"(multi-regime minus single-anchor)")
    print(f"  AR(1) null / shuffled-real control = {diag['ar1_null_rho']:+.3f} / "
          f"{diag['shuffled_real_rho']:+.3f}  (known-bad — must rank low)")
    print(f"\n{'rank':>4} {'tier':>4}  {'competitor':22s} {'rho':>7} {'CI95(tgt+synth)':>18}")
    for i, (name, kind, rho, draws) in enumerate(rows, 1):
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        tag = "  <== CONTROL" if kind == "control" else ""
        print(f"{i:>4} {tiers[i-1]:>4}  {name:22s} {rho:+7.3f} [{lo:+.2f},{hi:+.2f}]{tag}")

    nrk = next(i for i, r in enumerate(rows, 1) if r[0].startswith("AR1-NULL"))
    print(f"\nAR(1) null rank: #{nrk} of {len(rows)}. "
          f"Rows within a shared tier are a statistical tie (flip-prob > 0.10).")


if __name__ == "__main__":
    _board()
