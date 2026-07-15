"""T5 — VaR/ES risk backtesting (fully-established protocol, adopted wholesale).

The generator under test is treated as a *risk model*: its synthetic paths define
the model's P&L distribution, and the 200 real OOS windows (2020–2023, COVID crash
+ 2022 bear included) are the backtest observations. A tail-under-dispersed
generator must show up here as excess VaR exceptions and negative Acerbi–Székely
Z2 (realized tail losses larger than the model's ES).

Tests implemented (exact sources)
---------------------------------
- Kupiec (1995), "Techniques for Verifying the Accuracy of Risk Measurement
  Models", J. of Derivatives 3(2) — proportion-of-failures LR_uc ~ chi2(1).
  Two-sided by construction: too FEW exceptions also rejects.
- Christoffersen (1998), "Evaluating Interval Forecasts", Int. Econ. Review 39(4)
  — first-order Markov independence LR_ind ~ chi2(1) and conditional coverage
  LR_cc = LR_uc + LR_ind ~ chi2(2), on the hit sequence ordered by anchor time.
- Acerbi & Székely (2014), "Backtesting Expected Shortfall", Risk Magazine —
  test 2 statistic: Z2 = sum_t(X_t * I_t) / (T * p_tail * ES) + 1, with
  I_t = 1{X_t < -VaR}, ES > 0 the model expected shortfall magnitude and
  E[Z2] = 0 under H0. Significance is obtained the way AS prescribe: Monte
  Carlo under H0 by resampling T observations from the MODEL (synth)
  distribution (M = 1000 draws, one-sided left tail: negative Z2 = model ES
  too thin). ES level = 97.5% (p_tail = 0.025), the FRTB/Basel-III ES level.
  This leg detects ES UNDER-estimation only; it is one-sided and — like any MC
  test at T = 200 — low-power against ES OVER-estimation (the null saturates at
  Z2 = +1 when the over-fat model produces few real exceptions), so it is paired
  with the over-fatness gate below.
- ES over-fatness gate (audit T5-ES-1): the AS-Z2 leg is blind to a tail-
  SELECTIVELY over-fat ES97.5 — the FRTB headline risk number. A zero-skill
  model identical to real except with a 2-3x too-fat ES97.5 (body and 95%/99%
  VaR left calibrated) passes every AS-Z2 cell and, absent this gate, the whole
  ES leg. We therefore mirror the two-sidedness Kupiec already gives the VaR
  cells (where too-FEW exceptions rejects an over-wide model): the ES cell also
  fails when the model's ES97.5 exceeds a POWERED upper band of the realized
  tail loss. The band is a circular block-bootstrap (block 5, 2000 draws) of the
  realized ES97.5 from the real OOS observations; over-fat := model ES97.5 >
  1.25 x the band's 99th percentile. Tuned so calibrated data (real-vs-real, all
  12 cells x 8 bootstrap seeds) trips it 0/96, while a >=1.25x-over-fat cell is
  rejected. Genuinely over-fat cells in real competitors (e.g. an ES that is
  ~1.7x the realized tail loss on a specific series) are demoted — the honest
  outcome, not a false positive.
- Basel Committee (1996, "Supervisory framework for the use of backtesting"),
  traffic-light zones on the 99%-VaR 1-day exception count. Zones are derived
  from the binomial CDF exactly as in the Basel document (green: cumulative
  probability <= 95%; yellow: <= 99.99%; red: beyond), evaluated at our actual
  n = 200 (which yields green 0–4 / yellow 5–9 / red >= 10, matching the
  canonical 250-day thresholds); the count linearly scaled to 250 obs is also
  reported in `detail` for legibility.

Protocol design (frozen for this scorer)
----------------------------------------
Data: (synth, real) pairs of shape (n_paths, 60, 7) in native increment space —
LOG-RETURNS for price-type features, simple DIFFS for levels (VIX, TNX). Only
price-type features are backtested: IWM, QQQ, SPY, TLT, DXY, plus an
equal-weight portfolio "EW-PORT" of IWM/QQQ/SPY/TLT (daily-rebalanced: daily
simple returns averaged, re-logged; exact simple-return aggregation, not a
log-return average).

Cells: series in {IWM, QQQ, SPY, TLT, DXY, EW-PORT} x horizon h in {1, 10} days
x alpha in {0.95, 0.99} (VaR), plus series x h for ES 97.5%.

Model distribution: for each cell, the model's h-day return sample pools all
NON-OVERLAPPING h-day blocks of every synth path (h=1: n_paths*60 samples;
h=10: n_paths*6). Pooling within the window assumes within-window stationarity
of the model — a deliberate choice to stabilize the 99% empirical quantile
(200 anchor-only samples would make the 99% VaR the 2nd order statistic).
VaR_alpha = -(empirical (1-alpha) quantile); ES = -mean of the sub-quantile
tail. Exception: real return strictly below the quantile.

Backtest observations: ONE observation per real window = the cumulative return
over the FIRST h days after the anchor, taken in anchor-time order (verified:
archive axis 0 is chronologically ordered; consecutive anchors are ~1–15
trading days apart, mean ~4.7).

Overlap caveat (documented honestly): at h=10 the anchor stride (~4.7 d) is
smaller than the horizon, so consecutive 10-day observations share days. This
mechanically clusters exceptions and over-rejects the Christoffersen
independence test. Therefore the PASS rule per VaR cell is:
  h=1 : Kupiec LR_uc p >= 0.05 AND Christoffersen LR_cc p >= 0.05
  h=10: Kupiec LR_uc p >= 0.05 only (LR_ind/LR_cc still computed and reported
        in `detail`, but excluded from scoring at h=10).
At h=1 observations sit on distinct days (stride >= 1), so the hit sequence is
a genuine (if unevenly-spaced) chronological sequence; Kupiec is unaffected by
spacing, Christoffersen's Markov transition estimates treat the sequence as
consecutive — a mild approximation given the uneven stride, also noted here.

Composite score (0–1, higher = better risk model), per (synth, real) pair:
  var_score = fraction of the 24 VaR cells passing the rule above
  es_score  = fraction of the 12 ES cells passing BOTH legs: AS-Z2 Monte-Carlo
              p >= 0.05 (ES under-estimation) AND not over-fat per the gate
              above (ES over-estimation). A cell fails if the tail is either too
              thin or too fat — symmetric, like Kupiec on the VaR cells.
  tl_score  = mean over the 6 Basel traffic-light cells (h=1, alpha=0.99):
              green = 1.0, yellow = 0.5, red = 0.0
  composite = (var_score + es_score + tl_score) / 3
score()["mean"/"std"] = mean/std of the composite across seeds (pairs).

Over-dispersion demotion — scope of the claim (audit T5-ES-1). The published
over-dispersion ladder (real x1.5 -> ~0.65 ... x5 -> ~0.17, monotone) holds ONLY
for WHOLE-distribution scaling, which fattens the powered 95%/body/traffic-light
cells and is what those numbers measure. A tail-SELECTIVE over-fattening (ES97.5
inflated while the body and 95%/99% VaR stay calibrated) escapes every VaR/TL
cell and, before the fix, every ES cell too — scoring at or above the calibrated
ceiling. The es over-fatness gate above closes that path: whole-distribution
over-dispersion is still demoted monotonically, AND tail-selective over-fatness
is now demoted on the ES leg rather than rewarded.

Degeneracy handling: non-finite synth h-day returns are dropped (count
reported); a cell whose synth tail is degenerate (zero-size or ES <= 0) fails
that cell and is flagged. Zero real exceptions: Kupiec still evaluates (rejects
if 0 is too few), LR_ind := 0, AS Z2 = +1 (over-conservative, not a left-tail
rejection). Monte Carlo RNG is seeded deterministically per pair.

Run from finbench/ root:
    .venv/bin/python -m benchmark.scorers.t5_var_es
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, chi2

# ---- canonical v1-panel semantics -------------------------------------------
FEATURE_ORDER = ["IWM", "QQQ", "SPY", "TLT", "VIX", "TNX", "DXY"]
PRICE_FEATURES = ["IWM", "QQQ", "SPY", "TLT", "DXY"]   # native log-returns
PORTFOLIO_LEGS = ["IWM", "QQQ", "SPY", "TLT"]           # equal-weight portfolio
PORTFOLIO_NAME = "EW-PORT"

HORIZONS = (1, 10)
VAR_ALPHAS = (0.95, 0.99)
ES_LEVEL = 0.975            # Acerbi–Szekely on ES_97.5 (p_tail = 0.025)
AS_MC_DRAWS = 1000
PASS_P = 0.05

# ES over-fatness gate (audit T5-ES-1). The AS-Z2 leg is one-sided (thin tails)
# and low-power on the fat side, so a tail-selectively over-fat ES97.5 escapes
# it. Mirror Kupiec's two-sidedness on the VaR cells: the ES cell also fails when
# the model ES97.5 exceeds a powered upper band of the REALIZED tail loss,
# block-bootstrapped from the real OOS observations. Params tuned so calibrated
# data trips it 0/96 (12 cells x 8 seeds) while a >=1.25x-over-fat cell rejects.
ES_OVERFAT_BAND_Q = 0.99     # upper percentile of the block-bootstrap band
ES_OVERFAT_CUSHION = 1.25    # model ES97.5 must exceed cushion x band to gate
ES_OVERFAT_BLOCK = 5         # circular block length (volatility clustering)
ES_OVERFAT_NBOOT = 2000      # bootstrap draws (band is stable well below this)

_EPS = 1e-300


# ---- statistical tests -------------------------------------------------------

def kupiec_pof(x: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test. Returns (LR_uc, p-value)."""
    pi = x / n
    # log-likelihood under H0 (exception prob p) vs MLE (pi); 0*log(0) := 0
    ll0 = (n - x) * np.log(max(1 - p, _EPS)) + x * np.log(max(p, _EPS))
    ll1 = (n - x) * np.log(max(1 - pi, _EPS)) + x * np.log(max(pi, _EPS))
    lr = max(0.0, -2.0 * (ll0 - ll1))
    return lr, float(chi2.sf(lr, df=1))


def christoffersen(hits: np.ndarray, p: float) -> dict:
    """Christoffersen (1998) independence + conditional coverage on a 0/1 hit
    sequence in time order. Returns LR_uc/LR_ind/LR_cc and p-values."""
    h = np.asarray(hits, dtype=int)
    n, x = len(h), int(h.sum())
    lr_uc, p_uc = kupiec_pof(x, n, p)

    # first-order Markov transition counts
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))

    def _ll(k1, k0, q):  # k1 successes, k0 failures at prob q; 0*log0 := 0
        return k1 * np.log(max(q, _EPS)) + k0 * np.log(max(1 - q, _EPS))

    denom = n00 + n01 + n10 + n11
    if denom == 0 or (n01 + n11) == 0:
        # no exceptions (or single obs): independence is vacuous
        lr_ind, p_ind = 0.0, 1.0
    else:
        pi01 = n01 / max(n00 + n01, 1)
        pi11 = n11 / max(n10 + n11, 1)
        pi_ = (n01 + n11) / denom
        ll_ind = _ll(n01, n00, pi01) + _ll(n11, n10, pi11)
        ll_pool = _ll(n01 + n11, n00 + n10, pi_)
        lr_ind = max(0.0, -2.0 * (ll_pool - ll_ind))
        p_ind = float(chi2.sf(lr_ind, df=1))

    lr_cc = lr_uc + lr_ind
    return {
        "n": n, "exceptions": x, "expected": n * p,
        "lr_uc": float(lr_uc), "p_uc": float(p_uc),
        "lr_ind": float(lr_ind), "p_ind": float(p_ind),
        "lr_cc": float(lr_cc), "p_cc": float(chi2.sf(lr_cc, df=2)),
    }


def acerbi_szekely_z2(real_obs: np.ndarray, synth_sample: np.ndarray,
                      p_tail: float, rng: np.random.Generator,
                      n_draws: int = AS_MC_DRAWS) -> dict:
    """Acerbi–Szekely (2014) test 2 for ES, with Monte-Carlo significance under
    H0 drawn from the model (synth) distribution. One-sided left tail: negative
    Z2 = realized tail losses exceed model ES (under-dispersed generator)."""
    var_q = np.quantile(synth_sample, p_tail)          # left-tail quantile (a return)
    tail = synth_sample[synth_sample <= var_q]
    es_pos = -float(tail.mean()) if tail.size else np.nan
    if not np.isfinite(es_pos) or es_pos <= 0:
        return {"z2": np.nan, "p_mc": 0.0, "es_model": es_pos,
                "var_model": float(-var_q), "exceptions": None,
                "degenerate": True}

    T = len(real_obs)

    def _z2(x):
        i = x < var_q
        return float(np.sum(x * i) / (T * p_tail * es_pos) + 1.0)

    z_obs = _z2(real_obs)
    sims = rng.choice(synth_sample, size=(n_draws, T), replace=True)
    z_sim = np.sum(sims * (sims < var_q), axis=1) / (T * p_tail * es_pos) + 1.0
    # one-sided MC p-value with the standard +1 correction
    p_mc = float((np.sum(z_sim <= z_obs) + 1) / (n_draws + 1))
    return {"z2": z_obs, "p_mc": p_mc, "es_model": es_pos,
            "var_model": float(-var_q),
            "exceptions": int(np.sum(real_obs < var_q)), "degenerate": False}


def es_overfatness(real_obs: np.ndarray, es_model: float, p_tail: float,
                   rng: np.random.Generator, block: int = ES_OVERFAT_BLOCK,
                   n_boot: int = ES_OVERFAT_NBOOT, band_q: float = ES_OVERFAT_BAND_Q,
                   cushion: float = ES_OVERFAT_CUSHION) -> dict:
    """ES-over-fatness gate (audit T5-ES-1). Circular block-bootstrap the
    realized ES97.5 from the real OOS observations to get a POWERED upper band of
    the tail loss the data actually supports; the model's ES97.5 is 'over-fat'
    when it exceeds `cushion` x that upper band. This mirrors Kupiec's two-
    sidedness on the VaR cells (too-few exceptions rejects an over-wide model)
    for the ES cell, where the AS-Z2 MC test is one-sided and low-power on the
    fat side. Returns `over_fat` plus diagnostics for the board."""
    obs = np.asarray(real_obs, dtype=np.float64)
    T = obs.size
    k = max(1, int(round(T * p_tail)))                 # tail size at this level
    es_real = -float(np.sort(obs)[:k].mean())          # realized ES97.5 point est
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]
           ).reshape(n_boot, -1)[:, :T] % T            # circular block indices
    boot_es = -np.sort(obs[idx], axis=1)[:, :k].mean(axis=1)
    band_hi = float(np.nanquantile(boot_es, band_q))
    over = bool(np.isfinite(es_model) and es_model > cushion * band_hi)
    return {"over_fat": over, "es_real_point": es_real, "es_band_hi": band_hi,
            "es_overfat_ratio": float(es_model / es_real) if es_real > 0
            else float("inf")}


def basel_traffic_light(x: int, n: int, p: float = 0.01) -> dict:
    """Basel (1996) traffic-light zone from the binomial CDF at the actual n
    (green: CDF <= 0.95; yellow: <= 0.9999; red: beyond); scaled-to-250 count
    reported alongside."""
    cdf = float(binom.cdf(x, n, p))
    zone = "green" if cdf <= 0.95 else ("yellow" if cdf <= 0.9999 else "red")
    return {"exceptions": x, "n": n, "binom_cdf": cdf, "zone": zone,
            "scaled_250": round(x * 250.0 / n, 2)}


# ---- series / observation construction ---------------------------------------

def _resolve_series(tensor: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    """Map (n, 60, 7) native increments -> {series_name: (n, 60) daily log-returns}."""
    idx = {f: feature_names.index(f) for f in PRICE_FEATURES}
    out = {f: tensor[:, :, idx[f]].astype(np.float64) for f in PRICE_FEATURES}
    # equal-weight daily-rebalanced portfolio, exact simple-return aggregation
    legs = np.stack([tensor[:, :, feature_names.index(f)].astype(np.float64)
                     for f in PORTFOLIO_LEGS], axis=-1)      # (n, 60, 4) log-rets
    with np.errstate(over="ignore", invalid="ignore"):
        simple = np.expm1(legs).mean(axis=-1)                 # (n, 60) simple
        out[PORTFOLIO_NAME] = np.log1p(simple)                # back to daily log
    return out


def _synth_h_sample(daily: np.ndarray, h: int) -> tuple[np.ndarray, int]:
    """Pool all non-overlapping h-day cumulative returns from every synth path.
    Returns (finite sample, n_dropped_nonfinite)."""
    n, t = daily.shape
    k = t // h
    blocks = daily[:, : k * h].reshape(n, k, h).sum(axis=2).ravel()
    finite = np.isfinite(blocks)
    return blocks[finite], int((~finite).sum())


def _real_h_obs(daily: np.ndarray, h: int) -> np.ndarray:
    """One backtest observation per real window: cumulative return over the
    first h days after the anchor, in anchor-time (axis 0) order."""
    return daily[:, :h].sum(axis=1)


# ---- per-pair scoring ---------------------------------------------------------

def _score_pair(synth: np.ndarray, real: np.ndarray, feature_names: list[str],
                pair_idx: int) -> dict:
    rng = np.random.default_rng(20260710 + pair_idx)
    s_series = _resolve_series(synth, feature_names)
    r_series = _resolve_series(real, feature_names)

    var_cells, es_cells, tl_cells = {}, {}, {}
    for name in list(PRICE_FEATURES) + [PORTFOLIO_NAME]:
        for h in HORIZONS:
            s_h, dropped = _synth_h_sample(s_series[name], h)
            r_h = _real_h_obs(r_series[name], h)
            degenerate = s_h.size < 50

            # --- VaR cells: Kupiec + Christoffersen -------------------------
            for alpha in VAR_ALPHAS:
                key = f"{name}|h={h}|a={alpha}"
                if degenerate:
                    var_cells[key] = {"degenerate": True, "pass": False,
                                      "nonfinite_dropped": dropped}
                    continue
                q = np.quantile(s_h, 1.0 - alpha)
                hits = (r_h < q).astype(int)
                ct = christoffersen(hits, 1.0 - alpha)
                # pass rule: h=1 -> uc AND cc; h=10 -> uc only (overlap caveat)
                ok = ct["p_uc"] >= PASS_P and (h != 1 or ct["p_cc"] >= PASS_P)
                var_cells[key] = {**ct, "var_model": float(-q), "pass": bool(ok),
                                  "nonfinite_dropped": dropped,
                                  "ind_scored": h == 1, "degenerate": False}

            # --- ES cell: Acerbi–Szekely Z2 at 97.5% ------------------------
            key = f"{name}|h={h}|ES{ES_LEVEL}"
            if degenerate:
                es_cells[key] = {"degenerate": True, "pass": False}
            else:
                asr = acerbi_szekely_z2(r_h, s_h, 1.0 - ES_LEVEL, rng)
                if asr["degenerate"]:
                    asr["over_fat"] = False
                    asr["pass"] = False
                else:
                    # two legs: AS-Z2 (too thin) AND over-fatness gate (too fat)
                    gate = es_overfatness(r_h, asr["es_model"], 1.0 - ES_LEVEL, rng)
                    asr.update(gate)
                    asr["pass"] = bool(asr["p_mc"] >= PASS_P and not gate["over_fat"])
                es_cells[key] = asr

            # --- Basel traffic light: h=1, 99% VaR --------------------------
            if h == 1:
                key = f"{name}|h=1|TL99"
                if degenerate:
                    tl_cells[key] = {"degenerate": True, "zone": "red"}
                else:
                    q99 = np.quantile(s_h, 0.01)
                    tl_cells[key] = basel_traffic_light(int(np.sum(r_h < q99)),
                                                        len(r_h))

    var_score = float(np.mean([c["pass"] for c in var_cells.values()]))
    es_score = float(np.mean([c["pass"] for c in es_cells.values()]))
    _tl_val = {"green": 1.0, "yellow": 0.5, "red": 0.0}
    tl_score = float(np.mean([_tl_val[c["zone"]] for c in tl_cells.values()]))
    composite = (var_score + es_score + tl_score) / 3.0

    return {"composite": composite, "var_score": var_score,
            "es_score": es_score, "tl_score": tl_score,
            "var_cells": var_cells, "es_cells": es_cells, "tl_cells": tl_cells}


# ---- public interface ---------------------------------------------------------

def score(loaded, feature_names=None) -> dict:
    """Score one competitor's list of (synth, real) pairs. Returns
    {"mean","std","n","detail"} — mean composite (0-1, higher better) over pairs."""
    feature_names = list(feature_names) if feature_names else FEATURE_ORDER
    per_seed = [_score_pair(s, r, feature_names, i)
                for i, (s, r) in enumerate(loaded)]
    comps = np.array([p["composite"] for p in per_seed])
    return {
        "mean": float(comps.mean()) if comps.size else float("nan"),
        "std": float(comps.std(ddof=0)) if comps.size else float("nan"),
        "n": int(comps.size),
        "detail": {
            "per_seed": per_seed,
            "subscores_mean": {
                k: float(np.mean([p[k] for p in per_seed]))
                for k in ("var_score", "es_score", "tl_score")
            } if per_seed else {},
        },
    }


if __name__ == "__main__":
    from benchmark.registry import available_competitors

    rows = []
    for comp in available_competitors():
        res = score(comp.load())
        sub = res["detail"]["subscores_mean"]
        # aggregate 99%/h=1 exception count (Basel cell) for tail-dispersion legibility
        exc = [c["exceptions"] for p in res["detail"]["per_seed"]
               for c in p["tl_cells"].values() if not c.get("degenerate")]
        rows.append((comp.name, res["mean"], res["std"], res["n"],
                     sub["var_score"], sub["es_score"], sub["tl_score"],
                     float(np.mean(exc)) if exc else float("nan")))

    rows.sort(key=lambda r: -r[1])
    hdr = (f"{'rank':<5}{'competitor':<18}{'T5 score':>9}{'std':>7}{'n':>3}"
           f"{'VaR-pass':>10}{'ES-pass':>9}{'TL':>7}{'x99/200':>9}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r[0]:<18}{r[1]:>9.3f}{r[2]:>7.3f}{r[3]:>3d}"
              f"{r[4]:>10.3f}{r[5]:>9.3f}{r[6]:>7.3f}{r[7]:>9.1f}")
    print("\n(expected 99%/h=1 exceptions per 200 obs if calibrated: 2.0; "
          "excess => tail under-dispersion)")
