"""F4 — Distributional distances: Wasserstein-1 / sliced-W1 / RBF-MMD^2 / signature-MMD.

Task F4 of BENCHMARK_TASKS.md ("Distributional distance"): Wasserstein-1 / MMD /
signature-MMD (Sig-W1). Lower distance = better; the reported score inverts to
0-1 (higher = better).

SHAPE / SCALE SEPARATION. A distributional distance must
NOT pay a bonus for shrinking. The OLD build standardized both samples by the REAL
per-feature std, so every W1/MMD/signature component grew ~linearly (or, for the
level-2 signature, ~quadratically) in the synth-vs-real scale factor c — and |c-1| is
NOT symmetric in log c, so under-dispersion (c<1) was penalized LESS than the same
log-magnitude over-dispersion (c>1). Measured: an equal-log-factor under/over pair
scored up to ~0.167 apart in favour of under-dispersion — the exact failure mode this
suite exists to catch, and the reason a moment-matched Gaussian null used to top the
F4 field. The rebuild splits SHAPE from SCALE:

  * every shape distance (w1_step, w1_cum, sliced_w1, mmd2_rbf, sig_mmd2) is computed
    after standardizing EACH sample by ITS OWN pooled per-step per-feature std, so the
    shape components are SCALE-INVARIANT: for a pure rescale synth = c*real they are
    identically zero and score(c*x) == score(x/c) exactly (a permanent regression
    assert enforces this — see `_self_test`);
  * dispersion is scored SEPARATELY by an explicit, exactly log-symmetric per-feature
    term `disp` = mean_f |log(std_synth_f / std_real_f)| (|log c| = |log(1/c)|), so
    over- and under-dispersion of the same log magnitude are penalized EQUALLY.

Statistics implemented (all synth-vs-real, computed per seed pair, averaged over seeds;
inputs are the OWN-std-standardized increments unless noted):

1. **Wasserstein-1** (`scipy.stats.wasserstein_distance`; Villani, *Optimal
   Transport*, 2009): (a) per feature on POOLED per-step increments; (b) per feature on
   H-day CUMULATIVE returns (sum over the T-step window), one value per path.
2. **Sliced Wasserstein-1** (Rabin et al., SSVM 2011; Bonneel et al., JMIV 2015): W1
   between 1-D projections of the pooled per-step F-dim increment vectors along
   `N_PROJ = 64` fixed random unit directions, averaged over directions.
3. **RBF-MMD^2, unbiased estimator** (Gretton, Borgwardt, Rasch, Schölkopf, Smola,
   "A Kernel Two-Sample Test", JMLR 13, 2012 — Eq. 3 / Lemma 6) on flattened windows
   (T*F = 420-dim). MULTI-BANDWIDTH Gaussian mixture: k = sum over
   sigma in `sigma0 * {0.25,0.5,1,2,4}` so the statistic does not
   saturate beyond ~3x and remains sensitive across scales. `sigma0` is PINNED ONCE
   from the REAL sample only (median pairwise Euclidean distance within real), so
   EVERY competitor is scored under the IDENTICAL kernel — the OLD median-heuristic on
   the pooled (synth ∪ real) sample gave each competitor a DIFFERENT bandwidth, making
   MMD^2 not comparable across the field (not a metric on the field). Both samples are
   subsampled to `MMD_MAX_N = 200` windows (= the canonical real's n_paths, seeded), so
   the estimator is pinned at a fixed sample size and cannot be gamed by submitting
   more/fewer paths.
4. **Signature-MMD (Sig-W1-style JOINT path statistic)** — Ni, Szpruch, Wiese, Liao,
   Xiao, "Sig-Wasserstein GANs for Time Series Generation" (arXiv 2006.05421 /
   ICAIF'21); Chevyrev & Oberhauser, "Signature moments to characterize laws of
   stochastic processes" (JMLR 2022).
   COMPUTED ON THE JOINT time-augmented path of all F features at once (one
   (F+1)-dimensional path per window: channel 0 = time t_k = k/T, channels 1..F = the
   cumsum of the OWN-std-standardized increments / sqrt(T)), NOT feature-by-feature.
   Truncated DEPTH-2 signatures give level-1 (F+1 coords, the endpoint) and level-2
   ((F+1)^2 coords). The level-2 block's ANTISYMMETRIC part is the matrix of pairwise
   Lévy areas — the signed area between every pair of channels — the ORDER-DEPENDENT,
   CROSS-ASSET lead-lag content a per-marginal statistic cannot see. For F=7 this is
   8 + 64 = 72 signature coordinates per window.
   ESTIMATOR: the UNBIASED two-sample MMD^2 (Gretton et al. 2012, Lemma 6) between the
   two clouds of per-window signature vectors. "Unbiased" matters: the biased plug-in
   carries a +Var(S)/n_paths term that depends on each competitor's own signature
   variance and path count (so it would partly score dispersion and vary with
   n_paths); the unbiased estimator has expectation 0 when the two laws match,
   so an INDEPENDENT resample of the SAME distribution scores ~0 (asserted in
   `_self_test`), not merely an identical sample, and the sample size is PINNED to
   `MMD_MAX_N`. The kernel is the multi-bandwidth RBF with `sigma0` pinned from the REAL
   signatures only , NOT the LINEAR / expected-signature kernel
   K=<S_i,S_j>: the linear kernel compares only the MEAN signature and is BLIND to a
   temporal-order scramble that preserves marginals + cross-section (verified — it fails
   the order control in `_self_test`), which would lose the order-sensitivity;
   the RBF form is the SAME Lemma-6 unbiased estimator but distributional and
   order-sensitive.
   Signature coordinates are WHITENED by the REAL panel's per-coordinate std before the
   kernel (real-only => no synth-scale leakage) and divided by sqrt(n_coords) so the
   kernel distance is the MEAN per-coordinate whitened squared difference over the 72
   coordinates: the OLD raw-L2 sum let one feature/level carry up to
   ~77% of the component, and the c^2/c^4 level-2 magnitudes dominate; per-coordinate
   whitening + averaging equalizes levels 1 and 2 and all channels, matching the equal
   per-feature weighting of the other components. Per-feature signature distances (the
   pinned-RBF MMD^2 restricted to each feature's channels) are exposed in `detail`.
   This keeps the metric order-sensitive (a per-marginal mean-only form is blind —
   distance 0.0 — to a perturbation that destroys contemporaneous cross-asset
   dependence while preserving marginals; the joint form scores it worse: see the
   order/joint scramble controls in `_self_test`).
   CAVEAT (honest scope): on the shipped daily panel a pure temporal-order scramble
   that preserves the cross-section costs the FULL F4 combined score little, because
   (a) daily returns have genuinely weak autocorrelation and (b) most F4 components are
   marginal/exchangeable by construction — F4 remains a distributional task; F2/F5 own
   single-series temporal structure. This component's added power is chiefly on JOINT /
   cross-asset dependence, which the per-marginal form could not represent at all.
   DEPENDENCY ROUTE (documented per task instructions): `pip install iisignature` into
   the venv FAILED to build its C extension (no wheel for this platform/Python; source
   build errors at get-requirements); `signatory` is likewise unavailable. Signatures
   are therefore computed in pure numpy via Chen's identity, verified against a
   brute-force iterated-sum reference (see `_self_test`).
   PATH SCALING (in the spirit of Ni et al.'s pre-signature path normalization): the
   standardized-increment cumsum channels are divided by sqrt(T) so terminal values
   have ~unit variance and level-2 signature coordinates are O(1).

Combined score (score()["mean"]): each of the SIX raw distances d (w1_step, w1_cum
— both feature-averaged in own-std units — sliced_w1, mmd2_rbf, sig_mmd2, and the
log-symmetric dispersion term disp) is inverted via s = 1/(1+d) in [0,1] (d=0 => 1,
monotone decreasing), and the six inverted components are averaged with equal weight,
per seed pair; "mean"/"std" are over seed pairs. Raw per-component distances are
exposed in `detail`.

Run from the finbench repo root:
    .venv/bin/python -m benchmark.scorers.f4_distances
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

FEATURES_DEFAULT = ["IWM", "QQQ", "SPY", "TLT", "VIX", "TNX", "DXY"]

N_PROJ = 64          # sliced-W1 random projections
PROJ_SEED = 20260710  # fixed => identical slices for every competitor
MMD_MAX_N = 200      # subsample cap per side = canonical real n_paths => estimator
                     # pinned at a fixed sample size (independent of n_paths)
MMD_SEED = 20260710
SIG_SEED = 20260711  # subsample seed for the signature two-sample statistic
SIG_DEPTH = 2        # truncated signature depth (depth-2 on the JOINT (F+1)-dim path)
MMD_BW_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)  # multi-bandwidth mixture
_EPS = 1e-12


# ---------------------------------------------------------------------------
# distance components
# ---------------------------------------------------------------------------

def _w1_per_feature_step(synth: np.ndarray, real: np.ndarray) -> np.ndarray:
    """W1 per feature on pooled per-step increments. (F,) array."""
    F = synth.shape[2]
    return np.array([
        wasserstein_distance(synth[:, :, f].ravel(), real[:, :, f].ravel())
        for f in range(F)
    ])


def _w1_per_feature_cum(synth: np.ndarray, real: np.ndarray) -> np.ndarray:
    """W1 per feature on H-day cumulative returns (sum over the T-step window)."""
    s_cum = synth.sum(axis=1)  # (n, F)
    r_cum = real.sum(axis=1)
    F = synth.shape[2]
    return np.array([
        wasserstein_distance(s_cum[:, f], r_cum[:, f]) for f in range(F)
    ])


def _sliced_w1(synth: np.ndarray, real: np.ndarray, rng: np.random.Generator) -> float:
    """Sliced-W1 over N_PROJ random unit projections of pooled per-step F-dim vectors."""
    F = synth.shape[2]
    dirs = rng.standard_normal((N_PROJ, F))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    s = synth.reshape(-1, F)  # (n*T, F)
    r = real.reshape(-1, F)
    s_proj = s @ dirs.T       # (n*T, N_PROJ)
    r_proj = r @ dirs.T
    return float(np.mean([
        wasserstein_distance(s_proj[:, k], r_proj[:, k]) for k in range(N_PROJ)
    ]))


def _pin_sigma_real(Y: np.ndarray) -> float:
    """Bandwidth PINNED from the REAL sample only: median pairwise Euclidean distance
    WITHIN real. Real is fixed across the field, so every competitor is scored under
    the identical kernel."""
    sq = (Y * Y).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Y @ Y.T)
    np.maximum(D2, 0.0, out=D2)
    iu = np.triu_indices(len(Y), k=1)
    sigma = float(np.median(np.sqrt(D2[iu])))
    return sigma if (np.isfinite(sigma) and sigma > 0) else 1.0


def _subsample(A: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return A[rng.choice(len(A), n, replace=False)] if len(A) > n else A


def _rbf_mmd2_multi_pre(X: np.ndarray, Y: np.ndarray, sigma0: float) -> float:
    """Unbiased multi-bandwidth RBF-MMD^2 (Gretton et al. 2012, Lemma 6) on ALREADY
    sample-sized X:(n,D), Y:(m,D).

    Kernel = sum over sigma in sigma0*MMD_BW_FACTORS of the Gaussian kernel — a mixture
    that does not saturate beyond ~3x. sigma0 is pinned from REAL
    only by the caller (identical kernel for every competitor). Unbiased => expectation
    0 under matching laws."""
    n, m = len(X), len(Y)
    Z = np.concatenate([X, Y], axis=0).astype(np.float64)
    sq = (Z * Z).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(D2, 0.0, out=D2)
    total = 0.0
    for fac in MMD_BW_FACTORS:
        s = sigma0 * fac
        K = np.exp(-D2 / (2.0 * s * s))
        Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
        total += ((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
                  + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
                  - 2.0 * Kxy.mean())
    return float(total)


def _rbf_mmd2_multi(X: np.ndarray, Y: np.ndarray, sigma0: float,
                    rng: np.random.Generator) -> float:
    """Subsample X, Y to a PINNED n = min(len X, len Y, MMD_MAX_N) then unbiased
    multi-bandwidth RBF-MMD^2 (fixed sample size, independent of n_paths).
    sigma0 pinned from REAL only by the caller."""
    n = min(len(X), len(Y), MMD_MAX_N)
    return _rbf_mmd2_multi_pre(_subsample(X, n, rng), _subsample(Y, n, rng), sigma0)


# ---------------------------------------------------------------------------
# truncated depth-2 JOINT-path signatures via Chen's identity (pure numpy)
# ---------------------------------------------------------------------------

def _signatures_depth2(paths: np.ndarray) -> np.ndarray:
    """Depth-2 truncated signatures of piecewise-linear d-dim paths.

    paths: (N, P, d) — P points per path (P-1 linear segments).
    Returns (N, d + d^2) flattened signature levels 1..2.

    Chen's identity for appending a linear segment with increment a
    (segment signature levels a, a(x)a/2):
        S2 <- S2 + S1 (x) a + a(x)a/2
        S1 <- S1 + a
    (update order matters: S2 uses the OLD S1). The antisymmetric part of the
    level-2 block, 0.5*(S2 - S2^T), is the matrix of pairwise Levy areas — the
    order-dependent, cross-channel lead-lag content.
    """
    N, P, d = paths.shape
    dX = np.diff(paths.astype(np.float64), axis=1)   # (N, P-1, d)
    S1 = np.zeros((N, d))
    S2 = np.zeros((N, d, d))
    for t in range(P - 1):
        a = dX[:, t, :]                               # (N, d)
        a2 = 0.5 * np.einsum("ni,nj->nij", a, a)      # a(x)a/2!
        S2 += np.einsum("ni,nj->nij", S1, a) + a2
        S1 += a
    return np.concatenate([S1.reshape(N, -1), S2.reshape(N, -1)], axis=1)


def _sig_features(x: np.ndarray) -> np.ndarray:
    """Per-window signature features of the JOINT time-augmented path.

    x: (n, T, F) own-std-standardized increments. Builds ONE (F+1)-dimensional path per
    window — channel 0 = time (k/T), channels 1..F = cumsum(x)/sqrt(T) — and returns
    its depth-2 signature: (n, (F+1) + (F+1)^2). Because the path is joint over all
    features, the level-2 Levy areas carry cross-asset and lead-lag structure that a
    per-feature signature cannot represent (see module docstring)."""
    n, T, F = x.shape
    t_axis = np.linspace(0.0, 1.0, T + 1)
    cum = np.concatenate([np.zeros((n, 1, F)), np.cumsum(x, axis=1)], axis=1)  # (n,T+1,F)
    cum = cum / np.sqrt(T)
    time_ch = np.broadcast_to(t_axis, (n, T + 1))[..., None]                   # (n,T+1,1)
    paths = np.concatenate([time_ch, cum], axis=2)    # (n, T+1, F+1) JOINT path
    return _signatures_depth2(paths)                  # (n, (F+1)+(F+1)^2)


def _sig_channel_of_coord(F: int) -> np.ndarray:
    """For each of the (F+1)+(F+1)^2 depth-2 signature coords, the set of channels it
    involves (as a boolean (n_coords, F+1) mask). Channel 0 = time, 1..F = features."""
    d = F + 1
    n_coords = d + d * d
    mask = np.zeros((n_coords, d), dtype=bool)
    # level 1: coord k -> channel k
    for k in range(d):
        mask[k, k] = True
    # level 2: coord d + i*d + j -> channels i and j
    for i in range(d):
        for j in range(d):
            c = d + i * d + j
            mask[c, i] = True
            mask[c, j] = True
    return mask


def _sig_mmd2(synth_std: np.ndarray, real_std: np.ndarray,
              rng: np.random.Generator,
              per_feature: bool = False):
    """Distributional signature distance: UNBIASED two-sample MMD^2 (Gretton et al.
    2012, Lemma 6) over the per-window JOINT-path depth-2 signature vectors, with the
    bandwidth PINNED from the REAL signatures only + the multi-bandwidth mixture. Unbiased => an independent resample of the same law scores ~0; the sample size is PINNED to MMD_MAX_N.

    The RBF (not linear/expected-signature) kernel is used deliberately: the linear
    kernel compares only the MEAN signature, which is BLIND to a temporal-order
    scramble that preserves marginals+cross-section (verified: it fails the order
    control in `_self_test`), losing the order-sensitivity. The RBF form is
    the same Lemma-6 unbiased estimator but distributional and order-sensitive.

    Signature coordinates are WHITENED by the REAL panel's per-coordinate std
    (real-only => no synth-scale leakage) AND divided by sqrt(n_coords), so the kernel's
    squared distance is the MEAN per-coordinate whitened squared difference — no single
    feature/level can carry the component and levels 1/2 are equalized.
    If `per_feature`, also returns a (F,) vector: the same pinned-RBF MMD^2
    restricted to the signature coords whose channels involve that feature."""
    n = min(len(synth_std), len(real_std), MMD_MAX_N)
    S = _sig_features(_subsample(synth_std, n, rng))
    R = _sig_features(_subsample(real_std, n, rng))
    sd = R.std(axis=0)
    sd = np.where(sd > _EPS, sd, 1.0)
    D = S.shape[1]
    Sw = S / (sd * np.sqrt(D))     # real-whitened + averaged over coords
    Rw = R / (sd * np.sqrt(D))
    sigma0 = _pin_sigma_real(Rw)   # bandwidth from REAL signatures only
    d = _rbf_mmd2_multi_pre(Sw, Rw, sigma0)
    if not per_feature:
        return d
    F = synth_std.shape[2]
    mask = _sig_channel_of_coord(F)                        # (D, F+1); channel 0 = time
    pf = np.empty(F)
    for f in range(F):
        cols = mask[:, 1 + f]
        Sf, Rf = S[:, cols] / sd[cols], R[:, cols] / sd[cols]
        Sf = Sf / np.sqrt(Sf.shape[1]); Rf = Rf / np.sqrt(Rf.shape[1])
        pf[f] = _rbf_mmd2_multi_pre(Sf, Rf, _pin_sigma_real(Rf))
    return d, pf


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

def _invert(d: float) -> float:
    """Distance -> 0-1 score: 1/(1+d)."""
    return 1.0 / (1.0 + max(d, 0.0))


def _own_std(a: np.ndarray) -> np.ndarray:
    """Pooled per-step per-feature std of a (n,T,F) tensor, floored away from 0."""
    s = a.reshape(-1, a.shape[2]).std(axis=0)
    return np.where(s > _EPS, s, 1.0)


def score(loaded, feature_names=None) -> dict:
    """F4 distributional-distance score for one competitor.

    loaded: list of (synth, real) numpy pairs, each (n_paths, T, F).
    Returns {"mean","std","n","detail"} — mean/std of the combined 0-1 score over
    seed pairs (higher = better); detail carries every raw distance.
    """
    feats = list(feature_names) if feature_names else FEATURES_DEFAULT
    per_pair = []      # combined 0-1 per seed
    comps = ("w1_step", "w1_cum", "sliced_w1", "mmd2_rbf", "sig_mmd2", "disp")
    raw = {k: [] for k in comps}
    w1_step_raw_feat, w1_cum_raw_feat = [], []   # native units, per feature
    disp_feat, sig_feat = [], []                 # per-feature dispersion + sig
    n_dropped = 0

    for si, (synth, real) in enumerate(loaded):
        synth = np.asarray(synth, dtype=np.float64)
        real = np.asarray(real, dtype=np.float64)
        ok = np.isfinite(synth).all(axis=(1, 2))
        n_dropped += int((~ok).sum())
        synth = synth[ok]
        if len(synth) < 2:
            continue

        # native-unit per-feature W1 (reported, not combined)
        w1_step_raw_feat.append(_w1_per_feature_step(synth, real))
        w1_cum_raw_feat.append(_w1_per_feature_cum(synth, real))

        # SHAPE / SCALE separation. Standardize EACH sample by ITS
        # OWN per-feature std => shape distances are scale-invariant (score(c*x) exactly
        # symmetric); dispersion is scored separately by the log-symmetric `disp` term.
        s_scale, r_scale = _own_std(synth), _own_std(real)
        s_std, r_std = synth / s_scale, real / r_scale
        d_disp_feat = np.abs(np.log(s_scale) - np.log(r_scale))   # (F,) |log ratio|
        d_disp = float(d_disp_feat.mean())

        rng = np.random.default_rng(PROJ_SEED + si)
        mmd_rng = np.random.default_rng(MMD_SEED + si)
        sig_rng = np.random.default_rng(SIG_SEED + si)

        d_w1_step = float(_w1_per_feature_step(s_std, r_std).mean())
        d_w1_cum = float(_w1_per_feature_cum(s_std, r_std).mean())
        d_sw1 = _sliced_w1(s_std, r_std, rng)
        # MMD bandwidth PINNED from real only (identical kernel for every competitor)
        r_flat = r_std.reshape(len(r_std), -1)
        sigma0 = _pin_sigma_real(_subsample(r_flat, MMD_MAX_N,
                                            np.random.default_rng(MMD_SEED)))
        d_mmd = _rbf_mmd2_multi(s_std.reshape(len(s_std), -1), r_flat, sigma0, mmd_rng)
        d_sig, pf_sig = _sig_mmd2(s_std, r_std, sig_rng, per_feature=True)

        for k, v in zip(comps, (d_w1_step, d_w1_cum, d_sw1, d_mmd, d_sig, d_disp)):
            raw[k].append(v)
        disp_feat.append(d_disp_feat)
        sig_feat.append(pf_sig)
        per_pair.append(float(np.mean([_invert(d) for d in
                                       (d_w1_step, d_w1_cum, d_sw1, d_mmd, d_sig, d_disp)])))

    if not per_pair:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "detail": {"error": "no usable (synth, real) pairs"}}

    detail = {
        # raw distances (own-std units), averaged over seeds + inverted components
        **{k: float(np.mean(v)) for k, v in raw.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in raw.items()},
        "components_inverted": {k: _invert(float(np.mean(v))) for k, v in raw.items()},
        # native-unit per-feature W1s, averaged over seeds
        "w1_step_per_feature_raw": dict(zip(feats, np.mean(w1_step_raw_feat, axis=0).tolist())),
        "w1_cum_per_feature_raw": dict(zip(feats, np.mean(w1_cum_raw_feat, axis=0).tolist())),
        "disp_per_feature": dict(zip(feats, np.mean(disp_feat, axis=0).tolist())),
        "sig_mmd2_per_feature": dict(zip(feats, np.mean(sig_feat, axis=0).tolist())),
        "n_paths_dropped_nonfinite": n_dropped,
        "n_projections": N_PROJ,
        "mmd_max_n": MMD_MAX_N,
        "mmd_bandwidth_factors": list(MMD_BW_FACTORS),
        "sig_depth": SIG_DEPTH,
        "sig_route": ("numpy Chen depth-2 signatures of the JOINT (F+1)-dim "
                      "time-augmented path; unbiased linear-kernel two-sample MMD "
                      "(iisignature build failed; signatory absent)"),
        "sig_form": ("joint-path unbiased expected-signature MMD (distributional, "
                     "order- and cross-asset-sensitive, real-whitened per-coord)"),
        "scale_handling": ("shape distances on OWN-std-standardized increments "
                           "(scale-invariant); dispersion via log-symmetric disp term"),
    }
    return {"mean": float(np.mean(per_pair)), "std": float(np.std(per_pair)),
            "n": len(per_pair), "detail": detail}


# ---------------------------------------------------------------------------
# self-test + CLI
# ---------------------------------------------------------------------------

def _self_test():
    """Verify the Chen recursion, the scale-symmetry regression, the unbiased-estimator
    independent-resample zero, and the order/joint signature sensitivity."""
    rng = np.random.default_rng(0)
    path = rng.standard_normal((1, 6, 3)).cumsum(axis=1)
    got = _signatures_depth2(path)[0]
    dx = np.diff(path[0], axis=0)
    # brute force: compose segments sequentially with explicit tensor algebra.
    d = 3
    S1 = np.zeros(d); S2 = np.zeros((d, d))
    for a in dx:
        a2 = np.tensordot(a, a, 0) / 2.0
        S2 = S2 + np.tensordot(S1, a, 0) + a2
        S1 = S1 + a
    ref = np.concatenate([S1.ravel(), S2.ravel()])
    assert np.allclose(got, ref), "vectorized Chen recursion mismatch"
    # analytic check on a single straight segment: S = (a, a⊗a/2)
    a = np.array([0.3, -1.1])
    seg = np.stack([np.zeros(2), a])[None]
    got1 = _signatures_depth2(seg)[0]
    ref1 = np.concatenate([a, (np.tensordot(a, a, 0) / 2).ravel()])
    assert np.allclose(got1, ref1), "single-segment signature mismatch"
    # Lévy area is order-sensitive: the two orderings of a 2-segment path must differ
    # in the antisymmetric part of level 2, but share the endpoint (level 1).
    p_fwd = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])  # right then up
    p_rev = np.array([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # up then right
    s_fwd, s_rev = _signatures_depth2(p_fwd)[0], _signatures_depth2(p_rev)[0]
    assert np.allclose(s_fwd[:2], s_rev[:2]), "endpoints (level 1) should match"
    assert not np.allclose(s_fwd, s_rev), "Lévy area must distinguish orderings"

    # identical samples => (near-)zero distances
    x = rng.standard_normal((50, 60, 7))
    s = score([(x, x.copy())])
    assert s["mean"] > 0.999, f"self-distance should score ~1, got {s['mean']}"

    # --- control: SCALE SYMMETRY. score(c*x) == score(x/c) exactly, for real ---
    xr = rng.standard_normal((200, 60, 7)) * (1.0 + rng.random(7))  # heteroscale feats
    for c in (0.5, 0.25, 1.0 / 3.0, 2.0 / 3.0):
        su = score([(xr * c, xr.copy())])["mean"]
        so = score([(xr / c, xr.copy())])["mean"]
        assert abs(su - so) < 1e-6, (
            f"F4 not scale-symmetric at c={c}: under={su:.6f} over={so:.6f}")
    # and shape components are literally scale-invariant (== faithful) for pure rescale
    d_faith = score([(xr, xr.copy())])["detail"]
    d_scaled = score([(xr * 3.0, xr.copy())])["detail"]
    for k in ("w1_step", "w1_cum", "sliced_w1", "mmd2_rbf", "sig_mmd2"):
        assert abs(d_faith[k] - d_scaled[k]) < 1e-6, f"{k} not scale-invariant"
    assert d_scaled["disp"] > 1.0, "disp must charge a pure 3x rescale"

    # --- control: UNBIASED => independent resample of the SAME dist scores ~0 ---
    base = rng.standard_normal((400, 60, 7))
    A = base[:200]; B = base[200:]            # two independent draws, same law
    sc = _own_std(A)
    d_indep = _sig_mmd2(A / sc, B / sc, np.random.default_rng(3))
    assert abs(d_indep) < 5e-3, f"independent resample sig_mmd2 should be ~0, got {d_indep}"
    d_mmd_indep = _rbf_mmd2_multi(
        (A / sc).reshape(200, -1), (B / sc).reshape(200, -1),
        _pin_sigma_real((B / sc).reshape(200, -1)), np.random.default_rng(4))
    assert abs(d_mmd_indep) < 0.05, f"independent resample mmd should be ~0, got {d_mmd_indep}"

    # ORDER + JOINT SENSITIVITY CONTROLS. Build a panel with strong
    # within-window temporal SHAPE (up/flat/down drift profile) AND strong
    # contemporaneous cross-asset correlation (a shared per-step factor).
    T2, F2 = 60, 7
    shape = np.concatenate([np.ones(T2 // 3), np.zeros(T2 - 2 * (T2 // 3)),
                            -np.ones(T2 // 3)])                       # (T,)
    amp = rng.standard_normal((200, 1, F2)) * 2.0
    step_fac = rng.standard_normal((200, T2, 1))                      # shared across feats
    real = (0.4 * rng.standard_normal((200, T2, F2))
            + amp * shape[None, :, None]
            + 1.2 * step_fac)
    srng = np.random.default_rng(7)

    # (A) TEMPORAL scramble: same permutation per path across features => per-step
    # marginals AND the contemporaneous cross-section preserved EXACTLY, only temporal
    # order destroyed. A genuine path statistic must now score this worse.
    scr = np.empty_like(real)
    for i in range(real.shape[0]):
        scr[i] = real[i, srng.permutation(T2), :]
    assert np.allclose(np.sort(real.reshape(-1, F2), 0),
                       np.sort(scr.reshape(-1, F2), 0)), \
        "temporal scramble must preserve per-step marginals exactly"

    # (B) JOINT scramble: permute the PATH index independently per feature, keeping the
    # time order => each feature's full marginal AND its temporal dynamics are intact,
    # only the contemporaneous cross-asset dependence is destroyed. This is the
    # perturbation the OLD per-marginal signature was EXACTLY blind to (it scored 0.0);
    # the joint-path signature must now see it.
    jnt = np.empty_like(real)
    for f in range(F2):
        jnt[:, :, f] = real[srng.permutation(real.shape[0]), :, f]
    assert np.allclose(np.sort(real.reshape(-1, F2), 0),
                       np.sort(jnt.reshape(-1, F2), 0)), \
        "joint scramble must preserve per-step marginals exactly"

    s_faithful = score([(real, real.copy())])
    d_faithful = s_faithful["detail"]["sig_mmd2"]
    d_temporal = score([(scr, real.copy())])["detail"]["sig_mmd2"]
    s_joint = score([(jnt, real.copy())])
    d_joint = s_joint["detail"]["sig_mmd2"]
    assert d_temporal > d_faithful + 0.03, (
        f"signature component not order-sensitive: faithful sig_mmd2={d_faithful:.4f} "
        f"temporal-scramble sig_mmd2={d_temporal:.4f}")
    assert d_joint > d_faithful + 0.005, (
        f"signature component not joint-sensitive: faithful sig_mmd2={d_faithful:.4f} "
        f"joint-scramble sig_mmd2={d_joint:.4f}")
    assert s_joint["mean"] < s_faithful["mean"] - 0.005, (
        f"destroying cross-asset dependence must cost F4 score: "
        f"faithful={s_faithful['mean']:.4f} joint-scramble={s_joint['mean']:.4f}")


if __name__ == "__main__":
    from benchmark.registry import available_competitors

    _self_test()
    print("self-test OK\n")

    rows = []
    for comp in available_competitors():
        res = score(comp.load())
        rows.append((comp.name, res))
        d = res["detail"]
        print(f"{comp.name:18s} mean={res['mean']:.4f} std={res['std']:.4f} n={res['n']}  "
              f"[w1_step={d['w1_step']:.4f} w1_cum={d['w1_cum']:.4f} "
              f"sliced_w1={d['sliced_w1']:.4f} mmd2={d['mmd2_rbf']:.4f} "
              f"sig_mmd2={d['sig_mmd2']:.4f} disp={d['disp']:.4f}]")

    rows.sort(key=lambda r: r[1]["mean"], reverse=True)
    print("\n=== F4 distributional distances — combined 0-1 (higher = better) ===")
    print(f"{'rank':>4}  {'competitor':18s} {'mean':>8} {'std':>8} {'n':>3}")
    for i, (name, res) in enumerate(rows, 1):
        print(f"{i:>4}  {name:18s} {res['mean']:>8.4f} {res['std']:>8.4f} {res['n']:>3}")
