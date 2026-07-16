"""Memorization / OOS-copy guard.

The public repo ships the exact OOS scoring tensor (``registry.CANON_REAL``),
so without a gate a submitter could copy (or lightly warp / time-shuffle /
block-bootstrap) the scoring reference and top every board — a copied
reference outscores every honest generator on most tasks by construction.
Train-side NN distance alone does not catch these OOS-copy strategies, so
this module is the hard gate the runner wires in front of ranking. The
COPYING VERDICT is decided ONLY by the two copy-specific checks (b) and (c);
check (a) is a reported DIAGNOSTIC that does NOT gate (see the rationale
below).

(a) TRAIN-CLOSENESS DIAGNOSTIC — DISPERSION, NOT COPYING (reported, NEVER
    gates). This "train-check ratio" is a DISPERSION statistic normalised by a
    regime-shift baseline, not a copying statistic, so it informs but does not
    gate. Its construction (the train-boundary NN-distance protocol):
    sliding 60-day windows are built
    from the panel's train split (``sablier_flow.demo_data`` <= 2019-12-31, same
    return construction and windowing as the reference); the mean NN distance of
    synth windows to train windows is compared with the mean NN distance of real
    OOS windows to train windows, ``ratio = d_syn_to_train / d_oos_to_train``.
    WHY IT WAS DEMOTED: a provably non-copying iid Gaussian (stores only a
    7-vector mean + 7x7 covariance, no data) earns "MEMORISATION" on this ratio
    the instant its covariance is shrunk below the (more volatile) OOS regime —
    verified: shrink 0.6 -> ratio 0.56 (would-be SUSPICIOUS), shrink 0.25 ->
    ratio 0.34 (would-be MEMORISATION), both HEALTHY on (b)+(c). The ratio
    penalises faithful, under-dispersed train reproduction and rewards
    over-dispersion; under-dispersion is a QUALITY failure that finval / F1 / F5
    / tail already price. It is reported as a dispersion band for information
    only and mapped to a dispersion label, never to a copy severity:
        ratio <  0.50  -> "severely under-dispersed"   (diagnostic)
        ratio <  0.85  -> "under-dispersed"            (diagnostic)
        ratio <= 1.15  -> "dispersion balanced"        (diagnostic)
        ratio >  1.15  -> "over-dispersed / regime-shifted" (diagnostic)
    True TRAIN memorization (verbatim replay of specific train WINDOWS) is a
    near-zero-distance/duplicate phenomenon, NOT a low-ratio one; it is not
    gated here because it cannot game the OOS board (scoring is against OOS; the
    regime shift penalises it and finval prices the resulting under-dispersion),
    and an absolute-band train-duplicate DQ would falsely erase legitimate
    historical-resampling baselines (Historical-Sim / Block-Bootstrap / FHS,
    which honestly replay train history).

(b) OOS-COPY (the copy-specific check train-side NN distance cannot see):
    synth windows are compared against the CANON REAL windows — the public
    scoring tensor itself — in two geometries:
      * RAW window space (catches verbatim / lightly-warped copies);
      * SORTED-per-window space: each window's per-feature return set sorted
        along time, an order-invariant fingerprint (catches TIME-SHUFFLED
        copies, whose raw distance looks honest but whose sorted distance
        is ~0).
    A window counts as a DUPLICATE when its NN distance is inside the
    near-zero band ``DUP_BAND_FRAC x median(real-to-real self-NN distance)``
    of the same geometry. This is deliberately an ABSOLUTE near-zero band
    rather than a low quantile of the real-to-real self-NN distribution:
    the canon windows are overlapping rolling windows whose self-NN
    distances are large, and honestly under-dispersed generators (measured:
    KoVAE 26%, TimeVAE 100% of windows) fall below the 0.5th-percentile
    quantile without copying anything, while every copy/warp attack sits
    ~1000x closer. Calibration on the archived field: honest minima are
    >= 0.47 x median (raw) and >= 1.2 x DUP band (sorted); verbatim and
    shuffled copies sit at < 0.001 x median.
        duplicate fraction > COPY_DUP_FRAC (5%)  -> COPY
        duplicate fraction > SUSP_DUP_FRAC (1%)  -> SUSPICIOUS
    Additionally a BLOCK-level scan catches stitched near-copies (e.g. a
    block-bootstrap of the real tensor + 5% noise, which defeats whole-window
    distances): every synth 20-day sub-block is matched against all real
    20-day sub-blocks; a synth set whose blocks sit at a fraction
    < BLOCK_CLOSE_FRAC_OF_BASELINE (0.35) of the fresh-real baseline distance
    (train blocks vs canon blocks) for more than BLOCK_SUSP_FRAC (25%) of its
    blocks is at least SUSPICIOUS. Measured margins: fresh-real and honest
    FLOW blocks flag 0%, a 3x20d block-bootstrap of canon + 5% noise flags
    100% (median block distance 0.52 vs honest 4.25).

Final verdict = the worst of the COPY-SPECIFIC checks (b) and (c) with severity
    COPY > MEMORISATION > SUSPICIOUS > HEALTHY.
Check (a) contributes a dispersion diagnostic string only; it never raises the
verdict above HEALTHY.

Interface:
    memorization_guard(name, loaded, canon_real)
        -> {"verdict": "HEALTHY"|"SUSPICIOUS"|"MEMORISATION"|"COPY",
            "detail": str}
where ``loaded`` is the registry's list of (synth, real) pairs for ONE
competitor and ``canon_real`` is the canonical scoring tensor (array or path).
Synth tensors are pooled across seeds. All statistics are deterministic (no
RNG); panel windows are cached per process.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

PANEL_NAME = "us_equities_macro_2010_2023"
HORIZON = 60
TRAIN_END = "2019-12-31"  # train split boundary (protocol constant)
FEATURE_TYPES = {
    "IWM": "price", "QQQ": "price", "SPY": "price", "TLT": "price",
    "VIX": "level", "TNX": "level", "DXY": "price",
}

# -- check (a): TRAIN-CLOSENESS DIAGNOSTIC bands (DISPERSION, not copying) -----
# These do not gate the verdict (dispersion diagnostic). They map the
# train-check ratio to a human-readable DISPERSION label only. Kept as constants
# so the reported diagnostic stays comparable with historical logs.
TRAIN_SEVERE_UNDERDISP_RATIO = 0.50  # was TRAIN_MEMORISATION_RATIO (demoted)
TRAIN_UNDERDISP_RATIO = 0.85         # was TRAIN_SUSPICIOUS_RATIO (demoted)
TRAIN_BALANCED_RATIO = 1.15          # above this = over-dispersed / regime shift

# -- check (b): OOS-COPY thresholds (calibration in the module docstring) ----
DUP_BAND_FRAC = 0.25         # duplicate = NN dist < this x median(r2r self-NN)
COPY_DUP_FRAC = 0.05         # >5% duplicate windows           -> COPY
SUSP_DUP_FRAC = 0.01         # >1% duplicate windows           -> SUSPICIOUS
BLOCK_LEN = 20               # sub-window length for the stitched-copy scan
BLOCK_QUERY_STRIDE = 20      # synth blocks: non-overlapping
BLOCK_GALLERY_STRIDE = 1     # real blocks: every offset
BLOCK_CLOSE_FRAC_OF_BASELINE = 0.35  # "close" = < this x fresh-real baseline med
BLOCK_SUSP_FRAC = 0.25       # >25% close blocks               -> SUSPICIOUS
TRAIN_BASELINE_WINDOW_STRIDE = 6     # train subsample for the block baseline

# -- check (c): VERBATIM SHORT-SUBSEQUENCE replay against the OOS reference -----
# A block-bootstrap of the PUBLIC scoring tensor tops every fidelity board (F1=0.97)
# yet evades checks (a)+(b): no full 60-step window is copied (window-level NN sees
# nothing) and no exact 60-step duplicate exists. But it splices REAL sub-blocks, so
# short synth subsequences are byte-identical to the reference. Honest continuous
# generators produce ~0% verbatim k-step matches (float coincidence is ~impossible);
# any resampling of the OOS tensor produces tens of %. Measured 2026-07-13: every
# archived generator + the TRAIN-based bootstraps = 0.0%; the OOS-bootstrap attack
# = 54.9%. So any threshold in (0, 0.5) is safe. (TRAIN resamplers score 0 here
# because they replay the train window, not the OOS scoring tensor — as they should.)
VERBATIM_K = 10              # subsequence length probed for exact match
VERBATIM_N_PROBE = 3000      # random synth subsequences sampled
VERBATIM_COPY_FRAC = 0.20    # >20% verbatim -> COPY (replaying the OOS reference)
VERBATIM_SUSP_FRAC = 0.02    # >2% verbatim  -> SUSPICIOUS

_SEVERITY = {"HEALTHY": 0, "SUSPICIOUS": 1, "MEMORISATION": 2, "COPY": 3}


# ---------------------------------------------------------------------------
# window construction (train / OOS) — the train-boundary NN-distance protocol
# ---------------------------------------------------------------------------

def _to_returns(df):
    import pandas as pd
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    for c in df.columns:
        if FEATURE_TYPES[c] == "price":
            out[c] = np.log(df[c]).diff()
        else:
            out[c] = df[c].diff()
    return out.dropna().astype(np.float32)


def _sliding_windows(arr: np.ndarray, L: int = HORIZON) -> np.ndarray:
    T = arr.shape[0]
    return np.stack([arr[i:i + L] for i in range(T - L + 1)], axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def _train_oos_windows() -> tuple[np.ndarray, np.ndarray]:
    """(train_windows, oos_windows), the reference protocol's split."""
    import pandas as pd
    import sablier_flow
    df = sablier_flow.demo_data(PANEL_NAME)
    df.index = pd.to_datetime(df.index)
    train = df.loc[df.index <= TRAIN_END]
    oos = df.loc[df.index > TRAIN_END]
    return (_sliding_windows(_to_returns(train).values),
            _sliding_windows(_to_returns(oos).values))


# ---------------------------------------------------------------------------
# distance machinery
# ---------------------------------------------------------------------------

def _flat(w: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(w.reshape(w.shape[0], -1), dtype=np.float32)


def _flat_sorted(w: np.ndarray) -> np.ndarray:
    """Order-invariant fingerprint: sort each window along TIME per feature."""
    return np.ascontiguousarray(
        np.sort(w, axis=1).reshape(w.shape[0], -1), dtype=np.float32)


def _nn_distances(query: np.ndarray, gallery: np.ndarray,
                  chunk: int = 512) -> np.ndarray:
    """Per-row Euclidean NN distance from query rows to gallery rows."""
    g_norm = (gallery ** 2).sum(axis=1)
    out = np.empty(query.shape[0], dtype=np.float32)
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        d2 = (q ** 2).sum(axis=1, keepdims=True) + g_norm[None, :] - 2.0 * q @ gallery.T
        out[i:i + chunk] = np.sqrt(np.maximum(d2, 0.0).min(axis=1))
    return out


def _self_nn_distances(g: np.ndarray) -> np.ndarray:
    g_norm = (g ** 2).sum(axis=1)
    d2 = g_norm[:, None] + g_norm[None, :] - 2.0 * g @ g.T
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.maximum(d2, 0.0).min(axis=1))


def _blocks(w: np.ndarray, stride: int) -> np.ndarray:
    """All BLOCK_LEN-day sub-blocks of (n, HORIZON, F) windows, flattened."""
    parts = [w[:, s:s + BLOCK_LEN, :].reshape(w.shape[0], -1)
             for s in range(0, w.shape[1] - BLOCK_LEN + 1, stride)]
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


# ---------------------------------------------------------------------------
# cached canon-side reference statistics
# ---------------------------------------------------------------------------

_CANON_STATS_CACHE: dict = {}


def _canon_stats(canon: np.ndarray) -> dict:
    key = (canon.shape, canon.dtype.str, hash(canon.tobytes()[:4096]),
           float(canon.sum()))
    if key in _CANON_STATS_CACHE:
        return _CANON_STATS_CACHE[key]
    train_w, _ = _train_oos_windows()
    raw = _flat(canon)
    srt = _flat_sorted(canon)
    block_gallery = _blocks(canon, BLOCK_GALLERY_STRIDE)
    # fresh-real baseline for block distances: TRAIN blocks vs canon blocks
    train_blocks = _blocks(train_w[::TRAIN_BASELINE_WINDOW_STRIDE],
                           BLOCK_QUERY_STRIDE)
    stats = {
        "raw": raw,
        "sorted": srt,
        "r2r_med_raw": float(np.median(_self_nn_distances(raw))),
        "r2r_med_sorted": float(np.median(_self_nn_distances(srt))),
        "block_gallery": block_gallery,
        "block_baseline_med": float(np.median(
            _nn_distances(train_blocks, block_gallery))),
    }
    _CANON_STATS_CACHE[key] = stats
    return stats


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

def _train_check(synth_flat: np.ndarray) -> tuple[str, str, dict]:
    """Check (a) DIAGNOSTIC (does NOT gate): NN-distance-to-TRAIN ratio of
    the train-boundary NN-distance protocol, reported as a DISPERSION label.

    Returns ("HEALTHY", msg, {...}) unconditionally — the copying verdict is
    owned by checks (b)+(c). The dispersion label is informational: it measures
    how tightly synth windows sit against the train manifold relative to the
    (more volatile) OOS regime, which is a QUALITY signal finval/F1/F5/tail
    already score, NOT evidence of copying."""
    train_w, oos_w = _train_oos_windows()
    train_flat = _flat(train_w)
    d_oos = _nn_distances(_flat(oos_w), train_flat)
    d_syn = _nn_distances(synth_flat, train_flat)
    ratio = float(d_syn.mean() / d_oos.mean())
    if ratio < TRAIN_SEVERE_UNDERDISP_RATIO:
        disp = "severely under-dispersed"
    elif ratio < TRAIN_UNDERDISP_RATIO:
        disp = "under-dispersed"
    elif ratio <= TRAIN_BALANCED_RATIO:
        disp = "dispersion balanced"
    else:
        disp = "over-dispersed / regime-shifted"
    msg = (f"train-closeness DIAGNOSTIC (dispersion, NOT copying) ratio={ratio:.3f} "
           f"(d_syn_to_train={float(d_syn.mean()):.3f} / "
           f"d_oos_to_train={float(d_oos.mean()):.3f}) -> {disp} "
           f"[non-gating; scored by finval]")
    # Verdict is fixed HEALTHY so this check can never raise the final verdict.
    return "HEALTHY", msg, {"ratio": ratio, "dispersion": disp}


def _oos_copy_check(synth: np.ndarray, canon: np.ndarray) -> tuple[str, str, dict]:
    """Check (b): duplicate/NN scan against the public OOS scoring tensor."""
    cs = _canon_stats(canon)

    d_raw = _nn_distances(_flat(synth), cs["raw"])
    d_srt = _nn_distances(_flat_sorted(synth), cs["sorted"])
    dup_raw = float((d_raw < DUP_BAND_FRAC * cs["r2r_med_raw"]).mean())
    dup_srt = float((d_srt < DUP_BAND_FRAC * cs["r2r_med_sorted"]).mean())
    dup = max(dup_raw, dup_srt)

    d_blk = _nn_distances(_blocks(synth, BLOCK_QUERY_STRIDE), cs["block_gallery"])
    blk_close = float(
        (d_blk < BLOCK_CLOSE_FRAC_OF_BASELINE * cs["block_baseline_med"]).mean())

    if dup > COPY_DUP_FRAC:
        verdict = "COPY"
    elif dup > SUSP_DUP_FRAC or blk_close > BLOCK_SUSP_FRAC:
        verdict = "SUSPICIOUS"
    else:
        verdict = "HEALTHY"
    msg = (f"oos-copy-check dup_frac raw={dup_raw:.1%} sorted={dup_srt:.1%} "
           f"(band {DUP_BAND_FRAC} x r2r-med raw={cs['r2r_med_raw']:.3f}/"
           f"sorted={cs['r2r_med_sorted']:.3f}), "
           f"close-block frac={blk_close:.1%} "
           f"(< {BLOCK_CLOSE_FRAC_OF_BASELINE} x baseline "
           f"{cs['block_baseline_med']:.3f}) -> {verdict}")
    return verdict, msg, {"dup_frac_raw": dup_raw, "dup_frac_sorted": dup_srt,
                          "block_close_frac": blk_close}


def _verbatim_subseq_check(synth: np.ndarray, canon: np.ndarray,
                           rng: np.random.Generator) -> tuple[str, str, dict]:
    """Check (c): fraction of synth K-step subsequences that are byte-identical to a
    subsequence of the OOS reference. Catches block-bootstrap / spliced replay of the
    public scoring tensor, which evades the window-level checks (a)+(b)."""
    k = VERBATIM_K
    H = canon.shape[1]
    if H < k:
        return "HEALTHY", f"verbatim-check n/a (H={H} < k={k})", {"verbatim_frac": 0.0}
    ref = set()
    for w in canon:
        for t in range(H - k + 1):
            ref.add(w[t:t + k].tobytes())
    n = min(VERBATIM_N_PROBE, synth.shape[0] * (H - k + 1))
    hits = 0
    for _ in range(n):
        i = int(rng.integers(0, synth.shape[0])); t = int(rng.integers(0, H - k + 1))
        if synth[i, t:t + k].tobytes() in ref:
            hits += 1
    frac = hits / n if n else 0.0
    if frac > VERBATIM_COPY_FRAC:
        verdict = "COPY"
    elif frac > VERBATIM_SUSP_FRAC:
        verdict = "SUSPICIOUS"
    else:
        verdict = "HEALTHY"
    msg = (f"verbatim-{k}step frac={frac:.1%} (COPY>{VERBATIM_COPY_FRAC:.0%} / "
           f"SUSP>{VERBATIM_SUSP_FRAC:.0%}; honest generators ~0%) -> {verdict}")
    return verdict, msg, {"verbatim_frac": frac}


def memorization_guard(name: str, loaded, canon_real) -> dict:
    """Run both checks for one competitor; return the worst verdict.

    Args:
        name: competitor's public label (for the detail string only).
        loaded: list of (synth, real) numpy pairs, as returned by
            ``Competitor.load()``; synth tensors are pooled across seeds.
        canon_real: the canonical OOS scoring tensor — an (n, HORIZON, F)
            array or a path to the ``.npy``.

    Returns:
        {"verdict": "HEALTHY"|"SUSPICIOUS"|"MEMORISATION"|"COPY",
         "detail": str}
    """
    if isinstance(canon_real, (str, bytes)):
        canon_real = np.load(canon_real)
    canon = np.asarray(canon_real, dtype=np.float32)

    synths = [np.asarray(s, dtype=np.float32) for s, _ in loaded]
    if not synths:
        return {"verdict": "SUSPICIOUS",
                "detail": f"{name}: no synth tensors loaded — cannot audit"}
    synth = np.concatenate(synths, axis=0)

    _v_a, msg_a, _ = _train_check(_flat(synth))  # diagnostic only; does NOT gate
    v_b, msg_b, _ = _oos_copy_check(synth, canon)
    v_c, msg_c, _ = _verbatim_subseq_check(synth, canon, np.random.default_rng(0))

    # Copying verdict = worst of the COPY-SPECIFIC checks only (the
    # dispersion rationale above). Check (a) is a diagnostic, appended to
    # the detail but never contributing to the verdict.
    verdict = max((v_b, v_c), key=_SEVERITY.__getitem__)
    detail = (f"{name}: {verdict} | {msg_a} | {msg_b} | {msg_c} | "
              f"n_synth_windows={synth.shape[0]} (pooled over {len(synths)} seeds)")
    return {"verdict": verdict, "detail": detail}


if __name__ == "__main__":
    from benchmark.registry import available_competitors, CANON_REAL

    canon = np.load(CANON_REAL)
    print("Memorization / OOS-copy guard over the archived field:\n")
    for comp in available_competitors():
        try:
            loaded = comp.load()
        except Exception as e:  # e.g. registry tensor-sanity gate (F-1)
            print(f"  {comp.name:18s} {'N/A':12s} not auditable: "
                  f"{type(e).__name__}: {e}")
            continue
        res = memorization_guard(comp.name, loaded, canon)
        print(f"  {comp.name:18s} {res['verdict']:12s} {res['detail']}")
