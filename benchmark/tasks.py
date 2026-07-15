"""Task registry.

A task scores a competitor's archived paths and returns (score_mean, score_std,
n) or None if the task cannot be run for that competitor (logged as a coverage
gap, never silently dropped). `higher_better` fixes the ranking direction.

The suite ships seven tasks — four fidelity (F1, F2, F4, F5) and three utility
(T2, T3, T5) — each with its metric and ranking direction fixed. A task returns
None for any competitor it cannot score (a logged coverage gap, never a silent
drop).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np


class ScorerError:
    """Sentinel returned by Task.score when the scorer RAISED (or produced a
    non-finite mean). Distinct from None (= task inapplicable / not wired):
    the runner counts a ScorerError as a MISSING cell — audit F-04/AGG-4 —
    so a crash can never improve a competitor's aggregate, and F-03's
    NaN-rank-poisoning path is closed before ranking."""

    def __init__(self, error_class, detail):
        self.error_class = error_class
        self.detail = detail

    def __repr__(self):
        return f"ScorerError({self.error_class}: {self.detail})"


class Task:
    def __init__(self, tid, name, unit, higher_better, scorer=None, note=None):
        self.tid = tid
        self.name = name
        self.unit = unit
        self.higher_better = higher_better
        self._scorer = scorer
        self.note = note   # optional board annotation (e.g. T4's regen-pending status)

    def score(self, competitor):
        """Return (mean, std, n), None (task inapplicable — a legitimate
        coverage gap), or ScorerError (scorer raised / non-finite output —
        counts as a missing cell, printed with its error class)."""
        if self._scorer is None:
            return None
        try:
            r = self._scorer(competitor)
        except Exception as e:  # scorer raised => missing cell, never a waiver
            print(f"    [{self.tid}] {competitor.name}: SCORER ERROR "
                  f"({type(e).__name__}: {e})")
            return ScorerError(type(e).__name__, str(e))
        if r is None:
            return None
        if not np.isfinite(r[0]):  # F-03: non-finite mean must not reach ranking
            print(f"    [{self.tid}] {competitor.name}: SCORER ERROR "
                  f"(NonFiniteScore: mean={r[0]!r})")
            return ScorerError("NonFiniteScore", f"mean={r[0]!r}")
        return r


# ---- F1: synthetic-data quality — finval 0.6.1 pooled overall_score ----------
# (18 WEIGHTED metrics of 32 run; NO weight on the 14 diagnostics incl. c2st,
# far_tail_quantiles, hill_tail_index, coskewness, signature_distance,
# long_memory, variogram_score) — GATE-PENALIZED.
#
# F1-GATE-3: finval defines a set of HARD-GATE metrics on which a "poor" verdict
# is UN-averageable — under finval's own semantics one poor hard gate forces
# overall_quality="poor" regardless of how high the weighted mean is.
# `validate_paths` publishes ONLY the weighted overall_score and applies NONE of
# these gates, so a competitor that fails a tail / memorization / drawdown gate
# still posts a high headline. We enforce the gates HERE so the F1 board cannot
# rank a model that badly fails a gate above one that passes cleanly.
#
# Enforcement (F1-GATE-3b, 2026-07-15 — the CONTINUOUS penalty). The gate PRINCIPLE
# is preserved: a model that fails a hard gate cannot rank as if it passed. But the
# original implementation applied the gate as a HARD, per-SEED cut — a full score-band
# (1.0) subtracted from any seed that failed any computable gate. Because a metric
# sitting NEAR its threshold flips quality between 'acceptable' and 'poor' from one
# seed's generation noise to the next, that binary cut made a borderline model FLICKER
# between ~overall and ~overall−1 across seeds, producing a bimodal board: clean models
# at the top, then a cliff to 0.05–0.45 rows carrying ENORMOUS ±0.5 seed stds (a
# coin-flip, not generation variance) and meaningless CIs — exactly what a skeptical
# referee attacks. This is a PRESENTATION artifact, not a property of the models.
#
# We replace the binary cut with a CONTINUOUS, MONOTONE penalty on the gate MARGIN,
# applied to the PER-SEED score BEFORE averaging. For each COMPUTABLE gate metric that
# is 'poor' on a seed (value >= its 'acceptable' threshold — all finval gate metrics
# are lower-is-better with 0<=excellent<good<acceptable), we standardize the violation
# in units of the metric's OWN band width — the SAME severity scale finval uses for its
# within-grade `MetricResult.score` decay (scale = acceptable − excellent):
#     z = max(0, (value − acceptable) / (acceptable − excellent))     # 0 at the boundary
#     penalty_metric = exp(−LN2 · z)                                  # 1 at the boundary
# and the seed score is  overall_score · Πgates penalty_metric  (floored, below).
# Properties, all verified on the field (see the F1-GATE-3b proof in the audit trail):
#   • A metric INSIDE its gate contributes penalty 1 — a model that passes ALL gates is
#     UNCHANGED (FLOW-H, FLOW-P2 identical to before, penalty ≡ 1). No cliff at the
#     boundary: penalty is continuous there (=1), so a seed straddling the threshold no
#     longer swings by 1.0 — the ±0.5 seed stds collapse to genuine generation variance.
#   • MONOTONE + graded: worse violation → strictly larger demotion. LN2 calibrates it so
#     a full-band violation (z=1) HALVES the score; a metric far past the gate decays
#     toward — but never reaches — 0, so severity keeps registering (a mild fail and a
#     severe fail get PROPORTIONALLY different penalties, not both ~0).
#   • Ordering is preserved for real failures: a model that FAILS a gate is always demoted
#     (penalty<1), and on the panel every gate-failing model ranks below every clean one.
#     It is not a new gaming vector — the only way to lift a demoted score is to stop
#     failing the gate, exactly the intended behaviour. Scores now stay on finval's [0,1]
#     scale (a gated model lands at a REDUCED positive score, not a negative band), so the
#     row is directly comparable to the honest real-vs-real noise floor.
#
# c2st is EXCLUDED from the applied gate on this panel. c2st is finval-
# SAMPLE_HUNGRY (needs >=50 INDEPENDENT real windows); the v1 panel's 200
# reference paths are stride-1 OVERLAPPING windows cut from one ~1004-day series
# => only ~16 truly-independent windows. finval stamps `effective_n` with the
# count it was GIVEN (200, overlapping) and thus reports c2st `assessable=True`,
# but independence is the CALLER's responsibility, so we treat c2st as
# NOT-ASSESSABLE here and never gate on it (it is zero-weight in the headline
# anyway, so excluding it does not move overall_score). The other four gate
# metrics ARE nonzero-weight and computable on this panel.
_FINVAL_GATE_METRICS = (
    "memorization", "tail_quantiles", "tail_dependence_lower", "drawdown_distribution",
)
# metric -> reason it is NOT gated on the v1 panel (published for referees).
_FINVAL_GATE_NOT_ASSESSABLE = {
    "c2st": "SAMPLE_HUNGRY: ~16 independent windows < finval's 50 "
            "(effective_n=200 counts overlapping stride-1 windows)",
}
_GATE_LAMBDA = float(np.log(2.0))  # z=1 (one-band violation) halves the score
_GATE_PENALTY_FLOOR = 0.02         # product floor: worst case is small, never exactly 0
_GATE_FALLBACK_PENALTY = 0.5       # a 'poor' gate lacking usable thresholds → one-band demotion


def _gate_penalty(metric):
    """Continuous demotion factor in (0, 1] for one finval gate MetricResult.

    Returns (penalty, detail) where detail is None if the metric does not gate
    (penalty 1.0) or a dict {value, threshold, z, penalty} disclosing the applied
    demotion. A metric only gates when it is applicable, assessable AND 'poor'
    (value >= its 'acceptable' threshold) — this preserves the exact set of gates
    the binary cut used (c2st excluded upstream); we change only HOW a failing
    gate maps to a score: a smooth exp(-LN2·z) in the band-standardized margin,
    not a coin-flip band. Passing/inapplicable/not-assessable → penalty 1.0."""
    if metric is None:
        return 1.0, None
    if not (metric.applicable and metric.assessable) or metric.quality != "poor":
        return 1.0, None
    th = metric.thresholds or {}
    try:
        e, a = float(th["excellent"]), float(th["acceptable"])
    except (KeyError, TypeError, ValueError):
        e = a = None
    v = float(metric.value)
    # Fallback: a failing gate whose thresholds are missing/non-standard (can't form a
    # margin) still gets a fixed, documented demotion so the gate keeps its teeth.
    if a is None or not (0.0 <= e < a):
        return _GATE_FALLBACK_PENALTY, {"value": v, "threshold": a, "z": None,
                                        "penalty": _GATE_FALLBACK_PENALTY}
    if not np.isfinite(v):  # total blow-up (inf value) → strongest per-metric demotion
        return _GATE_PENALTY_FLOOR, {"value": v, "threshold": a, "z": float("inf"),
                                     "penalty": _GATE_PENALTY_FLOOR}
    z = max(0.0, (v - a) / (a - e))       # violation in units of the metric's OWN band
    p = float(np.exp(-_GATE_LAMBDA * z))
    return p, {"value": v, "threshold": a, "z": z, "penalty": p}


def _seed_gate_penalty(report):
    """Product of the per-gate continuous penalties for one seed's finval report,
    floored at _GATE_PENALTY_FLOOR ('small but not exactly 0'). Returns
    (penalty, details) where details maps each FAILING gate to its {value,threshold,
    z,penalty} disclosure (empty if the seed passes every gate → penalty 1.0)."""
    p, details = 1.0, {}
    for name in _FINVAL_GATE_METRICS:
        pi, det = _gate_penalty(report.metrics.get(name))
        p *= pi
        if det is not None:
            details[name] = det
    return max(_GATE_PENALTY_FLOOR, p), details


def _failing_gates(report):
    """Names of the COMPUTABLE finval hard-gate metrics this report FAILS
    (quality 'poor', applicable, assessable). c2st is excluded — see
    _FINVAL_GATE_NOT_ASSESSABLE."""
    out = []
    for name in _FINVAL_GATE_METRICS:
        m = report.metrics.get(name)
        if m is None:
            continue
        if m.applicable and m.assessable and m.quality == "poor":
            out.append(name)
    return out


def _score_finval(competitor):
    """F1: finval pooled overall_score, GATE-PENALIZED (see the block comment above).

    Publishes the finval weighted overall_score, then multiplies each SEED's score by
    a continuous, monotone penalty on how badly that seed VIOLATES any computable hard
    gate (band-standardized margin → exp(-LN2·z), floored). A model that passes all
    gates is unchanged; one that fails is demoted in proportion to the violation, so a
    badly-failing model can never rank on F1 above a clean one — WITHOUT the per-seed
    pass/fail flicker that made the old hard cut post ±0.5 seed stds. Higher is better.
    Prints a per-competitor gate summary (mean applied penalty + worst violation) so the
    demotion is disclosed at run time; the (mean, std, n) tuple contract is unchanged."""
    import finval
    seed_scores, gate_hits, applied_penalties, n_gated = [], {}, [], 0
    for s, r in competitor.load():
        rep = finval.validate_paths(s, r)
        penalty, details = _seed_gate_penalty(rep)
        seed_scores.append(rep.overall_score * penalty)
        if details:
            n_gated += 1
            applied_penalties.append(penalty)
            for g in details:
                gate_hits[g] = gate_hits.get(g, 0) + 1
    if not seed_scores:
        return None
    if n_gated:
        summary = ", ".join(f"{g}×{c}" for g, c in sorted(gate_hits.items()))
        print(f"    [F1] {competitor.name}: gate-penalized on {n_gated}/"
              f"{len(seed_scores)} seed(s) [{summary}] — mean penalty "
              f"{np.mean(applied_penalties):.3f} (strongest {min(applied_penalties):.3f})")
    return (float(np.mean(seed_scores)), float(np.std(seed_scores)), len(seed_scores))


def _wrap(module_name):
    """Adapt a scorers/<module>.score(loaded) -> {"mean","std","n",...} to the
    Task scorer contract (competitor) -> (mean, std, n). Lazy import so one
    broken scorer doesn't take down the runner."""
    def scorer(competitor):
        import importlib
        mod = importlib.import_module(f"benchmark.scorers.{module_name}")
        r = mod.score(competitor.load())
        if int(r["n"]) == 0:      # no scorable seed => clean coverage gap, not a
            return None           # scorer error (T4's single-anchor archives; a
                                  # T6/T7 competitor whose every seed diverged)
        return (float(r["mean"]), float(r["std"]), int(r["n"]))
    return scorer


# Two layers (see ../BENCHMARK_TASKS.md): fidelity (F*) — mandatory + legibility;
# utility (T*) — economic downstream. Directions follow what each scorer RETURNS
# (F1 = finval score, higher better; F2 = normalized distance, lower better;
# F4/F5 = 0-1 composites, higher better; T2 = IV RMSE in bps, lower better;
# T3 = TSTR rank correlation, higher better; T5 = VaR/ES backtest score, higher
# better). Protocols + citations live in each scorer module's docstring.
# This version publishes ONLY the tasks that measurably resolve the field (each
# clears its honest real-vs-real noise floor) and are fairly comparable across
# families.
TASKS = [
    # fidelity
    Task("F1", "Synthetic-data quality (finval 0.6.1, gate-penalized)", "score", True,  _score_finval),
    Task("F2", "Stylized-facts battery (Cont 2001)",     "dist",  False, _wrap("f2_stylized_facts")),
    Task("F4", "Distributional distance (W1/MMD/SigW1)",  "score", True,  _wrap("f4_distances")),
    Task("F5", "Martingale / no-drift check",            "score", True,  _wrap("f5_martingale")),
    # utility
    Task("T2", "Options pricing / IV smile",             "bps",   False, _wrap("t2_options")),
    Task("T3", "Predictive validity (TSTR, multi-family)","rho",   True,  _wrap("t3_tstr")),
    Task("T5", "VaR/ES risk backtesting",                "score", True,  _wrap("t5_var_es")),
]

TASKS_BY_ID = {t.tid: t for t in TASKS}
