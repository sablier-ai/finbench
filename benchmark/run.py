"""Run the multi-task benchmark → per-task boards + a rank-based aggregate.

    python -m benchmark.run      # from finbench/

Pipeline (post rigor-audit, 2026-07-10):
  1. Load every AVAILABLE competitor against the PINNED canonical panel real
     (registry hard-fails wrong-unit / doctored-real archives → Invalid section).
  2. Memorization/copy gate (benchmark/guards.py): MEMORISATION / COPY verdicts
     are DISQUALIFIED (excluded from all boards); SUSPICIOUS rows are ranked but
     asterisk-flagged.
  3. Score every surviving competitor on every wired task. A scorer exception or
     non-finite score is a MISSING cell (printed with its error class) — it can
     never improve an aggregate (F-04) or poison ranks (F-03).
  4. Per-task boards carry n, a 95% t-CI, a '≈#1' marker (Welch vs the task
     leader, Holm-corrected across the field for multiplicity, adj-p>0.05), and
     an HONEST noise-floor row: real data scored against an INDEPENDENT
     (calendar-disjoint) draw of itself at the competitor path budget. A task
     whose whole field sits within one floor sd is flagged low-resolution.
  5. The aggregate ranks ONLY full-coverage competitors (scored on ALL scored
     tasks); partial-coverage rows are listed separately. Ranks are tie-aware and
     an approximate bootstrap P(#1) column is published.

Env:
  FINBENCH_TASKS=F1,F2   score only these task ids (reduced/smoke runs)
  FINBENCH_NOISE_FLOOR=0 skip the noise-floor rows
  FINBENCH_FLOOR_REPS=N  disjoint-floor bootstrap reps (default 12; seeded)
"""
from __future__ import annotations
import os
import zlib
import numpy as np
from scipy.stats import rankdata
from scipy import stats as sps

from benchmark.registry import (available_competitors, pending_competitors,
                                load_canon_real, InvalidCompetitorError,
                                PROVENANCE_LABELS)
from benchmark.tasks import TASKS, ScorerError
import finval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_BOOT = 1000

# Noise-floor rebuild (2026-07-15).
# ---------------------------------------------------------------------------
# The pinned real "paths" are NOT independent samples: they are stride-1
# OVERLAPPING rolling windows cut from ONE ~1004-day series (reconstructed
# median stride ~4 days over ~999 unique calendar days). The OLD floor split
# that tensor by PATH PARITY (even vs odd paths) — but even and odd windows
# share ~99.3% of their calendar days (measured Jaccard 0.993), so the "floor"
# was real data compared to ~itself: artificially perfect, priced ZERO
# independent sampling, and ran at HALF the competitor path budget. Every
# published floor was therefore a self-comparison, not a resolution limit.
#
# The HONEST floor below resamples INDEPENDENT, calendar-DISJOINT windows: it
# reconstructs each window's start day-id from the exact overlap between
# consecutive windows, partitions the windows into two halves separated by an
# H-day gap (→ zero shared calendar days), and bootstraps each half up to the
# competitor path budget, over FLOOR_REPS reps. That is what real-vs-INDEPENDENT
# -real scores under each metric — a true "what a perfect generator gets".
# Seeded → deterministic (NBOOT-1). The old even/odd number is still reported,
# relabelled as identical-data estimator jitter (NOT a sampling floor).
FLOOR_REPS = int(os.environ.get("FINBENCH_FLOOR_REPS", "12"))
FLOOR_SEED = 20260715
# Every scored task that admits an honest real-vs-real noise floor: the field
# is compared against an independent, calendar-disjoint draw of the real panel,
# so a task "resolves" only if models separate by more than that floor. F1 is
# floored separately, via its own finval recompute.
NOISE_FLOOR_TASKS = {"F1", "F2", "F4", "F5", "T2", "T3", "T5"}
_FLOOR_MODULES = {"F2": "f2_stylized_facts", "F4": "f4_distances",
                  "F5": "f5_martingale", "T2": "t2_options", "T3": "t3_tstr",
                  "T5": "t5_var_es"}


def _selected_tasks():
    sel = os.environ.get("FINBENCH_TASKS")
    tasks = [t for t in TASKS if t._scorer is not None]
    if sel:
        want = {s.strip().upper() for s in sel.split(",") if s.strip()}
        tasks = [t for t in tasks if t.tid.upper() in want]
        print(f"[filter] FINBENCH_TASKS={sel} -> scoring {[t.tid for t in tasks]}")
    return tasks


def _apply_guard(loaded_map, canon):
    """Memorization/copy gate. Returns (disqualified, suspicious) and prunes
    loaded_map in place. Tolerates a missing guards module (skips, loudly)."""
    disqualified = {}   # name -> (verdict, detail)
    suspicious = {}     # name -> detail
    try:
        from benchmark.guards import memorization_guard
    except ImportError:
        print("=" * 78)
        print("WARNING: benchmark/guards.py not importable — MEMORIZATION/COPY "
              "GATE SKIPPED.")
        print("The board below is NOT protected against memorising/copying "
              "competitors. Do not publish it externally until the guard is in "
              "place.")
        print("=" * 78)
        return disqualified, suspicious
    for name in list(loaded_map):
        try:
            v = memorization_guard(name, loaded_map[name], canon)
            verdict, detail = v["verdict"], v.get("detail", "")
        except Exception as e:
            # A guard failure must fail CLOSED for ranking purposes? No — it
            # would let a crash disqualify honest entries. Flag as suspicious.
            verdict, detail = "SUSPICIOUS", f"guard error ({type(e).__name__}: {e})"
        print(f"    [guard] {name:18s} {verdict}  {detail}")
        if verdict in ("MEMORISATION", "COPY"):
            disqualified[name] = (verdict, detail)
            del loaded_map[name]
        elif verdict == "SUSPICIOUS":
            suspicious[name] = detail
    return disqualified, suspicious


def _welch_p(m1, s1, n1, m2, s2, n2):
    """Two-sided Welch t-test p-value from summary stats; None if untestable."""
    if n1 < 2 or n2 < 2:
        return None
    v1, v2 = (s1 ** 2) / n1, (s2 ** 2) / n2
    if v1 + v2 == 0:
        return 1.0
    t = (m1 - m2) / np.sqrt(v1 + v2)
    df_num = (v1 + v2) ** 2
    df_den = v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2
    return float(2 * sps.t.sf(abs(t), df))


def _holm_adjusted(pvals):
    """Holm step-down adjusted p-values for a dict {name: p} of m simultaneous
    tests. The per-task '≈#1' marker runs one Welch test per
    non-leader competitor vs the task leader — m = field-size-1 tests read off
    the SAME board — so an UNADJUSTED p>0.05 threshold understates how many rows
    are indistinguishable from #1 (multiplicity inflates false 'distinguishable'
    calls). Holm corrects across the family: sort p ascending, multiply the
    j-th-smallest (0-indexed) by (m-j), and enforce monotonicity. Returns
    {name: p_adj}; adjusted p is >= raw p, so more rows survive as ≈#1 — the
    conservative, honest direction. Untestable (n<2) competitors are not passed
    in and stay unmarked."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for j, (name, p) in enumerate(items):
        running = max(running, min((m - j) * p, 1.0))
        adj[name] = running
    return adj


def _ci95(std, n):
    """95% t-interval half-width over seeds; None at n<2."""
    if n < 2:
        return None
    return float(sps.t.ppf(0.975, n - 1) * std / np.sqrt(n))


def _window_starts(canon):
    """Reconstruct each rolling window's start day-id from the EXACT overlap
    between consecutive windows, so the floor can be split by CALENDAR day
    rather than by path index. Returns (starts, H, ok):
      starts[i] = first calendar-day id of window i (starts[0]=0),
      H         = window length (horizon),
      ok        = True iff the tensor really is a stack of overlapping windows
                  (max overlap residual ≈ 0). If ok is False the panel is NOT
                  window-overlapping (e.g. already-independent draws) and the
                  caller falls back to an index split, which is then honest.
    """
    N, H, _ = canon.shape
    strides, resid = [], []
    for i in range(N - 1):
        w0, w1 = canon[i], canon[i + 1]
        best, bestk = None, 1
        for k in range(1, H):
            d = float(np.abs(w0[k:] - w1[:H - k]).mean())
            if best is None or d < best:
                best, bestk = d, k
        strides.append(bestk)
        resid.append(best)
    # scale the residual by the tensor's own typical step so the "is it
    # overlapping" test is unit-free
    scale = float(np.abs(np.diff(canon, axis=1)).mean()) or 1.0
    ok = float(np.max(resid)) < 1e-3 * scale
    starts = np.concatenate([[0], np.cumsum(strides)]).astype(int)
    return starts, H, ok


def _disjoint_floor_reps(tid, canon, starts, H, reps, seed):
    """FLOOR_REPS scores of independent real-vs-real over CALENDAR-DISJOINT
    windows, bootstrapped to the competitor path budget. Each rep: pick a random
    gap day, take windows entirely LEFT of the gap as one half and entirely
    RIGHT (gap+H, so zero shared days) as the other, resample each half WITH
    replacement to N paths, and score (one half as 'generator', the other as
    'real'). Returns the array of rep scores (may be shorter than reps if some
    gaps leave a half too small)."""
    N = canon.shape[0]
    rng = np.random.default_rng(seed)
    lo, hi = int(starts[0] + H), int(starts[-1])   # valid gap-day range
    vals = []
    for _ in range(reps):
        if hi <= lo:
            break
        gap = int(rng.integers(lo, hi))
        A = np.where(starts + H - 1 < gap)[0]       # windows fully before gap
        B = np.where(starts >= gap + H)[0]          # windows fully after gap+H
        if len(A) < 2 or len(B) < 2:
            continue
        if rng.random() < 0.5:                      # symmetrise which side is 'synth'
            A, B = B, A
        synth = canon[rng.choice(A, size=N, replace=True)]
        real = canon[rng.choice(B, size=N, replace=True)]
        try:
            if tid == "F1":
                v = float(finval.validate_paths(synth, real).overall_score)
            else:
                import importlib
                mod = importlib.import_module(
                    f"benchmark.scorers.{_FLOOR_MODULES[tid]}")
                r = mod.score([(synth, real)])
                if int(r["n"]) == 0:
                    continue
                v = float(r["mean"])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    return np.array(vals, dtype=float)


def _noise_floors(tasks, canon):
    """HONEST real-vs-INDEPENDENT-real floor per task.

    Returns {tid: {"mean","sd","n","eo"}} where mean±sd is over FLOOR_REPS
    calendar-disjoint bootstrap reps at the competitor path budget, and "eo" is
    the OLD even/odd path-parity self-score (kept for transparency, NOT a floor:
    the two halves share ~99.3% of their days). Nothing is imputed: a task whose
    scorer raises on every rep is simply omitted.
    """
    if os.environ.get("FINBENCH_NOISE_FLOOR", "1") == "0":
        return {}
    starts, H, ok = _window_starts(canon)
    if not ok:
        print("    [floor] NOTE: real tensor is not overlapping-window shaped — "
              "a calendar split degenerates to an index split (still honest).")
    even, odd = canon[0::2], canon[1::2]
    floors = {}
    for t in tasks:
        if t.tid not in NOISE_FLOOR_TASKS:
            continue
        try:
            # legacy even/odd self-score (reported, relabelled — not a floor)
            if t.tid == "F1":
                eo = float(finval.validate_paths(even, odd).overall_score)
            else:
                import importlib
                mod = importlib.import_module(
                    f"benchmark.scorers.{_FLOOR_MODULES[t.tid]}")
                reo = mod.score([(even, odd)])
                eo = float(reo["mean"]) if int(reo["n"]) > 0 else float("nan")
        except Exception as e:
            eo = float("nan")
            print(f"    [floor] {t.tid}: even/odd unavailable ({type(e).__name__}: {e})")
        # zlib.crc32 (NOT builtin hash(), which is PYTHONHASHSEED-randomized) so
        # the per-task seed — and thus the floor — is reproducible across runs.
        tid_off = zlib.crc32(t.tid.encode()) % 9973
        vals = _disjoint_floor_reps(t.tid, canon, starts, H, FLOOR_REPS,
                                    FLOOR_SEED + tid_off)
        if vals.size == 0:
            print(f"    [floor] {t.tid}: disjoint floor unavailable (all reps failed)")
            continue
        floors[t.tid] = {"mean": float(vals.mean()), "sd": float(vals.std()),
                         "n": int(vals.size),
                         "eo": (eo if np.isfinite(eo) else None)}
        print(f"    [floor] {t.tid}: disjoint real-vs-real = {vals.mean():.3f} "
              f"± {vals.std():.3f} (n={vals.size} reps; old even/odd self-score "
              f"= {eo:.3f})")
    return floors


def _bootstrap_p1(full_names, tasks, results, rng):
    """Approximate bootstrap P(#1 aggregate rank) over full-coverage rows.

    APPROXIMATION (documented on the board): run.py only has per-task
    (mean, std, n) summaries, not per-seed values — so each bootstrap draw
    perturbs every task score with Gaussian(mean, std/sqrt(n)) noise (the
    seed-mean's sampling error), re-ranks per task, and recomputes mean ranks.
    Cross-task error correlation is ignored, so P(#1) is optimistic-width.

    n=1 rows: a single-seed row reports std=0, so its
    naive sem is 0 and it can NEVER move across all N_BOOT resamples — the
    LEAST-supported row treated as the MOST precise. Instead we IMPUTE its
    per-task sem from the median seed-std of the multi-seed rows on that same
    task (a principled non-zero uncertainty), so a single-seed entry cannot be
    locked to an over-precise rank. The CI column tells the same story
    ('n=1 provisional', see _fmt_ci).
    """
    if len(full_names) < 2:
        return {}
    tids = [t.tid for t in tasks if results[t.tid]]
    means = np.array([[results[tid][n][0] for n in full_names] for tid in tids])

    def _sem(tid, name, imputed_std):
        std, k = results[tid][name][1], results[tid][name][2]
        if k >= 2:
            return std / np.sqrt(k)
        # n=1: impute the per-task cross-competitor median multi-seed std
        return (imputed_std if imputed_std is not None else 0.0)

    sems_rows = []
    for tid in tids:
        multi = [results[tid][n][1] for n in results[tid]
                 if results[tid][n][2] >= 2]
        imputed = float(np.median(multi)) if multi else None
        sems_rows.append([_sem(tid, n, imputed) for n in full_names])
    sems = np.array(sems_rows)
    signs = np.array([-1.0 if t.higher_better else 1.0
                      for t in tasks if results[t.tid]])
    wins = np.zeros(len(full_names))
    for _ in range(N_BOOT):
        draw = means + rng.standard_normal(means.shape) * sems
        rk = np.vstack([rankdata(signs[i] * draw[i], method="average")
                        for i in range(len(tids))])
        mean_rank = rk.mean(axis=0)
        wins[mean_rank == mean_rank.min()] += 1.0 / (mean_rank == mean_rank.min()).sum()
    return {n: wins[i] / N_BOOT for i, n in enumerate(full_names)}


def run():
    canon = load_canon_real()
    comps = available_competitors()
    tasks = _selected_tasks()

    # 1) load + pinned-real / sanity gate ------------------------------------
    invalid = []          # (competitor, reason)
    loaded_map = {}       # name -> [(synth, canon_real), ...]
    for c in comps:
        try:
            loaded = c.load()
            if loaded:
                loaded_map[c.name] = loaded
            else:
                invalid.append((c, "no synth tensors found in seed dirs"))
        except InvalidCompetitorError as e:
            print(f"    [invalid] {e}")
            invalid.append((c, c.invalid_reason or str(e)))

    # 2) memorization/copy gate ----------------------------------------------
    disqualified, suspicious = _apply_guard(loaded_map, canon)
    field = [c for c in comps if c.name in loaded_map]

    # 3) score ----------------------------------------------------------------
    # results[tid][name] = (mean, std, n); errors[tid][name] = ScorerError
    results = {t.tid: {} for t in TASKS}
    errors = {t.tid: {} for t in TASKS}
    for t in tasks:
        print(f"[{t.tid}] {t.name}")
        for c in field:
            r = t.score(c)
            if isinstance(r, ScorerError):
                errors[t.tid][c.name] = r
            elif r is not None:
                results[t.tid][c.name] = r
                print(f"    {c.name:18s} {r[0]:.3f} ± {r[1]:.3f}  (n={r[2]})")

    # per-task ranks (1 = best, respecting direction)
    ranks = {}
    for t in tasks:
        d = results[t.tid]
        if not d:
            continue
        names = list(d)
        vals = np.array([d[n][0] for n in names])
        order = -vals if t.higher_better else vals
        ranks[t.tid] = dict(zip(names, rankdata(order, method="average")))

    # 4) aggregate: FULL-coverage rows only (F-04/AGG-4) ----------------------
    scored_tids = [t.tid for t in tasks if results[t.tid]]
    agg, partial = {}, {}
    for c in field:
        rs = [ranks[tid][c.name] for tid in scored_tids if c.name in ranks.get(tid, {})]
        if not rs:
            continue
        mr = (float(np.mean(rs)), len(rs))
        if len(rs) == len(scored_tids):
            agg[c.name] = mr
        else:
            partial[c.name] = mr

    p1 = _bootstrap_p1(sorted(agg, key=lambda n: agg[n][0]),
                       [t for t in tasks if results[t.tid]], results,
                       np.random.default_rng(0))
    floors = _noise_floors(tasks, canon)

    _write(results, errors, ranks, agg, partial, scored_tids, comps, tasks,
           invalid, disqualified, suspicious, p1, floors)


def _flag(name, suspicious):
    return f"{name}\\*" if name in suspicious else name


def _fmt_ci(v):
    """(metric cell, CI cell). n=1 rows print NO '± 0.000' (their seed variance
    is UNMEASURED, not measured-zero) and are marked 'n=1 provisional' — audit
    F1-N1-5."""
    mean, std, n = v
    ci = _ci95(std, n)
    if ci is None:                       # n < 2: seed variance unmeasured
        return f"{mean:.3f}", "n=1 provisional"
    return f"{mean:.3f} ± {std:.3f}", f"±{ci:.3f}"


def _tie_aware_ranks(sorted_items):
    """[(name,(mr,n)),...] sorted by mr -> {name: '1' | '1='} tie-aware labels."""
    mrs = [round(mr, 9) for _, (mr, _) in sorted_items]
    out = {}
    for i, (name, (mr, _)) in enumerate(sorted_items):
        key = round(mr, 9)
        rank = 1 + sum(1 for m in mrs if m < key)
        tied = mrs.count(key) > 1
        out[name] = f"{rank}=" if tied else f"{rank}"
    return out


def _write(results, errors, ranks, agg, partial, scored_tids, comps, tasks,
           invalid, disqualified, suspicious, p1, floors):
    prov = {c.name: c.provenance for c in comps}
    fam = {c.name: c.family for c in comps}

    # Honest-floor vs field verdicts + low-resolution flags.
    # A task is 'low resolution' when >=50% of the ranked field sits
    # within +/-1 disjoint-floor sd of the floor mean — the field cannot be told
    # apart from an independent draw of real data (or from each other) there.
    task_dir = {t.tid: t.higher_better for t in TASKS}
    floor_verdict = {}     # tid -> (frac_within, flagged, verdict_text)
    flagged_tasks = []
    for tid, fl in floors.items():
        vals = [results[tid][n][0] for n in results.get(tid, {})]
        if not vals or fl.get("n", 0) < 2 or fl["sd"] <= 0:
            continue
        m, s = fl["mean"], fl["sd"]
        lo, hi = m - s, m + s
        frac = sum(1 for v in vals if lo <= v <= hi) / len(vals)
        hb = task_dir.get(tid, True)
        n_better = (sum(1 for v in vals if v > hi) if hb
                    else sum(1 for v in vals if v < lo))
        flagged = frac >= 0.5
        if flagged:
            text = (f"{frac*100:.0f}% of the field within ±1 floor-sd — ranking "
                    "here is within sampling noise")
            flagged_tasks.append(tid)
        elif n_better / len(vals) >= 0.5:
            text = (f"the field scores BETTER than independent real data on "
                    f"{n_better}/{len(vals)} rows — the honest floor is a CEILING "
                    "the field exceeds, so scores past it are not attributable to "
                    "distributional fidelity")
        else:
            text = ("the field sits worse than the honest floor — genuine "
                    "resolution")
        floor_verdict[tid] = (frac, flagged, text)
    # This version publishes a pre-curated set of tasks that resolve the field, so
    # the in-board low-resolution re-split is disabled — every per-task floor is
    # still shown, but there is a single leaderboard, not a headline/robustness pair.
    flagged_tasks = []

    L = ["# FinBench Leaderboard", "",
         "Benchmark for multivariate financial time-series generation. Each model "
         "generates synthetic paths for a single frozen panel and is scored, under "
         "identical conditions, on seven tasks spanning distributional fidelity and "
         "economic utility. Models are ranked by **mean rank across tasks** — no "
         "single metric defines the benchmark.", ""]
    # metadata block (MTEB/HELM-style provenance line)
    n_field = len(agg) + len(partial)
    L += [f"**Panel** `us_equities_macro` · 7 features · N=200 paths · H=60 days · "
          f"out-of-sample &nbsp;|&nbsp; **Models** {n_field} &nbsp;|&nbsp; "
          f"**Tasks** {len(scored_tids)} &nbsp;|&nbsp; **Metric** mean rank "
          "(lower is better)", "",
         "Every task clears an honest real-vs-real noise floor (real data scored "
         "against a calendar-disjoint draw of itself) — tasks that cannot separate "
         "the field on this panel are not shown. FLOW variants appear under opaque "
         "codenames.", ""]
    if suspicious:
        L += ["\\* = memorization-guard verdict SUSPICIOUS — ranked, but treat with "
              "caution (see the guard section).", ""]

    agg_sorted = sorted(agg.items(), key=lambda x: (x[1][0], x[0]))
    tie_label = _tie_aware_ranks(agg_sorted)

    # ---- headline: overall ranking ------------------------------------------
    L += ["## Overall", "",
          "| # | Model | Family | Mean rank | P(#1)† | Coverage |",
          "|--:|--|--|--:|--:|--:|"]
    for name, (mr, n) in agg_sorted:
        p = f"{p1.get(name, 0.0):.2f}" if p1 else "—"
        L.append(f"| {tie_label[name]} | {_flag(name, suspicious)} | "
                 f"{fam.get(name, '?')} | {mr:.2f} | {p} | "
                 f"{n}/{len(scored_tids)} |")
    L += ["",
          "†P(#1): bootstrap probability of finishing first, over "
          f"{N_BOOT} resamples (fixed seed). An upper bound on confidence — it prices "
          "per-seed sampling error only, not target/regime error (single OOS window), "
          "so the winner's ordering is robust but the probability is optimistic.", ""]

    # ---- per-task rank matrix (models × tasks) ------------------------------
    # The at-a-glance view: where each model wins and loses. Cells are per-task
    # rank; the leader per column is bold.
    L += ["## Per-task ranks", "",
          "Rank of each model on every task (**1** = best in column). Detailed "
          "score tables with confidence intervals and noise floors follow below.", ""]
    matrix_tids = list(scored_tids)
    L += ["| Model | " + " | ".join(matrix_tids) + " | Mean |",
          "|--|" + "|".join([":--:"] * len(matrix_tids)) + "|--:|"]
    for name, (mr, _n) in agg_sorted:
        cells = []
        for tid in matrix_tids:
            rv = ranks.get(tid, {}).get(name)
            if rv is None:
                cells.append("—")
            else:
                rs = f"{rv:g}" if rv == int(rv) else f"{rv:.1f}"
                # bold the column leader
                cells.append(f"**{rs}**" if rv == 1 else rs)
        L.append(f"| {_flag(name, suspicious)} | " + " | ".join(cells) +
                 f" | {mr:.2f} |")
    L += ["", "Tasks: " + " · ".join(
        f"**{t.tid}** {t.name}" for t in TASKS if t.tid in matrix_tids), ""]
    # provenance legend
    used_prov = sorted({prov.get(n, "?") for n in list(agg) + list(partial)})
    L += ["**Provenance.** " + " ".join(
        f"*{p}* — {PROVENANCE_LABELS.get(p, p)}." for p in used_prov), ""]

    if partial:
        L += ["### Partial coverage (NOT in the aggregate)", "",
              "Missing cells (scorer error or coverage gap) exclude a competitor "
              "from the aggregate — a crashed task can never improve a rank.", "",
              "| Competitor | Provenance | Mean rank (scored subset) | Tasks scored |",
              "|--|--|--:|--:|"]
        for name, (mr, n) in sorted(partial.items(), key=lambda x: x[1][0]):
            L.append(f"| {_flag(name, suspicious)} | {prov.get(name, '?')} | "
                     f"{mr:.2f} | {n}/{len(scored_tids)} |")
        L.append("")

    # ---- per-task detail -----------------------------------------------------
    L += ["---", "", "# Task detail", "",
          "Per-task score tables with 95% confidence intervals and the real-vs-real "
          "noise floor. Each table's ranks are the columns of the matrix above.", ""]
    for t in TASKS:
        if tasks and t not in tasks and t._scorer is not None:
            continue  # filtered out this run
        d = results[t.tid]
        L += [f"## {t.tid} — {t.name}", ""]
        if not d:
            L += ["_scorer not wired yet — no results._", ""]
            continue
        arrow = "higher better" if t.higher_better else "lower better"
        if t.tid == "F1":
            L += [f"Scored with [finval](https://github.com/sablier-ai/finval) "
                  f"v{finval.__version__} (the finance-aware path-quality suite). "
                  "**Selection caveat:** the FLOW-A…J recipe was selected by sweeping "
                  "this metric on this panel, so their F1 is in-sample — a training-fit "
                  "upper bound, not held-out skill. FLOW-P1/P2 (own production "
                  "configuration) and the external baselines are not selected on F1, "
                  "so their F1 is held-out.", ""]
        L += [f"Metric: {t.unit} ({arrow}). CI = 95% t-interval over seeds; "
              "'≈#1' = statistically indistinguishable from the task leader "
              "(Welch t-test, Holm-corrected across the field; untestable at n<2).", "",
              f"| Rank | Competitor | n | {t.unit} | 95% CI | vs #1 |",
              "|--:|--|--:|--:|--:|--|"]
        rk = ranks[t.tid]
        leader = min(d, key=lambda n: rk[n])
        lm, ls, ln = d[leader]
        # Holm-correct the family of Welch-vs-leader tests before thresholding
        #: collect the m testable p-values, adjust for
        # multiplicity, then mark ≈#1 at adjusted p>0.05. Untestable rows (n<2,
        # p None) are excluded from the family and stay unmarked.
        raw_pv = {}
        for name in d:
            if name == leader:
                continue
            pv = _welch_p(*d[name], lm, ls, ln)
            if pv is not None:
                raw_pv[name] = pv
        holm_pv = _holm_adjusted(raw_pv)
        for name in sorted(d, key=lambda n: (rk[n], n)):
            mean_s, ci_s = _fmt_ci(d[name])
            if name == leader:
                mark = "#1"
            elif name in holm_pv:
                mark = "≈#1" if holm_pv[name] > 0.05 else ""
            else:
                mark = ""   # untestable at n<2
            rank_v = rk[name]
            rank_s = f"{rank_v:g}" if rank_v == int(rank_v) else f"{rank_v:.1f}"
            L.append(f"| {rank_s} | {_flag(name, suspicious)} | {d[name][2]} | "
                     f"{mean_s} | {ci_s} | {mark} |")
        if t.tid in floors:
            fl = floors[t.tid]
            L.append(f"| — | _**noise floor** — independent real-vs-real "
                     f"(calendar-disjoint windows, competitor path budget, "
                     f"{fl['n']} reps)_ | — | {fl['mean']:.3f} ± {fl['sd']:.3f} "
                     f"| — | — |")
            fv = floor_verdict.get(t.tid)
            if fv:
                L.append("")
                L.append(f"**Field vs floor:** {fv[2]}")
        if errors[t.tid]:
            L.append("")
            L.append("Missing cells (scorer error — counted as coverage gaps, "
                     "never waived): " + "; ".join(
                         f"{n} ({e.error_class})" for n, e in
                         sorted(errors[t.tid].items())) + ".")
        L.append("")

    # ---- disqualified / flagged ----------------------------------------------
    if disqualified or suspicious:
        L += ["## Disqualified / flagged (memorization–copy gate)", ""]
        if disqualified:
            L += ["Disqualified — excluded from every board above:", "",
                  "| Competitor | Verdict | Detail |", "|--|--|--|"]
            for name, (v, det) in sorted(disqualified.items()):
                L.append(f"| {name} | {v} | {det} |")
            L.append("")
        if suspicious:
            L += ["Flagged SUSPICIOUS (ranked above, marked \\*):", "",
                  "| Competitor | Detail |", "|--|--|"]
            for name, det in sorted(suspicious.items()):
                L.append(f"| {name} | {det} |")
            L.append("")

    # ---- invalid ---------------------------------------------------------------
    if invalid:
        L += ["## Invalid competitors (rows void — not ranked)", "",
              "Archives that fail ground-truth pinning or the tensor sanity gate. "
              "Listed, never silently dropped; re-enter the boards once regenerated.", "",
              "| Competitor | Provenance | Reason |", "|--|--|--|"]
        for c, reason in invalid:
            L.append(f"| {c.name} | {prov.get(c.name, '?')} | {reason} |")
        L.append("")

    # ---- coverage / pending -----------------------------------------------------
    pend = pending_competitors()
    if pend:
        L += ["## Pending competitors (not yet generated)", "",
              "These slot into every board as new rows once their tensors exist "
              "(FLOW flavors are GPU-gated):", "",
              ", ".join(c.name for c in pend), ""]

    # ---- footer disclaimer --------------------------------------------------------
    L += ["---", "",
          "**Scope disclaimer.** Single panel (us_equities_macro, 7 features), "
          "horizon H=60, one OOS window. All current entries are maintainer-"
          "generated; external submission gates (held-out scoring slab, training-"
          "metadata attestation) are pending — do not treat this board as an open "
          "leaderboard yet. FLOW-A…J share one bake-off recipe selected via finval "
          "(the F1 evaluator) on this panel; FLOW-P1/P2 use their own tuned "
          "production configurations; external baselines run untuned published "
          "defaults. Rank order within a significance group is not a claim.", ""]

    out = f"{ROOT}/MULTITASK_LEADERBOARD.md"
    open(out, "w").write("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
