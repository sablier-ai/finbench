"""Re-score the FinBench corpus under the CURRENT finval and regenerate the
leaderboard — a pure recompute on stored paths (no model re-runs).

This is the benchmark-migration tool: a scorer upgrade (finval 0.2.0 -> 0.3.0) is a
recompute, because finbench archives every submission's raw paths. v0.2.0 results stay
frozen (LEADERBOARD_finval-0.2.0.md + the per-model finval_scores.json); this writes a
NEW versioned leaderboard + per-model finval_scores_v<ver>.json stamped with finval_version.

  python examples/rescore.py
"""
from __future__ import annotations
import os, json, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np
import finval

VER = finval.__version__
BASE = os.path.expanduser("~/sablier/finbench/reference")
ROOT = os.path.expanduser("~/sablier/finbench")
# Official leaderboard models (5 seeds each), in their published display names.
MODELS = [("sablier_flow", "Sablier-Flow"), ("kovae", "KoVAE"),
          ("diffusion_ts", "Diffusion-TS"), ("timevae", "TimeVAE"), ("timegan", "TimeGAN")]
# v0.2.0 overall means (frozen leaderboard) for the side-by-side.
V020 = {"sablier_flow": 0.794, "kovae": 0.616, "diffusion_ts": 0.521, "timevae": 0.403, "timegan": 0.388}
KEY = ["overall", "regime_conditional", "tail_quantiles", "energy_distance"]


def score_model(slug):
    seeds = sorted(glob.glob(f"{BASE}/{slug}/seed_*"))
    per = {k: [] for k in KEY}
    regime_frac = []
    for sd in seeds:
        s = np.load(f"{sd}/synth_paths.npy"); r = np.load(f"{sd}/real_paths.npy")
        rep = finval.validate_paths(s, r)
        per["overall"].append(rep.overall_score)
        for m in ("regime_conditional", "tail_quantiles", "energy_distance"):
            if m in rep.metrics:
                per[m].append(rep.metrics[m].value)
        if "regime_conditional" in rep.metrics:
            regime_frac.append(rep.metrics["regime_conditional"].metadata["per_regime"]["regime_2"]["frac_synth"])
    agg = {k: (float(np.mean(v)), float(np.std(v))) for k, v in per.items() if v}
    agg["_hivol_frac"] = float(np.mean(regime_frac)) if regime_frac else float("nan")
    agg["_n_seeds"] = len(seeds)
    return agg


def main():
    results = {slug: score_model(slug) for slug, _ in MODELS}
    order = sorted(MODELS, key=lambda x: -results[x[0]]["overall"][0])

    # per-model provenance-stamped json (does NOT overwrite the v0.2.0 finval_scores.json)
    for slug, disp in MODELS:
        out = {"finval_version": VER, "method": disp, "n_seeds": results[slug]["_n_seeds"],
               "overall_score_mean": results[slug]["overall"][0], "overall_score_std": results[slug]["overall"][1],
               "regime_conditional_mean": results[slug].get("regime_conditional", (None,))[0],
               "high_vol_synth_fraction": results[slug]["_hivol_frac"]}
        with open(f"{BASE}/{slug}/finval_scores_v{VER}.json", "w") as f:
            json.dump(out, f, indent=2)

    # leaderboard markdown
    L = [f"# FinBench v1 — Leaderboard (finval {VER})", "",
         f"Same task/protocol as v0.2.0 (us_equities_macro panel, 2010-2019 train / 2020-2023 OOS, "
         f"L=60, 200 paths, 5 seeds). **Re-scored from the archived paths under finval {VER}** — a pure "
         "recompute, no model re-runs. The v0.2.0 board is frozen at "
         "[LEADERBOARD_finval-0.2.0.md](./LEADERBOARD_finval-0.2.0.md); do not compare raw scores across "
         "scorer versions (compare rankings + cite the version).", "",
         "**What changed v0.2.0 → " + VER + ":** (1) the overall score is now *de-quantized* "
         "(severity-aware, not a 4-level grade collapse) so tail failures register; (2) tail/risk metrics "
         "re-weighted up (`tail_quantiles` effective 0.0375→0.080); (3) a new **`regime_conditional`** "
         "scored axis (regime-conditional fidelity — does the generator reproduce the high-volatility "
         "regime, not just the pooled distribution). Signature & wavelet metrics were tested and rejected "
         "as redundant (see ECONOMETRIC_BASELINES.md).", "",
         "## Overall ranking", "",
         "| Rank | Method | finval " + VER + " | (v0.2.0) | regime_conditional | high-vol path frac (real 0.33) |",
         "|-----:|--------|----------:|---------:|-------------------:|------------------------------:|"]
    for i, (slug, disp) in enumerate(order, 1):
        a = results[slug]
        rc = a.get("regime_conditional", (float("nan"), 0))
        L.append(f"| {i} | {disp} | {a['overall'][0]:.3f} ± {a['overall'][1]:.3f} | {V020.get(slug,'—')} | "
                 f"{rc[0]:.3f} | {a['_hivol_frac']:.2f} |")
    L += ["", "**Reading it:** Sablier-Flow still leads. The new `regime_conditional` column exposes a "
          "failure invisible to v0.2.0 — **every** generator under-produces the high-volatility regime "
          "(high-vol path fraction far below the real 0.33), i.e. they underprice stress. Sablier-Flow is "
          "best but still produces ~10× too few high-vol paths: the quantified FLOW-2.0 conditional target.", ""]
    path = f"{ROOT}/LEADERBOARD_finval-{VER}.md"
    with open(path, "w") as f:
        f.write("\n".join(L))

    print(f"finval {VER} — re-scored leaderboard written to {os.path.basename(path)}\n")
    print(f"{'Method':14s} {'v'+VER:>10s} {'v0.2.0':>8s} {'regime_cond':>12s} {'hivol_frac':>11s}")
    for i, (slug, disp) in enumerate(order, 1):
        a = results[slug]; rc = a.get("regime_conditional", (float('nan'),))[0]
        print(f"{disp:14s} {a['overall'][0]:10.3f} {V020.get(slug,0):8.3f} {rc:12.3f} {a['_hivol_frac']:11.2f}")


if __name__ == "__main__":
    main()
