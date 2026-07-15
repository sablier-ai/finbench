"""Aggregate the per-seed FinBench scores for a single method.

Run this AFTER you've called ``submit.py`` once per seed. It walks
``reference/<method>/seed_*/finval_scores.json`` and produces a single
``reference/<method>/finval_scores.json`` with the mean ± std across
seeds — the row that lands on the LEADERBOARD.

Usage:

    python examples/score.py --name your_method
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any


def aggregate(method_dir: Path) -> dict[str, Any]:
    seed_files = sorted(method_dir.glob("seed_*/finval_scores.json"))
    if not seed_files:
        raise SystemExit(
            f"no per-seed scores under {method_dir} — run examples/submit.py for each seed first"
        )

    per_seed: list[dict[str, Any]] = []
    for f in seed_files:
        per_seed.append(json.loads(f.read_text()))

    n = len(per_seed)
    # Aggregate metrics (mean + std) across seeds.
    metric_names = sorted({m for r in per_seed for m in r.get("metrics", {})})
    agg_metrics: dict[str, dict[str, float]] = {}
    for nm in metric_names:
        vals = [r["metrics"][nm]["value"]
                for r in per_seed
                if nm in r["metrics"] and r["metrics"][nm]["value"] is not None]
        if not vals:
            continue
        agg_metrics[nm] = {
            "mean": stats.mean(vals),
            "std": stats.stdev(vals) if len(vals) > 1 else 0.0,
        }

    overall_scores = [r["overall_score"] for r in per_seed]
    pass_rates = [r["pass_rate"] for r in per_seed]
    qualities = [r["overall_quality"] for r in per_seed]

    out = {
        "method": method_dir.name,
        "n_seeds": n,
        "overall_quality": max(set(qualities), key=qualities.count),  # mode
        "overall_score_mean": stats.mean(overall_scores),
        "overall_score_std": stats.stdev(overall_scores) if n > 1 else 0.0,
        "pass_rate_mean": stats.mean(pass_rates),
        "pass_rate_std": stats.stdev(pass_rates) if n > 1 else 0.0,
        "metrics": agg_metrics,
    }
    return out


def render_leaderboard_row(agg: dict[str, Any]) -> str:
    """Print a markdown row ready to paste into LEADERBOARD.md."""
    pass_n = int(round(agg["pass_rate_mean"] * 14))
    pieces = [
        agg["method"],
        agg["overall_quality"],
        f"{agg['overall_score_mean']:.2f}",
        f"{pass_n} / 14",
    ]
    return "| " + " | ".join(pieces) + " |"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="method name; matches reference/<name>/")
    p.add_argument("--root", type=Path, default=Path("reference"),
                   help="reference root (default: reference/)")
    args = p.parse_args()

    method_dir = args.root / args.name
    agg = aggregate(method_dir)

    out_path = method_dir / "finval_scores.json"
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"wrote {out_path}")
    print()
    print("Leaderboard row (copy into LEADERBOARD.md):")
    print()
    print(render_leaderboard_row(agg))
    print()
    print(f"Per-metric (mean ± std across {agg['n_seeds']} seeds):")
    for nm, m in agg["metrics"].items():
        print(f"  {nm:30s}  {m['mean']:9.4f} ± {m['std']:.4f}")


if __name__ == "__main__":
    main()
