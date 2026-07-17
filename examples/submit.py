"""FinBench submission template.

Run this script to score your synthetic outputs against the v1 panel.
Outputs a ``meta.json`` + ``finval_scores.json`` that you can commit
under ``reference/<your_method>/``.

Usage:

    python examples/submit.py \\
        --synth path/to/your_synth.npy \\   # (200, 60, 7) float32 returns
        --name your_method \\
        --seed 0 \\
        --paper https://arxiv.org/abs/YOUR_PAPER

Your ``synth`` array must be in NATIVE return space:

    IWM / QQQ / SPY / TLT / DXY:  log-returns
    VIX / TNX:                    first-differences

Match the shape exactly: ``(200, 60, 7)``. The feature order is
``[IWM, QQQ, SPY, TLT, VIX, TNX, DXY]``.

This script does NOT require your model code. It only consumes the
synthetic-output numpy file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


PANEL_NAME = "us_equities_macro_2010_2023"
HORIZON = 60
N_PATHS = 200
N_FEATURES = 7
FEATURE_ORDER = ["IWM", "QQQ", "SPY", "TLT", "VIX", "TNX", "DXY"]
FEATURE_TYPES = {
    "SPY": "price", "QQQ": "price", "IWM": "price", "TLT": "price",
    "VIX": "level", "TNX": "level", "DXY": "price",
}
OOS_SPLIT_DATE = "2020-01-01"


def load_real_reference() -> np.ndarray:
    """Load the pinned canonical FinBench v1 real reference.

    The v1 panel real lives at ``reference/panels/us_equities_macro/real_paths.npy``
    and is sha256-verified by the runner. Loading the file directly (rather than
    rebuilding it from ``sablier_flow.demo_data`` + random subsampling) guarantees
    every scored submission uses the SAME ground truth — a submitter can never
    supply their own.
    """
    canon = Path(__file__).resolve().parent.parent / "reference" / "panels" / "us_equities_macro" / "real_paths.npy"
    if not canon.exists():
        raise FileNotFoundError(f"canonical panel real missing: {canon}")
    return np.load(canon).astype(np.float32)


def score_submission(synth: np.ndarray, real: np.ndarray) -> dict:
    """Run finval on the (synth, real) pair and return a serialisable
    score dict."""
    import finval

    assert synth.shape == real.shape == (N_PATHS, HORIZON, N_FEATURES), (
        f"shape mismatch: synth {synth.shape} real {real.shape}; "
        f"expected ({N_PATHS}, {HORIZON}, {N_FEATURES})"
    )
    report = finval.validate_paths(synth, real)
    return {
        # Stamp the validator version: a finbench edition is only comparable
        # within a single finval version (scores shift across versions). Every
        # output records which finval produced it so editions are self-documenting.
        "finval_version": finval.__version__,
        "overall_quality": report.overall_quality,
        "overall_score": float(report.overall_score),
        "pass_rate": float(report.pass_rate),
        "metrics": {
            nm: {
                "value": float(m.value) if m.value is not None else None,
                "quality": m.quality,
                "passed": bool(m.passed),
            }
            for nm, m in report.metrics.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synth", required=True, type=Path,
                   help="(200, 60, 7) float32 .npy of synthetic returns")
    p.add_argument("--name", required=True,
                   help="method name (matches the reference/<name>/ dir)")
    p.add_argument("--seed", type=int, required=True,
                   help="which seed this submission corresponds to (0..4)")
    p.add_argument("--paper", default=None, help="paper URL")
    p.add_argument("--code", default=None, help="code URL (optional)")
    p.add_argument("--framework", default=None,
                   help="e.g. 'pytorch 2.7.0' (optional, for meta.json)")
    p.add_argument("--gpu", default=None, help="e.g. 'NVIDIA A100 SXM4 40GB'")
    p.add_argument("--wall_seconds", type=float, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("reference"),
                   help="root directory; defaults to reference/")
    args = p.parse_args()

    synth = np.load(args.synth).astype(np.float32)
    real = load_real_reference()
    print(f"synth shape: {synth.shape}  real shape: {real.shape}")

    t0 = time.time()
    score = score_submission(synth, real)
    print(f"scored in {time.time() - t0:.1f}s")
    print(f"  overall: {score['overall_quality']} ({score['overall_score']:.3f}, "
          f"{int(score['pass_rate'] * 14)}/14 pass)")

    seed_dir = args.out_dir / args.name / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.save(seed_dir / "synth_paths.npy", synth)
    # NO per-seed real_paths.npy — the runner uses the pinned canonical panel real,
    # sha256-verified. Duplicating it per seed would let a submitter smuggle in
    # their own ground truth.
    (seed_dir / "meta.json").write_text(json.dumps({
        "method": args.name, "seed": args.seed,
        "paper": args.paper, "code": args.code,
        "framework": args.framework, "gpu": args.gpu,
        "wall_seconds": args.wall_seconds,
        "hyperparameters_tuned": False,
        "horizon": HORIZON, "n_paths": N_PATHS,
        "feature_order": FEATURE_ORDER,
        "panel": PANEL_NAME,
    }, indent=2))
    (seed_dir / "finval_scores.json").write_text(json.dumps(score, indent=2))
    print(f"wrote {seed_dir}")


if __name__ == "__main__":
    main()
