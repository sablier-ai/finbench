"""Competitor registry + path loading.

A competitor is a source of archived synthetic path tensors. Sablier-Flow rows
are Sablier's three published entries: the current production model, the top
research candidate, and the previous production model. External baselines keep
their real published names.

Ground truth is PINNED: every competitor on the v1 panel is scored against the
canonical panel real at `reference/panels/us_equities_macro/real_paths.npy`,
whose sha256 is asserted at load time. A per-seed `real*.npy` in a competitor's
archive is tolerated only if byte-identical to the canonical real — any
mismatch hard-fails the competitor into an "invalid" state; a submitter never
supplies their own ground truth.

Every loaded synth tensor passes a sanity gate: non-finite values or per-feature
variance far outside the canonical real's scale mark the competitor INVALID
with a reason. Invalid competitors are listed in a dedicated section, never
ranked.

Every competitor carries a `provenance` field:
  - "production":         Sablier-Flow — currently shipping in the Sablier SDK.
  - "research":           Sablier-Flow-Next — top research candidate for the
                          next production version.
  - "production-legacy":  Sablier-Flow-Old — previous production model, kept
                          for reference and progression comparison.
  - "published-defaults": external baselines at untuned published defaults.
  - "replay-resampling":  resampling / replay methods (historical simulation
                          family); memorization-guard flags are expected by
                          construction.

Every competitor is scored identically; nothing here is task-aware.
"""
from __future__ import annotations
import os, glob, hashlib
import numpy as np

# Resolve BASE from this file, not the user's home: a clone
# elsewhere must find its own reference/, not silently write an empty board.
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference")

# The canonical us_equities_macro OOS real reference, pinned by full sha256.
CANON_REAL = f"{BASE}/panels/us_equities_macro/real_paths.npy"
CANON_REAL_SHA256 = "34f9063f5b6c61a9aae501b84e001464b31ee1abbfde5ea232e429d0e5ed5fdc"

# Sanity-gate thresholds: per-feature std ratio synth/real outside
# [1/100, 100] is a gross units/scaling violation (e.g. annualized-vs-daily,
# percent-vs-decimal), not a (bad) generator. The band is deliberately WIDE:
# honest-but-severely-under-dispersed generators (TimeVAE, min ratio
# 0.0165-0.0193 across seeds) must pass here and be handled by the
# memorization/quality gates instead. The min-max-scaled Diffusion-TS class
# (min ratio 0.0122-0.0146) overlaps that range, so it is caught primarily by
# the all-positive-"returns" check below, which is specific to that failure.
_VAR_RATIO_MIN = 1.0 / 100.0
_VAR_RATIO_MAX = 100.0

PROVENANCE_LABELS = {
    "production":
        "currently shipping in the Sablier SDK",
    "research":
        "top research candidate for the next production version",
    "production-legacy":
        "previous production model, kept for progression comparison",
    "published-defaults":
        "external baseline at untuned published defaults",
    "replay-resampling":
        "resamples/replays real training data (historical simulation family); "
        "memorization-guard flags are expected by construction",
}


class InvalidCompetitorError(Exception):
    """A competitor whose archive fails ground-truth pinning or tensor sanity.

    Raised by Competitor.load(); the runner lists these in a dedicated
    'invalid' board section — they are never ranked and never silently dropped.
    """


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_CANON_CACHE = None


def load_canon_real() -> np.ndarray:
    """Load the pinned canonical real tensor, asserting its sha256."""
    global _CANON_CACHE
    if _CANON_CACHE is None:
        if not os.path.exists(CANON_REAL):
            raise RuntimeError(f"canonical real missing: {CANON_REAL}")
        got = _sha256(CANON_REAL)
        if got != CANON_REAL_SHA256:
            raise RuntimeError(
                f"canonical real hash mismatch at {CANON_REAL}: "
                f"expected {CANON_REAL_SHA256}, got {got}")
        _CANON_CACHE = np.load(CANON_REAL)
    return _CANON_CACHE


def _sanity_reason(synth: np.ndarray, canon: np.ndarray) -> str | None:
    """Return a human-readable reason if the tensor fails basic sanity, else None."""
    if synth.ndim != canon.ndim or synth.shape[-1] != canon.shape[-1]:
        return (f"shape mismatch vs canonical real: synth {synth.shape} "
                f"vs real {canon.shape}")
    if not np.isfinite(synth).all():
        n_bad = int((~np.isfinite(synth)).sum())
        return f"non-finite values in synth tensor ({n_bad} entries)"
    if not (synth < 0).any():
        return ("all-positive 'returns' tensor — wrong units/scaling "
                "(real returns are ~49% negative); min-max-scaled output?")
    r_std = canon.std(axis=(0, 1))
    s_std = synth.std(axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r_std > 0, s_std / r_std, np.inf)
    if (ratio < _VAR_RATIO_MIN).any() or (ratio > _VAR_RATIO_MAX).any():
        j = int(np.argmax(np.abs(np.log(np.maximum(ratio, 1e-300)))))
        return (f"per-feature scale violation: feature {j} std ratio "
                f"synth/real = {ratio[j]:.4g} (allowed "
                f"[{_VAR_RATIO_MIN:.3g}, {_VAR_RATIO_MAX:.3g}]) — wrong units?")
    return None


class Competitor:
    def __init__(self, name, family, seed_glob, *, provenance="published-defaults",
                 note="", invalid_reason=None):
        self.name = name                 # PUBLIC label (codename for FLOW flavors)
        self.family = family             # "flow" | "neural" | "classical"
        self.seed_glob = seed_glob       # glob of seed dirs OR None if not yet generated
        self.provenance = provenance     # see PROVENANCE_LABELS
        self.note = note
        # Pre-declared (e.g. Diffusion-TS wrong-unit archive) or set at load time.
        self.invalid_reason = invalid_reason

    @property
    def available(self) -> bool:
        return bool(self.seed_glob) and bool(glob.glob(f"{BASE}/{self.seed_glob}"))

    def _fail(self, reason: str):
        self.invalid_reason = reason
        raise InvalidCompetitorError(f"{self.name}: {reason}")

    def load(self):
        """Yield (synth, real) per seed.

        `real` is ALWAYS the pinned canonical panel real (v1 panel). A per-seed
        real*.npy in the archive is only tolerated if sha256-identical to the
        canonical real; otherwise the competitor is hard-failed into the
        'invalid' state (a submitter must not control their own
        ground truth). Each synth tensor must pass the sanity gate.
        """
        if self.invalid_reason:
            raise InvalidCompetitorError(f"{self.name}: {self.invalid_reason}")
        canon = load_canon_real()
        out = []
        for sd in sorted(glob.glob(f"{BASE}/{self.seed_glob}")):
            s = glob.glob(f"{sd}/synth*.npy")
            if not s:
                continue
            r = glob.glob(f"{sd}/real*.npy")
            if r:
                got = _sha256(r[0])
                if got != CANON_REAL_SHA256:
                    self._fail(
                        f"per-seed real {os.path.relpath(r[0], BASE)} is not the "
                        f"canonical panel real (sha256 {got[:12]}… != "
                        f"{CANON_REAL_SHA256[:12]}…) — submitter-supplied ground "
                        f"truth is rejected")
            synth = np.load(s[0])
            reason = _sanity_reason(synth, canon)
            if reason is not None:
                self._fail(f"seed dir {os.path.relpath(sd, BASE)}: {reason}")
            out.append((synth, canon))
        return out


# ---- the v1-panel competitor field -----------------------------------------
# Sablier's three published Sablier-Flow rows. Sablier-Flow is what the SDK
# currently produces; Sablier-Flow-Next is our top research candidate for the
# next production release; Sablier-Flow-Old is the previous production model,
# kept so readers can see the progression.
_SABLIER_FLOW = [
    Competitor("Sablier-Flow", "flow", "sablier-flow/seed_*",
               provenance="production",
               note="currently shipping in the Sablier SDK"),
    Competitor("Sablier-Flow-Next", "flow", "sablier-flow-next/seed_*",
               provenance="research",
               note="top research candidate for the next production version"),
    Competitor("Sablier-Flow-Old", "flow", "sablier-flow-old/seed_*",
               provenance="production-legacy",
               note="previous production model, kept for progression comparison"),
]

_BASELINES = [
    Competitor("KoVAE",        "neural", "kovae/seed_*"),
    Competitor("Diffusion-TS", "neural", "diffusion_ts/seed_*",
               note="authors' stocks config at published defaults"),
    Competitor("TimeVAE",      "neural", "timevae/seed_*"),
    Competitor("TimeGAN",      "neural", "timegan/seed_*"),
    Competitor("TimeGAN-600",  "neural", "timegan_600/seed_*"),
    Competitor("QuantGAN",     "neural", "quantgan/seed_*"),
    Competitor("ImagenTime",   "neural", "imagentime/seed_*",
               note="NeurIPS'24 diffusion"),
    Competitor("FM-TS",        "neural", "fm_ts/seed_*",
               note="flow-based TS baseline"),
    Competitor("GARCH-t",      "classical", "garch_t/seed_*",
               note="per-asset GJR-GARCH(1,1,1)-t, assets independent"),
    Competitor("DCC-t",        "classical", "dcc_t/seed_*",
               note="GJR-GARCH(1,1,1)-t marginals + DCC Student-t copula"),
    # Replay/resampling family: what a practitioner uses INSTEAD of a deep
    # generator. Ranked on every board they apply to, provenance-disclosed;
    # memorization-guard flags are expected by construction.
    Competitor("Historical-Sim",  "classical", "historical_sim/seed_*",
               provenance="replay-resampling",
               note="overlapping historical windows replayed verbatim"),
    Competitor("Block-Bootstrap", "classical", "block_bootstrap/seed_*",
               provenance="replay-resampling",
               note="stationary/block bootstrap of real return blocks"),
    Competitor("FHS",             "classical", "fhs/seed_*",
               provenance="replay-resampling",
               note="filtered historical simulation (GARCH-filtered residual resampling)"),
    Competitor("Gaussian-iid",    "classical", "gaussian_iid/seed_*",
               note="iid multivariate Gaussian fit to train moments"),
    Competitor("t-Copula",        "classical", "t_copula/seed_*",
               note="Student-t copula + empirical marginals, iid in time"),
]

COMPETITORS = _SABLIER_FLOW + _BASELINES


def available_competitors():
    return [c for c in COMPETITORS if c.available]


def pending_competitors():
    return [c for c in COMPETITORS if not c.available]
