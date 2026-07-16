"""Competitor registry + path loading.

A competitor is a source of archived synthetic path tensors. FLOW flavors carry
OPAQUE codenames only — the arch mapping is private (see ../BENCHMARK_TASKS.md §4).
Baselines keep their real published names.

Ground truth is PINNED: every competitor on the v1 panel is
scored against the canonical panel real at `reference/panels/us_equities_macro/
real_paths.npy`, whose full sha256 is asserted at load time. A per-seed
`real*.npy` inside a competitor archive is tolerated only if it is byte-identical
(sha256) to the canonical real — any mismatch hard-fails the competitor into a
listed "invalid" state; it is never used as the scoring reference.

Every loaded synth tensor passes a sanity gate:
non-finite values, per-feature variance far outside the canonical real's scale,
or an all-positive "returns" tensor (the Diffusion-TS min-max-scaling class)
mark the competitor INVALID with a reason. Invalid competitors are listed on the
board in a dedicated section, never ranked.

Every competitor carries a `provenance` field:
  - "recipe-controlled":     FLOW-A..J — one shared bake-off recipe, selected by
                             sweeping finval (the F1 evaluator) on this panel.
                             Disclosed on the board.
  - "production-reference":  FLOW-P1 / FLOW-P2 — own tuned production configuration.
  - "published-defaults":    external baselines at untuned published defaults.
  - "invalid-pending-regen": rows VOID (excluded from all boards, listed with
                             reason) until regenerated — currently Diffusion-TS,
                             whose archived tensors are min-max [0,1], not returns.

Every competitor is scored identically by every task; nothing here is task-aware.
"""
from __future__ import annotations
import os, glob, hashlib
import numpy as np

# Resolve BASE from this file, not the user's home: a clone
# elsewhere must find its own reference/, not silently write an empty board.
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference")

# The canonical us_equities_macro OOS real reference, moved out of any
# competitor's archive to a panel path and pinned by full sha256
# (data-bytes md5 509088fee22a68952b2ef6e26db987de, kept for continuity with
# older notes; the sha256 below is authoritative and asserted at load).
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
    "recipe-controlled":
        "one shared FLOW bake-off recipe, selected via finval (the F1 evaluator) on this panel",
    "production-reference":
        "own tuned production configuration",
    "published-defaults":
        "external baseline at untuned published defaults",
    "invalid-pending-regen":
        "rows void until regenerated (see Invalid section)",
    "replay-resampling":
        "resamples/replays real training data (industry practice: historical "
        "simulation family); scenarios are drawn from history rather than "
        "generated, so memorization-guard flags are expected by construction",
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
# FLOW production entries (P1 = first production config, P2 = tuned production
# config), each under an opaque codename. The FLOW flavors (FLOW-A…J) are
# separate architecture-sweep configurations; they slot into every board as
# additional rows with no code change.
_FLOW_PRODUCTION = [
    Competitor("FLOW-P1", "flow", "sablier_flow/seed_*",
               provenance="production-reference", note="production configuration"),
    Competitor("FLOW-P2", "flow", "sablier_flow_v2/seed_*",
               provenance="production-reference", note="tuned production configuration"),
]

# Flavor codenames — arch mapping is PRIVATE. seed_glob=None => pending (GPU-gated).
FLOW_FLAVOR_CODENAMES = [f"FLOW-{c}" for c in "ABCDEFGHIJ"]
_FLOW_FLAVORS = [
    Competitor(cn, "flow", f"flavors/{cn.lower()}/seed_*",
               provenance="recipe-controlled", note="pending GPU generation")
    for cn in FLOW_FLAVOR_CODENAMES
]

_BASELINES = [
    Competitor("KoVAE",        "neural", "kovae/seed_*"),
    Competitor("Diffusion-TS", "neural", "diffusion_ts/seed_*",
               note="REGENERATED in native return space 2026-07-13 (void lifted); "
                    "authors' stocks config at published defaults"),
    Competitor("TimeVAE",      "neural", "timevae/seed_*"),
    Competitor("TimeGAN",      "neural", "timegan/seed_*"),
    Competitor("TimeGAN-600",  "neural", "timegan_600/seed_*"),
    Competitor("QuantGAN",     "neural", "quantgan/seed_*"),
    # GARCH-t / DCC-t: archived tensors generated by scripts/flow/classical_baselines_run.py
    # (generators verbatim from examples/multi_universe_leaderboard.py, published defaults).
    Competitor("GARCH-t",      "classical", "garch_t/seed_*",
               note="per-asset GJR-GARCH(1,1,1)-t, assets independent"),
    Competitor("DCC-t",        "classical", "dcc_t/seed_*",
               note="GJR-GARCH(1,1,1)-t marginals + DCC Student-t copula"),
    # Replay/resampling family: what a practitioner uses INSTEAD of a deep
    # generator (historical simulation & friends). Ranked on every board they
    # apply to, provenance-disclosed; memorization-guard flags are expected by
    # construction, not a defect.
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
               note="Student-t copula + EMPIRICAL marginals (semi-parametric: simulated "
                    "values interpolate train order statistics, so a SUSPICIOUS "
                    "memorization flag is expected by construction), iid in time"),
]

# Research-surfaced baselines to ADD. Each is a repo to
# integrate + a generation run, so PENDING (seed_glob points at where its tensors
# will land). Tier 1 competes on our turf / reviewers expect it; Tier 2 = legibility.
_PENDING_BASELINES = [
    # Tier 1
    Competitor("ImagenTime",  "neural", "imagentime/seed_*",         note="NeurIPS'24 diffusion — modern unconditional SOTA"),
    Competitor("PCF-GAN",     "neural", "pending/pcf_gan/seed_*",    note="NeurIPS'23 signature — standard sig baseline; distinct discriminator"),
    Competitor("Tail-GAN",    "neural", "pending/tail_gan/seed_*",   note="Mgmt Sci'25 — optimizes multivariate VaR/ES tail fidelity; official repo chaozhang-ox/Tail-GAN (2 bugs to patch)"),
    # Tier 2 (optional / legibility)
    Competitor("FM-TS",         "neural", "fm_ts/seed_*",                 note="rectified-flow representative"),
    Competitor("Fourier-Flows", "neural", "pending/fourier_flows/seed_*", note="the normalizing-flow reference baseline"),
    Competitor("TimeVQVAE",     "neural", "pending/timevqvae/seed_*",     note="discrete-latent + transformer prior; different family"),
    # NOTE: SigCWGAN — add only if our sig-WGAN is the UNCONDITIONAL variant (2111.01207).
]

# SUBMISSION-ONLY (decision 2026-07-13): repos verified working + specced, but their
# code carries NO LICENSE, so we do not train/publish results on them. They are not
# omitted from the field on merit — the authors can enter their own tensors via the
# public submission path if/when they choose. Do NOT add to COMPETITORS.
#   TSFlow  (ICLR'25 flow-matching, marcelkollovieh/TSFlow — no license)
#   SBBTS   (arXiv 2604.07159 Schrödinger–Bass bridge, alexouadi/SBBTS — no license)
SUBMISSION_ONLY_NO_LICENSE = ["TSFlow", "SBBTS"]

# NOT benchmarked — acknowledged in the paper as non-comparable:
#   forecasters (Chronos, Moirai, TimesFM, Lag-Llama, TimeGPT) — conditional path
#   extension, not unconditional joint generation; closed finance vendors (Skanalytix,
#   Synthera); microstructure sims (Simudyne, ABIDES). Kept out of COMPETITORS by design.
ACKNOWLEDGE_NOT_BENCHMARKED = [
    "Chronos", "Moirai", "TimesFM", "Lag-Llama", "TimeGPT",  # forecasters, not generators
    "Skanalytix", "Synthera AI",                              # closed finance vendors
    "Simudyne Horizon", "ABIDES",                             # microstructure sims
]

COMPETITORS = _FLOW_PRODUCTION + _FLOW_FLAVORS + _BASELINES + _PENDING_BASELINES


def available_competitors():
    return [c for c in COMPETITORS if c.available]


def pending_competitors():
    return [c for c in COMPETITORS if not c.available]
