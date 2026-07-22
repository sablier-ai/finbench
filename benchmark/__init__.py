"""FinBench multi-task benchmark framework.

Fixed tasks, identical conditions, every competitor scored the same way; a
per-task leaderboard plus a rank-based aggregate (see ../BENCHMARK_TASKS.md).
"""

# --- finval version gate ----------------------------------------------------
# Scores are only comparable within a finval minor version, and the scorers read
# fields that do not exist in every release: `tasks.py` gates on
# MetricResult.assessable, which finval added in 0.6.0. Under an older finval that
# is an AttributeError raised inside the per-rep loop in `run.py`, where a bare
# `except Exception: continue` swallows it — so the wrong finval does not crash,
# it silently yields an EMPTY noise floor and a board that looks published but is
# empty or incomparable. That is the worst failure mode for a public benchmark,
# so check once, loudly, at import. requirements.txt pins the same range.

_FINVAL_MIN = (0, 6, 0)
_FINVAL_MAX_EXCLUSIVE = (0, 7, 0)


def _parse_version(v: str) -> tuple:
    """Leading numeric components only, so '0.6.1rc1' -> (0, 6, 1)."""
    parts = []
    for chunk in str(v).split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_finval() -> None:
    try:
        import finval
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "FinBench requires the `finval` scoring library.\n"
            "    pip install -r requirements.txt"
        ) from exc

    raw = getattr(finval, "__version__", "")
    ver = _parse_version(raw)
    if not ver:  # unparseable — allow through rather than block an unusual install
        return
    padded = ver + (0,) * (3 - len(ver))
    if not (_FINVAL_MIN <= padded < _FINVAL_MAX_EXCLUSIVE):
        want = ">={}.{}.{},<{}.{}".format(*_FINVAL_MIN, *_FINVAL_MAX_EXCLUSIVE[:2])
        raise ImportError(
            f"FinBench requires finval {want}, but finval {raw} is installed.\n"
            f"Scores are only comparable within a finval minor version, and the "
            f"scorers read fields (e.g. MetricResult.assessable) that other releases "
            f"may not have — a mismatch would silently produce an empty or "
            f"incomparable leaderboard rather than an error.\n"
            f"    pip install -r requirements.txt"
        )


_check_finval()
del _check_finval
