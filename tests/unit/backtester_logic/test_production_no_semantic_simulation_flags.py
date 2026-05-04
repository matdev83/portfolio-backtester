"""Production sources must not expose semantic opt-in flags for alternate simulation paths."""

from __future__ import annotations

import pathlib

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[3] / "src" / "portfolio_backtester"

_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "ndarray_simulation",
    "array_first_portfolio_prep",
    "enable_vectorized_tracking",
    "use_vectorized_tracking",
    "vectorized_tracking_enabled",
)


def test_portfolio_backtester_package_has_no_semantic_simulation_flag_names() -> None:
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        bad = tuple(s for s in _FORBIDDEN_SUBSTRINGS if s.lower() in lower)
        assert not bad, f"{path} must not reference semantic simulation flags: {bad}"
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert (
            "ENABLE_NUMBA_WALKFORWARD" not in text
        ), f"{path} must not reference ENABLE_NUMBA_WALKFORWARD"
        assert (
            "DISABLE_NUMBA_WALKFORWARD" not in text
        ), f"{path} must not reference DISABLE_NUMBA_WALKFORWARD"
