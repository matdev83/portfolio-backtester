# This file makes the directory a Python package.

# ---------------------------------------------------------------------------
# Lazy attribute access to avoid side-effects at package import time.
# This prevents the RuntimeWarning shown when executing
#   python -m portfolio_backtester.spy_holdings
# because the sub-module is no longer imported automatically.
# ---------------------------------------------------------------------------

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:  # pragma: no cover – static type checkers only
    from .universe_data.spy_holdings import get_top_weight_sp500_components  # noqa: F401


_LAZY_ATTRS = {
    "get_top_weight_sp500_components": "spy_holdings",
}


def __getattr__(name: str) -> Any:  # noqa: D401 – simple function
    """Dynamically import *name* from its sub-module on first access."""
    if name in _LAZY_ATTRS:
        module_name = _LAZY_ATTRS[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[name])
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent look-ups
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
