from __future__ import annotations

from .base_strategy import BaseStrategy
from .momentum_strategy import MomentumStrategy
from .momentum_unfiltered_atr_strategy import MomentumUnfilteredAtrStrategy
from .sharpe_momentum_strategy import SharpeMomentumStrategy
from .vams_momentum_strategy import VAMSMomentumStrategy
from .sortino_momentum_strategy import SortinoMomentumStrategy
from .calmar_momentum_strategy import CalmarMomentumStrategy
from .vams_no_downside_strategy import VAMSNoDownsideStrategy
from .momentum_dvol_sizer_strategy import MomentumDvolSizerStrategy
from .filtered_lagged_momentum_strategy import FilteredLaggedMomentumStrategy
from .ema_crossover_strategy import EMAStrategy
from .low_volatility_factor_strategy import LowVolatilityFactorStrategy

__all__ = [
    "BaseStrategy",
    "MomentumStrategy",
    "MomentumUnfilteredAtrStrategy",
    "SharpeMomentumStrategy",
    "VAMSMomentumStrategy",
    "SortinoMomentumStrategy",
    "CalmarMomentumStrategy",
    "VAMSNoDownsideStrategy",
    "MomentumDvolSizerStrategy",
    "FilteredLaggedMomentumStrategy",
    "EMAStrategy",
    "LowVolatilityFactorStrategy",
]

# --------------------------------------------------------------------
# Dynamic discovery helper
# --------------------------------------------------------------------

import inspect
import importlib
import pkgutil
import re
from typing import Dict, List


def _camel_to_snake(name: str) -> str:
    """Convert *CamelCase* class names to *snake_case* identifiers.

    The trailing "_strategy" suffix (case-insensitive) is stripped so that the
    resulting name matches the identifiers used in configuration files (e.g.
    "MomentumStrategy" -> "momentum", "VAMSMomentumStrategy" -> "vams_momentum").
    """
    # Insert underscore boundaries between camelcase transitions
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    snake = s2.lower()
    if snake.endswith("_strategy"):
        snake = snake[:-9]
    return snake


def enumerate_strategies_with_params() -> Dict[str, List[str]]:  # pragma: no cover
    """Enumerate all concrete strategy classes and their tunable parameters.

    The function performs the following steps:
    1. Dynamically imports every module in this *strategies* package so that all
       subclasses of :class:`~portfolio_backtester.strategies.base_strategy.BaseStrategy`
       are registered.
    2. Traverses the inheritance tree to collect every *concrete* subclass
       (i.e. not abstract) of ``BaseStrategy``.
    3. Converts each class name to its canonical *snake_case* identifier and
       queries its :py:meth:`tunable_parameters` hook.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from strategy identifier (snake case) to a **sorted** list of
        tunable parameter names.
    """
    # 1. Import all sub-modules to ensure classes are registered
    package_name = __name__
    for module_info in pkgutil.walk_packages(__path__, prefix=f"{package_name}."):
        try:
            importlib.import_module(module_info.name)
        except Exception:
            # If a module fails to import we skip it so that discovery continues
            continue

    # Local import to avoid circular dependencies at module import time
    from .base_strategy import BaseStrategy

    def is_concrete_strategy(cls) -> bool:
        """Check if *cls* is a concrete subclass of BaseStrategy."""
        return (
            inspect.isclass(cls)
            and issubclass(cls, BaseStrategy)
            and cls is not BaseStrategy
            and not inspect.isabstract(cls)
        )

    # 2. Collect subclasses (recursively)
    discovered: Dict[str, List[str]] = {}
    pending = list(BaseStrategy.__subclasses__())
    visited: set[type] = set()

    while pending:
        cls = pending.pop()
        if cls in visited:
            continue
        visited.add(cls)
        pending.extend(cls.__subclasses__())  # recurse

        if not is_concrete_strategy(cls):
            continue

        strategy_key = _camel_to_snake(cls.__name__)
        try:
            params = sorted(cls.tunable_parameters())  # type: ignore[attr-defined]
        except Exception:
            params = []
        discovered[strategy_key] = params

    return discovered


# Ensure the helper is publicly importable
__all__.append("enumerate_strategies_with_params")
