from __future__ import annotations

# Only import the base classes that are commonly used
from .base.base_strategy import BaseStrategy
from .base.portfolio_strategy import PortfolioStrategy
from .base.signal_strategy import SignalStrategy

__all__ = [
    "BaseStrategy",
    "PortfolioStrategy", 
    "SignalStrategy",
    "enumerate_strategies_with_params",
]

# --------------------------------------------------------------------
# Dynamic discovery helper
# --------------------------------------------------------------------

import importlib
import inspect
import pkgutil
import re
from typing import Dict, List
import os

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


def enumerate_strategies_with_params() -> Dict[str, type]:  # pragma: no cover
    """Enumerate all concrete strategy classes and their tunable parameters.

    The function performs the following steps:
    1. Dynamically imports every module in this *strategies* package so that all
       subclasses of :class:`~portfolio_backtester.strategies.base.base_strategy.BaseStrategy`
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
    package_path = os.path.dirname(__file__)

    for subdir in ['portfolio', 'signal']:
        subdir_path = os.path.join(package_path, subdir)
        if os.path.isdir(subdir_path):
            for module_info in pkgutil.walk_packages([subdir_path], prefix=f"{package_name}.{subdir}."):
                try:
                    importlib.import_module(module_info.name)
                except Exception:
                    # If a module fails to import we skip it so that discovery continues
                    continue

    # Local import to avoid circular dependencies at module import time
    from .base.base_strategy import BaseStrategy

    def is_concrete_strategy(cls) -> bool:
        """Check if *cls* is a concrete subclass of BaseStrategy."""
        return (
            inspect.isclass(cls)
            and issubclass(cls, BaseStrategy)
            and cls is not BaseStrategy
            and not inspect.isabstract(cls)
        )

    # 2. Collect subclasses (recursively)
    discovered: Dict[str, type] = {}
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
        discovered[strategy_key] = cls

    # Add manual mappings for naming inconsistencies between config files and auto-generated names
    strategy_aliases = {
        'ema_crossover': 'ema',  # EMAStrategy should be accessible as both 'ema' and 'ema_crossover'
        'ema_roro': 'ema_ro_ro',  # EMARoRoStrategy should be accessible as both 'ema_ro_ro' and 'ema_roro'
    }
    
    # Add aliases to the discovered strategies
    for alias, canonical_name in strategy_aliases.items():
        if canonical_name in discovered:
            discovered[alias] = discovered[canonical_name]

    return discovered


# The enumerate_strategies_with_params function is already included in __all__ above