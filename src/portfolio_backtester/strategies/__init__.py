from __future__ import annotations

# Re-export base types from new core path to preserve import surface
from ._core.base import (
    BaseStrategy,
    PortfolioStrategy,
    SignalStrategy,
    BaseMetaStrategy,
)

# Backward-compatibility aliases for moved modules
import sys as _sys
import importlib as _importlib

# strategies.base -> strategies._core.base.base
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base.base_strategy",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base.base_strategy"),
)
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base.portfolio_strategy",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base.portfolio_strategy"),
)
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base.signal_strategy",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base.signal_strategy"),
)
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base.meta_strategy",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base.meta_strategy"),
)

# Expose moved trade utilities for legacy imports
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base.trade_aggregator",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base.trade_aggregator"),
)
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base.trade_interceptor",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base.trade_interceptor"),
)
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base.trade_record",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base.trade_record"),
)

# strategies.registry.* -> strategies._core.registry.*
_sys.modules.setdefault(
    "portfolio_backtester.strategies.registry",
    _importlib.import_module("portfolio_backtester.strategies._core.registry"),
)
_sys.modules.setdefault(
    "src.portfolio_backtester.strategies.registry",
    _importlib.import_module("portfolio_backtester.strategies._core.registry"),
)

# strategies.strategy_factory -> strategies._core.strategy_factory
_sys.modules.setdefault(
    "portfolio_backtester.strategies.strategy_factory",
    _importlib.import_module("portfolio_backtester.strategies._core.strategy_factory"),
)
_sys.modules.setdefault(
    "portfolio_backtester.strategies.base",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base"),
)

# strategies.candidate_weights / leverage_and_smoothing -> strategies._core.*
_sys.modules.setdefault(
    "portfolio_backtester.strategies.candidate_weights",
    _importlib.import_module("portfolio_backtester.strategies._core.candidate_weights"),
)
_sys.modules.setdefault(
    "portfolio_backtester.strategies.leverage_and_smoothing",
    _importlib.import_module("portfolio_backtester.strategies._core.leverage_and_smoothing"),
)

# Also support src.portfolio_backtester.* legacy namespace used in some tests
_sys.modules.setdefault(
    "src.portfolio_backtester.strategies.strategy_factory",
    _importlib.import_module("portfolio_backtester.strategies._core.strategy_factory"),
)
_sys.modules.setdefault(
    "src.portfolio_backtester.strategies.base",
    _importlib.import_module("portfolio_backtester.strategies._core.base.base"),
)

_sys.modules.setdefault(
    "src.portfolio_backtester.strategies.candidate_weights",
    _importlib.import_module("portfolio_backtester.strategies._core.candidate_weights"),
)
_sys.modules.setdefault(
    "src.portfolio_backtester.strategies.leverage_and_smoothing",
    _importlib.import_module("portfolio_backtester.strategies._core.leverage_and_smoothing"),
)

# Expose commonly patched submodules as attributes on this package
strategy_factory = _importlib.import_module(
    "portfolio_backtester.strategies._core.strategy_factory"
)
registry = _importlib.import_module("portfolio_backtester.strategies._core.registry")
base = _importlib.import_module("portfolio_backtester.strategies._core.base.base")

__all__ = [
    "BaseStrategy",
    "PortfolioStrategy",
    "SignalStrategy",
    "BaseMetaStrategy",
]
