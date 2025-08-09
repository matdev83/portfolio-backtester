"""Removed legacy shim module.

Alias/compat imports have been removed in alpha to avoid maintaining legacy paths.
Use `portfolio_backtester.testing.strategies.dummy_signal_strategy.DummySignalStrategy` directly.
"""

from ..builtins.signal.dummy_signal_strategy import DummySignalStrategy  # noqa: F401

__all__ = ["DummySignalStrategy"]
