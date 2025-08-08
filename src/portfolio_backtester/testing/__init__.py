"""
Testing infrastructure for the portfolio backtester.

This module contains testing strategies and fixtures that were moved from
the strategies folder to improve code organization and prevent test dependencies
on user-modifiable strategies.
"""

from .strategies.dummy_signal_strategy import DummySignalStrategy
from .strategies.stop_loss_tester_strategy import StopLossTesterStrategy

__all__ = [
    "DummySignalStrategy",
    "StopLossTesterStrategy",
]