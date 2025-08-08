"""
Testing strategies for the portfolio backtester.

These strategies are designed specifically for testing the framework
and should not be used for actual trading.
"""

from .dummy_signal_strategy import DummySignalStrategy
from .stop_loss_tester_strategy import StopLossTesterStrategy

__all__ = [
    "DummySignalStrategy", 
    "StopLossTesterStrategy",
]