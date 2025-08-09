"""
Simple EMA Crossover Strategy

This strategy uses exponential moving average crossovers to generate buy/sell signals.
- Long signal: Fast EMA crosses above Slow EMA
- Exit signal: Fast EMA crosses below Slow EMA

SOLID Refactoring: This strategy now uses polymorphic interfaces instead of isinstance checks:
- Uses ISignalPriceExtractor for polymorphic price extraction
- Uses IColumnHandler for polymorphic column operations
"""

import logging



# Import strategy base interface for composition instead of inheritance

# Legacy shim: re-export class from builtins
from ..builtins.signal.ema_crossover_signal_strategy import (  # noqa: F401
    EmaCrossoverSignalStrategy,
)

__all__ = ["EmaCrossoverSignalStrategy"]

logger = logging.getLogger(__name__)
