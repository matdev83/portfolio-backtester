"""
Core module providing the main Backtester interface.

This module now uses the refactored BacktesterFacade that follows SOLID principles
while maintaining backward compatibility with the original API.

The original monolithic 1481-line Backtester class has been decomposed into:
- DataFetcher: Handles data fetching and preprocessing
- StrategyManager: Manages strategy creation and validation
- EvaluationEngine: Handles performance evaluation
- BacktestRunner: Executes core backtest logic
- OptimizationOrchestrator: Manages optimization workflows
- BacktesterFacade: Provides unified interface maintaining API compatibility
"""

import logging

# Import the new facade that replaces the monolithic Backtester
from .backtester_logic.backtester_facade import BacktesterFacade

logger = logging.getLogger(__name__)

# Backward compatibility alias: Backtester now uses the refactored facade
# This ensures all existing code continues to work without changes
Backtester = BacktesterFacade

# Export the main class for external use
__all__ = ["Backtester"]
