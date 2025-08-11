"""
Core module providing the main Backtester interface.

This module provides the main Backtester class that follows SOLID principles
with a clean, modular architecture.

The Backtester class coordinates specialized components:
- DataFetcher: Handles data fetching and preprocessing
- StrategyManager: Manages strategy creation and validation
- EvaluationEngine: Handles performance evaluation
- BacktestRunner: Executes core backtest logic
- OptimizationOrchestrator: Manages optimization workflows
"""

import logging

# Import the main Backtester class
from .backtester_logic.backtester_facade import Backtester

logger = logging.getLogger(__name__)

# Export the main class for external use
__all__ = ["Backtester"]
