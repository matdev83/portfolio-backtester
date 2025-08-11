"""
Structured result data classes for optimization.

This module defines all result data structures used throughout the
optimization process, ensuring type safety and clear data flow between components.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
import pandas as pd

from ..backtesting.results import WindowResult
from .wfo_window import WFOWindow


@dataclass
class OptimizationDataContext:
    """Lightweight, serializable context for parallel optimization workers.

    Contains file paths to memory-mapped NumPy arrays and pickled metadata,
    allowing workers to reconstruct the full `OptimizationData` without
    incurring the high cost of pickling large DataFrames.

    Attributes:
        daily_data_path: Path to the memory-mapped file for daily OHLC data.
        monthly_data_path: Path to the memory-mapped file for monthly data.
        returns_data_path: Path to the memory-mapped file for daily returns data.
        metadata_path: Path to the pickled file containing index/column info.
        windows: List of walk-forward window definitions (small enough to pickle).
    """

    daily_data_path: str
    monthly_data_path: str
    returns_data_path: str
    metadata_path: str
    windows: List[WFOWindow]


@dataclass
class EvaluationResult:
    """Result of evaluating a parameter set across all windows.

    Aggregates the results from evaluating a single parameter set
    across all walk-forward windows in an optimization.

    Attributes:
        objective_value: Single value or tuple for multi-objective optimization
        metrics: Aggregated performance metrics across all windows
        window_results: List of individual window results
    """

    objective_value: Union[float, List[float]]
    metrics: Dict[str, float]
    window_results: List[WindowResult]


@dataclass
class OptimizationResult:
    """Final optimization result with best parameters.

    Contains the final results from a complete optimization run,
    including the best parameters found and optimization history.

    Attributes:
        best_parameters: Dictionary of optimal parameter values
        best_value: Best objective value(s) achieved
        n_evaluations: Total number of parameter evaluations performed
        optimization_history: History of all evaluations performed
        best_trial: Reference to the best trial object (optimizer-specific)
    """

    best_parameters: Dict[str, Any]
    best_value: Union[float, List[float]]
    n_evaluations: int
    optimization_history: List[Dict[str, Any]]
    best_trial: Optional[Any] = None


@dataclass
class OptimizationData:
    """Data container for optimization process.

    Contains all the data needed for running an optimization,
    including price data and walk-forward windows.

    Attributes:
        monthly: Monthly price data
        daily: Daily OHLC price data
        returns: Daily returns data
        windows: List of walk-forward window definitions
    """

    monthly: pd.DataFrame
    daily: pd.DataFrame
    returns: pd.DataFrame
    windows: List[WFOWindow]
