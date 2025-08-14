"""
EvaluationStrategy interface and implementations for polymorphic evaluation methods.

Replaces isinstance checks in evaluation logic with proper polymorphic behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IEvaluationStrategy(ABC):
    """Abstract interface for evaluation strategies."""

    @abstractmethod
    def evaluate_performance(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate performance metrics for given returns.

        Args:
            returns: Portfolio returns (Series or DataFrame)
            benchmark_returns: Optional benchmark returns for comparison
            metrics: List of metrics to calculate

        Returns:
            Dictionary of metric names to values

        Raises:
            ValueError: If returns data is invalid
        """
        pass

    @abstractmethod
    def evaluate_parameters(
        self, parameters: Dict[str, Any], scenario_config: Dict[str, Any], data: Any
    ) -> Dict[str, float]:
        """
        Evaluate a specific parameter set.

        Args:
            parameters: Parameter dictionary to evaluate
            scenario_config: Scenario configuration
            data: Data for evaluation

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If parameters or data are invalid
        """
        pass


class StandardEvaluationStrategy(IEvaluationStrategy):
    """Standard implementation of evaluation strategy."""

    def __init__(self, global_config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation strategy.

        Args:
            global_config: Global configuration dictionary
        """
        self.global_config = global_config or {}

    def evaluate_performance(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate performance using standard metrics.

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            metrics: List of metrics to calculate

        Returns:
            Dictionary of performance metrics
        """
        try:
            from ..reporting.metrics import calculate_metrics

            # Convert DataFrame to Series if needed
            if isinstance(returns, pd.DataFrame):
                if returns.shape[1] == 1:
                    returns = returns.iloc[:, 0]
                else:
                    # Use first column or combine somehow
                    returns = returns.iloc[:, 0]

            # Calculate standard metrics
            result = calculate_metrics(
                returns,
                benchmark_returns,
                "SPY",
                name="Strategy",  # Default benchmark ticker name
            )

            # Ensure we return Dict[str, float]
            if not isinstance(result, dict):
                return {"error": 0.0}

            return {k: float(v) if v is not None else 0.0 for k, v in result.items()}

        except Exception as e:
            logger.error(f"Failed to evaluate performance: {e}")
            raise ValueError(f"Performance evaluation failed: {e}")

    def evaluate_parameters(
        self, parameters: Dict[str, Any], scenario_config: Dict[str, Any], data: Any
    ) -> Dict[str, float]:
        """
        Evaluate parameters using scenario configuration.

        Args:
            parameters: Parameters to evaluate
            scenario_config: Scenario configuration
            data: Evaluation data

        Returns:
            Evaluation results
        """
        try:
            # This would typically run a backtest with the parameters
            # and return performance metrics

            # Create temporary evaluator for this evaluation
            metrics_to_optimize = scenario_config.get("optimization", {}).get(
                "metrics", ["sharpe_ratio"]
            )
            # is_multi_objective = len(metrics_to_optimize) > 1

            # Create temporary evaluator for this evaluation
            # evaluator = BacktestEvaluator(
            #     metrics_to_optimize=metrics_to_optimize, is_multi_objective=is_multi_objective
            # )

            # Evaluate parameters (this would need the actual implementation)
            # For now, return placeholder metrics based on objectives
            result = {}
            for metric in metrics_to_optimize:
                result[metric] = 0.0

            return result

        except Exception as e:
            logger.error(f"Failed to evaluate parameters: {e}")
            raise ValueError(f"Parameter evaluation failed: {e}")


class MultiObjectiveEvaluationStrategy(IEvaluationStrategy):
    """Evaluation strategy for multi-objective optimization."""

    def __init__(self, objectives: List[str], global_config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-objective evaluation strategy.

        Args:
            objectives: List of objective metrics
            global_config: Global configuration dictionary
        """
        self.objectives = objectives
        self.global_config = global_config or {}

    def evaluate_performance(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate performance for multiple objectives."""
        # Delegate to standard strategy but focus on multiple metrics
        standard_strategy = StandardEvaluationStrategy(self.global_config)
        return standard_strategy.evaluate_performance(
            returns, benchmark_returns, metrics or self.objectives
        )

    def evaluate_parameters(
        self, parameters: Dict[str, Any], scenario_config: Dict[str, Any], data: Any
    ) -> Dict[str, float]:
        """Evaluate parameters for multiple objectives."""
        standard_strategy = StandardEvaluationStrategy(self.global_config)
        return standard_strategy.evaluate_parameters(parameters, scenario_config, data)


class EvaluationStrategyFactory:
    """Factory for creating appropriate evaluation strategies."""

    @staticmethod
    def create_strategy(
        scenario_config: Dict[str, Any], global_config: Optional[Dict[str, Any]] = None
    ) -> IEvaluationStrategy:
        """
        Create appropriate evaluation strategy based on configuration.

        Args:
            scenario_config: Scenario configuration
            global_config: Global configuration

        Returns:
            Appropriate evaluation strategy
        """
        optimization_config = scenario_config.get("optimization", {})
        metrics = optimization_config.get("metrics", ["sharpe_ratio"])

        if len(metrics) > 1:
            return MultiObjectiveEvaluationStrategy(metrics, global_config)
        else:
            return StandardEvaluationStrategy(global_config)
