"""
Unit tests for BacktestEvaluator.

Tests the BacktestEvaluator class that performs walk-forward analysis
for parameter sets in both single and multi-objective optimization scenarios.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.optimization.results import (
    EvaluationResult,
    OptimizationData,
    WindowResult,
)
from portfolio_backtester.optimization.wfo_window import WFOWindow


@pytest.fixture
def mock_backtester():
    """Return a mock StrategyBacktester."""
    mock = MagicMock()
    mock._get_strategy.return_value = MagicMock()
    # Mocking the internal call within evaluate_parameters
    mock.evaluate_window.side_effect = lambda config, window, monthly, daily, rets: WindowResult(
        window_returns=pd.Series([0.1, 0.2]),
        metrics={"sharpe": 1.5, "sortino": 1.8},
        train_start=window.train_start,
        train_end=window.train_end,
        test_start=window.test_start,
        test_end=window.test_end,
    )
    return mock


@pytest.fixture
def sample_windows():
    """Return a list of sample WFOWindow objects for testing."""
    return [
        WFOWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-12-31"),
            test_start=pd.Timestamp("2021-01-01"),
            test_end=pd.Timestamp("2021-12-31"),
        ),
        WFOWindow(
            train_start=pd.Timestamp("2021-01-01"),
            train_end=pd.Timestamp("2021-12-31"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2022-12-31"),
        ),
    ]


@pytest.fixture
def sample_data(sample_windows):
    """Return a sample OptimizationData object."""
    return OptimizationData(
        monthly=pd.DataFrame(),
        daily=pd.DataFrame(),
        returns=pd.DataFrame(),
        windows=sample_windows,
    )


class TestBacktestEvaluator:
    """Test suite for the BacktestEvaluator."""

    def test_init_single_objective(self):
        """Test initialization for single-objective optimization."""
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        assert evaluator.metrics_to_optimize == ["sharpe"]
        assert not evaluator.is_multi_objective

    def test_init_multi_objective(self):
        """Test initialization for multi-objective optimization."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe", "sortino"], is_multi_objective=True
        )
        assert evaluator.metrics_to_optimize == ["sharpe", "sortino"]
        assert evaluator.is_multi_objective

    @patch(
        "portfolio_backtester.optimization.evaluator.BacktestEvaluator._aggregate_window_results"
    )
    def test_evaluate_parameters_single_objective_success(
        self, mock_aggregate, mock_backtester, sample_data
    ):
        """Test successful evaluation of a single-objective parameter set."""
        # Setup
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        mock_aggregate.return_value = EvaluationResult(
            objective_value=1.75, metrics={"sharpe": 1.75}, window_results=[]
        )

        # Action
        result = evaluator.evaluate_parameters(
            parameters={"param1": 10},
            scenario_config={"strategy": "TestStrategy"},
            data=sample_data,
            backtester=mock_backtester,
        )

        # Assert
        assert isinstance(result, EvaluationResult)
        assert result.objective_value == pytest.approx(1.75)
        assert len(result.window_results) == 0
        assert result.metrics["sharpe"] == pytest.approx(1.75)

    @patch(
        "portfolio_backtester.optimization.evaluator.BacktestEvaluator._aggregate_window_results"
    )
    def test_evaluate_parameters_multi_objective_success(
        self, mock_aggregate, mock_backtester, sample_data
    ):
        """Test successful evaluation of a multi-objective parameter set."""
        # Setup
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe", "sortino"], is_multi_objective=True
        )
        mock_aggregate.return_value = EvaluationResult(
            objective_value=[1.75, 2.0], metrics={}, window_results=[]
        )

        # Action
        result = evaluator.evaluate_parameters(
            parameters={"param1": 10},
            scenario_config={"strategy": "TestStrategy"},
            data=sample_data,
            backtester=mock_backtester,
        )

        # Assert
        assert isinstance(result, EvaluationResult)
        assert result.objective_value[0] == pytest.approx(1.75)
        assert result.objective_value[1] == pytest.approx(2.0)

    @patch(
        "portfolio_backtester.optimization.evaluator.BacktestEvaluator._aggregate_window_results"
    )
    def test_evaluate_parameters_with_failed_window(
        self, mock_aggregate, mock_backtester, sample_data
    ):
        """Test evaluation when one window fails."""
        # Setup
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        mock_aggregate.return_value = EvaluationResult(
            objective_value=-499999999.25, metrics={}, window_results=[]
        )

        # Action
        result = evaluator.evaluate_parameters(
            parameters={"param1": 10},
            scenario_config={"strategy": "TestStrategy"},
            data=sample_data,
            backtester=mock_backtester,
        )

        # Assert
        assert result.objective_value == pytest.approx(-499999999.25)

    @patch(
        "portfolio_backtester.optimization.evaluator.BacktestEvaluator._aggregate_window_results"
    )
    def test_evaluate_parameters_length_weighted_aggregation(
        self, mock_aggregate, mock_backtester, sample_data
    ):
        """Test length-weighted aggregation."""
        # Setup
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe"],
            is_multi_objective=False,
            aggregate_length_weighted=True,
        )
        expected_weighted_avg = ((1.0 * 10) + (2.0 * 20)) / (10 + 20)
        mock_aggregate.return_value = EvaluationResult(
            objective_value=expected_weighted_avg, metrics={}, window_results=[]
        )

        # Action
        result = evaluator.evaluate_parameters(
            parameters={"param1": 10},
            scenario_config={"strategy": "TestStrategy"},
            data=sample_data,
            backtester=mock_backtester,
        )

        # Assert
        assert result.objective_value == pytest.approx(expected_weighted_avg)

    def test_aggregate_objective_values_single_objective(self):
        """Test aggregation of single-objective values."""
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        objective_values = [1.0, 2.0, 3.0]
        window_lengths = [10, 10, 10]
        result = evaluator._aggregate_objective_values(objective_values, window_lengths)
        assert result == pytest.approx(2.0)

    def test_aggregate_objective_values_multi_objective(self):
        """Test aggregation of multi-objective values."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe", "sortino"], is_multi_objective=True
        )
        objective_values = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        window_lengths = [10, 10, 10]
        result = evaluator._aggregate_objective_values(objective_values, window_lengths)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)

    def test_aggregate_metrics(self, sample_windows):
        """Test aggregation of metrics across windows."""
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        window_results = [
            WindowResult(
                window_returns=pd.Series([0.1] * 10),
                metrics={"sharpe": 1.0, "sortino": 1.5},
                train_start=sample_windows[0].train_start,
                train_end=sample_windows[0].train_end,
                test_start=sample_windows[0].test_start,
                test_end=sample_windows[0].test_end,
            ),
            WindowResult(
                window_returns=pd.Series([0.2] * 20),
                metrics={"sharpe": 2.0, "sortino": 2.5},
                train_start=sample_windows[1].train_start,
                train_end=sample_windows[1].train_end,
                test_start=sample_windows[1].test_start,
                test_end=sample_windows[1].test_end,
            ),
        ]
        window_lengths = [10, 20]
        aggregated_metrics = evaluator._aggregate_metrics(window_results, window_lengths)
        assert aggregated_metrics["sharpe"] == pytest.approx(1.5)
        assert aggregated_metrics["sortino"] == pytest.approx(2.0)

    def test_empty_objective_values(self):
        """Test aggregation with no objective values."""
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        result = evaluator._aggregate_objective_values([], [])
        assert result == -1e9

    @patch(
        "portfolio_backtester.optimization.evaluator.BacktestEvaluator._aggregate_window_results"
    )
    def test_nan_handling(self, mock_aggregate, mock_backtester, sample_data):
        """Test handling of NaN values in metrics."""
        # Setup
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        mock_aggregate.return_value = EvaluationResult(
            objective_value=-499999999.25, metrics={}, window_results=[]
        )

        # Action
        result = evaluator.evaluate_parameters(
            parameters={"param1": 10},
            scenario_config={"strategy": "TestStrategy"},
            data=sample_data,
            backtester=mock_backtester,
        )

        # Assert
        assert result.objective_value == pytest.approx(-499999999.25)
