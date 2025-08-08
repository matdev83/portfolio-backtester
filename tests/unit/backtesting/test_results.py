"""
Unit tests for result data classes.

These tests verify that all result data structures work correctly
and maintain proper type safety.
"""

import pytest
import pandas as pd
import numpy as np

from portfolio_backtester.optimization.results import (
    EvaluationResult,
    OptimizationResult,
    OptimizationData,
)
from portfolio_backtester.backtesting.results import (
    BacktestResult,
    WindowResult,
)

class TestBacktestResult:
    """Test BacktestResult dataclass."""
    
    def test_backtest_result_creation(self):
        """Test BacktestResult can be created with all required fields."""
        returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range('2020-01-01', periods=3))
        metrics = {'sharpe_ratio': 1.2, 'max_drawdown': -0.05}
        trade_history = pd.DataFrame({'ticker': ['AAPL'], 'quantity': [100]})
        performance_stats = {'total_return': 0.15}
        charts_data = {'cumulative_returns': returns.cumsum()}
        
        result = BacktestResult(
            returns=returns,
            metrics=metrics,
            trade_history=trade_history,
            performance_stats=performance_stats,
            charts_data=charts_data
        )
        
        assert isinstance(result.returns, pd.Series)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.trade_history, pd.DataFrame)
        assert isinstance(result.performance_stats, dict)
        assert isinstance(result.charts_data, dict)
        
        pd.testing.assert_series_equal(result.returns, returns)
        assert result.metrics == metrics
        pd.testing.assert_frame_equal(result.trade_history, trade_history)
        assert result.performance_stats == performance_stats
        assert result.charts_data == charts_data
    
    def test_backtest_result_empty(self):
        """Test BacktestResult with empty data."""
        result = BacktestResult(
            returns=pd.Series(dtype=float),
            metrics={},
            trade_history=pd.DataFrame(),
            performance_stats={},
            charts_data={}
        )
        
        assert result.returns.empty
        assert result.metrics == {}
        assert result.trade_history.empty
        assert result.performance_stats == {}
        assert result.charts_data == {}


class TestWindowResult:
    """Test WindowResult dataclass."""
    
    def test_window_result_creation(self):
        """Test WindowResult can be created with all required fields."""
        window_returns = pd.Series([0.01, 0.02], index=pd.date_range('2020-07-01', periods=2))
        metrics = {'return': 0.03, 'volatility': 0.15}
        train_start = pd.Timestamp('2020-01-01')
        train_end = pd.Timestamp('2020-06-30')
        test_start = pd.Timestamp('2020-07-01')
        test_end = pd.Timestamp('2020-12-31')
        
        result = WindowResult(
            window_returns=window_returns,
            metrics=metrics,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end
        )
        
        pd.testing.assert_series_equal(result.window_returns, window_returns)
        assert result.metrics == metrics
        assert result.train_start == train_start
        assert result.train_end == train_end
        assert result.test_start == test_start
        assert result.test_end == test_end
    
    def test_window_result_timestamp_types(self):
        """Test that WindowResult properly handles timestamp types."""
        result = WindowResult(
            window_returns=pd.Series(dtype=float),
            metrics={},
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-06-30'),
            test_start=pd.Timestamp('2020-07-01'),
            test_end=pd.Timestamp('2020-12-31')
        )
        
        assert isinstance(result.train_start, pd.Timestamp)
        assert isinstance(result.train_end, pd.Timestamp)
        assert isinstance(result.test_start, pd.Timestamp)
        assert isinstance(result.test_end, pd.Timestamp)


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""
    
    def test_evaluation_result_single_objective(self):
        """Test EvaluationResult with single objective value."""
        window_results = [
            WindowResult(
                window_returns=pd.Series([0.01]),
                metrics={'return': 0.01},
                train_start=pd.Timestamp('2020-01-01'),
                train_end=pd.Timestamp('2020-06-30'),
                test_start=pd.Timestamp('2020-07-01'),
                test_end=pd.Timestamp('2020-12-31')
            )
        ]
        
        result = EvaluationResult(
            objective_value=0.15,
            metrics={'avg_return': 0.15, 'avg_volatility': 0.12},
            window_results=window_results
        )
        
        assert result.objective_value == 0.15
        assert isinstance(result.objective_value, float)
        assert result.metrics == {'avg_return': 0.15, 'avg_volatility': 0.12}
        assert len(result.window_results) == 1
        assert isinstance(result.window_results[0], WindowResult)
    
    def test_evaluation_result_multi_objective(self):
        """Test EvaluationResult with multiple objective values."""
        result = EvaluationResult(
            objective_value=[0.15, -0.05, 1.2],  # return, drawdown, sharpe
            metrics={'avg_return': 0.15, 'max_drawdown': -0.05, 'sharpe_ratio': 1.2},
            window_results=[]
        )
        
        assert result.objective_value == [0.15, -0.05, 1.2]
        assert isinstance(result.objective_value, list)
        assert len(result.objective_value) == 3


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_single_objective(self):
        """Test OptimizationResult with single objective."""
        best_params = {'lookback': 20, 'num_holdings': 10}
        optimization_history = [
            {'params': {'lookback': 15, 'num_holdings': 5}, 'value': 0.10},
            {'params': {'lookback': 20, 'num_holdings': 10}, 'value': 0.15}
        ]
        
        result = OptimizationResult(
            best_parameters=best_params,
            best_value=0.15,
            n_evaluations=2,
            optimization_history=optimization_history,
            best_trial=None
        )
        
        assert result.best_parameters == best_params
        assert result.best_value == 0.15
        assert result.n_evaluations == 2
        assert len(result.optimization_history) == 2
        assert result.best_trial is None
    
    def test_optimization_result_multi_objective(self):
        """Test OptimizationResult with multiple objectives."""
        result = OptimizationResult(
            best_parameters={'param1': 1.0},
            best_value=[0.15, -0.05, 1.2],
            n_evaluations=100,
            optimization_history=[],
            best_trial="mock_trial_object"
        )
        
        assert isinstance(result.best_value, list)
        assert len(result.best_value) == 3
        assert result.best_trial == "mock_trial_object"
    
    def test_optimization_result_default_best_trial(self):
        """Test OptimizationResult with default best_trial value."""
        result = OptimizationResult(
            best_parameters={},
            best_value=0.0,
            n_evaluations=0,
            optimization_history=[]
        )
        
        assert result.best_trial is None


class TestOptimizationData:
    """Test OptimizationData dataclass."""
    
    def test_optimization_data_creation(self):
        """Test OptimizationData can be created with all required fields."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create sample data
        monthly_data = pd.DataFrame(
            np.random.randn(12, len(tickers)),
            index=pd.date_range('2020-01-01', '2020-12-01', freq='MS'),
            columns=tickers
        )
        
        daily_data = pd.DataFrame(
            np.random.randn(len(dates), len(tickers)),
            index=dates,
            columns=tickers
        )
        
        returns_data = daily_data.pct_change().fillna(0)
        
        windows = [
            (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-30'),
             pd.Timestamp('2020-07-01'), pd.Timestamp('2020-12-31'))
        ]
        
        opt_data = OptimizationData(
            monthly=monthly_data,
            daily=daily_data,
            returns=returns_data,
            windows=windows
        )
        
        pd.testing.assert_frame_equal(opt_data.monthly, monthly_data)
        pd.testing.assert_frame_equal(opt_data.daily, daily_data)
        pd.testing.assert_frame_equal(opt_data.returns, returns_data)
        assert opt_data.windows == windows
    
    def test_optimization_data_windows_format(self):
        """Test that OptimizationData properly handles window tuples."""
        windows = [
            (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-03-31'),
             pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30')),
            (pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'),
             pd.Timestamp('2020-07-01'), pd.Timestamp('2020-09-30'))
        ]
        
        opt_data = OptimizationData(
            monthly=pd.DataFrame(),
            daily=pd.DataFrame(),
            returns=pd.DataFrame(),
            windows=windows
        )
        
        assert len(opt_data.windows) == 2
        for window in opt_data.windows:
            assert len(window) == 4  # train_start, train_end, test_start, test_end
            assert all(isinstance(ts, pd.Timestamp) for ts in window)
    
    def test_optimization_data_empty(self):
        """Test OptimizationData with empty data."""
        opt_data = OptimizationData(
            monthly=pd.DataFrame(),
            daily=pd.DataFrame(),
            returns=pd.DataFrame(),
            windows=[]
        )
        
        assert opt_data.monthly.empty
        assert opt_data.daily.empty
        assert opt_data.returns.empty
        assert opt_data.windows == []


class TestResultDataClassesIntegration:
    """Test integration between different result data classes."""
    
    def test_window_results_in_evaluation_result(self):
        """Test that WindowResult objects work properly within EvaluationResult."""
        window_results = []
        
        # Use valid date ranges
        date_ranges = [
            ('2020-01-01', '2020-03-31', '2020-04-01', '2020-06-30'),
            ('2020-04-01', '2020-06-30', '2020-07-01', '2020-09-30'),
            ('2020-07-01', '2020-09-30', '2020-10-01', '2020-12-31')
        ]
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(date_ranges):
            window_result = WindowResult(
                window_returns=pd.Series([0.01 * (i + 1)]),
                metrics={'return': 0.01 * (i + 1)},
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end)
            )
            window_results.append(window_result)
        
        eval_result = EvaluationResult(
            objective_value=0.06,  # Sum of window returns
            metrics={'avg_return': 0.02},
            window_results=window_results
        )
        
        assert len(eval_result.window_results) == 3
        assert all(isinstance(wr, WindowResult) for wr in eval_result.window_results)
        
        # Verify we can access individual window data
        first_window = eval_result.window_results[0]
        assert first_window.metrics['return'] == 0.01
    
    def test_result_classes_type_safety(self):
        """Test that result classes maintain proper type safety."""
        # This test ensures that the dataclasses properly enforce types
        
        # Test that we can create valid instances
        backtest_result = BacktestResult(
            returns=pd.Series([0.01]),
            metrics={'test': 1.0},
            trade_history=pd.DataFrame(),
            performance_stats={},
            charts_data={}
        )
        
        window_result = WindowResult(
            window_returns=pd.Series([0.01]),
            metrics={'test': 1.0},
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-06-30'),
            test_start=pd.Timestamp('2020-07-01'),
            test_end=pd.Timestamp('2020-12-31')
        )
        
        eval_result = EvaluationResult(
            objective_value=1.0,
            metrics={'test': 1.0},
            window_results=[window_result]
        )
        
        opt_result = OptimizationResult(
            best_parameters={'test': 1.0},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[]
        )
        
        opt_data = OptimizationData(
            monthly=pd.DataFrame(),
            daily=pd.DataFrame(),
            returns=pd.DataFrame(),
            windows=[]
        )
        
        # Verify all instances were created successfully
        assert isinstance(backtest_result, BacktestResult)
        assert isinstance(window_result, WindowResult)
        assert isinstance(eval_result, EvaluationResult)
        assert isinstance(opt_result, OptimizationResult)
        assert isinstance(opt_data, OptimizationData)


if __name__ == '__main__':
    pytest.main([__file__])