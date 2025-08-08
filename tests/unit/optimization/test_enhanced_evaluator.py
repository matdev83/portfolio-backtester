"""
Integration tests for enhanced BacktestEvaluator with daily evaluation support.

Tests the integration between the enhanced evaluator, window evaluator, and position tracker.
"""

import pandas as pd
from unittest.mock import Mock, patch
from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.optimization.results import EvaluationResult


class TestEnhancedBacktestEvaluator:
    """Test cases for enhanced BacktestEvaluator with daily evaluation."""
    
    def test_determine_evaluation_frequency_intramonth(self):
        """Test evaluation frequency determination for intramonth strategies."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Test intramonth strategy class
        scenario_config = {
            'strategy_class': 'SeasonalSignalStrategy',
            'strategy': 'momentum'
        }
        
        freq = evaluator._determine_evaluation_frequency(scenario_config)
        assert freq == 'D'
        
        # Test intramonth in strategy name
        scenario_config = {
            'strategy_class': 'SomeStrategy',
            'strategy': 'intramonth_momentum'
        }
        
        freq = evaluator._determine_evaluation_frequency(scenario_config)
        assert freq == 'D'
    
    def test_determine_evaluation_frequency_signal_based(self):
        """Test evaluation frequency determination for signal-based strategies."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Test daily signal-based strategy
        scenario_config = {
            'strategy_class': 'SignalStrategy',
            'timing_config': {
                'mode': 'signal_based',
                'scan_frequency': 'D'
            }
        }
        
        freq = evaluator._determine_evaluation_frequency(scenario_config)
        assert freq == 'D'
        
        # Test weekly signal-based strategy
        scenario_config = {
            'strategy_class': 'SignalStrategy',
            'timing_config': {
                'mode': 'signal_based',
                'scan_frequency': 'W'
            }
        }
        
        freq = evaluator._determine_evaluation_frequency(scenario_config)
        assert freq == 'M'  # Falls back to monthly
    
    def test_determine_evaluation_frequency_daily_rebalance(self):
        """Test evaluation frequency determination for daily rebalance strategies."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        scenario_config = {
            'strategy_class': 'SimpleMomentumPortfolioStrategy',
            'rebalance_frequency': 'D'
        }
        
        freq = evaluator._determine_evaluation_frequency(scenario_config)
        assert freq == 'D'
    
    def test_determine_evaluation_frequency_default(self):
        """Test evaluation frequency determination defaults to monthly."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        scenario_config = {
            'strategy_class': 'SimpleMomentumPortfolioStrategy',
            'rebalance_frequency': 'M'
        }
        
        freq = evaluator._determine_evaluation_frequency(scenario_config)
        assert freq == 'M'
    
    def test_get_universe_tickers_default(self):
        """Test getting universe tickers with default fallback."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Mock strategy without get_universe method
        mock_strategy = Mock()
        del mock_strategy.get_universe  # Ensure method doesn't exist
        
        tickers = evaluator._get_universe_tickers(mock_strategy)
        
        # Should return default tickers
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert 'SPY' in tickers
    
    def test_get_universe_tickers_from_strategy(self):
        """Test getting universe tickers from strategy method."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Mock strategy with get_universe method
        mock_strategy = Mock()
        mock_strategy.get_universe.return_value = ['TLT', 'GLD', 'VTI']
        
        tickers = evaluator._get_universe_tickers(mock_strategy)
        
        # Should return strategy's universe
        assert tickers == ['TLT', 'GLD', 'VTI']
        mock_strategy.get_universe.assert_called_once()
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Create test returns (1% daily for 10 days)
        returns = pd.Series([0.01] * 10)
        trades = []
        
        metrics = evaluator._calculate_metrics(returns, trades)
        
        # Check basic metrics exist
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # Total return should be approximately (1.01)^10 - 1
        expected_total_return = (1.01 ** 10) - 1
        assert abs(metrics['total_return'] - expected_total_return) < 1e-6
        
        # Max drawdown should be 0 (no drawdowns with constant positive returns)
        assert metrics['max_drawdown'] == 0.0
    
    def test_calculate_metrics_with_trades(self):
        """Test metrics calculation with trade data."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Create test returns
        returns = pd.Series([0.01] * 10)
        
        # Create mock trades
        mock_trades = []
        for i in range(3):
            mock_trade = Mock()
            mock_trade.duration_days = 5 + i  # 5, 6, 7 days
            mock_trades.append(mock_trade)
        
        metrics = evaluator._calculate_metrics(returns, mock_trades)
        
        # Check trade metrics
        assert 'num_trades' in metrics
        assert 'avg_trade_duration' in metrics
        assert 'max_trade_duration' in metrics
        assert 'min_trade_duration' in metrics
        
        assert metrics['num_trades'] == 3
        assert metrics['avg_trade_duration'] == 6.0  # (5+6+7)/3
        assert metrics['max_trade_duration'] == 7
        assert metrics['min_trade_duration'] == 5
    
    def test_calculate_metrics_empty_returns(self):
        """Test metrics calculation with empty returns."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        returns = pd.Series(dtype=float)
        trades = []
        
        metrics = evaluator._calculate_metrics(returns, trades)
        
        # Should return default values
        assert metrics['total_return'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['max_drawdown'] == 0.0
    
    @patch('portfolio_backtester.optimization.evaluator.WFO_ENHANCEMENT_AVAILABLE', True)
    def test_evaluate_parameters_chooses_daily_evaluation(self):
        """Test that evaluate_parameters chooses daily evaluation for intramonth strategies."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Mock scenario config for intramonth strategy
        scenario_config = {
            'strategy_class': 'SeasonalSignalStrategy',
            'name': 'test_strategy'
        }
        
        # Mock data with daily data
        mock_data = Mock()
        mock_data.daily = pd.DataFrame({'TLT': [100, 101, 102]})
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Mock the daily evaluation method
        evaluator._evaluate_parameters_daily = Mock(return_value=Mock())
        evaluator._evaluate_parameters_monthly = Mock(return_value=Mock())
        
        # Call evaluate_parameters
        evaluator.evaluate_parameters({}, scenario_config, mock_data, mock_backtester)
        
        # Should call daily evaluation
        evaluator._evaluate_parameters_daily.assert_called_once()
        evaluator._evaluate_parameters_monthly.assert_not_called()
    
    def test_evaluate_parameters_intramonth_always_daily(self):
        """After removal of dual-path logic, intramonth strategies should always use daily evaluation."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)

        scenario_config = {
            'strategy_class': 'SeasonalSignalStrategy',
            'name': 'test_strategy'
        }

        mock_data = Mock()
        mock_data.daily = pd.DataFrame({'TLT': [100, 101, 102]})
        mock_backtester = Mock()

        evaluator._evaluate_parameters_daily = Mock(return_value=Mock())
        evaluator._evaluate_parameters_monthly = Mock(return_value=Mock())

        evaluator.evaluate_parameters({}, scenario_config, mock_data, mock_backtester)

        evaluator._evaluate_parameters_daily.assert_called_once()
        evaluator._evaluate_parameters_monthly.assert_not_called()
    
    def test_aggregate_window_results_single_objective(self):
        """Test aggregation of window results for single objective optimization."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Create mock window results
        window_results = []
        for i in range(3):
            result = Mock()
            result.window_returns = pd.Series([0.01, 0.02, 0.01])  # Some returns
            result.trades = []
            window_results.append(result)
        
        # Mock the metrics calculation
        evaluator._calculate_metrics = Mock(return_value={'sharpe_ratio': 1.5, 'total_return': 0.1})
        
        # Aggregate results
        evaluation_result = evaluator._aggregate_window_results(window_results, {'param1': 1.0})
        
        # Check result structure
        assert isinstance(evaluation_result, EvaluationResult)
        assert evaluation_result.objective_value == 1.5  # sharpe_ratio value
        assert 'sharpe_ratio' in evaluation_result.metrics
        assert evaluation_result.window_results == window_results
    
    def test_aggregate_window_results_multi_objective(self):
        """Test aggregation of window results for multi-objective optimization."""
        evaluator = BacktestEvaluator(['sharpe_ratio', 'max_drawdown'], True)
        
        # Create mock window results
        window_results = []
        for i in range(2):
            result = Mock()
            result.window_returns = pd.Series([0.01, 0.02])
            result.trades = []
            window_results.append(result)
        
        # Mock the metrics calculation
        evaluator._calculate_metrics = Mock(return_value={
            'sharpe_ratio': 1.5, 
            'max_drawdown': -0.05,
            'total_return': 0.1
        })
        
        # Aggregate results
        evaluation_result = evaluator._aggregate_window_results(window_results, {'param1': 1.0})
        
        # Check result structure
        assert isinstance(evaluation_result, EvaluationResult)
        assert isinstance(evaluation_result.objective_value, list)
        assert len(evaluation_result.objective_value) == 2
        assert evaluation_result.objective_value[0] == 1.5  # sharpe_ratio
        assert evaluation_result.objective_value[1] == -0.05  # max_drawdown
    
    def test_aggregate_window_results_empty_returns(self):
        """Test aggregation when window results have no returns."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Create mock window results with empty returns
        window_results = []
        for i in range(2):
            result = Mock()
            result.window_returns = pd.Series(dtype=float)  # Empty series
            result.trades = []
            window_results.append(result)
        
        # Mock the metrics calculation
        evaluator._calculate_metrics = Mock(return_value={'sharpe_ratio': 0.0, 'total_return': 0.0})
        
        # Aggregate results
        evaluation_result = evaluator._aggregate_window_results(window_results, {'param1': 1.0})
        
        # Should handle empty returns gracefully
        assert evaluation_result.objective_value == 0.0
        evaluator._calculate_metrics.assert_called_once()
        
        # Check that empty series was passed to metrics calculation
        args, kwargs = evaluator._calculate_metrics.call_args
        returns_arg = args[0]
        assert len(returns_arg) == 0
        assert isinstance(returns_arg, pd.Series)