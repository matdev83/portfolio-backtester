"""
Unit tests for StrategyBacktester class.

These tests verify that the pure backtesting engine works correctly
in complete isolation from optimization components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.backtesting.results import BacktestResult, WindowResult


class TestStrategyBacktester:
    """Test suite for StrategyBacktester class."""
    
    @pytest.fixture
    def mock_global_config(self):
        """Mock global configuration."""
        return {
            'benchmark': 'SPY',
            'universe': ['AAPL', 'MSFT', 'GOOGL'],
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'portfolio_value': 100000.0
        }
    
    @pytest.fixture
    def mock_data_source(self):
        """Mock data source."""
        mock_source = Mock()
        mock_source.get_data.return_value = pd.DataFrame()
        return mock_source
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
        
        # Create daily data
        daily_data = {}
        for ticker in tickers:
            np.random.seed(42)  # For reproducible tests
            prices = 100 * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
            daily_data[ticker] = prices
        
        daily_df = pd.DataFrame(daily_data, index=dates)
        
        # Create monthly data
        monthly_df = daily_df.resample('BME').last()
        
        # Create returns data
        returns_df = daily_df.pct_change().fillna(0)
        
        return daily_df, monthly_df, returns_df
    
    @pytest.fixture
    def mock_strategy(self):
        """Mock strategy for testing."""
        strategy = Mock()
        strategy.get_universe.return_value = [('AAPL', 1.0), ('MSFT', 1.0), ('GOOGL', 1.0)]
        strategy.get_timing_controller.return_value = Mock()
        return strategy
    
    @pytest.fixture
    def strategy_config(self):
        """Sample strategy configuration."""
        return {
            'name': 'test_strategy',
            'strategy': 'momentum_strategy',
            'strategy_params': {
                'lookback_period': 20,
                'num_holdings': 10
            },
            'universe': ['AAPL', 'MSFT', 'GOOGL'],
            'rebalance_frequency': 'monthly'
        }
    
    @pytest.fixture
    def backtester(self, mock_global_config, mock_data_source):
        """Create StrategyBacktester instance for testing."""
        with patch('portfolio_backtester.backtesting.strategy_backtester.get_strategy_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.get_all_strategies.return_value = {'dummy': Mock}
            mock_get_registry.return_value = mock_registry
            
            backtester = StrategyBacktester(mock_global_config, mock_data_source)
            return backtester

    def test_initialization(self, mock_global_config, mock_data_source):
        """Test StrategyBacktester initialization."""
        with patch('portfolio_backtester.backtesting.strategy_backtester.get_strategy_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.get_all_strategies.return_value = {'dummy': Mock}
            mock_get_registry.return_value = mock_registry
            
            backtester = StrategyBacktester(mock_global_config, mock_data_source)
            
            assert backtester.global_config == mock_global_config
            assert backtester.data_source == mock_data_source
            assert isinstance(backtester.strategy_map, dict)
            assert backtester.data_cache is not None
    
    def test_get_strategy_success(self, backtester, mock_strategy):
        """Test successful strategy retrieval."""
        # Make mock_strategy inherit from BaseStrategy
        from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy
        mock_strategy.__class__ = type('MockStrategy', (BaseStrategy,), {})
        
        # Mock the registry get_strategy_class method
        backtester._registry.get_strategy_class.return_value = lambda params: mock_strategy
        
        result = backtester._get_strategy('test_strategy', {'param1': 'value1'})
        
        assert result == mock_strategy
    
    def test_get_strategy_unsupported(self, backtester):
        """Test error handling for unsupported strategy."""
        # Mock the registry to return None for nonexistent strategy
        backtester._registry.get_strategy_class.return_value = None
        
        with pytest.raises(ValueError, match="Unsupported strategy: nonexistent_strategy"):
            backtester._get_strategy('nonexistent_strategy', {})
    
    def test_get_strategy_wrong_type(self, backtester):
        """Test error handling when strategy class returns wrong type."""
        # Mock the registry to return a function that doesn't return a BaseStrategy
        backtester._registry.get_strategy_class.return_value = lambda params: "not_a_strategy"
        
        with pytest.raises(TypeError, match="did not return a BaseStrategy instance"):
            backtester._get_strategy('bad_strategy', {})
    
    @patch('portfolio_backtester.backtesting.strategy_backtester.generate_signals')
    @patch('portfolio_backtester.backtesting.strategy_backtester.size_positions')
    @patch('portfolio_backtester.backtesting.strategy_backtester.calculate_portfolio_returns')
    @patch('portfolio_backtester.backtesting.strategy_backtester.prepare_scenario_data')
    @patch('portfolio_backtester.backtesting.strategy_backtester.calculate_metrics')
    def test_backtest_strategy_success(
        self, 
        mock_calc_metrics,
        mock_prepare_data,
        mock_calc_returns,
        mock_size_positions,
        mock_generate_signals,
        backtester, 
        strategy_config, 
        sample_price_data,
        mock_strategy
    ):
        """Test successful backtest execution."""
        daily_df, monthly_df, returns_df = sample_price_data
        
        # Make mock_strategy inherit from BaseStrategy
        from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy
        mock_strategy.__class__ = type('MockStrategy', (BaseStrategy,), {})
        
        # Setup mocks
        backtester._registry.get_strategy_class.return_value = lambda params: mock_strategy
        mock_prepare_data.return_value = (monthly_df, returns_df)
        mock_generate_signals.return_value = pd.DataFrame()
        mock_size_positions.return_value = pd.DataFrame()
        
        # Create mock portfolio returns
        portfolio_returns = pd.Series(
            np.random.randn(len(daily_df)) * 0.01,
            index=daily_df.index
        )
        mock_calc_returns.return_value = portfolio_returns
        
        # Mock metrics
        mock_calc_metrics.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05
        }
        
        # Execute backtest
        result = backtester.backtest_strategy(strategy_config, monthly_df, daily_df, returns_df)
        
        # Verify result structure
        assert isinstance(result, BacktestResult)
        assert isinstance(result.returns, pd.Series)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.trade_history, pd.DataFrame)
        assert isinstance(result.performance_stats, dict)
        assert isinstance(result.charts_data, dict)
        
        # Verify method calls
        mock_generate_signals.assert_called_once()
        mock_size_positions.assert_called_once()
        mock_calc_returns.assert_called_once()
        mock_calc_metrics.assert_called_once()
    
    def test_backtest_strategy_no_universe_tickers(self, backtester, strategy_config, sample_price_data):
        """Test backtest with no valid universe tickers."""
        daily_df, monthly_df, returns_df = sample_price_data
        
        # Create a proper mock strategy that inherits from BaseStrategy
        from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy
        mock_strategy = Mock()
        mock_strategy.__class__ = type('MockStrategy', (BaseStrategy,), {})
        backtester._registry.get_strategy_class.return_value = lambda params: mock_strategy
        
        # Remove all universe tickers from data
        strategy_config['universe'] = ['INVALID1', 'INVALID2']
        
        result = backtester.backtest_strategy(strategy_config, monthly_df, daily_df, returns_df)
        
        # Should return empty result
        assert isinstance(result, BacktestResult)
        assert result.returns.empty
        assert result.metrics == {}
        assert result.trade_history.empty
    
    @patch('portfolio_backtester.backtesting.strategy_backtester.calculate_metrics')
    def test_evaluate_window_success(
        self, 
        mock_calc_metrics,
        backtester, 
        strategy_config, 
        sample_price_data,
        mock_strategy
    ):
        """Test successful window evaluation."""
        daily_df, monthly_df, returns_df = sample_price_data
        
        # Setup mocks
        backtester._registry.get_strategy_class.return_value = lambda params: mock_strategy
        with patch.object(backtester.data_cache, 'get_window_returns_by_dates', return_value=returns_df):
            # Mock the scenario run
            portfolio_returns = pd.Series(
                np.random.randn(len(daily_df)) * 0.01,
                index=daily_df.index
            )
            
            with patch.object(backtester, '_run_scenario_for_window', return_value=portfolio_returns):
                mock_calc_metrics.return_value = {
                    'total_return': 0.05,
                    'sharpe_ratio': 0.8
                }
                
                # Define window
                window = (
                    pd.Timestamp('2020-01-01'),
                    pd.Timestamp('2020-06-30'),
                    pd.Timestamp('2020-07-01'),
                    pd.Timestamp('2020-12-31')
                )
                
                result = backtester.evaluate_window(strategy_config, window, monthly_df, daily_df, returns_df)
                
                # Verify result
                assert isinstance(result, WindowResult)
                assert isinstance(result.window_returns, pd.Series)
                assert isinstance(result.metrics, dict)
                assert result.train_start == window[0]
                assert result.train_end == window[1]
                assert result.test_start == window[2]
                assert result.test_end == window[3]
    
    def test_evaluate_window_empty_returns(self, backtester, strategy_config, sample_price_data):
        """Test window evaluation with empty returns."""
        daily_df, monthly_df, returns_df = sample_price_data
        
        # Mock empty returns
        with patch.object(backtester, '_run_scenario_for_window', return_value=None):
            window = (
                pd.Timestamp('2020-01-01'),
                pd.Timestamp('2020-06-30'),
                pd.Timestamp('2020-07-01'),
                pd.Timestamp('2020-12-31')
            )
            
            result = backtester.evaluate_window(strategy_config, window, monthly_df, daily_df, returns_df)
            
            # Should return empty window result
            assert isinstance(result, WindowResult)
            assert result.window_returns.empty
            assert result.metrics == {}
    
    def test_create_performance_stats(self, backtester):
        """Test performance statistics creation."""
        # Create sample returns
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015], 
                          index=pd.date_range('2020-01-01', periods=5))
        metrics = {'sharpe_ratio': 1.2}
        
        stats = backtester._create_performance_stats(returns, metrics)
        
        assert isinstance(stats, dict)
        assert 'total_return' in stats
        assert 'annualized_return' in stats
        assert 'annualized_volatility' in stats
        assert 'max_drawdown' in stats
        assert 'num_observations' in stats
        assert 'start_date' in stats
        assert 'end_date' in stats
        
        assert stats['num_observations'] == 5
        assert stats['start_date'] == returns.index.min()
        assert stats['end_date'] == returns.index.max()
    
    def test_create_charts_data(self, backtester):
        """Test charts data creation."""
        # Create sample returns
        portfolio_returns = pd.Series([0.01, -0.005, 0.02], 
                                    index=pd.date_range('2020-01-01', periods=3))
        benchmark_returns = pd.Series([0.008, -0.003, 0.015], 
                                    index=pd.date_range('2020-01-01', periods=3))
        
        charts_data = backtester._create_charts_data(portfolio_returns, benchmark_returns)
        
        assert isinstance(charts_data, dict)
        assert 'portfolio_cumulative' in charts_data
        assert 'benchmark_cumulative' in charts_data
        assert 'drawdown' in charts_data
        assert 'rolling_sharpe' in charts_data
        
        # Verify cumulative returns calculation
        expected_portfolio_cum = (1 + portfolio_returns).cumprod()
        pd.testing.assert_series_equal(charts_data['portfolio_cumulative'], expected_portfolio_cum)
    
    def test_calculate_max_drawdown(self, backtester):
        """Test maximum drawdown calculation."""
        # Create returns with known drawdown
        returns = pd.Series([0.1, -0.05, -0.1, 0.05, 0.02])
        
        max_dd = backtester._calculate_max_drawdown(returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
    
    def test_calculate_max_drawdown_empty(self, backtester):
        """Test maximum drawdown with empty series."""
        returns = pd.Series(dtype=float)
        
        max_dd = backtester._calculate_max_drawdown(returns)
        
        assert max_dd == 0.0
    
    def test_create_trade_history(self, backtester, sample_price_data):
        """Test trade history creation."""
        daily_df, _, _ = sample_price_data
        
        # Create sample sized signals
        sized_signals = pd.DataFrame({
            'AAPL': [0.5, 0.3, 0.0],
            'MSFT': [0.3, 0.4, 0.6],
            'GOOGL': [0.2, 0.3, 0.4]
        }, index=daily_df.index[:3])
        
        trade_history = backtester._create_trade_history(sized_signals, daily_df)
        
        assert isinstance(trade_history, pd.DataFrame)
        assert 'date' in trade_history.columns
        assert 'ticker' in trade_history.columns
        assert 'position' in trade_history.columns
        assert 'price' in trade_history.columns
        
        # Should have entries for non-zero positions
        assert len(trade_history) > 0


class TestStrategyBacktesterSeparationOfConcerns:
    """Test separation of concerns - verify backtester works without optimization components."""
    
    @pytest.fixture
    def mock_global_config(self):
        """Mock global configuration."""
        return {
            'benchmark': 'SPY',
            'universe': ['AAPL', 'MSFT', 'GOOGL'],
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'portfolio_value': 100000.0
        }
    
    @pytest.fixture
    def mock_data_source(self):
        """Mock data source."""
        mock_source = Mock()
        mock_source.get_data.return_value = pd.DataFrame()
        return mock_source

    def test_backtester_runs_without_optimizers_disabled(self, mock_global_config, mock_data_source):
        """Test that backtester works with all optimizers disabled via feature flags."""
        # This test simulates having optimization components disabled
        with patch('portfolio_backtester.backtesting.strategy_backtester.get_strategy_registry') as mock_get_registry:
            # Mock the strategy enumerator
            mock_registry = Mock()
            mock_registry.get_all_strategies.return_value = {'dummy': Mock}
            mock_get_registry.return_value = mock_registry
            
            # Simulate feature flags disabling optimizers
            with patch.dict('os.environ', {'DISABLE_OPTUNA': '1', 'DISABLE_PYGAD': '1'}):
                backtester = StrategyBacktester(mock_global_config, mock_data_source)
                
                # Verify backtester can be created and initialized
                assert backtester is not None
                assert backtester.global_config == mock_global_config
                assert backtester.data_source == mock_data_source
    
    def test_no_optimization_imports_in_backtester(self):
        """Test that StrategyBacktester has no optimization-related imports."""
        import inspect
        import portfolio_backtester.backtesting.strategy_backtester as backtester_module
        
        # Get the source code of the entire module
        source = inspect.getsource(backtester_module)
        
        # Check that no optimization-related imports are present in import statements
        import_lines = [line.strip() for line in source.split('\n') if line.strip().startswith(('import ', 'from '))]
        
        optimization_imports = [
            'optuna',
            'pygad',
            'parameter_generator'
        ]
        
        for import_line in import_lines:
            for opt_import in optimization_imports:
                assert opt_import not in import_line.lower(), f"Found optimization import in: {import_line}"
    
    def test_backtester_module_independence(self):
        """Test that backtesting module can be imported without optimization dependencies."""
        # This test verifies that the backtesting module doesn't have hard dependencies
        # on optimization components at import time
        
        try:
            from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
            from portfolio_backtester.backtesting.results import BacktestResult, WindowResult
            
            # If we can import these without errors, the separation is working
            assert StrategyBacktester is not None
            assert BacktestResult is not None
            assert WindowResult is not None
            
        except ImportError as e:
            pytest.fail(f"Backtesting module has unwanted dependencies: {e}")
    
    @patch('portfolio_backtester.backtesting.strategy_backtester.get_strategy_registry')
    def test_backtester_works_in_isolation(self, mock_get_registry):
        """Test that backtester can work completely in isolation."""
        # Setup mocks to simulate minimal dependencies
        mock_registry = Mock()
        mock_registry.get_all_strategies.return_value = {'test_strategy': Mock}
        mock_get_registry.return_value = mock_registry
        
        global_config = {
            'benchmark': 'SPY',
            'universe': ['AAPL', 'MSFT'],
            'start_date': '2020-01-01',
            'end_date': '2023-12-31'
        }
        
        data_source = Mock()
        
        # Create backtester in isolation
        backtester = StrategyBacktester(global_config, data_source)
        
        # Verify it can perform basic operations
        assert backtester._get_strategy.__name__ == '_get_strategy'
        assert backtester._create_empty_backtest_result().__class__.__name__ == 'BacktestResult'
        
        # Test error handling works
        # Make sure get_strategy_class returns None for nonexistent strategy
        mock_registry.get_strategy_class.return_value = None
        with pytest.raises(ValueError):
            backtester._get_strategy('nonexistent_strategy', {})


if __name__ == '__main__':
    pytest.main([__file__])