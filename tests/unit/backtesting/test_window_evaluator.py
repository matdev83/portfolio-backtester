"""
Unit tests for WindowEvaluator class.

Tests the window evaluation engine for daily strategy evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from src.portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from src.portfolio_backtester.optimization.wfo_window import WFOWindow
from src.portfolio_backtester.backtesting.results import WindowResult


class TestWindowEvaluator:
    """Test cases for WindowEvaluator class."""
    
    def test_window_evaluator_initialization(self):
        """Test basic window evaluator initialization."""
        evaluator = WindowEvaluator()
        
        assert evaluator.data_cache == {}
        
        # Test with custom cache
        custom_cache = {'test': 'data'}
        evaluator_with_cache = WindowEvaluator(data_cache=custom_cache)
        assert evaluator_with_cache.data_cache == custom_cache
    
    def test_get_historical_data_caching(self):
        """Test that historical data is cached properly."""
        evaluator = WindowEvaluator()
        
        # Create test data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        data = pd.DataFrame({
            'TLT': np.random.randn(len(dates)),
            'SPY': np.random.randn(len(dates))
        }, index=dates)
        
        train_start = pd.Timestamp('2024-01-01')
        current_date = pd.Timestamp('2024-06-30')
        
        # First call should cache the data
        result1 = evaluator._get_historical_data(data, current_date, train_start)
        cache_key = f"{train_start}_{current_date}"
        assert cache_key in evaluator.data_cache
        
        # Second call should return cached data
        result2 = evaluator._get_historical_data(data, current_date, train_start)
        assert result1.equals(result2)
        
        # Verify data is correctly filtered
        expected_data = data.loc[(data.index >= train_start) & (data.index <= current_date)]
        assert result1.equals(expected_data)
    
    def test_get_current_prices(self):
        """Test getting current prices for a specific date."""
        evaluator = WindowEvaluator()
        
        # Create test price data
        dates = pd.date_range('2025-01-01', '2025-01-10', freq='D')
        price_data = pd.DataFrame({
            'TLT': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'SPY': [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
        }, index=dates)
        
        current_date = pd.Timestamp('2025-01-05')
        
        # Get current prices
        current_prices = evaluator._get_current_prices(price_data, current_date)
        
        # Should return DataFrame with single row for current date
        assert len(current_prices) == 1
        assert current_prices.index[0] == current_date
        assert current_prices.loc[current_date, 'TLT'] == 104
        assert current_prices.loc[current_date, 'SPY'] == 404
    
    def test_get_current_prices_missing_date(self):
        """Test getting current prices when date is not available."""
        evaluator = WindowEvaluator()
        
        # Create test price data
        dates = pd.date_range('2025-01-01', '2025-01-10', freq='D')
        price_data = pd.DataFrame({
            'TLT': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'SPY': [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
        }, index=dates)
        
        missing_date = pd.Timestamp('2025-01-15')
        
        # Get current prices for missing date
        current_prices = evaluator._get_current_prices(price_data, missing_date)
        
        # Should return None
        assert current_prices is None
    
    def test_calculate_daily_return_single_level_columns(self):
        """Test daily return calculation with single-level columns."""
        evaluator = WindowEvaluator()
        
        # Create test data
        dates = pd.date_range('2025-01-01', '2025-01-10', freq='D')
        price_data = pd.DataFrame({
            'TLT': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],  # 2% daily return
            'SPY': [400, 404, 408, 412, 416, 420, 424, 428, 432, 436]   # 1% daily return
        }, index=dates)
        
        # Create weights (50% TLT, 50% SPY)
        weights = pd.Series({'TLT': 0.5, 'SPY': 0.5})
        universe_tickers = ['TLT', 'SPY']
        
        current_date = pd.Timestamp('2025-01-02')
        previous_date = pd.Timestamp('2025-01-01')
        
        # Calculate daily return
        daily_return = evaluator._calculate_daily_return(
            weights, price_data, current_date, previous_date, universe_tickers
        )
        
        # Expected: 0.5 * (102-100)/100 + 0.5 * (404-400)/400 = 0.5 * 0.02 + 0.5 * 0.01 = 0.015
        expected_return = 0.5 * 0.02 + 0.5 * 0.01
        assert abs(daily_return - expected_return) < 1e-6
    
    def test_calculate_daily_return_multi_level_columns(self):
        """Test daily return calculation with multi-level columns."""
        evaluator = WindowEvaluator()
        
        # Create test data with MultiIndex columns
        dates = pd.date_range('2025-01-01', '2025-01-10', freq='D')
        columns = pd.MultiIndex.from_product([['TLT', 'SPY'], ['Close']], names=['Ticker', 'Field'])
        price_data = pd.DataFrame({
            ('TLT', 'Close'): [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            ('SPY', 'Close'): [400, 404, 408, 412, 416, 420, 424, 428, 432, 436]
        }, index=dates, columns=columns)
        
        # Create weights
        weights = pd.Series({'TLT': 0.5, 'SPY': 0.5})
        universe_tickers = ['TLT', 'SPY']
        
        current_date = pd.Timestamp('2025-01-02')
        previous_date = pd.Timestamp('2025-01-01')
        
        # Calculate daily return
        daily_return = evaluator._calculate_daily_return(
            weights, price_data, current_date, previous_date, universe_tickers
        )
        
        # Expected: 0.5 * (102-100)/100 + 0.5 * (404-400)/400 = 0.015
        expected_return = 0.5 * 0.02 + 0.5 * 0.01
        assert abs(daily_return - expected_return) < 1e-6
    
    def test_calculate_window_metrics(self):
        """Test calculation of window metrics."""
        evaluator = WindowEvaluator()
        
        # Create test daily returns (5% total return over 10 days)
        daily_returns = [0.005] * 10  # 0.5% daily return
        
        metrics = evaluator._calculate_window_metrics(daily_returns)
        
        # Check metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'volatility' in metrics
        assert 'num_periods' in metrics
        
        # Total return should be approximately (1.005)^10 - 1 ≈ 0.0511
        expected_total_return = (1.005 ** 10) - 1
        assert abs(metrics['total_return'] - expected_total_return) < 1e-4
        
        # Number of periods should match
        assert metrics['num_periods'] == 10
        
        # Volatility should be very close to 0 (constant returns)
        assert abs(metrics['volatility']) < 1e-10
    
    def test_calculate_window_metrics_empty_returns(self):
        """Test calculation of window metrics with empty returns."""
        evaluator = WindowEvaluator()
        
        metrics = evaluator._calculate_window_metrics([])
        
        # Should return default values
        assert metrics['total_return'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['volatility'] == 0.0
    
    def test_create_empty_result(self):
        """Test creation of empty window result."""
        evaluator = WindowEvaluator()
        
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D'
        )
        
        result = evaluator._create_empty_result(window)
        
        # Check result structure
        assert isinstance(result, WindowResult)
        assert len(result.window_returns) == 0
        assert result.train_start == window.train_start
        assert result.train_end == window.train_end
        assert result.test_start == window.test_start
        assert result.test_end == window.test_end
        assert result.trades == []
        assert result.final_weights == {}
        
        # Check metrics
        assert result.metrics['total_return'] == 0.0
        assert result.metrics['sharpe_ratio'] == 0.0
        assert result.metrics['volatility'] == 0.0
    
    def test_evaluate_window_with_mock_strategy(self):
        """Test window evaluation with a mock strategy."""
        evaluator = WindowEvaluator()
        
        # Create test window
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-03'),  # Short window for testing
            evaluation_frequency='D'
        )
        
        # Create test data
        dates = pd.date_range('2024-01-01', '2025-01-10', freq='D')
        daily_data = pd.DataFrame({
            'TLT': np.random.randn(len(dates)) * 0.01 + 1.0001,  # Small random returns around 1
            'SPY': np.random.randn(len(dates)) * 0.01 + 1.0001
        }, index=dates)
        daily_data = daily_data.cumprod() * 100  # Convert to price series starting at 100
        
        benchmark_data = daily_data[['SPY']].copy()
        
        # Create mock strategy
        mock_strategy = Mock()
        
        # Mock strategy returns simple signals (buy TLT on first day, hold, then sell)
        def mock_generate_signals(*args, **kwargs):
            current_date = kwargs.get('current_date')
            if current_date == pd.Timestamp('2025-01-01'):
                return pd.DataFrame({'TLT': [1.0], 'SPY': [0.0]}, index=[current_date])
            elif current_date == pd.Timestamp('2025-01-03'):
                return pd.DataFrame({'TLT': [0.0], 'SPY': [0.0]}, index=[current_date])
            else:
                return pd.DataFrame({'TLT': [1.0], 'SPY': [0.0]}, index=[current_date])
        
        mock_strategy.generate_signals = mock_generate_signals
        
        # Evaluate window
        result = evaluator.evaluate_window(
            window=window,
            strategy=mock_strategy,
            daily_data=daily_data,
            benchmark_data=benchmark_data,
            universe_tickers=['TLT', 'SPY'],
            benchmark_ticker='SPY'
        )
        
        # Check result structure
        assert isinstance(result, WindowResult)
        assert result.train_start == window.train_start
        assert result.train_end == window.train_end
        assert result.test_start == window.test_start
        assert result.test_end == window.test_end
        
        # Should have some returns (2 days of returns for 3-day window)
        assert len(result.window_returns) == 2
        
        # Should have metrics
        assert 'total_return' in result.metrics
        assert 'sharpe_ratio' in result.metrics
        assert 'volatility' in result.metrics
    
    def test_evaluate_window_no_evaluation_dates(self):
        """Test window evaluation when no evaluation dates are available."""
        evaluator = WindowEvaluator()
        
        # Create window with dates outside available data
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2026-01-01'),  # Future date
            test_end=pd.Timestamp('2026-01-31'),
            evaluation_frequency='D'
        )
        
        # Create test data that doesn't cover the test window
        dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')
        daily_data = pd.DataFrame({
            'TLT': np.ones(len(dates)) * 100,
            'SPY': np.ones(len(dates)) * 400
        }, index=dates)
        
        benchmark_data = daily_data[['SPY']].copy()
        mock_strategy = Mock()
        
        # Evaluate window
        result = evaluator.evaluate_window(
            window=window,
            strategy=mock_strategy,
            daily_data=daily_data,
            benchmark_data=benchmark_data,
            universe_tickers=['TLT', 'SPY'],
            benchmark_ticker='SPY'
        )
        
        # Should return empty result
        assert len(result.window_returns) == 0
        assert result.trades == []
        assert result.final_weights == {}
        assert result.metrics['total_return'] == 0.0