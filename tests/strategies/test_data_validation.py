"""
Test suite for strategy data validation.

This module tests that all strategies properly validate their data requirements
and handle insufficient historical data gracefully.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.portfolio_backtester.strategies import (
    MomentumStrategy,
    MomentumUnfilteredAtrStrategy,
    CalmarMomentumStrategy,
    SortinoMomentumStrategy,
    SharpeMomentumStrategy,
    VAMSMomentumStrategy,
    VAMSNoDownsideStrategy,
    FilteredLaggedMomentumStrategy,
    MomentumDvolSizerStrategy,
)


class TestStrategyDataValidation:
    """Test data validation for all strategy implementations."""

    @pytest.fixture
    def sample_dates(self):
        """Generate sample date range for testing."""
        start_date = datetime(2020, 1, 31)
        end_date = datetime(2023, 12, 31)
        return pd.date_range(start=start_date, end=end_date, freq='ME')

    @pytest.fixture
    def insufficient_dates(self):
        """Generate insufficient date range for testing."""
        start_date = datetime(2023, 10, 31)
        end_date = datetime(2023, 12, 31)
        return pd.date_range(start=start_date, end=end_date, freq='ME')

    @pytest.fixture
    def sample_tickers(self):
        """Sample tickers for testing."""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    @pytest.fixture
    def sufficient_universe_data(self, sample_dates, sample_tickers):
        """Generate sufficient universe data for testing."""
        np.random.seed(42)
        data = {}
        
        for ticker in sample_tickers:
            # Generate realistic price data
            base_price = 100
            returns = np.random.normal(0.01, 0.05, len(sample_dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data[ticker] = {
                'Open': np.array(prices) * np.random.uniform(0.99, 1.01, len(prices)),
                'High': np.array(prices) * np.random.uniform(1.00, 1.05, len(prices)),
                'Low': np.array(prices) * np.random.uniform(0.95, 1.00, len(prices)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(prices))
            }
        
        # Create MultiIndex DataFrame
        arrays = [
            np.repeat(sample_tickers, 5),  # Ticker level
            np.tile(['Open', 'High', 'Low', 'Close', 'Volume'], len(sample_tickers))  # Field level
        ]
        columns = pd.MultiIndex.from_arrays(arrays, names=['Ticker', 'Field'])
        
        # Flatten data for DataFrame creation
        df_data = []
        for ticker in sample_tickers:
            for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_data.append(data[ticker][field])
        
        df_data = np.column_stack(df_data)
        return pd.DataFrame(df_data, index=sample_dates, columns=columns)

    @pytest.fixture
    def insufficient_universe_data(self, insufficient_dates, sample_tickers):
        """Generate insufficient universe data for testing."""
        np.random.seed(42)
        data = {}
        
        for ticker in sample_tickers:
            # Generate minimal price data
            base_price = 100
            returns = np.random.normal(0.01, 0.05, len(insufficient_dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data[ticker] = {
                'Open': np.array(prices) * np.random.uniform(0.99, 1.01, len(prices)),
                'High': np.array(prices) * np.random.uniform(1.00, 1.05, len(prices)),
                'Low': np.array(prices) * np.random.uniform(0.95, 1.00, len(prices)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(prices))
            }
        
        # Create MultiIndex DataFrame
        arrays = [
            np.repeat(sample_tickers, 5),  # Ticker level
            np.tile(['Open', 'High', 'Low', 'Close', 'Volume'], len(sample_tickers))  # Field level
        ]
        columns = pd.MultiIndex.from_arrays(arrays, names=['Ticker', 'Field'])
        
        # Flatten data for DataFrame creation
        df_data = []
        for ticker in sample_tickers:
            for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_data.append(data[ticker][field])
        
        df_data = np.column_stack(df_data)
        return pd.DataFrame(df_data, index=insufficient_dates, columns=columns)

    @pytest.fixture
    def benchmark_data(self, sample_dates):
        """Generate benchmark data for testing."""
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.008, 0.04, len(sample_dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'Open': np.array(prices) * np.random.uniform(0.99, 1.01, len(prices)),
            'High': np.array(prices) * np.random.uniform(1.00, 1.05, len(prices)),
            'Low': np.array(prices) * np.random.uniform(0.95, 1.00, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(5000000, 50000000, len(prices))
        }, index=sample_dates)

    @pytest.fixture
    def insufficient_benchmark_data(self, insufficient_dates):
        """Generate insufficient benchmark data for testing."""
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.008, 0.04, len(insufficient_dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'Open': np.array(prices) * np.random.uniform(0.99, 1.01, len(prices)),
            'High': np.array(prices) * np.random.uniform(1.00, 1.05, len(prices)),
            'Low': np.array(prices) * np.random.uniform(0.95, 1.00, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(5000000, 50000000, len(prices))
        }, index=insufficient_dates)

    @pytest.mark.parametrize("strategy_class,strategy_config", [
        (MomentumStrategy, {"strategy_params": {"lookback_months": 12, "skip_months": 1}}),
        (MomentumUnfilteredAtrStrategy, {"strategy_params": {"lookback_months": 12, "atr_length": 14}}),
        (CalmarMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (SortinoMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (SharpeMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (VAMSMomentumStrategy, {"strategy_params": {"lookback_months": 6}}),
        (VAMSNoDownsideStrategy, {"strategy_params": {"lookback_months": 6}}),
        (FilteredLaggedMomentumStrategy, {
            "strategy_params": {
                "momentum_lookback_standard": 6,
                "momentum_lookback_predictive": 6,
                "momentum_skip_standard": 1,
                "momentum_skip_predictive": 0,
                "sma_filter_window": 10
            }
        }),
        (MomentumDvolSizerStrategy, {
            "strategy_params": {"lookback_months": 12, "sizer_dvol_window": 12},
            "position_sizer": "rolling_downside_volatility"
        }),
    ])
    def test_strategy_minimum_required_periods(self, strategy_class, strategy_config):
        """Test that all strategies properly calculate minimum required periods."""
        strategy = strategy_class(strategy_config)
        min_periods = strategy.get_minimum_required_periods()
        
        # All strategies should require at least some historical data
        assert min_periods > 0, f"{strategy_class.__name__} should require positive minimum periods"
        
        # Minimum periods should be reasonable (not too small or too large)
        assert 2 <= min_periods <= 50, f"{strategy_class.__name__} minimum periods should be reasonable: {min_periods}"

    @pytest.mark.parametrize("strategy_class,strategy_config", [
        (MomentumStrategy, {"strategy_params": {"lookback_months": 12, "skip_months": 1}}),
        (MomentumUnfilteredAtrStrategy, {"strategy_params": {"lookback_months": 12, "atr_length": 14}}),
        (CalmarMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (SortinoMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (SharpeMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (VAMSMomentumStrategy, {"strategy_params": {"lookback_months": 6}}),
        (VAMSNoDownsideStrategy, {"strategy_params": {"lookback_months": 6}}),
        (FilteredLaggedMomentumStrategy, {
            "strategy_params": {
                "momentum_lookback_standard": 6,
                "momentum_lookback_predictive": 6,
                "momentum_skip_standard": 1,
                "momentum_skip_predictive": 0,
                "sma_filter_window": 10
            }
        }),
        (MomentumDvolSizerStrategy, {
            "strategy_params": {"lookback_months": 12, "sizer_dvol_window": 12},
            "position_sizer": "rolling_downside_volatility"
        }),
    ])
    def test_strategy_validates_sufficient_data(self, strategy_class, strategy_config, 
                                               sufficient_universe_data, benchmark_data, sample_dates):
        """Test that strategies work correctly with sufficient data."""
        strategy = strategy_class(strategy_config)
        current_date = sample_dates[-1]
        
        # Test validation method directly
        is_sufficient, reason = strategy.validate_data_sufficiency(
            sufficient_universe_data, benchmark_data, current_date
        )
        
        assert is_sufficient, f"{strategy_class.__name__} should validate sufficient data as sufficient: {reason}"
        
        # Test generate_signals with sufficient data
        signals = strategy.generate_signals(
            sufficient_universe_data, benchmark_data, current_date
        )
        
        assert isinstance(signals, pd.DataFrame), f"{strategy_class.__name__} should return DataFrame"
        assert len(signals) == 1, f"{strategy_class.__name__} should return one row for current_date"
        assert signals.index[0] == current_date, f"{strategy_class.__name__} should return signals for current_date"

    @pytest.mark.parametrize("strategy_class,strategy_config", [
        (MomentumStrategy, {"strategy_params": {"lookback_months": 12, "skip_months": 1}}),
        (MomentumUnfilteredAtrStrategy, {"strategy_params": {"lookback_months": 12, "atr_length": 14}}),
        (CalmarMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (SortinoMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (SharpeMomentumStrategy, {"strategy_params": {"rolling_window": 6}}),
        (VAMSMomentumStrategy, {"strategy_params": {"lookback_months": 6}}),
        (VAMSNoDownsideStrategy, {"strategy_params": {"lookback_months": 6}}),
        (FilteredLaggedMomentumStrategy, {
            "strategy_params": {
                "momentum_lookback_standard": 6,
                "momentum_lookback_predictive": 6,
                "momentum_skip_standard": 1,
                "momentum_skip_predictive": 0,
                "sma_filter_window": 10
            }
        }),
        (MomentumDvolSizerStrategy, {
            "strategy_params": {"lookback_months": 12, "sizer_dvol_window": 12},
            "position_sizer": "rolling_downside_volatility"
        }),
    ])
    def test_strategy_handles_insufficient_data(self, strategy_class, strategy_config, 
                                               insufficient_universe_data, insufficient_benchmark_data, 
                                               insufficient_dates, sample_tickers):
        """Test that strategies handle insufficient data gracefully."""
        strategy = strategy_class(strategy_config)
        current_date = insufficient_dates[-1]
        
        # Test validation method directly
        is_sufficient, reason = strategy.validate_data_sufficiency(
            insufficient_universe_data, insufficient_benchmark_data, current_date
        )
        
        assert not is_sufficient, f"{strategy_class.__name__} should detect insufficient data"
        assert reason, f"{strategy_class.__name__} should provide a reason for insufficient data"
        
        # Test generate_signals with insufficient data
        signals = strategy.generate_signals(
            insufficient_universe_data, insufficient_benchmark_data, current_date
        )
        
        assert isinstance(signals, pd.DataFrame), f"{strategy_class.__name__} should return DataFrame even with insufficient data"
        assert len(signals) == 1, f"{strategy_class.__name__} should return one row for current_date"
        assert signals.index[0] == current_date, f"{strategy_class.__name__} should return signals for current_date"
        
        # All weights should be zero when data is insufficient
        assert (signals.iloc[0] == 0).all(), f"{strategy_class.__name__} should return zero weights with insufficient data"

    @pytest.mark.parametrize("strategy_class,strategy_config", [
        (MomentumStrategy, {"strategy_params": {"lookback_months": 12, "sma_filter_window": 20}}),
        (FilteredLaggedMomentumStrategy, {
            "strategy_params": {
                "momentum_lookback_standard": 6,
                "momentum_lookback_predictive": 6,
                "sma_filter_window": 20
            }
        }),
    ])
    def test_strategy_validates_benchmark_data_for_sma_filter(self, strategy_class, strategy_config,
                                                             sufficient_universe_data, insufficient_benchmark_data,
                                                             sample_dates):
        """Test that strategies requiring SMA filtering validate benchmark data."""
        strategy = strategy_class(strategy_config)
        current_date = sample_dates[-1]
        
        # Test with insufficient benchmark data but sufficient universe data
        is_sufficient, reason = strategy.validate_data_sufficiency(
            sufficient_universe_data, insufficient_benchmark_data, current_date
        )
        
        assert not is_sufficient, f"{strategy_class.__name__} should detect insufficient benchmark data for SMA filtering"
        assert "benchmark" in reason.lower(), f"{strategy_class.__name__} should mention benchmark in error reason"

    def test_empty_data_handling(self, sample_dates):
        """Test that strategies handle completely empty data gracefully."""
        strategy = MomentumStrategy({"strategy_params": {"lookback_months": 6}})
        current_date = sample_dates[-1]
        
        empty_df = pd.DataFrame()
        
        is_sufficient, reason = strategy.validate_data_sufficiency(
            empty_df, empty_df, current_date
        )
        
        assert not is_sufficient, "Strategy should detect empty data as insufficient"
        assert "no historical data" in reason.lower(), "Error message should mention no historical data"

    def test_data_validation_with_future_date(self, sufficient_universe_data, benchmark_data):
        """Test validation when current_date is beyond available data."""
        strategy = MomentumStrategy({"strategy_params": {"lookback_months": 6}})
        future_date = sufficient_universe_data.index[-1] + pd.DateOffset(months=6)
        
        is_sufficient, reason = strategy.validate_data_sufficiency(
            sufficient_universe_data, benchmark_data, future_date
        )
        
        assert not is_sufficient, "Strategy should detect future date as having insufficient data" 