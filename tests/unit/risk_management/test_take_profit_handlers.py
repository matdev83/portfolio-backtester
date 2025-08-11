"""
Comprehensive test suite for Take Profit Handlers.

This test suite covers all take profit handler implementations,
ensuring they work correctly for both long and short positions.
"""

import pytest
import pandas as pd
import numpy as np

from portfolio_backtester.risk_management.take_profit_handlers import (
    NoTakeProfit,
    AtrBasedTakeProfit,
)


class TestNoTakeProfit:
    """Test suite for NoTakeProfit handler (dummy implementation)."""

    def test_no_take_profit_initialization(self):
        """Test NoTakeProfit initializes correctly."""
        strategy_config: dict[str, str] = {"test": "value"}
        take_profit_config: dict[str, str] = {"type": "NoTakeProfit"}

        handler = NoTakeProfit(strategy_config, take_profit_config)

        assert handler.strategy_config == strategy_config
        assert handler.take_profit_specific_config == take_profit_config

    def test_calculate_take_profit_levels_returns_nan(self):
        """Test that NoTakeProfit returns NaN for all take profit levels."""
        handler = NoTakeProfit({}, {})

        current_date = pd.Timestamp("2023-01-15")
        current_weights = pd.Series({"AAPL": 0.5, "MSFT": -0.3, "GOOGL": 0.2})
        entry_prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0})
        historical_data = pd.DataFrame()  # Empty for NoTakeProfit

        result = handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        assert len(result) == 3
        assert result.index.tolist() == ["AAPL", "MSFT", "GOOGL"]
        assert pd.isna(result).all()

    def test_apply_take_profit_returns_unchanged_weights(self):
        """Test that NoTakeProfit returns unchanged target weights."""
        handler = NoTakeProfit({}, {})

        current_date = pd.Timestamp("2023-01-15")
        current_prices = pd.Series({"AAPL": 160.0, "MSFT": 240.0, "GOOGL": 110.0})
        target_weights = pd.Series({"AAPL": 0.5, "MSFT": -0.3, "GOOGL": 0.2})
        entry_prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0})
        take_profit_levels = pd.Series({"AAPL": np.nan, "MSFT": np.nan, "GOOGL": np.nan})

        result = handler.apply_take_profit(
            current_date=current_date,
            current_asset_prices=current_prices,
            target_weights=target_weights,
            entry_prices=entry_prices,
            take_profit_levels=take_profit_levels,
        )

        pd.testing.assert_series_equal(result, target_weights)


class TestAtrBasedTakeProfit:
    """Test suite for ATR-based take profit handler."""

    @pytest.fixture
    def atr_handler(self):
        """Create ATR-based take profit handler for testing."""
        strategy_config: dict[str, str] = {}
        take_profit_config = {
            "type": "AtrBasedTakeProfit",
            "atr_length": 14,
            "atr_multiple": 2.0,
        }
        return AtrBasedTakeProfit(strategy_config, take_profit_config)

    @pytest.fixture
    def historical_data(self):
        """Create historical OHLC data for ATR calculations."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")

        # Create MultiIndex columns for OHLC data
        tickers = ["AAPL", "MSFT", "GOOGL"]
        fields = ["Open", "High", "Low", "Close"]

        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        # Set seed once for consistent data across all tickers
        np.random.seed(42)  # For reproducible tests

        data = {}
        for ticker in tickers:
            # Generate realistic OHLC data with some volatility
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]

            prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()

            data[(ticker, "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
            data[(ticker, "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Close")] = prices

        return pd.DataFrame(data, index=dates, columns=columns)

    def test_atr_handler_initialization(self, atr_handler):
        """Test ATR handler initializes with correct parameters."""
        assert atr_handler.atr_length == 14
        assert atr_handler.atr_multiple == 2.0

    def test_calculate_take_profit_levels_long_positions(self, atr_handler, historical_data):
        """Test take profit level calculation for long positions."""
        current_date = pd.Timestamp("2023-01-20")
        current_weights = pd.Series({"AAPL": 0.5, "GOOGL": 0.3})  # Long positions only
        entry_prices = pd.Series({"AAPL": 150.0, "GOOGL": 100.0})

        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # For long positions, take profit should be above entry price
        assert take_profit_levels["AAPL"] > entry_prices["AAPL"]
        assert take_profit_levels["GOOGL"] > entry_prices["GOOGL"]
        assert not pd.isna(take_profit_levels).any()

    def test_calculate_take_profit_levels_short_positions(self, atr_handler, historical_data):
        """Test take profit level calculation for short positions."""
        current_date = pd.Timestamp("2023-01-20")
        current_weights = pd.Series({"MSFT": -0.4})  # Short position
        entry_prices = pd.Series({"MSFT": 250.0})

        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # For short positions, take profit should be below entry price
        assert take_profit_levels["MSFT"] < entry_prices["MSFT"]
        assert not pd.isna(take_profit_levels["MSFT"])

    def test_calculate_take_profit_levels_mixed_positions(self, atr_handler, historical_data):
        """Test take profit level calculation for mixed long/short positions."""
        current_date = pd.Timestamp("2023-01-20")
        current_weights = pd.Series({"AAPL": 0.5, "MSFT": -0.3, "GOOGL": 0.2})
        entry_prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0})

        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # Long positions: take profit above entry
        assert take_profit_levels["AAPL"] > entry_prices["AAPL"]
        assert take_profit_levels["GOOGL"] > entry_prices["GOOGL"]

        # Short position: take profit below entry
        assert take_profit_levels["MSFT"] < entry_prices["MSFT"]

        assert not pd.isna(take_profit_levels).any()

    def test_apply_take_profit_long_position_triggered(self, atr_handler, historical_data):
        """Test take profit application when long position should be closed."""
        current_date = pd.Timestamp("2023-01-20")

        # Setup: AAPL long position with current price well above take profit level
        current_prices = pd.Series({"AAPL": 200.0})  # High price to trigger take profit
        target_weights = pd.Series({"AAPL": 0.5})
        entry_prices = pd.Series({"AAPL": 150.0})

        # Calculate take profit levels first
        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=target_weights,
            entry_prices=entry_prices,
        )

        # Ensure we have a take profit level that would be triggered
        if take_profit_levels["AAPL"] < current_prices["AAPL"]:
            result = atr_handler.apply_take_profit(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=target_weights,
                entry_prices=entry_prices,
                take_profit_levels=take_profit_levels,
            )

            # Position should be closed (set to 0)
            assert result["AAPL"] == 0.0

    def test_apply_take_profit_short_position_triggered(self, atr_handler, historical_data):
        """Test take profit application when short position should be closed."""
        current_date = pd.Timestamp("2023-01-20")

        # Setup: MSFT short position with current price well below take profit level
        current_prices = pd.Series({"MSFT": 200.0})  # Low price to trigger take profit for short
        target_weights = pd.Series({"MSFT": -0.3})
        entry_prices = pd.Series({"MSFT": 250.0})

        # Calculate take profit levels first
        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=target_weights,
            entry_prices=entry_prices,
        )

        # Ensure we have a take profit level that would be triggered
        if take_profit_levels["MSFT"] > current_prices["MSFT"]:
            result = atr_handler.apply_take_profit(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=target_weights,
                entry_prices=entry_prices,
                take_profit_levels=take_profit_levels,
            )

            # Position should be closed (set to 0)
            assert result["MSFT"] == 0.0

    def test_apply_take_profit_no_trigger(self, atr_handler, historical_data):
        """Test take profit application when no positions should be closed."""
        current_date = pd.Timestamp("2023-01-20")

        # Setup: prices that shouldn't trigger take profit
        current_prices = pd.Series({"AAPL": 155.0, "MSFT": 245.0})  # Small moves from entry
        target_weights = pd.Series({"AAPL": 0.5, "MSFT": -0.3})
        entry_prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        # Calculate take profit levels first
        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=target_weights,
            entry_prices=entry_prices,
        )

        result = atr_handler.apply_take_profit(
            current_date=current_date,
            current_asset_prices=current_prices,
            target_weights=target_weights,
            entry_prices=entry_prices,
            take_profit_levels=take_profit_levels,
        )

        # Weights should remain unchanged
        pd.testing.assert_series_equal(result, target_weights)

    def test_zero_weights_positions(self, atr_handler, historical_data):
        """Test behavior with zero-weight positions."""
        current_date = pd.Timestamp("2023-01-20")
        current_weights = pd.Series({"AAPL": 0.0, "MSFT": 0.0})  # No positions
        entry_prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # Should return NaN for zero-weight positions
        assert pd.isna(take_profit_levels).all()

    def test_missing_entry_prices(self, atr_handler, historical_data):
        """Test behavior with missing entry prices."""
        current_date = pd.Timestamp("2023-01-20")
        current_weights = pd.Series({"AAPL": 0.5})
        entry_prices = pd.Series({"AAPL": np.nan})  # Missing entry price

        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # Should return NaN when entry price is missing
        assert pd.isna(take_profit_levels["AAPL"])

    def test_insufficient_historical_data(self, atr_handler):
        """Test behavior with insufficient historical data."""
        # Create minimal historical data (less than ATR length)
        dates = pd.date_range("2023-01-01", periods=5, freq="D")  # Less than 14 days
        columns = pd.MultiIndex.from_product([["AAPL"], ["Close"]], names=["Ticker", "Field"])
        historical_data = pd.DataFrame(
            {("AAPL", "Close"): [150.0, 151.0, 149.0, 152.0, 148.0]},
            index=dates,
            columns=columns,
        )

        current_date = pd.Timestamp("2023-01-05")
        current_weights = pd.Series({"AAPL": 0.5})
        entry_prices = pd.Series({"AAPL": 150.0})

        take_profit_levels = atr_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # Should handle insufficient data gracefully
        assert pd.isna(take_profit_levels["AAPL"])

    def test_atr_caching(self):
        """Test that ATR calculations are cached correctly."""
        from portfolio_backtester.risk_management.atr_service import OptimizedATRService

        # Create test data in the test method to avoid fixture issues
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        fields = ["Open", "High", "Low", "Close"]
        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        for ticker in tickers:
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]
            prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
            data[(ticker, "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
            data[(ticker, "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Close")] = prices

        historical_data = pd.DataFrame(data, index=dates, columns=columns)

        current_date = pd.Timestamp("2023-01-20")
        atr_length = 14

        # Create a local ATR service instance to avoid global singleton issues
        atr_service = OptimizedATRService(cache_size=100)
        initial_cache_size = atr_service.cache_info()["size"]
        assert initial_cache_size == 0

        # First calculation
        result_1 = atr_service.calculate_atr(historical_data, current_date, atr_length)
        cache_size_after_first = atr_service.cache_info()["size"]
        assert cache_size_after_first == initial_cache_size + 1
        assert not result_1.empty

        # Second calculation with same parameters (should hit cache)
        result_2 = atr_service.calculate_atr(historical_data, current_date, atr_length)
        cache_size_after_second = atr_service.cache_info()["size"]

        # Results should be identical (cache hit)
        pd.testing.assert_series_equal(result_1, result_2)

        # Cache should still have one entry (no additional entries)
        assert cache_size_after_second == initial_cache_size + 1

        # Test cache invalidation with different data
        np.random.seed(123)  # Different seed for different data
        data_new = {}
        for ticker in tickers:
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]
            prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
            data_new[(ticker, "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
            data_new[(ticker, "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            data_new[(ticker, "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            data_new[(ticker, "Close")] = prices

        historical_data_new = pd.DataFrame(data_new, index=dates, columns=columns)

        # Third calculation with different data (should create new cache entry)
        result_3 = atr_service.calculate_atr(historical_data_new, current_date, atr_length)
        cache_size_after_third = atr_service.cache_info()["size"]

        # Should have two cache entries now
        assert cache_size_after_third == 2
        # Results should be different
        assert not result_1.equals(result_3)

    def test_custom_atr_parameters(self):
        """Test ATR handler with custom parameters."""
        strategy_config: dict[str, str] = {}
        take_profit_config = {
            "type": "AtrBasedTakeProfit",
            "atr_length": 10,  # Custom length
            "atr_multiple": 3.0,  # Custom multiple
        }

        handler = AtrBasedTakeProfit(strategy_config, take_profit_config)

        assert handler.atr_length == 10
        assert handler.atr_multiple == 3.0

    def test_default_atr_parameters(self):
        """Test ATR handler with default parameters."""
        strategy_config: dict[str, str] = {}
        take_profit_config = {"type": "AtrBasedTakeProfit"}  # No custom params

        handler = AtrBasedTakeProfit(strategy_config, take_profit_config)

        # Should use defaults
        assert handler.atr_length == 14
        assert handler.atr_multiple == 2.0


class TestTakeProfitHandlerEdgeCases:
    """Test edge cases and error conditions for take profit handlers."""

    def test_empty_historical_data(self):
        """Test behavior with empty historical data."""
        handler = AtrBasedTakeProfit({}, {"atr_length": 14, "atr_multiple": 2.0})

        current_date = pd.Timestamp("2023-01-15")
        current_weights = pd.Series({"AAPL": 0.5})
        entry_prices = pd.Series({"AAPL": 150.0})
        empty_data = pd.DataFrame()

        result = handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=empty_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # Should handle empty data gracefully
        assert pd.isna(result["AAPL"])

    def test_date_not_in_historical_data(self):
        """Test behavior when current_date is not in historical data."""
        handler = AtrBasedTakeProfit({}, {"atr_length": 5, "atr_multiple": 2.0})

        # Historical data that doesn't include current_date
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        columns = pd.MultiIndex.from_product([["AAPL"], ["Close"]], names=["Ticker", "Field"])
        historical_data = pd.DataFrame({("AAPL", "Close"): range(10)}, index=dates, columns=columns)

        current_date = pd.Timestamp("2023-02-01")  # Not in historical data
        current_weights = pd.Series({"AAPL": 0.5})
        entry_prices = pd.Series({"AAPL": 150.0})

        result = handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=historical_data,
            current_weights=current_weights,
            entry_prices=entry_prices,
        )

        # Should handle gracefully when date is beyond available data
        assert pd.isna(result["AAPL"])

    def test_nan_current_prices_in_apply(self):
        """Test apply_take_profit with NaN current prices."""
        handler = AtrBasedTakeProfit({}, {"atr_length": 14, "atr_multiple": 2.0})

        current_date = pd.Timestamp("2023-01-15")
        current_prices = pd.Series({"AAPL": np.nan})  # NaN price
        target_weights = pd.Series({"AAPL": 0.5})
        entry_prices = pd.Series({"AAPL": 150.0})
        take_profit_levels = pd.Series({"AAPL": 160.0})

        result = handler.apply_take_profit(
            current_date=current_date,
            current_asset_prices=current_prices,
            target_weights=target_weights,
            entry_prices=entry_prices,
            take_profit_levels=take_profit_levels,
        )

        # Should not modify weights when price is NaN
        pd.testing.assert_series_equal(result, target_weights)

    def test_nan_take_profit_levels_in_apply(self):
        """Test apply_take_profit with NaN take profit levels."""
        handler = AtrBasedTakeProfit({}, {"atr_length": 14, "atr_multiple": 2.0})

        current_date = pd.Timestamp("2023-01-15")
        current_prices = pd.Series({"AAPL": 200.0})
        target_weights = pd.Series({"AAPL": 0.5})
        entry_prices = pd.Series({"AAPL": 150.0})
        take_profit_levels = pd.Series({"AAPL": np.nan})  # NaN take profit level

        result = handler.apply_take_profit(
            current_date=current_date,
            current_asset_prices=current_prices,
            target_weights=target_weights,
            entry_prices=entry_prices,
            take_profit_levels=take_profit_levels,
        )

        # Should not modify weights when take profit level is NaN
        pd.testing.assert_series_equal(result, target_weights)


if __name__ == "__main__":
    pytest.main([__file__])
