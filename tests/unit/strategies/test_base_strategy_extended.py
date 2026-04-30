import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.strategies._core.base.base.base_strategy import (
    BaseStrategy,
    TradeDirectionConfigurationError,
    TradeDirectionViolationError,
)


# Concrete implementation for testing (name avoids PytestCollectionWarning:
# classes named Test* with __init__ are treated as test classes.)
class ConcreteBaseStrategyStub(BaseStrategy):
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        # Simple implementation that returns fixed weights
        # We'll control this via strategy_params for testing

        weights = self.strategy_params.get("test_weights", {})
        if not weights:
            return pd.DataFrame()

        return pd.DataFrame([weights], index=[current_date])


class TestBaseStrategy:
    @pytest.fixture
    def strategy_params(self):
        return {"strategy_params": {"trade_longs": True, "trade_shorts": True}}

    @pytest.fixture
    def strategy(self, strategy_params):
        # Mock providers initialization to avoid complex dependencies
        with patch.object(BaseStrategy, "_initialize_providers"):
            with patch.object(BaseStrategy, "_initialize_timing_controller"):
                strategy = ConcreteBaseStrategyStub(strategy_params)
                # Manually set mocked providers
                strategy._universe_provider = MagicMock()
                strategy._position_sizer_provider = MagicMock()
                strategy._stop_loss_provider = MagicMock()
                strategy._take_profit_provider = MagicMock()
                strategy._risk_off_signal_provider = MagicMock()
                return strategy

    def test_trade_direction_initialization(self):
        # Default: True/True
        s1 = ConcreteBaseStrategyStub({"strategy_params": {}})
        assert s1.trade_longs is True
        assert s1.trade_shorts is True

        # Custom
        s2 = ConcreteBaseStrategyStub(
            {"strategy_params": {"trade_longs": False, "trade_shorts": True}}
        )
        assert s2.trade_longs is False
        assert s2.trade_shorts is True

    def test_invalid_trade_direction_config(self):
        # Both False
        with pytest.raises(
            TradeDirectionConfigurationError, match="Both trade_longs and trade_shorts are False"
        ):
            ConcreteBaseStrategyStub(
                {"strategy_params": {"trade_longs": False, "trade_shorts": False}}
            )

        # Invalid Types
        with pytest.raises(TradeDirectionConfigurationError, match="must be boolean"):
            ConcreteBaseStrategyStub(
                {"strategy_params": {"trade_longs": "yes", "trade_shorts": True}}
            )

    def test_validate_signal_constraints_long_only(self, strategy):
        # Set to Long Only
        strategy.trade_longs = True
        strategy.trade_shorts = False

        # Valid: Positive weights
        valid_signals = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.5]})
        strategy._validate_signal_constraints(valid_signals)  # Should not raise

        # Invalid: Negative weights
        invalid_signals = pd.DataFrame({"AAPL": [-0.5], "MSFT": [0.5]})
        with pytest.raises(TradeDirectionViolationError, match="Generated .* negative .* signals"):
            strategy._validate_signal_constraints(invalid_signals)

    def test_validate_signal_constraints_short_only(self, strategy):
        # Set to Short Only
        strategy.trade_longs = False
        strategy.trade_shorts = True

        # Valid: Negative weights
        valid_signals = pd.DataFrame({"AAPL": [-0.5], "MSFT": [-0.5]})
        strategy._validate_signal_constraints(valid_signals)

        # Invalid: Positive weights
        invalid_signals = pd.DataFrame({"AAPL": [0.5], "MSFT": [-0.5]})
        with pytest.raises(TradeDirectionViolationError, match="Generated .* positive .* signals"):
            strategy._validate_signal_constraints(invalid_signals)

    def test_validate_data_sufficiency(self, strategy):
        # Mock min periods
        with patch.object(
            ConcreteBaseStrategyStub, "get_minimum_required_periods", return_value=12
        ):

            current_date = pd.Timestamp("2021-01-01")

            # 1. Empty Universe Data
            is_valid, msg = strategy.validate_data_sufficiency(
                pd.DataFrame(), pd.DataFrame(), current_date
            )
            assert not is_valid
            assert "No historical data available for universe" in msg

            # 2. Sufficient Data (> 12 months)
            dates = pd.date_range("2019-01-01", "2021-01-01", freq="D")
            data = pd.DataFrame(1.0, index=dates, columns=["AAPL"])

            is_valid, msg = strategy.validate_data_sufficiency(data, pd.DataFrame(), current_date)
            assert is_valid, f"Validation failed for sufficient data: {msg}"

            # 3. Insufficient Data (< 12 months)
            short_dates = pd.date_range("2020-06-01", "2021-01-01", freq="D")
            short_data = pd.DataFrame(1.0, index=short_dates, columns=["AAPL"])

            is_valid, msg = strategy.validate_data_sufficiency(
                short_data, pd.DataFrame(), current_date
            )
            assert not is_valid
            assert "Insufficient historical data" in msg

    def test_filter_universe_by_data_availability(self, strategy):
        current_date = pd.Timestamp("2021-01-01")

        # Create data where AAPL has enough history, but IPO_STOCK is new
        dates_long = pd.date_range("2019-01-01", "2021-01-01", freq="D")
        dates_short = pd.date_range("2020-12-01", "2021-01-01", freq="D")

        df_long = pd.DataFrame({"Close": 100.0}, index=dates_long)
        df_short = pd.DataFrame({"Close": 100.0}, index=dates_short)

        # Combine into MultiIndex DataFrame
        # Columns: (Ticker, Field)
        data = pd.concat([df_long, df_short], axis=1, keys=["AAPL", "IPO_STOCK"])
        data.columns.names = ["Ticker", "Field"]

        with patch.object(
            ConcreteBaseStrategyStub, "get_minimum_required_periods", return_value=6
        ):  # 6 months required
            valid_assets = strategy.filter_universe_by_data_availability(data, current_date)

            assert "AAPL" in valid_assets
            assert "IPO_STOCK" not in valid_assets

    def test_run_logic_kernel(self):
        # Test the numba-optimized logic (static method)
        signals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        w_prev = np.zeros(5)
        num_holdings = 2
        top_decile = 0.0  # Unused if num_holdings set
        trade_shorts = False
        leverage = 1.0
        smoothing = 0.0  # Immediate transition

        w_new = BaseStrategy.run_logic(
            signals, w_prev, num_holdings, top_decile, trade_shorts, leverage, smoothing
        )

        # Should pick top 2 winners (indices 3 and 4)
        # Weights should be 1/2 = 0.5 each
        expected = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        np.testing.assert_array_almost_equal(w_new, expected)

        # Test with shorts
        trade_shorts = True
        w_new_shorts = BaseStrategy.run_logic(
            signals, w_prev, num_holdings, top_decile, trade_shorts, leverage, smoothing
        )
        # Top 2 winners (indices 3, 4) -> 0.5
        # Bottom 2 losers (indices 0, 1) -> -0.5
        expected_shorts = np.array([-0.5, -0.5, 0.0, 0.5, 0.5])
        np.testing.assert_array_almost_equal(w_new_shorts, expected_shorts)
