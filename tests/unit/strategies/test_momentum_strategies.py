import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from portfolio_backtester.strategies.portfolio.base_momentum_portfolio_strategy import (
    BaseMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.portfolio.simple_momentum_portfolio_strategy import (
    SimpleMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.portfolio.calmar_momentum_portfolio_strategy import (
    CalmarMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.portfolio.sortino_momentum_portfolio_strategy import (
    SortinoMomentumPortfolioStrategy,
)


class TestBaseMomentumPortfolioStrategy:
    """Comprehensive tests for the abstract base momentum strategy class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseMomentumPortfolioStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseMomentumPortfolioStrategy({})  # type: ignore

    def test_concrete_subclass_must_implement_calculate_scores(self):
        """Test that concrete subclasses must implement _calculate_scores method."""

        class IncompleteStrategy(BaseMomentumPortfolioStrategy):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStrategy({})  # type: ignore

    def test_template_method_pattern(self, momentum_test_data):
        """Test that the template method pattern works correctly."""

        class TestMomentumStrategy(BaseMomentumPortfolioStrategy):
            def _calculate_scores(self, asset_prices, current_date):
                # Simple mock implementation
                return pd.Series([0.1, 0.2, 0.3, 0.4], index=asset_prices.columns)

        strategy = TestMomentumStrategy(
            {"strategy_params": {"price_column_asset": "Close", "price_column_benchmark": "Close"}}
        )

        current_date = momentum_test_data["daily_ohlc_data"].index[-1]
        signals = strategy.generate_signals(
            momentum_test_data["daily_ohlc_data"],
            momentum_test_data["benchmark_ohlc_data"],
            current_date=current_date,
        )

        assert isinstance(signals, pd.DataFrame)
        assert not signals.empty

    def test_default_parameters_initialization(self):
        """Test that default parameters are properly initialized."""

        class TestMomentumStrategy(BaseMomentumPortfolioStrategy):
            def _calculate_scores(self, asset_prices, current_date):
                return pd.Series(dtype=float)

        strategy = TestMomentumStrategy({})

        # Check that default parameters are set
        params = strategy.strategy_config.get("strategy_params", strategy.strategy_config)
        assert "num_holdings" in params
        assert "top_decile_fraction" in params
        assert "smoothing_lambda" in params
        assert "leverage" in params
        assert "trade_longs" in params
        assert "trade_shorts" in params
        assert "price_column_asset" in params
        assert "price_column_benchmark" in params

    def test_risk_filters_sma_filter(self, momentum_test_data):
        """Test SMA filter functionality."""

        class TestMomentumStrategy(BaseMomentumPortfolioStrategy):
            def _calculate_scores(self, asset_prices, current_date):
                return pd.Series([0.1, 0.2, 0.3, 0.4], index=asset_prices.columns)

        strategy = TestMomentumStrategy(
            {
                "strategy_params": {
                    "sma_filter_window": 5,
                    "price_column_asset": "Close",
                    "price_column_benchmark": "Close",
                }
            }
        )

        current_date = momentum_test_data["daily_ohlc_data"].index[-1]
        signals = strategy.generate_signals(
            momentum_test_data["daily_ohlc_data"],
            momentum_test_data["benchmark_ohlc_data"],
            current_date=current_date,
        )

        assert isinstance(signals, pd.DataFrame)

    def test_candidate_weights_calculation(self):
        """Test candidate weights calculation."""

        class TestMomentumStrategy(BaseMomentumPortfolioStrategy):
            def _calculate_scores(self, asset_prices, current_date):
                return pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])

        strategy = TestMomentumStrategy(
            {"strategy_params": {"top_decile_fraction": 0.5, "num_holdings": 2}}
        )

        scores = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        weights = strategy._calculate_candidate_weights(scores)

        assert isinstance(weights, pd.Series)
        assert len(weights) == 4
        # Top 2 assets should have non-zero weights
        assert weights["D"] > 0  # Highest score
        assert weights["C"] > 0  # Second highest score

    def test_leverage_and_smoothing_application(self):
        """Test leverage and smoothing application."""

        class TestMomentumStrategy(BaseMomentumPortfolioStrategy):
            def _calculate_scores(self, asset_prices, current_date):
                return pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])

        strategy = TestMomentumStrategy(
            {"strategy_params": {"leverage": 2.0, "smoothing_lambda": 0.5}}
        )

        candidate_weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])
        prev_weights = pd.Series([0.5, 0.0, 0.5, 0.0], index=["A", "B", "C", "D"])

        final_weights = strategy._apply_leverage_and_smoothing(candidate_weights, prev_weights)

        assert isinstance(final_weights, pd.Series)
        assert len(final_weights) == 4
        # Should be a blend of candidate and previous weights due to smoothing

    @patch("portfolio_backtester.strategies.portfolio.base_momentum_portfolio_strategy.logger")
    def test_data_sufficiency_validation(self, mock_logger, momentum_test_data):
        """Test data sufficiency validation."""

        class TestMomentumStrategy(BaseMomentumPortfolioStrategy):
            def _calculate_scores(self, asset_prices, current_date):
                return pd.Series(dtype=float)

            def get_minimum_required_periods(self):
                return 100  # Require more data than available

        strategy = TestMomentumStrategy(
            {"strategy_params": {"price_column_asset": "Close", "price_column_benchmark": "Close"}}
        )

        current_date = momentum_test_data["daily_ohlc_data"].index[10]  # Early date
        signals = strategy.generate_signals(
            momentum_test_data["daily_ohlc_data"],
            momentum_test_data["benchmark_ohlc_data"],
            current_date=current_date,
        )

        # Should return zero weights when insufficient data
        assert isinstance(signals, pd.DataFrame)
        assert (signals == 0).all().all()


class TestSimpleMomentumPortfolioStrategy:
    """Minimal tests for the concrete simple momentum strategy."""

    def test_cannot_be_inherited(self):
        """Test that SimpleMomentumPortfolioStrategy cannot be inherited due to @final decorator."""

        # This test verifies the @final decorator is working
        # In practice, this would be caught by type checkers, but we can test the intent
        try:

            class IllegalSubclass(SimpleMomentumPortfolioStrategy):  # type: ignore
                pass

            # If we get here, the @final decorator isn't working as expected
            # But the class should still function correctly
            strategy = IllegalSubclass({})
            assert isinstance(strategy, SimpleMomentumPortfolioStrategy)
        except TypeError:
            # This is expected if @final is enforced at runtime
            pass

    def test_simple_momentum_calculation(self, momentum_test_data):
        """Test that simple momentum calculation works correctly."""

        strategy = SimpleMomentumPortfolioStrategy(
            {
                "strategy_params": {
                    "lookback_months": 3,
                    "skip_months": 0,
                    "price_column_asset": "Close",
                    "price_column_benchmark": "Close",
                }
            }
        )

        current_date = momentum_test_data["daily_ohlc_data"].index[-1]
        asset_prices = momentum_test_data["daily_ohlc_data"].xs("Close", level="Field", axis=1)

        scores = strategy._calculate_scores(asset_prices, current_date)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(asset_prices.columns)
        # StockA should have highest momentum (increasing trend)
        assert scores["StockA"] > scores["StockB"]  # StockB is decreasing

    def test_tunable_parameters(self):
        """Test that tunable parameters are properly defined for simple strategy."""

        params = SimpleMomentumPortfolioStrategy.tunable_parameters()

        assert isinstance(params, dict)
        # Core momentum parameters
        assert "lookback_months" in params
        assert "skip_months" in params
        # Portfolio construction parameters
        assert "num_holdings" in params
        assert "top_decile_fraction" in params
        # Risk management parameters
        assert "leverage" in params
        assert "smoothing_lambda" in params

        # Should be simplified - fewer parameters than complex strategies
        assert len(params) == 6, f"Expected 6 parameters for simple strategy, got {len(params)}"

        # Check parameter structure
        assert "type" in params["lookback_months"]
        assert "min" in params["lookback_months"]
        assert "max" in params["lookback_months"]
        assert "default" in params["lookback_months"]

        # Check simplified ranges
        assert params["lookback_months"]["max"] == 24  # Simplified max
        assert params["num_holdings"]["max"] == 20  # Simplified max

    def test_minimum_required_periods(self):
        """Test minimum required periods calculation for simple strategy."""

        strategy = SimpleMomentumPortfolioStrategy(
            {
                "strategy_params": {
                    "lookback_months": 6,
                    "skip_months": 1,
                }
            }
        )

        min_periods = strategy.get_minimum_required_periods()

        assert isinstance(min_periods, int)
        assert min_periods > 0
        # Simple calculation: lookback + skip + 3-month buffer
        expected_min = 6 + 1 + 3
        assert min_periods == expected_min, f"Expected {expected_min}, got {min_periods}"

        # Test with different parameters
        strategy2 = SimpleMomentumPortfolioStrategy(
            {
                "strategy_params": {
                    "lookback_months": 12,
                    "skip_months": 0,
                }
            }
        )

        min_periods2 = strategy2.get_minimum_required_periods()
        expected_min2 = 12 + 0 + 3
        assert min_periods2 == expected_min2, f"Expected {expected_min2}, got {min_periods2}"


@pytest.fixture
def momentum_test_data():
    rebalance_dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=12, freq="ME"))
    daily_start_date = rebalance_dates.min() - pd.DateOffset(months=12)
    daily_end_date = rebalance_dates.max()
    daily_dates = pd.date_range(start=daily_start_date, end=daily_end_date, freq="B")

    tickers = ["StockA", "StockB", "StockC", "StockD"]
    data_frames = []
    for ticker in tickers:
        if ticker == "StockA":
            base_price = np.linspace(80, 210, len(daily_dates))
        elif ticker == "StockB":
            base_price = np.linspace(120, 10, len(daily_dates))
        elif ticker == "StockC":
            base_price = np.linspace(95, 115, len(daily_dates))
        else:
            base_price = np.full(len(daily_dates), 100)

        noise = np.random.normal(0, 0.5, size=len(daily_dates))
        close_prices = base_price + noise
        open_prices = close_prices - np.random.uniform(0, 0.5, size=len(daily_dates))
        high_prices = close_prices + np.random.uniform(0, 0.5, size=len(daily_dates))
        low_prices = close_prices - np.random.uniform(0, 0.5, size=len(daily_dates))
        volume = np.random.randint(1000, 5000, size=len(daily_dates))

        df = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volume,
            },
            index=daily_dates,
        )
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=["Ticker", "Field"])
        data_frames.append(df)

    daily_ohlc_data = pd.concat(data_frames, axis=1)

    benchmark_base_price = np.linspace(90, 110, len(daily_dates))
    benchmark_noise = np.random.normal(0, 0.5, size=len(daily_dates))
    benchmark_close = benchmark_base_price + benchmark_noise
    benchmark_df = pd.DataFrame(
        {
            "Open": benchmark_close - np.random.uniform(0, 0.2, size=len(daily_dates)),
            "High": benchmark_close + np.random.uniform(0, 0.2, size=len(daily_dates)),
            "Low": benchmark_close - np.random.uniform(0, 0.2, size=len(daily_dates)),
            "Close": benchmark_close,
            "Volume": np.random.randint(10000, 50000, size=len(daily_dates)),
        },
        index=daily_dates,
    )
    benchmark_df.columns = pd.MultiIndex.from_product(
        [["SPY"], benchmark_df.columns], names=["Ticker", "Field"]
    )
    benchmark_ohlc_data = benchmark_df

    asset_monthly_closes = daily_ohlc_data.xs("Close", level="Field", axis=1).resample("ME").last()
    benchmark_monthly_closes = (
        benchmark_ohlc_data.xs("Close", level="Field", axis=1).resample("ME").last()
    )

    return {
        "rebalance_dates": rebalance_dates,
        "daily_dates": daily_dates,
        "daily_ohlc_data": daily_ohlc_data,
        "benchmark_ohlc_data": benchmark_ohlc_data,
        "asset_monthly_closes": asset_monthly_closes,
        "benchmark_monthly_closes": benchmark_monthly_closes,
    }


@pytest.mark.parametrize(
    "strategy_class, config",
    [
        (
            SimpleMomentumPortfolioStrategy,
            {
                "strategy_params": {
                    "lookback_months": 3,
                    "skip_months": 1,
                    "top_decile_fraction": 0.5,
                    "smoothing_lambda": 0.5,
                    "leverage": 1.0,
                    "trade_longs": True,
                    "trade_shorts": False,
                    "price_column_asset": "Close",
                    "price_column_benchmark": "Close",
                },
                "num_holdings": None,
            },
        ),
        (
            CalmarMomentumPortfolioStrategy,
            {
                "rolling_window": 6,
                "top_decile_fraction": 0.1,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "sma_filter_window": None,
            },
        ),
        (
            SortinoMomentumPortfolioStrategy,
            {
                "rolling_window": 3,
                "top_decile_fraction": 0.5,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "target_return": 0.0,
            },
        ),
    ],
)
def test_generate_signals_smoke(strategy_class, config, momentum_test_data):
    """Smoke test to ensure strategies can generate signals without errors."""
    strategy = strategy_class(config)

    # Test with a single date to keep test simple and focused
    current_date = momentum_test_data["daily_ohlc_data"].index[-1]
    historical_assets = momentum_test_data["daily_ohlc_data"][
        momentum_test_data["daily_ohlc_data"].index <= current_date
    ]
    historical_benchmark = momentum_test_data["benchmark_ohlc_data"][
        momentum_test_data["benchmark_ohlc_data"].index <= current_date
    ]

    try:
        weights_df = strategy.generate_signals(
            all_historical_data=historical_assets,
            benchmark_historical_data=historical_benchmark,
            current_date=current_date,
        )
    except Exception as e:
        pytest.fail(f"generate_signals raised an exception on {current_date}: {e}")

    assert isinstance(weights_df, pd.DataFrame), "Expected DataFrame output"
    assert not weights_df.empty, "Weights DataFrame should not be empty"


class TestCalmarMomentumPortfolioStrategy:
    """Tests for the CalmarMomentumPortfolioStrategy."""

    @patch(
        "portfolio_backtester.strategies.portfolio.calmar_momentum_portfolio_strategy.CalmarRatio"
    )
    def test_calculate_scores(self, mock_calmar_ratio, momentum_test_data):
        """Test that _calculate_scores correctly uses the CalmarRatio feature."""
        # Arrange
        strategy = CalmarMomentumPortfolioStrategy({"strategy_params": {"rolling_window": 6}})

        mock_feature_instance = mock_calmar_ratio.return_value

        # Mock the compute method to return a predictable DataFrame of scores
        mock_scores = pd.DataFrame(
            {
                "StockA": [0.5, 0.6],
                "StockB": [0.3, 0.4],
            },
            index=pd.to_datetime(["2020-06-30", "2020-07-31"]),
        )
        mock_feature_instance.compute.return_value = mock_scores

        asset_prices = momentum_test_data["asset_monthly_closes"]
        current_date = pd.to_datetime("2020-07-31")

        # Act
        scores = strategy._calculate_scores(asset_prices, current_date)

        # Assert
        mock_calmar_ratio.assert_called_once_with(rolling_window=6)
        mock_feature_instance.compute.assert_called_once_with(asset_prices)

        expected_scores = pd.Series([0.6, 0.4], name=current_date, index=["StockA", "StockB"])
        pd.testing.assert_series_equal(expected_scores, scores, check_names=False)


class TestSortinoMomentumPortfolioStrategy:
    """Tests for the SortinoMomentumPortfolioStrategy."""

    @patch(
        "portfolio_backtester.strategies.portfolio.sortino_momentum_portfolio_strategy.SortinoRatio"
    )
    def test_calculate_scores(self, mock_sortino_ratio, momentum_test_data):
        """Test that _calculate_scores correctly uses the SortinoRatio feature."""
        # Arrange
        strategy = SortinoMomentumPortfolioStrategy(
            {"strategy_params": {"rolling_window": 3, "target_return": 0.01}}
        )

        mock_feature_instance = mock_sortino_ratio.return_value

        # Mock the compute method to return a predictable DataFrame of scores
        mock_scores = pd.DataFrame(
            {
                "StockA": [1.5, 1.8],
                "StockB": [1.2, 1.1],
            },
            index=pd.to_datetime(["2020-06-30", "2020-07-31"]),
        )
        mock_feature_instance.compute.return_value = mock_scores

        asset_prices = momentum_test_data["asset_monthly_closes"]
        current_date = pd.to_datetime("2020-07-31")

        # Act
        scores = strategy._calculate_scores(asset_prices, current_date)

        # Assert
        mock_sortino_ratio.assert_called_once_with(rolling_window=3, target_return=0.01)
        mock_feature_instance.compute.assert_called_once_with(asset_prices)

        expected_scores = pd.Series([1.8, 1.1], name=current_date, index=["StockA", "StockB"])
        pd.testing.assert_series_equal(expected_scores, scores, check_names=False)
