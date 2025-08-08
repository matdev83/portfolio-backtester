import unittest
import pandas as pd
import numpy as np
from pandas import DataFrame, Series  # Added for type hinting
from typing import Tuple  # Added for type hinting

from portfolio_backtester.portfolio.position_sizer import (
    EqualWeightSizer,
    RollingSharpeSizer,
    RollingSortinoSizer,
    RollingBetaSizer,
    RollingBenchmarkCorrSizer,
    RollingDownsideVolatilitySizer,
    SIZER_REGISTRY,
)


class TestPositionSizer(unittest.TestCase):

    def test_sizers_return_positive_weights(self):
        prices = pd.DataFrame(
            {"StockA": [100, 110, 100], "StockB": [100, 90, 100], "StockC": [100, 110, 90]},
            index=pd.to_datetime(["2023-01-01", "2023-01-31", "2023-02-28"]),
        )
        signals = pd.DataFrame(
            {"StockA": [-1, 1, 0], "StockB": [0, -1, 1], "StockC": [1, 0, -1]}, index=prices.index
        )
        benchmark = pd.Series([100, 105, 100])
        daily_prices = pd.DataFrame(
            {
                "StockA": [100, 105, 110, 105, 100],
                "StockB": [100, 95, 90, 95, 100],
                "StockC": [100, 105, 110, 100, 90],
            },
            index=pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
        )

        for name, sizer_class in SIZER_REGISTRY.items():
            with self.subTest(name=name):
                sizer = sizer_class()
                weights = sizer.calculate_weights(
                    signals=signals,
                    prices=prices,
                    benchmark=benchmark,
                    daily_prices_for_vol=daily_prices,
                    window=2,
                )
                self.assertTrue((weights >= 0).all().all(), f"{name} returned negative weights")

    def test_equal_weight_sizer(self):
        signals = pd.DataFrame({"StockA": [1, 1, 0], "StockB": [0, 1, 1], "StockC": [1, 0, 1]})
        # Create dummy prices data - EqualWeightSizer doesn't actually use prices
        prices = pd.DataFrame(
            {"StockA": [100, 110, 120], "StockB": [100, 105, 110], "StockC": [100, 95, 90]}
        )
        expected_weights = pd.DataFrame(
            {"StockA": [0.5, 1 / 2, 0], "StockB": [0, 1 / 2, 0.5], "StockC": [0.5, 0, 0.5]}
        )

        sizer = EqualWeightSizer()
        result_weights = sizer.calculate_weights(signals, prices)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def test_equal_weight_sizer_empty_signals(self):
        signals = pd.DataFrame()
        prices = pd.DataFrame()  # Empty prices to match empty signals
        expected_weights = pd.DataFrame()
        sizer = EqualWeightSizer()
        result_weights = sizer.calculate_weights(signals, prices)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def test_equal_weight_sizer_all_zeros(self):
        signals = pd.DataFrame({"StockA": [0, 0], "StockB": [0, 0]})
        # Create dummy prices data - EqualWeightSizer doesn't actually use prices
        prices = pd.DataFrame({"StockA": [100, 110], "StockB": [100, 105]})
        expected_weights = pd.DataFrame({"StockA": [0.0, 0.0], "StockB": [0.0, 0.0]})
        sizer = EqualWeightSizer()
        result_weights = sizer.calculate_weights(signals, prices)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def _create_price_data(
        self, num_months: int = 5, daily_points_per_month: int = 21
    ) -> Tuple[DataFrame, Series, DataFrame, DataFrame, Series]:
        # Create monthly dates for signals and monthly prices
        monthly_dates = pd.date_range("2020-01-31", periods=num_months, freq="ME")
        monthly_prices = pd.DataFrame(
            {
                "A": [100, 110, 132, 145, 160][:num_months],
                "B": [100, 102, 104, 105, 107][:num_months],
                "C": [100, 98, 97, 97, 98][:num_months],
            },
            index=monthly_dates,
        )
        monthly_benchmark = pd.Series([100, 101, 103, 105, 107][:num_months], index=monthly_dates)
        monthly_signals = pd.DataFrame(1, index=monthly_dates, columns=["A", "B", "C"])

        # Create daily dates and prices for more realistic daily volatility calculation
        # Ensure enough daily data for rolling windows (window * 21)
        start_date_daily = monthly_dates.min().to_period("M").start_time
        end_date_daily = monthly_dates.max().to_period("M").end_time
        daily_dates = pd.date_range(start_date_daily, end_date_daily, freq="B")  # Business days

        # Interpolate monthly prices to daily, then add some noise for realism
        daily_prices = monthly_prices.resample("B").ffill().reindex(daily_dates)
        daily_prices = daily_prices.apply(
            lambda x: x * (1 + np.random.randn(len(x)) * 0.001)
        )  # Add small noise
        daily_prices = daily_prices.ffill().bfill()  # Fill any NaNs from reindex

        daily_benchmark = monthly_benchmark.resample("B").ffill().reindex(daily_dates)
        daily_benchmark = daily_benchmark * (1 + np.random.randn(len(daily_benchmark)) * 0.001)
        daily_benchmark = daily_benchmark.ffill().bfill()

        return monthly_prices, monthly_benchmark, monthly_signals, daily_prices, daily_benchmark

    def test_rolling_sharpe_sizer(self):
        monthly_prices, monthly_benchmark, monthly_signals, daily_prices, daily_benchmark = (
            self._create_price_data()
        )
        window = 2
        rets = monthly_prices.pct_change(fill_method=None).fillna(0)
        mean = rets.rolling(window).mean()
        std = rets.rolling(window).std()
        sharpe = mean / std.replace(0, np.nan)

        # Expected calculation should match the sizer's logic
        weights = monthly_signals.abs().mul(sharpe.abs())
        expected = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)

        sizer = RollingSharpeSizer()
        result = sizer.calculate_weights(monthly_signals, monthly_prices, window=window)
        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_sortino_sizer(self):
        monthly_prices, _, monthly_signals, _, _ = self._create_price_data()
        window = 2
        target = 0.0

        # Use the original implementation logic to calculate expected values
        rets = monthly_prices.pct_change(fill_method=None).fillna(0)
        mean = rets.rolling(window).mean() - target

        def downside(series):
            d = series[series < target]
            if len(d) == 0:
                return np.nan
            return np.sqrt(np.mean((d - target) ** 2))

        downside_dev = rets.rolling(window).apply(downside, raw=False)
        sortino = mean / downside_dev.replace(0, np.nan)

        # Expected calculation should match the sizer's logic
        weights = monthly_signals.abs().mul(sortino.abs())
        expected = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)

        sizer = RollingSortinoSizer()
        result = sizer.calculate_weights(
            monthly_signals, monthly_prices, window=window, target_return=target
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_beta_sizer(self):
        monthly_prices, monthly_benchmark, monthly_signals, daily_prices, daily_benchmark = (
            self._create_price_data()
        )
        window = 2
        rets = monthly_prices.pct_change(fill_method=None).fillna(0)
        bench_rets = monthly_benchmark.pct_change(fill_method=None).fillna(0)
        beta = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            beta[col] = rets[col].rolling(window).cov(bench_rets) / bench_rets.rolling(window).var()
        factor = 1 / beta.abs().replace(0, np.nan)
        expected = (
            monthly_signals.mul(factor)
            .div(monthly_signals.mul(factor).abs().sum(axis=1), axis=0)
            .fillna(0.0)
        )

        sizer = RollingBetaSizer()
        result = sizer.calculate_weights(
            monthly_signals, monthly_prices, benchmark=monthly_benchmark, window=window
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_benchmark_corr_sizer(self):
        monthly_prices, monthly_benchmark, monthly_signals, daily_prices, daily_benchmark = (
            self._create_price_data()
        )
        window = 2
        rets = monthly_prices.pct_change(fill_method=None).fillna(0)
        bench_rets = monthly_benchmark.pct_change(fill_method=None).fillna(0)
        corr = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            corr[col] = rets[col].rolling(window).corr(bench_rets)
        factor = 1 / (corr.abs() + 1e-9)
        expected = (
            monthly_signals.mul(factor)
            .div(monthly_signals.mul(factor).abs().sum(axis=1), axis=0)
            .fillna(0.0)
        )

        sizer = RollingBenchmarkCorrSizer()
        result = sizer.calculate_weights(
            monthly_signals, monthly_prices, benchmark=monthly_benchmark, window=window
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_downside_volatility_sizer(self):
        monthly_prices, monthly_benchmark, monthly_signals, daily_prices, daily_benchmark = (
            self._create_price_data()
        )
        window = 2
        rets = monthly_prices.pct_change(fill_method=None).fillna(0)
        downside = rets.clip(upper=0)

        # Mimic the new logic from the sizer for expected calculation
        epsilon = 1e-9
        target_volatility = 1.0  # Default target_volatility in sizer
        max_leverage = 2.0  # Default max_leverage in sizer

        # Original dvol calculation was based on .mean()
        # The new sizer uses (downside_sq_sum / window).pow(0.5)
        # To align the test, we should use the same logic as in the sizer.
        downside_sq_sum = (downside**2).rolling(window).sum()
        dvol = (downside_sq_sum / window).pow(0.5)
        dvol = pd.DataFrame(
            dvol, index=monthly_prices.index, columns=monthly_prices.columns
        )  # Convert to DataFrame

        factor = target_volatility / np.maximum(dvol, epsilon)
        factor = pd.DataFrame(
            factor, index=monthly_prices.index, columns=monthly_prices.columns
        )  # Convert to DataFrame
        factor = factor.clip(upper=max_leverage)  # Added this line to match sizer logic
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)

        sized_initial = monthly_signals.mul(factor)

        # Now, replicate the daily volatility targeting logic from the sizer
        # daily_prices_for_vol = prices.resample('D').ffill().reindex(pd.date_range(prices.index.min(), prices.index.max(), freq='D'))
        # daily_prices_for_vol = daily_prices_for_vol.reindex(columns=prices.columns) # Ensure columns match

        daily_weights_from_sized_initial = sized_initial.reindex(daily_prices.index, method="ffill")
        daily_weights_from_sized_initial = daily_weights_from_sized_initial.shift(1).fillna(0.0)

        daily_rets_for_vol = daily_prices.pct_change(fill_method=None).fillna(0)
        daily_portfolio_returns_initial = (
            daily_weights_from_sized_initial * daily_rets_for_vol
        ).sum(axis=1)

        annualization_factor = np.sqrt(252)
        # Use a fixed window for daily vol calc in test to match sizer's approx monthly to daily window
        # The sizer uses window*21, so we'll use that here.
        actual_portfolio_vol = (
            daily_portfolio_returns_initial.rolling(window=window * 21).std() * annualization_factor
        )

        scaling_factor = target_volatility / np.maximum(actual_portfolio_vol, epsilon)
        max_leverage = 2.0  # Default max_leverage in sizer
        scaling_factor = scaling_factor.clip(upper=max_leverage)

        scaling_factor_monthly = scaling_factor.reindex(monthly_signals.index, method="ffill")

        expected = sized_initial.mul(scaling_factor_monthly, axis=0)
        expected = expected.clip(upper=max_leverage, lower=-max_leverage)
        expected = expected.fillna(0)

        sizer = RollingDownsideVolatilitySizer()
        result = sizer.calculate_weights(
            monthly_signals,
            monthly_prices,
            benchmark=monthly_benchmark,
            daily_prices_for_vol=daily_prices,
            window=window,
            target_volatility=target_volatility,
            max_leverage=max_leverage,
        )

        # Save to CSV for inspection
        import os

        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        expected.to_csv(os.path.join(output_dir, "expected_dvol_sizer.csv"))
        result.to_csv(os.path.join(output_dir, "result_dvol_sizer.csv"))

        pd.testing.assert_frame_equal(
            result, expected, check_dtype=False, check_exact=False, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
