
import unittest
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from src.portfolio_backtester.portfolio.position_sizer import (
    equal_weight_sizer,
    rolling_sharpe_sizer,
    rolling_sortino_sizer,
    rolling_beta_sizer,
    rolling_benchmark_corr_sizer,
    rolling_downside_volatility_sizer,
)

class TestPositionSizer(unittest.TestCase):

    def test_equal_weight_sizer(self):
        signals = pd.DataFrame({
            'StockA': [1, 1, 0],
            'StockB': [0, 1, 1],
            'StockC': [1, 0, 1]
        })
        expected_weights = pd.DataFrame({
            'StockA': [0.5, 1/2, 0],
            'StockB': [0, 1/2, 0.5],
            'StockC': [0.5, 0, 0.5]
        })
        
        result_weights = equal_weight_sizer(signals)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def test_equal_weight_sizer_empty_signals(self):
        signals = pd.DataFrame()
        expected_weights = pd.DataFrame()
        result_weights = equal_weight_sizer(signals)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def test_equal_weight_sizer_all_zeros(self):
        signals = pd.DataFrame({
            'StockA': [0, 0],
            'StockB': [0, 0]
        })
        expected_weights = pd.DataFrame({
            'StockA': [np.nan, np.nan],
            'StockB': [np.nan, np.nan]
        })
        result_weights = equal_weight_sizer(signals)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def _create_price_data(self):
        dates = pd.date_range('2020-01-31', periods=5, freq='ME')
        prices = pd.DataFrame({
            'A': [100, 110, 132, 145, 160],
            'B': [100, 102, 104, 105, 107],
            'C': [100, 98, 97, 97, 98]
        }, index=dates)
        benchmark = pd.Series([100, 101, 103, 105, 107], index=dates)
        signals = pd.DataFrame(1, index=dates, columns=['A', 'B', 'C'])
        return prices, benchmark, signals

    def test_rolling_sharpe_sizer(self):
        prices, bench, signals = self._create_price_data()
        window = 2
        rets = prices.pct_change(fill_method=None).fillna(0)
        mean = rets.rolling(window).mean()
        std = rets.rolling(window).std()
        sharpe = mean / std.replace(0, np.nan)
        expected = signals.mul(sharpe).div(signals.mul(sharpe).abs().sum(axis=1), axis=0)

        result = rolling_sharpe_sizer(signals, prices, window)
        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_sortino_sizer(self):
        prices, bench, signals = self._create_price_data()
        window = 2
        target = 0.0
        rets = prices.pct_change(fill_method=None).fillna(0)
        mean = rets.rolling(window).mean() - target

        def dd(series):
            downside = series[series < target]
            if len(downside) == 0:
                return np.nan
            return np.sqrt(np.mean((downside - target) ** 2))

        downside = rets.rolling(window).apply(dd, raw=False)
        sortino = mean / downside.replace(0, np.nan)
        expected = signals.mul(sortino).div(signals.mul(sortino).abs().sum(axis=1), axis=0)

        result = rolling_sortino_sizer(signals, prices, window, target_return=target)
        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_beta_sizer(self):
        prices, bench, signals = self._create_price_data()
        window = 2
        rets = prices.pct_change(fill_method=None).fillna(0)
        bench_rets = bench.pct_change(fill_method=None).fillna(0)
        beta = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            beta[col] = rets[col].rolling(window).cov(bench_rets) / bench_rets.rolling(window).var()
        factor = 1 / beta.abs().replace(0, np.nan)
        expected = signals.mul(factor).div(signals.mul(factor).abs().sum(axis=1), axis=0)

        result = rolling_beta_sizer(signals, prices, bench, window)
        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_benchmark_corr_sizer(self):
        prices, bench, signals = self._create_price_data()
        window = 2
        rets = prices.pct_change(fill_method=None).fillna(0)
        bench_rets = bench.pct_change(fill_method=None).fillna(0)
        corr = pd.DataFrame(index=rets.index, columns=rets.columns)
        for col in rets.columns:
            corr[col] = rets[col].rolling(window).corr(bench_rets)
        factor = 1 / (corr.abs() + 1e-9)
        expected = signals.mul(factor).div(signals.mul(factor).abs().sum(axis=1), axis=0)

        result = rolling_benchmark_corr_sizer(signals, prices, bench, window)
        pd.testing.assert_frame_equal(result, expected)

    def test_rolling_downside_volatility_sizer(self):
        prices, bench, signals = self._create_price_data()
        window = 2
        rets = prices.pct_change(fill_method=None).fillna(0)
        downside = rets.clip(upper=0)

        # Mimic the new logic from the sizer for expected calculation
        epsilon = 1e-9
        target_volatility = 1.0 # Default target_volatility in sizer

        # Original dvol calculation was based on .mean()
        # The new sizer uses (downside_sq_sum / window).pow(0.5)
        # To align the test, we should use the same logic as in the sizer.
        downside_sq_sum = (downside ** 2).rolling(window).sum()
        dvol = (downside_sq_sum / window).pow(0.5)
        dvol = pd.DataFrame(dvol, index=prices.index, columns=prices.columns) # Convert to DataFrame

        factor = target_volatility / np.maximum(dvol, epsilon)
        factor = pd.DataFrame(factor, index=prices.index, columns=prices.columns) # Convert to DataFrame
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)

        sized = signals.mul(factor)
        weight_sum = sized.abs().sum(axis=1)
        expected = sized.div(weight_sum, axis=0).fillna(0)

        # Explicitly pass target_volatility if the sizer uses it and it's part of the test parameters
        # In this case, the sizer has a default for target_volatility, so we ensure the test reflects that.
        
        # Create a dummy daily_prices_for_vol for the test
        daily_prices_for_vol = prices.resample('D').ffill().reindex(pd.date_range(prices.index.min(), prices.index.max(), freq='D'))
        daily_prices_for_vol = daily_prices_for_vol.reindex(columns=prices.columns) # Ensure columns match

        result = rolling_downside_volatility_sizer(signals, prices, bench, daily_prices_for_vol, window, target_volatility=target_volatility)
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

if __name__ == '__main__':
    unittest.main()
