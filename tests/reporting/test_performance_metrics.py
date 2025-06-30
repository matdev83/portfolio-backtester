import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.reporting.performance_metrics import calculate_metrics
import warnings

class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        # Create a sample returns series for the portfolio and benchmark
        dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=24, freq='ME'))
        self.rets = pd.Series([0.02, -0.01, 0.03, -0.02] * 6, index=dates, name='Portfolio')
        self.bench_rets = pd.Series([0.01, -0.005, 0.015, -0.01] * 6, index=dates, name='Benchmark')
        self.bench_ticker_name = 'Benchmark'

    def test_calculate_metrics_smoke(self):
        # Smoke test to ensure the function runs without errors
        try:
            calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        except Exception as e:
            self.fail(f"calculate_metrics raised an exception: {e}")

    def test_total_return(self):
        metrics = calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        # Expected: (1.02 * 0.99 * 1.03 * 0.98)^6 - 1
        expected_return = (1.02 * 0.99 * 1.03 * 0.98)**6 - 1
        self.assertAlmostEqual(metrics['Total Return'], expected_return, places=4)

    def test_annualized_return(self):
        metrics = calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        # Expected: ((1 + total_return)^(12/24)) - 1
        total_return = (1.02 * 0.99 * 1.03 * 0.98)**6 - 1
        expected_ann_return = (1 + total_return)**(12/24) - 1
        self.assertAlmostEqual(metrics['Ann. Return'], expected_ann_return, places=4)

    def test_sharpe_ratio(self):
        # Test with a zero volatility series
        zero_vol_rets = pd.Series([0.01] * 24, index=self.rets.index)
        
        # Expect RuntimeWarning from scipy.stats.skew/kurtosis for constant data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning) # Ignore other potential warnings
            with self.assertWarns(RuntimeWarning):
                metrics = calculate_metrics(zero_vol_rets, self.bench_rets, self.bench_ticker_name)
        
        self.assertEqual(metrics['Sharpe'], 0) # Should be 0 as ann_vol is not 0

if __name__ == '__main__':
    unittest.main()