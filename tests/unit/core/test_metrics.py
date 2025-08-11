import unittest
import pandas as pd
import numpy as np

from portfolio_backtester.reporting.metrics import calculate_metrics


class TestPerformanceMetrics(unittest.TestCase):
    """
    Test suite for the performance metrics calculations.
    """

    def setUp(self):
        """
        Set up mock returns data for testing.
        """
        self.mock_rets = pd.Series(
            np.random.randn(252) / 100,
            index=pd.to_datetime(pd.date_range("2020-01-01", periods=252)),
        )
        self.mock_bench_rets = pd.Series(
            np.random.randn(252) / 100,
            index=pd.to_datetime(pd.date_range("2020-01-01", periods=252)),
        )

    def test_calculate_metrics(self):
        """
        Test that the calculate_metrics function returns a Series of metrics.
        """
        metrics = calculate_metrics(self.mock_rets, self.mock_bench_rets, "SPY")

        self.assertIsInstance(metrics, pd.Series)
        self.assertIn("Sharpe", metrics)
        self.assertIn("Sortino", metrics)
        self.assertIn("Max Drawdown", metrics)
        self.assertFalse(pd.isna(metrics["Sharpe"]))


if __name__ == "__main__":
    unittest.main()
