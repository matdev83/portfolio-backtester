import unittest
from unittest import mock
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from portfolio_backtester.core import Backtester


class TestBacktesterCore(unittest.TestCase):
    """
    Test suite for the core Backtester class.
    """

    def setUp(self):
        """
        Set up a mock configuration and arguments for testing.
        """
        self.mock_global_config = {
            "data_source": "yfinance",
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
            "universe": ["AAPL", "GOOG"],
        }
        self.mock_scenarios = [
            {"name": "test_scenario", "strategy": "TestStrategy", "strategy_params": {}}
        ]
        self.mock_args = MagicMock()
        self.mock_args.timeout = 60
        self.mock_args.random_seed = 42
        self.mock_args.n_jobs = 1
        self.mock_args.early_stop_patience = 10
        self.mock_args.mode = "backtest"

    @mock.patch("portfolio_backtester.interfaces.create_strategy_resolver")
    @mock.patch("portfolio_backtester.data_sources.yfinance_data_source.YFinanceDataSource")
    @mock.patch("numpy.random.randint")
    def test_backtester_initialization(
        self, mock_randint, mock_data_source, mock_strategy_resolver
    ):
        """
        Test that the Backtester class initializes correctly.
        """
        mock_strategy_resolver.return_value.resolve_strategy.return_value = MagicMock()
        mock_data_source.return_value.get_data.return_value = pd.DataFrame(
            np.random.rand(10, 2),
            columns=["AAPL", "GOOG"],
            index=pd.to_datetime(pd.date_range("2020-01-01", periods=10)),
        )
        mock_data_source.__name__ = "YFinanceDataSource"
        mock_randint.return_value = 42

        backtester = Backtester(
            global_config=self.mock_global_config,
            scenarios=self.mock_scenarios,
            args=self.mock_args,
        )

        self.assertIsInstance(backtester, Backtester)
        self.assertEqual(backtester.random_state, 42)
        self.assertEqual(backtester.args.mode, "backtest")
        # After refactoring, enumerate_strategies_with_params may be called multiple times
        self.assertGreaterEqual(mock_strategy_resolver.call_count, 1)


if __name__ == "__main__":
    unittest.main()
