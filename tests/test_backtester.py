import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.config import GLOBAL_CONFIG, BACKTEST_SCENARIOS

class TestBacktester(unittest.TestCase):
    def setUp(self):
        """Set up a mock backtester and data for testing."""
        self.global_config = GLOBAL_CONFIG.copy()
        self.scenarios = [
            {
                "name": "Test_Momentum_WFO",
                "strategy": "momentum",
                "strategy_params": {
                    "lookback_months": 6,
                    "leverage": 1.0,
                    "smoothing_lambda": 0.5,
                    "long_only": True,
                    "top_decile_fraction": 0.1
                },
                "rebalance_frequency": "M",
                "position_sizer": "equal_weight",
                "transaction_costs_bps": 10,
                "train_window_months": 24,
                "test_window_months": 6,
                "optimize": [
                    {
                        "parameter": "lookback_months",
                        "metric": "Sharpe Ratio",
                        "min_value": 3,
                        "max_value": 9,
                        "step": 3
                    }
                ]
            }
        ]
        class MockArgs:
            def __init__(self):
                self.optimize_min_positions = 10
                self.optimize_max_positions = 30
                self.top_n_params = 3
        self.mock_args = MockArgs()
        self.backtester = Backtester(self.global_config, self.scenarios, self.mock_args)
        
        # Create mock data
        dates = pd.date_range(start="2020-01-01", periods=48, freq="ME")
        tickers = self.global_config["universe"] + [self.global_config["benchmark"]]
        self.mock_data = pd.DataFrame(
            np.random.randn(48, len(tickers)) / 100 + 0.001,
            index=dates,
            columns=tickers
        )

    def test_walk_forward_optimization_runs(self):
        """Test that walk-forward optimization completes and produces results."""
        # Pre-calculate returns for the mock data
        rets_full = self.mock_data.pct_change().fillna(0)
        
        self.backtester.run_optimization(self.scenarios[0], self.mock_data, rets_full)
        
        # Check that results are generated
        self.assertIn("Test_Momentum_WFO (Walk-Forward Optimized)", self.backtester.results)
        
        # Check that the returns are a pandas Series
        result_data = self.backtester.results["Test_Momentum_WFO (Walk-Forward Optimized)"]
        self.assertIsInstance(result_data["returns"], pd.Series)
        
        # Check that the returns are not empty
        self.assertFalse(result_data["returns"].empty)

if __name__ == '__main__':
    unittest.main()
