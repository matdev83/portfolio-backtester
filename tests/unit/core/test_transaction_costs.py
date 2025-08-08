import unittest
import pandas as pd

from portfolio_backtester.trading.transaction_costs import RealisticTransactionCostModel


class TestTransactionCosts(unittest.TestCase):
    """
    Test suite for the transaction cost models.
    """

    def setUp(self):
        """
        Set up a mock configuration and data for testing.
        """
        self.mock_global_config = {
            "slippage_bps": 2.0,
            "commission_min_per_order": 1.0,
            "commission_per_share": 0.005,
            "commission_max_percent_of_trade": 0.005,
        }
        self.mock_turnover = pd.Series(
            [0.1, 0.2], index=pd.to_datetime(["2020-01-02", "2020-01-03"])
        )
        self.mock_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.6], "GOOG": [0.5, 0.4]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        self.mock_prices = pd.DataFrame(
            {"AAPL": [100, 102], "GOOG": [200, 205]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )

    def test_realistic_model_calculation(self):
        """
        Test that the realistic transaction cost model calculates costs correctly.
        """
        model = RealisticTransactionCostModel(self.mock_global_config)
        total_costs, breakdown = model.calculate(
            turnover=self.mock_turnover,
            weights_daily=self.mock_weights,
            price_data=self.mock_prices,
            portfolio_value=100000.0,
        )

        self.assertIsInstance(total_costs, pd.Series)
        self.assertIsInstance(breakdown, dict)
        self.assertIn("commission_costs", breakdown)
        self.assertIn("slippage_costs", breakdown)
        self.assertIn("total_costs", breakdown)
        self.assertAlmostEqual(total_costs.sum(), 0.00006, places=5)


if __name__ == "__main__":
    unittest.main()
