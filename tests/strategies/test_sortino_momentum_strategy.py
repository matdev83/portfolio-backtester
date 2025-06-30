import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.sortino_momentum_strategy import SortinoMomentumStrategy

class TestSortinoMomentumStrategy(unittest.TestCase):

    def setUp(self):
        # Create a sample price dataframe with different volatility and return patterns
        dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))
        
        # StockA: High returns with high volatility (including downside)
        stock_a_prices = [100, 110, 95, 130, 120, 150, 140, 170, 160, 190, 180, 210]
        
        # StockB: Declining stock with high downside volatility
        stock_b_prices = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]
        
        # StockC: Steady growth with low volatility
        stock_c_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
        
        # StockD: Flat performance
        stock_d_prices = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        
        # StockE: High returns with low downside (good Sortino)
        stock_e_prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]
        
        self.data = pd.DataFrame({
            'StockA': stock_a_prices,
            'StockB': stock_b_prices,
            'StockC': stock_c_prices,
            'StockD': stock_d_prices,
            'StockE': stock_e_prices,
        }, index=dates)
        
        self.benchmark_data = pd.Series([100] * 12, index=dates)

    def test_generate_signals_smoke(self):
        """Smoke test to ensure the function runs without errors."""
        strategy_config = {
            'rolling_window': 3,
            'top_decile_fraction': 0.5,
            'smoothing_lambda': 0.5,
            'leverage': 1.0,
            'long_only': True,
            'target_return': 0.0
        }
        strategy = SortinoMomentumStrategy(strategy_config)
        try:
            weights = strategy.generate_signals(self.data, self.benchmark_data)
            self.assertIsInstance(weights, pd.DataFrame)
            self.assertEqual(weights.shape[1], len(self.data.columns))
        except Exception as e:
            self.fail(f"generate_signals raised an exception: {e}")

    def test_rolling_sortino_calculation(self):
        """Test that rolling Sortino calculation works correctly."""
        strategy_config = {
            'rolling_window': 3,
            'target_return': 0.0
        }
        strategy = SortinoMomentumStrategy(strategy_config)
        
        # Calculate returns
        rets = self.data.pct_change(fill_method=None)
        
        # Calculate rolling Sortino
        rolling_sortino = strategy._calculate_rolling_sortino(rets, 3, 0.0)
        
        # Check that the result has the right shape
        self.assertEqual(rolling_sortino.shape, rets.shape)
        
        # Check that early periods are NaN or 0 (due to insufficient data)
        self.assertTrue(pd.isna(rolling_sortino.iloc[0]).all() or (rolling_sortino.iloc[0] == 0).all())
        
        # Check that later periods have valid values for stocks with returns
        # StockE should have a good Sortino ratio (steady positive returns, low downside)
        final_sortino = rolling_sortino.iloc[-1]
        self.assertGreater(final_sortino['StockE'], final_sortino['StockB'])  # StockE > StockB

    def test_candidate_weights_calculation(self):
        """Test that candidate weights are calculated correctly."""
        strategy_config = {
            'num_holdings': 2,
            'long_only': True
        }
        strategy = SortinoMomentumStrategy(strategy_config)
        
        # Create a mock Sortino series
        sortino_scores = pd.Series({
            'StockA': 0.5,
            'StockB': -1.0,
            'StockC': 0.2,
            'StockD': 0.0,
            'StockE': 0.8
        })
        
        weights = strategy._calculate_candidate_weights(sortino_scores)
        
        # Should select top 2 stocks (StockE and StockA)
        self.assertEqual(weights['StockE'], 0.5)  # 1/2
        self.assertEqual(weights['StockA'], 0.5)  # 1/2
        self.assertEqual(weights['StockB'], 0.0)
        self.assertEqual(weights['StockC'], 0.0)
        self.assertEqual(weights['StockD'], 0.0)

    def test_leverage_and_smoothing(self):
        """Test leverage and smoothing application."""
        strategy_config = {
            'leverage': 0.5,
            'smoothing_lambda': 0.5
        }
        strategy = SortinoMomentumStrategy(strategy_config)
        
        # Create candidate weights
        cand = pd.Series({
            'StockA': 0.5,
            'StockB': 0.0,
            'StockC': 0.5,
            'StockD': 0.0,
            'StockE': 0.0
        })
        
        # Previous weights
        w_prev = pd.Series({
            'StockA': 0.0,
            'StockB': 0.25,
            'StockC': 0.0,
            'StockD': 0.25,
            'StockE': 0.0
        })
        
        w_new = strategy._apply_leverage_and_smoothing(cand, w_prev)
        
        # Check that smoothing was applied
        self.assertAlmostEqual(w_new['StockA'], 0.16666666666666666, places=2)  # (0.5 * 0.0 + 0.5 * 0.5) * (0.5 / 0.75) = 0.16666666666666666
        self.assertAlmostEqual(w_new['StockB'], 0.08333333333333333, places=2)  # (0.5 * 0.25 + 0.5 * 0.0) * (0.5 / 0.75) = 0.08333333333333333
        
        # Check that total leverage is approximately correct
        total_leverage = w_new[w_new > 0].sum()
        self.assertLessEqual(total_leverage, 0.5 + 1e-6)  # Should not exceed target leverage

    def test_sma_filter(self):
        """Test that SMA filter works correctly."""
        strategy_config = {
            'rolling_window': 3,
            'top_decile_fraction': 0.5,
            'smoothing_lambda': 0.0,  # No smoothing for clearer test
            'leverage': 1.0,
            'long_only': True,
            'sma_filter_window': 3,
            'target_return': 0.0
        }
        strategy = SortinoMomentumStrategy(strategy_config)
        
        # Create benchmark data that goes below SMA (risk-off period)
        benchmark_declining = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45], 
                                      index=self.data.index)
        
        weights = strategy.generate_signals(self.data, benchmark_declining)
        
        # In periods where benchmark is below SMA, weights should be zero
        # Check the last few periods where this condition should be true
        final_weights = weights.iloc[-1]
        self.assertTrue((final_weights == 0).all(), "All weights should be zero during risk-off periods")

    def test_target_return_parameter(self):
        """Test that different target returns affect Sortino calculation."""
        strategy_config_zero = {
            'rolling_window': 6,
            'target_return': 0.0
        }
        strategy_config_positive = {
            'rolling_window': 6,
            'target_return': 0.02  # 2% monthly target
        }
        
        strategy_zero = SortinoMomentumStrategy(strategy_config_zero)
        strategy_positive = SortinoMomentumStrategy(strategy_config_positive)
        
        rets = self.data.pct_change(fill_method=None)
        
        sortino_zero = strategy_zero._calculate_rolling_sortino(rets, 6, 0.0)
        sortino_positive = strategy_positive._calculate_rolling_sortino(rets, 6, 0.02)
        
        # With higher target return, Sortino ratios should generally be lower
        # (since excess return decreases and downside deviation may increase)
        final_sortino_zero = sortino_zero.iloc[-1]
        final_sortino_positive = sortino_positive.iloc[-1]
        
        # At least some stocks should have lower Sortino with higher target
        self.assertTrue((final_sortino_positive <= final_sortino_zero).any())

    def test_num_holdings_vs_top_decile_fraction(self):
        """Test that num_holdings parameter overrides top_decile_fraction."""
        strategy_config_num = {
            'num_holdings': 2,
            'top_decile_fraction': 0.8,  # This should be ignored
            'long_only': True
        }
        strategy_config_fraction = {
            'top_decile_fraction': 0.4,  # 40% of 5 stocks = 2 stocks
            'long_only': True
        }
        
        strategy_num = SortinoMomentumStrategy(strategy_config_num)
        strategy_fraction = SortinoMomentumStrategy(strategy_config_fraction)
        
        sortino_scores = pd.Series({
            'StockA': 0.5,
            'StockB': -1.0,
            'StockC': 0.2,
            'StockD': 0.0,
            'StockE': 0.8
        })
        
        weights_num = strategy_num._calculate_candidate_weights(sortino_scores)
        weights_fraction = strategy_fraction._calculate_candidate_weights(sortino_scores)
        
        # Both should select exactly 2 stocks
        self.assertEqual((weights_num > 0).sum(), 2)
        self.assertEqual((weights_fraction > 0).sum(), 2)
        
        # Should select the same top 2 stocks
        self.assertTrue((weights_num > 0).equals(weights_fraction > 0))

if __name__ == '__main__':
    unittest.main()