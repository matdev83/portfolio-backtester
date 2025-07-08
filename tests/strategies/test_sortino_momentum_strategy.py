import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.sortino_momentum_strategy import SortinoMomentumStrategy
# Removed SortinoRatio and precompute_features imports


class TestSortinoMomentumStrategy(unittest.TestCase):

    def setUp(self):
        self.dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))
        stock_a_prices = [100, 110, 95, 130, 120, 150, 140, 170, 160, 190, 180, 210]
        stock_b_prices = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]
        stock_c_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
        stock_d_prices = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        stock_e_prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]
        
        self.all_historical_data = pd.DataFrame({ # Renamed
            'StockA': stock_a_prices, 'StockB': stock_b_prices,
            'StockC': stock_c_prices, 'StockD': stock_d_prices,
            'StockE': stock_e_prices,
        }, index=self.dates)
        
        self.benchmark_historical_data = pd.DataFrame( # Made DataFrame
            {'SPY': [100] * 12}, index=self.dates
        )

        self.default_strategy_config = {
            'rolling_window': 3,
            'num_holdings': 2, # Using num_holdings
            'smoothing_lambda': 0.5,
            'leverage': 1.0,
            'long_only': True,
            'target_return': 0.0, # target_return_pct in strategy
            'apply_trading_lag': False
        }
        self.strategy = SortinoMomentumStrategy(self.default_strategy_config)
        self.rolling_window_periods = self.default_strategy_config['rolling_window']


    def test_generate_signals_smoke(self):
        """Smoke test for the refactored generate_signals."""
        strategy = SortinoMomentumStrategy(self.default_strategy_config)
        all_generated_weights = []
        try:
            for current_date in self.all_historical_data.index:
                weights_for_date = strategy.generate_signals(
                    self.all_historical_data,
                    self.benchmark_historical_data,
                    current_date
                )
                all_generated_weights.append(weights_for_date)

            final_weights_df = pd.concat(all_generated_weights)
            self.assertIsInstance(final_weights_df, pd.DataFrame)
            self.assertEqual(final_weights_df.shape[0], len(self.all_historical_data.index))
            self.assertEqual(final_weights_df.shape[1], len(self.all_historical_data.columns))
            self.assertTrue(np.isfinite(final_weights_df.values).all())

        except Exception as e:
            self.fail(f"generate_signals raised an exception: {e}")

    def test_internal_calculate_sortino_ratio(self): # Renamed
        """Test the internal Sortino ratio calculation."""
        strategy = SortinoMomentumStrategy(self.default_strategy_config)
        # The _calculate_sortino_ratio method expects price_data up to current_date
        # For testing the calculation over the whole period:
        rolling_sortino = strategy._calculate_sortino_ratio(self.all_historical_data)
        
        self.assertEqual(rolling_sortino.shape, self.all_historical_data.shape)
        
        # The refactored _calculate_sortino_ratio ends with fillna(0).
        # pct_change introduces 1 NaN row. Rolling window of N needs N periods.
        # So, first `rolling_window_periods` rows should be 0.
        self.assertTrue((rolling_sortino.iloc[:self.rolling_window_periods-1] == 0).all().all())
        
        final_sortino = rolling_sortino.iloc[-1]
        self.assertGreater(final_sortino['StockE'], final_sortino['StockB'])

    def test_candidate_weights_calculation(self):
        """Test _calculate_candidate_weights from BaseStrategy."""
        strategy = SortinoMomentumStrategy(self.default_strategy_config) # Uses num_holdings=2
        sortino_scores = pd.Series({
            'StockA': 0.5, 'StockB': -1.0, 'StockC': 0.2,
            'StockD': 0.0, 'StockE': 0.8
        })
        weights = strategy._calculate_candidate_weights(sortino_scores)
        self.assertEqual(weights['StockE'], 0.5)
        self.assertEqual(weights['StockA'], 0.5)
        self.assertEqual(weights['StockB'], 0.0)

    def test_leverage_and_smoothing(self):
        """Test _apply_leverage_and_smoothing from BaseStrategy."""
        config = {**self.default_strategy_config, 'leverage': 0.5, 'smoothing_lambda': 0.5}
        strategy = SortinoMomentumStrategy(config)
        cand = pd.Series({'StockA': 0.5, 'StockC': 0.5}, index=['StockA', 'StockB', 'StockC', 'StockD', 'StockE']).fillna(0.0)
        w_prev = pd.Series({'StockB': 0.25, 'StockD': 0.25}, index=['StockA', 'StockB', 'StockC', 'StockD', 'StockE']).fillna(0.0)
        w_new = strategy._apply_leverage_and_smoothing(cand, w_prev)
        
        # Expected from previous test: SA=0.166, SB=0.083, SC=0.166, SD=0.083
        # Here, cand sums to 1. prev_weights sums to 0.5.
        # w_new for StockA: 0.5 * 0 + 0.5 * 0.5 = 0.25
        # w_new for StockC: 0.5 * 0 + 0.5 * 0.5 = 0.25
        # w_new for StockB: 0.5 * 0.25 + 0.5 * 0 = 0.125
        # w_new for StockD: 0.5 * 0.25 + 0.5 * 0 = 0.125
        # Sum of positive new weights: 0.25+0.25 = 0.5. Sum of negative new weights: 0.
        # Leverage factor for positive: 0.5 / 0.5 = 1.0. No change.
        # So expected: SA=0.25, SC=0.25, SB=0.125, SD=0.125
        # The previous test's numbers were different due to different cand/w_prev.
        # Let's re-verify the logic for this specific case:
        # w_sm = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand
        # w_sm['StockA'] = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        # w_sm['StockB'] = 0.5 * 0.25 + 0.5 * 0.0 = 0.125
        # w_sm['StockC'] = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        # w_sm['StockD'] = 0.5 * 0.25 + 0.5 * 0.0 = 0.125
        # long_lev = w_sm[w_sm > 0].sum() = 0.25 + 0.125 + 0.25 + 0.125 = 0.75
        # leverage = 0.5. Factor = 0.5 / 0.75 = 2/3
        # w_new['StockA'] = 0.25 * (2/3) = 0.1666...
        # w_new['StockB'] = 0.125 * (2/3) = 0.0833...
        # w_new['StockC'] = 0.25 * (2/3) = 0.1666...
        # w_new['StockD'] = 0.125 * (2/3) = 0.0833...
        self.assertAlmostEqual(w_new['StockA'], 0.16666666, places=5)
        self.assertAlmostEqual(w_new['StockB'], 0.08333333, places=5)
        self.assertAlmostEqual(w_new['StockC'], 0.16666666, places=5)
        self.assertAlmostEqual(w_new['StockD'], 0.08333333, places=5)
        self.assertLessEqual(w_new[w_new > 0].sum(), 0.5 + 1e-6)

    # test_sma_filter removed as SMA filter is not part of this strategy's direct logic now

    def test_target_return_parameter_on_internal_calc(self): # Renamed
        """Test that different target returns affect internal Sortino calculation."""
        config_zero = {**self.default_strategy_config, 'target_return': 0.0, 'rolling_window': 6}
        config_positive = {**self.default_strategy_config, 'target_return': 0.02, 'rolling_window': 6} # 0.02 assumed per-period
        
        strategy_zero = SortinoMomentumStrategy(config_zero)
        strategy_positive = SortinoMomentumStrategy(config_positive)
        
        # Use a slice of data for calculation consistency
        data_slice = self.all_historical_data

        sortino_zero = strategy_zero._calculate_sortino_ratio(data_slice)
        sortino_positive = strategy_positive._calculate_sortino_ratio(data_slice)
        
        final_sortino_zero = sortino_zero.iloc[-1]
        final_sortino_positive = sortino_positive.iloc[-1]
        
        self.assertTrue((final_sortino_positive <= final_sortino_zero).any())
        # For StockE (steady positive returns), higher target should definitely lower Sortino
        if 'StockE' in final_sortino_zero and 'StockE' in final_sortino_positive:
             if final_sortino_zero['StockE'] > 0 : # Only if original sortino was positive
                self.assertLess(final_sortino_positive['StockE'], final_sortino_zero['StockE'])


    def test_num_holdings_vs_top_decile_fraction(self):
        """Test BaseStrategy's num_holdings override of top_decile_fraction."""
        # This tests BaseStrategy._calculate_candidate_weights behavior
        strategy_config_num = {'num_holdings': 2, 'top_decile_fraction': 0.8, 'long_only': True}
        strategy_config_fraction = {'num_holdings': None, 'top_decile_fraction': 0.4, 'long_only': True} # 0.4 of 5 is 2
        
        strategy_num = SortinoMomentumStrategy(strategy_config_num)
        strategy_fraction = SortinoMomentumStrategy(strategy_config_fraction)
        
        sortino_scores = pd.Series({'StockA': 0.5, 'StockB': -1.0, 'StockC': 0.2, 'StockD': 0.0, 'StockE': 0.8})
        
        weights_num = strategy_num._calculate_candidate_weights(sortino_scores)
        weights_fraction = strategy_fraction._calculate_candidate_weights(sortino_scores)
        
        self.assertEqual((weights_num > 0).sum(), 2)
        self.assertEqual((weights_fraction > 0).sum(), 2) # 0.4 * 5 stocks = 2
        self.assertTrue((weights_num.index[(weights_num > 0)] == weights_fraction.index[(weights_fraction > 0)]).all())

if __name__ == '__main__':
    unittest.main()