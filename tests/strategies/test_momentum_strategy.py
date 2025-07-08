import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
# from src.portfolio_backtester.feature_engineering import precompute_features # Removed

class TestMomentumStrategy(unittest.TestCase):

    def setUp(self):
        # Define rebalance dates (e.g., month ends)
        self.rebalance_dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))

        # Create daily dates covering the span of rebalance_dates with some buffer
        daily_start_date = self.rebalance_dates.min() - pd.DateOffset(months=12) # Need enough history for lookbacks
        daily_end_date = self.rebalance_dates.max()
        self.daily_dates = pd.date_range(start=daily_start_date, end=daily_end_date, freq='B') # Business days

        # --- Asset Data (Daily OHLCV) ---
        tickers = ['StockA', 'StockB', 'StockC', 'StockD']
        data_frames = []
        for ticker in tickers:
            # Create somewhat realistic daily data, ensuring values change for momentum
            # StockA: trending up
            # StockB: trending down
            # StockC: mild trend up
            # StockD: flat
            if ticker == 'StockA':
                base_price = np.linspace(80, 210, len(self.daily_dates))
            elif ticker == 'StockB':
                base_price = np.linspace(120, 10, len(self.daily_dates))
            elif ticker == 'StockC':
                base_price = np.linspace(95, 115, len(self.daily_dates))
            else: # StockD
                base_price = np.full(len(self.daily_dates), 100)

            noise = np.random.normal(0, 0.5, size=len(self.daily_dates))
            close_prices = base_price + noise
            open_prices = close_prices - np.random.uniform(0, 0.5, size=len(self.daily_dates))
            high_prices = close_prices + np.random.uniform(0, 0.5, size=len(self.daily_dates))
            low_prices = close_prices - np.random.uniform(0, 0.5, size=len(self.daily_dates))
            volume = np.random.randint(1000, 5000, size=len(self.daily_dates))

            df = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volume
            }, index=self.daily_dates)
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=['Ticker', 'Field'])
            data_frames.append(df)

        self.daily_ohlc_data = pd.concat(data_frames, axis=1)

        # --- Benchmark Data (Daily OHLCV) ---
        benchmark_base_price = np.linspace(90, 110, len(self.daily_dates))
        benchmark_noise = np.random.normal(0, 0.5, size=len(self.daily_dates))
        benchmark_close = benchmark_base_price + benchmark_noise
        benchmark_df = pd.DataFrame({
            'Open': benchmark_close - np.random.uniform(0,0.2, size=len(self.daily_dates)),
            'High': benchmark_close + np.random.uniform(0,0.2, size=len(self.daily_dates)),
            'Low': benchmark_close - np.random.uniform(0,0.2, size=len(self.daily_dates)),
            'Close': benchmark_close,
            'Volume': np.random.randint(10000, 50000, size=len(self.daily_dates))
        }, index=self.daily_dates)
        benchmark_df.columns = pd.MultiIndex.from_product([['SPY'], benchmark_df.columns], names=['Ticker', 'Field'])
        self.benchmark_ohlc_data = benchmark_df

        # Store monthly closes for convenience in some assertions if needed, derived from daily
        self.asset_monthly_closes = self.daily_ohlc_data.xs('Close', level='Field', axis=1).resample('ME').last()
        self.benchmark_monthly_closes = self.benchmark_ohlc_data.xs('Close', level='Field', axis=1).resample('ME').last()


    def test_generate_signals_smoke(self):
        strategy_config = {
            'strategy_params': { # Ensure params are nested correctly if strategy expects it
                'lookback_months': 3,
                'skip_months': 1,
                'top_decile_fraction': 0.5, # Effectively top 2 for 4 stocks
                'smoothing_lambda': 0.5,
                'leverage': 1.0,
                'long_only': True,
                'price_column_asset': 'Close',
                'price_column_benchmark': 'Close',
            },
            # Global config items if needed by strategy's base class or helpers directly
            'num_holdings': None,
        }
        strategy = MomentumStrategy(strategy_config)

        all_signal_weights = []
        for current_rebalance_date in self.rebalance_dates:
            if current_rebalance_date < self.daily_dates.min() + pd.DateOffset(months=strategy_config['strategy_params']['lookback_months']):
                # Skip dates where there isn't enough history for the initial lookback
                continue

            historical_assets = self.daily_ohlc_data[self.daily_ohlc_data.index <= current_rebalance_date]
            historical_benchmark = self.benchmark_ohlc_data[self.benchmark_ohlc_data.index <= current_rebalance_date]

            try:
                weights_df = strategy.generate_signals(
                    all_historical_data=historical_assets,
                    benchmark_historical_data=historical_benchmark,
                    current_date=current_rebalance_date
                )
                all_signal_weights.append(weights_df)
            except Exception as e:
                self.fail(f"generate_signals raised an exception on {current_rebalance_date}: {e}")

        self.assertTrue(len(all_signal_weights) > 0, "No signals were generated.")
        final_weights_df = pd.concat(all_signal_weights)
        self.assertFalse(final_weights_df.empty, "Concatenated weights DataFrame is empty.")


    def test_top_performer_selection(self):
        strategy_config = {
            'strategy_params': {
                'lookback_months': 3,
                'skip_months': 0,
                'num_holdings': 1, # Explicitly select only the top stock
                'smoothing_lambda': 0.0, # No smoothing
                'leverage': 1.0,
                'long_only': True,
                'price_column_asset': 'Close',
                'price_column_benchmark': 'Close',
            },
             # num_holdings can also be at the top level of strategy_config
            'num_holdings': 1,
        }
        strategy = MomentumStrategy(strategy_config)

        all_signal_weights = []
        # Use rebalance dates that have enough lookback history
        valid_rebalance_dates = [
            d for d in self.rebalance_dates
            if d >= self.daily_dates.min() + pd.DateOffset(months=strategy_config['strategy_params']['lookback_months'])
        ]

        for current_rebalance_date in valid_rebalance_dates:
            historical_assets = self.daily_ohlc_data[self.daily_ohlc_data.index <= current_rebalance_date]
            historical_benchmark = self.benchmark_ohlc_data[self.benchmark_ohlc_data.index <= current_rebalance_date]

            weights_df = strategy.generate_signals(
                all_historical_data=historical_assets,
                benchmark_historical_data=historical_benchmark,
                current_date=current_rebalance_date
            )
            all_signal_weights.append(weights_df)

        self.assertTrue(len(all_signal_weights) > 0, "No signals generated for top performer test.")
        final_weights_df = pd.concat(all_signal_weights)

        # Check weights at the last valid rebalance date
        # StockA is designed to be the top performer
        last_date_weights = final_weights_df.iloc[-1]
        self.assertAlmostEqual(last_date_weights['StockA'], 1.0, places=3)
        self.assertAlmostEqual(last_date_weights['StockB'], 0.0, places=5)
        self.assertAlmostEqual(last_date_weights['StockC'], 0.0, places=5) # StockC might be positive if num_holdings was >1
        self.assertAlmostEqual(last_date_weights['StockD'], 0.0, places=5)


    def test_generate_signals_with_nans_in_price(self):
        # Modify a copy of daily_ohlc_data to include NaNs
        data_with_nans = self.daily_ohlc_data.copy()

        # Introduce NaNs into StockB's Close prices for a few specific daily periods
        # Choose dates that fall within a rebalance period for testing
        nan_dates_stock_b = pd.to_datetime(['2020-03-10', '2020-03-11', '2020-07-15'])
        for date_idx in nan_dates_stock_b:
            if date_idx in data_with_nans.index:
                 data_with_nans.loc[date_idx, ('StockB', 'Close')] = np.nan
                 # Also affect H, L, O to be consistent if ATR uses them
                 data_with_nans.loc[date_idx, ('StockB', 'High')] = np.nan
                 data_with_nans.loc[date_idx, ('StockB', 'Low')] = np.nan
                 data_with_nans.loc[date_idx, ('StockB', 'Open')] = np.nan


        strategy_config = {
            'strategy_params': {
                'lookback_months': 3,
                'skip_months': 0,
                'top_decile_fraction': 0.5, # num_holdings will be ceil(0.5 * (4-potential_NaN_asset))
                'smoothing_lambda': 0.0, # No smoothing for clarity
                'leverage': 1.0,
                'long_only': True,
                'price_column_asset': 'Close',
                'price_column_benchmark': 'Close',
            }
        }
        strategy = MomentumStrategy(strategy_config)

        all_signal_weights = []
        valid_rebalance_dates = [
            d for d in self.rebalance_dates
            if d >= self.daily_dates.min() + pd.DateOffset(months=strategy_config['strategy_params']['lookback_months'])
        ]

        for current_rebalance_date in valid_rebalance_dates:
            historical_assets = data_with_nans[data_with_nans.index <= current_rebalance_date]
            historical_benchmark = self.benchmark_ohlc_data[self.benchmark_ohlc_data.index <= current_rebalance_date]

            weights_df = strategy.generate_signals(
                all_historical_data=historical_assets,
                benchmark_historical_data=historical_benchmark,
                current_date=current_rebalance_date
            )
            all_signal_weights.append(weights_df)

        self.assertTrue(len(all_signal_weights) > 0, "No signals generated with NaN data.")
        final_weights_df = pd.concat(all_signal_weights)

        # Assert that some trades are generated (weights are not all zero across the entire period)
        # And StockB might have zero weight on dates affected by its NaNs in lookback
        self.assertTrue(final_weights_df.sum().sum() > 0.0, "No trades generated with NaN data (all weights zero).")

        # Check a specific date where StockB's momentum might be affected by recent NaNs.
        # E.g. if 2020-07-31 is a rebalance date, and StockB had NaNs on 2020-07-15, its momentum score might be NaN or 0.
        # The strategy's _calculate_momentum_scores fills NaN scores with 0.0.
        # So StockB might not be selected. StockA should still be selected.
        if pd.Timestamp('2020-07-31') in final_weights_df.index:
            weights_on_july_31 = final_weights_df.loc[pd.Timestamp('2020-07-31')]
            if 'StockB' in weights_on_july_31:
                 # Depending on num_holdings and other scores, StockB might be 0
                 pass # Assertion here depends on specific expectation of num_holdings vs available assets
            self.assertTrue(weights_on_july_31.get('StockA', 0) > 0 or weights_on_july_31.get('StockC', 0) > 0,
                            "Expected StockA or StockC to be selected on 2020-07-31 despite NaNs in StockB")


if __name__ == '__main__':
    unittest.main()