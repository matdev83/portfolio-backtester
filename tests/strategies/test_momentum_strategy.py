import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
# No precompute_features needed

class TestMomentumStrategy(unittest.TestCase):

    def setUp(self):
        self.rebalance_dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))
        daily_start_date = self.rebalance_dates.min() - pd.DateOffset(months=12)
        daily_end_date = self.rebalance_dates.max()
        self.daily_dates = pd.date_range(start=daily_start_date, end=daily_end_date, freq='B')

        tickers = ['StockA', 'StockB', 'StockC', 'StockD']
        data_frames = []
        np.random.seed(42) # Ensure consistent test data
        for ticker in tickers:
            if ticker == 'StockA': base_price = np.linspace(80, 210, len(self.daily_dates))
            elif ticker == 'StockB': base_price = np.linspace(120, 10, len(self.daily_dates))
            elif ticker == 'StockC': base_price = np.linspace(95, 115, len(self.daily_dates))
            else: base_price = np.full(len(self.daily_dates), 100)
            noise = np.random.normal(0, 0.5, size=len(self.daily_dates))
            close_prices = base_price + noise
            df = pd.DataFrame({
                'Open': close_prices - np.random.uniform(0, 0.5, size=len(self.daily_dates)),
                'High': close_prices + np.random.uniform(0, 0.5, size=len(self.daily_dates)),
                'Low': close_prices - np.random.uniform(0, 0.5, size=len(self.daily_dates)),
                'Close': close_prices,
                'Volume': np.random.randint(1000, 5000, size=len(self.daily_dates))
            }, index=self.daily_dates)
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=['Ticker', 'Field'])
            data_frames.append(df)
        self.daily_ohlc_data = pd.concat(data_frames, axis=1)

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

        # Default config, can be overridden in tests
        self.default_strategy_config = {
            'lookback_months': 3,
            'skip_months': 1,
            'top_decile_fraction': 0.5,
            'num_holdings': None, # Allow top_decile_fraction to work
            'smoothing_lambda': 0.5,
            'leverage': 1.0,
            'long_only': True,
            'price_column_asset': 'Close',
            'price_column_benchmark': 'Close',
            'sma_filter_window': None, # No SMA filter by default
            'derisk_days_under_sma': 10,
            'apply_trading_lag': False,
             # Stop loss config can be added here if testing SL explicitly
            'stop_loss_config': {'type': 'NoStopLoss'} # Default to no stop loss for most tests
        }


    def test_generate_signals_smoke(self):
        # Use a flat strategy_config, MomentumStrategy handles nesting if "strategy_params" is present
        strategy = MomentumStrategy(self.default_strategy_config)

        all_signal_weights = []
        # Iterate through rebalance dates, ensuring enough history for the first lookback
        start_calculation_date = self.daily_dates.min() + pd.DateOffset(months=self.default_strategy_config['lookback_months'])

        for current_rebalance_date in self.rebalance_dates:
            if current_rebalance_date < start_calculation_date:
                continue

            # For generate_signals, pass the full history available up to current_rebalance_date
            # The method itself will handle slicing for momentum calculation based on its lookback/skip logic.
            # However, for SMA/RoRo/SL that also use this data, it should be the full history.
            # The current implementation of MomentumStrategy slices asset_prices_hist internally for _calculate_momentum_scores.
            # For other components (SMA, RoRo, SL), it uses all_historical_data[... <= current_date].

            try:
                weights_df = strategy.generate_signals(
                    all_historical_data=self.daily_ohlc_data, # Pass full daily OHLC
                    benchmark_historical_data=self.benchmark_ohlc_data, # Pass full daily benchmark OHLC
                    current_date=current_rebalance_date
                )
                all_signal_weights.append(weights_df)
            except Exception as e:
                self.fail(f"generate_signals raised an exception on {current_rebalance_date}: {e}")

        self.assertTrue(len(all_signal_weights) > 0, "No signals were generated.")
        final_weights_df = pd.concat(all_signal_weights)
        self.assertFalse(final_weights_df.empty, "Concatenated weights DataFrame is empty.")
        self.assertTrue(np.isfinite(final_weights_df.values).all(), "Signal weights contain NaNs or Infs.")


    def test_top_performer_selection(self):
        config = {
            **self.default_strategy_config,
            'lookback_months': 3,
            'skip_months': 0,
            'num_holdings': 1,
            'top_decile_fraction': None, # Ensure num_holdings is used
            'smoothing_lambda': 0.0,
        }
        strategy = MomentumStrategy(config)

        all_signal_weights = []
        start_calculation_date = self.daily_dates.min() + pd.DateOffset(months=config['lookback_months'])
        valid_rebalance_dates = [d for d in self.rebalance_dates if d >= start_calculation_date]

        for current_rebalance_date in valid_rebalance_dates:
            weights_df = strategy.generate_signals(
                all_historical_data=self.daily_ohlc_data,
                benchmark_historical_data=self.benchmark_ohlc_data,
                current_date=current_rebalance_date
            )
            all_signal_weights.append(weights_df)

        self.assertTrue(len(all_signal_weights) > 0, "No signals generated for top performer test.")
        final_weights_df = pd.concat(all_signal_weights)

        last_date_weights = final_weights_df.iloc[-1]
        self.assertEqual(last_date_weights['StockA'], 1.0)
        self.assertEqual(last_date_weights['StockB'], 0.0)
        self.assertEqual(last_date_weights['StockC'], 0.0)
        self.assertEqual(last_date_weights['StockD'], 0.0)


    def test_generate_signals_with_nans_in_price(self):
        data_with_nans = self.daily_ohlc_data.copy()
        nan_dates_stock_b = pd.to_datetime(['2020-03-10', '2020-03-11', '2020-07-15']) # Use daily dates
        for date_idx in nan_dates_stock_b:
            if date_idx in data_with_nans.index:
                 data_with_nans.loc[date_idx, ('StockB', 'Close')] = np.nan
                 data_with_nans.loc[date_idx, ('StockB', 'High')] = np.nan
                 data_with_nans.loc[date_idx, ('StockB', 'Low')] = np.nan
                 data_with_nans.loc[date_idx, ('StockB', 'Open')] = np.nan

        config = {
            **self.default_strategy_config,
            'lookback_months': 3,
            'skip_months': 0,
            'top_decile_fraction': 0.5,
            'num_holdings': None,
            'smoothing_lambda': 0.0,
        }
        strategy = MomentumStrategy(config)

        all_signal_weights = []
        start_calculation_date = self.daily_dates.min() + pd.DateOffset(months=config['lookback_months'])
        valid_rebalance_dates = [d for d in self.rebalance_dates if d >= start_calculation_date]

        for current_rebalance_date in valid_rebalance_dates:
            weights_df = strategy.generate_signals(
                all_historical_data=data_with_nans, # Use data with NaNs
                benchmark_historical_data=self.benchmark_ohlc_data,
                current_date=current_rebalance_date
            )
            all_signal_weights.append(weights_df)

        self.assertTrue(len(all_signal_weights) > 0, "No signals generated with NaN data.")
        final_weights_df = pd.concat(all_signal_weights)
        self.assertTrue(final_weights_df.sum().sum() > -1e-9, "Sum of weights should not be negative for long_only.")

        if pd.Timestamp('2020-07-31') in final_weights_df.index: # A rebalance date
            weights_on_july_31 = final_weights_df.loc[pd.Timestamp('2020-07-31')]
            # StockB had NaNs on 2020-07-15. Its momentum score calculated at 2020-07-31 might be 0.
            # StockA (strong uptrend) should likely be selected.
            self.assertTrue(weights_on_july_31.get('StockA', 0) > 0 or weights_on_july_31.get('StockC', 0) > 0,
                            "Expected StockA or StockC to be selected on 2020-07-31 despite NaNs in StockB's history.")
            if 'StockB' in weights_on_july_31 and config.get('num_holdings',0) is None : # if num_holdings not strictly limiting
                 # if top_decile_fraction is used, and StockB score is 0, it might not be picked.
                 # This assertion is weak without knowing other scores.
                 pass


if __name__ == '__main__':
    unittest.main()