import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.ema_roro_strategy import EMARoRoStrategy
from src.portfolio_backtester.roro_signals import DummyRoRoSignal


class TestEMARoRoStrategy(unittest.TestCase):

    def setUp(self):
        """Set up test data and strategy configuration."""
        self.strategy_config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "leverage": 2.0,
            "risk_off_leverage_multiplier": 0.5
        }
        self.strategy = EMARoRoStrategy(self.strategy_config)

        # Create test dates covering both risk-on and risk-off periods
        # Based on DummyRoRoSignal hardcoded dates:
        # Risk-on: 2006-01-01 to 2009-12-31, 2020-01-01 to 2020-04-01, 2022-01-01 to 2022-11-05
        self.test_dates = pd.to_datetime([
            "2019-12-31",  # Risk-off (before 2020 window)
            "2020-01-15",  # Risk-on (during 2020 window)
            "2020-02-15",  # Risk-on (during 2020 window)
            "2020-04-15",  # Risk-off (after 2020 window)
            "2021-12-31",  # Risk-off (before 2022 window)
            "2022-01-15",  # Risk-on (during 2022 window)
            "2022-06-15",  # Risk-on (during 2022 window)
            "2022-11-15",  # Risk-off (after 2022 window)
        ])

        # Create sample asset data with sufficient history for EMA calculations
        self.assets = ["AAPL", "MSFT", "GOOGL"]
        
        # Generate historical data starting well before test dates
        start_date = pd.Timestamp("2019-01-01")
        end_date = self.test_dates[-1]
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create MultiIndex columns for OHLCV data
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([self.assets, fields], names=['Ticker', 'Field'])
        
        # Generate realistic price data with trends
        np.random.seed(42)
        data = {}
        for asset in self.assets:
            base_price = 100 + np.random.normal(0, 20)  # Different starting prices
            prices = []
            current_price = base_price
            
            for i, date in enumerate(all_dates):
                # Add some trend and volatility
                daily_return = np.random.normal(0.0005, 0.02)  # ~0.05% daily return, 2% volatility
                current_price *= (1 + daily_return)
                
                # Create OHLCV data
                open_price = current_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
                close_price = current_price
                volume = int(1000000 + np.random.normal(0, 200000))
                
                data[(asset, 'Open')] = data.get((asset, 'Open'), []) + [open_price]
                data[(asset, 'High')] = data.get((asset, 'High'), []) + [high_price]
                data[(asset, 'Low')] = data.get((asset, 'Low'), []) + [low_price]
                data[(asset, 'Close')] = data.get((asset, 'Close'), []) + [close_price]
                data[(asset, 'Volume')] = data.get((asset, 'Volume'), []) + [volume]
        
        self.asset_data = pd.DataFrame(data, index=all_dates, columns=columns)
        
        # Create benchmark data (SPY)
        benchmark_columns = pd.MultiIndex.from_product([['SPY'], fields], names=['Ticker', 'Field'])
        benchmark_data = {}
        base_price = 300
        current_price = base_price
        
        for i, date in enumerate(all_dates):
            daily_return = np.random.normal(0.0003, 0.015)  # Slightly lower volatility for benchmark
            current_price *= (1 + daily_return)
            
            open_price = current_price * (1 + np.random.normal(0, 0.003))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.008)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.008)))
            close_price = current_price
            volume = int(50000000 + np.random.normal(0, 10000000))
            
            benchmark_data[('SPY', 'Open')] = benchmark_data.get(('SPY', 'Open'), []) + [open_price]
            benchmark_data[('SPY', 'High')] = benchmark_data.get(('SPY', 'High'), []) + [high_price]
            benchmark_data[('SPY', 'Low')] = benchmark_data.get(('SPY', 'Low'), []) + [low_price]
            benchmark_data[('SPY', 'Close')] = benchmark_data.get(('SPY', 'Close'), []) + [close_price]
            benchmark_data[('SPY', 'Volume')] = benchmark_data.get(('SPY', 'Volume'), []) + [volume]
        
        self.benchmark_data = pd.DataFrame(benchmark_data, index=all_dates, columns=benchmark_columns)

    def test_roro_signal_integration(self):
        """Test that the strategy correctly integrates RoRo signals."""
        # Verify that the strategy has the RoRo signal class set
        self.assertEqual(self.strategy.roro_signal_class, DummyRoRoSignal)
        
        # Verify that get_roro_signal returns a DummyRoRoSignal instance
        roro_instance = self.strategy.get_roro_signal()
        self.assertIsInstance(roro_instance, DummyRoRoSignal)

    def test_leverage_adjustment_during_risk_periods(self):
        """Test that leverage is adjusted correctly during risk-on and risk-off periods."""
        
        # Expected risk-on dates based on DummyRoRoSignal windows
        risk_on_dates = pd.to_datetime([
            "2020-01-15", "2020-02-15",  # During 2020-01-01 to 2020-04-01 window
            "2022-01-15", "2022-06-15"   # During 2022-01-01 to 2022-11-05 window
        ])
        
        risk_off_dates = pd.to_datetime([
            "2019-12-31", "2020-04-15",  # Outside risk-on windows
            "2021-12-31", "2022-11-15"
        ])
        
        # Test risk-on periods (should use full leverage)
        for date in risk_on_dates:
            signals = self.strategy.generate_signals(
                all_historical_data=self.asset_data,
                benchmark_historical_data=self.benchmark_data,
                current_date=date
            )
            
            # Check that signals are generated (non-zero weights possible)
            total_weight = abs(signals.iloc[0]).sum()
            
            # If there are any positions, the total leverage should be close to base leverage
            if total_weight > 0:
                expected_max_leverage = self.strategy_config["leverage"]
                self.assertLessEqual(total_weight, expected_max_leverage + 0.01, 
                                   f"Risk-on leverage should not exceed {expected_max_leverage} on {date}")
        
        # Test risk-off periods (should use reduced leverage)
        for date in risk_off_dates:
            signals = self.strategy.generate_signals(
                all_historical_data=self.asset_data,
                benchmark_historical_data=self.benchmark_data,
                current_date=date
            )
            
            total_weight = abs(signals.iloc[0]).sum()
            
            # If there are any positions, the total leverage should be reduced
            if total_weight > 0:
                expected_max_leverage = (self.strategy_config["leverage"] * 
                                       self.strategy_config["risk_off_leverage_multiplier"])
                self.assertLessEqual(total_weight, expected_max_leverage + 0.01,
                                   f"Risk-off leverage should not exceed {expected_max_leverage} on {date}")

    def test_ema_signal_generation(self):
        """Test that the underlying EMA logic still works correctly."""
        # Test on a risk-on date to ensure full functionality
        test_date = pd.Timestamp("2020-02-15")  # Risk-on period
        
        signals = self.strategy.generate_signals(
            all_historical_data=self.asset_data,
            benchmark_historical_data=self.benchmark_data,
            current_date=test_date
        )
        
        # Verify signals DataFrame structure
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), 1)  # Should have one row for the current date
        self.assertEqual(signals.index[0], test_date)
        self.assertEqual(len(signals.columns), len(self.assets))
        
        # Verify that weights are numeric and finite
        for col in signals.columns:
            weight = signals.iloc[0][col]
            self.assertTrue(pd.isna(weight) or np.isfinite(weight), 
                          f"Weight for {col} should be finite or NaN")

    def test_strategy_string_representation(self):
        """Test the string representation of the strategy."""
        strategy_str = str(self.strategy)
        expected_str = f"EMARoRo({self.strategy_config['fast_ema_days']},{self.strategy_config['slow_ema_days']},mult={self.strategy_config['risk_off_leverage_multiplier']})"
        self.assertEqual(strategy_str, expected_str)

    def test_tunable_parameters(self):
        """Test that tunable parameters include both EMA and RoRo parameters."""
        params = self.strategy.tunable_parameters()
        expected_params = {'fast_ema_days', 'slow_ema_days', 'leverage', 'risk_off_leverage_multiplier'}
        self.assertTrue(expected_params.issubset(params))


if __name__ == '__main__':
    unittest.main()