"""
Test suite for UVXY RSI Strategy
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.portfolio_backtester.strategies.uvxy_rsi_strategy import UvxyRsiStrategy


class TestUvxyRsiStrategy(unittest.TestCase):
    """Test cases for UVXY RSI Strategy"""

    def setUp(self):
        """Set up test fixtures"""
        self.strategy_config = {
            "strategy_params": {
                "rsi_period": 2,
                "rsi_threshold": 30.0,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "long_only": False,
                "leverage": 1.0,
                "smoothing_lambda": 0.0,
                "position_sizer": "equal_weight",
            }
        }
        self.strategy = UvxyRsiStrategy(self.strategy_config)

    def test_strategy_initialization(self):
        """Test strategy initializes correctly"""
        self.assertIsInstance(self.strategy, UvxyRsiStrategy)
        self.assertEqual(self.strategy.strategy_config["strategy_params"]["rsi_period"], 2)
        self.assertEqual(self.strategy.strategy_config["strategy_params"]["rsi_threshold"], 30.0)
        self.assertFalse(self.strategy.strategy_config["strategy_params"]["long_only"])

    def test_tunable_parameters(self):
        """Test tunable parameters are correctly defined"""
        params = UvxyRsiStrategy.tunable_parameters()
        expected_params = {"rsi_period", "rsi_threshold"}
        self.assertEqual(params, expected_params)

    def test_non_universe_data_requirements(self):
        """Test non-universe data requirements"""
        requirements = self.strategy.get_non_universe_data_requirements()
        self.assertEqual(requirements, ["SPY"])

    def test_rsi_calculation(self):
        """Test RSI calculation function"""
        # Create test price series
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], index=dates)
        
        # Calculate RSI with period 2
        rsi = UvxyRsiStrategy._calculate_rsi(prices, 2)
        
        # RSI should be calculated for the last few values
        self.assertFalse(rsi.empty)
        self.assertTrue(all(rsi.dropna() >= 0))
        self.assertTrue(all(rsi.dropna() <= 100))

    def test_rsi_calculation_insufficient_data(self):
        """Test RSI calculation with insufficient data"""
        # Create test price series with only 2 points (need at least period+1)
        dates = pd.date_range('2023-01-01', periods=2, freq='D')
        prices = pd.Series([100, 102], index=dates)
        
        # Calculate RSI with period 2 (should return empty series)
        rsi = UvxyRsiStrategy._calculate_rsi(prices, 2)
        
        # Should return series with NaN values
        self.assertTrue(rsi.isna().all())

    def test_extract_spy_prices_multiindex(self):
        """Test SPY price extraction from MultiIndex DataFrame"""
        # Create test data with MultiIndex columns
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        columns = pd.MultiIndex.from_product([['SPY'], ['Open', 'High', 'Low', 'Close']], 
                                           names=['Ticker', 'Field'])
        data = np.random.randn(5, 4) * 0.01 + 100  # Small random changes around 100
        df = pd.DataFrame(data, index=dates, columns=columns)
        
        current_date = dates[-1]
        spy_prices = self.strategy._extract_spy_prices(df, current_date)
        
        self.assertIsNotNone(spy_prices)
        self.assertEqual(len(spy_prices), 5)
        self.assertTrue(all(spy_prices > 0))

    def test_extract_spy_prices_single_index(self):
        """Test SPY price extraction from single index DataFrame"""
        # Create test data with single index columns
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = np.random.randn(5, 1) * 0.01 + 100
        df = pd.DataFrame(data, index=dates, columns=['SPY'])
        
        current_date = dates[-1]
        spy_prices = self.strategy._extract_spy_prices(df, current_date)
        
        self.assertIsNotNone(spy_prices)
        self.assertEqual(len(spy_prices), 5)
        self.assertTrue(all(spy_prices > 0))

    def test_generate_signals_short_entry(self):
        """Test signal generation for short entry"""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        current_date = dates[-1]
        
        # UVXY universe data (MultiIndex)
        uvxy_columns = pd.MultiIndex.from_product([['UVXY'], ['Open', 'High', 'Low', 'Close']], 
                                                names=['Ticker', 'Field'])
        uvxy_data = np.random.randn(10, 4) * 0.02 + 50
        all_historical_data = pd.DataFrame(uvxy_data, index=dates, columns=uvxy_columns)
        
        # SPY data for signal generation - create declining prices for low RSI
        spy_columns = pd.MultiIndex.from_product([['SPY'], ['Open', 'High', 'Low', 'Close']], 
                                                names=['Ticker', 'Field'])
        # Create declining prices to generate low RSI
        spy_prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]  # Declining trend
        spy_data = np.column_stack([spy_prices, spy_prices, spy_prices, spy_prices])
        non_universe_data = pd.DataFrame(spy_data, index=dates, columns=spy_columns)
        
        # Empty benchmark data
        benchmark_data = pd.DataFrame(index=dates)
        
        # Generate signals
        signals = self.strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_data,
            current_date=current_date,
            non_universe_historical_data=non_universe_data
        )
        
        # Should generate short signal (negative weight)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), 1)
        self.assertIn('UVXY', signals.columns)
        # Should be short (negative) or zero
        self.assertTrue(signals.loc[current_date, 'UVXY'] <= 0)

    def test_generate_signals_no_spy_data(self):
        """Test signal generation with no SPY data"""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        current_date = dates[-1]
        
        # UVXY universe data
        uvxy_columns = pd.MultiIndex.from_product([['UVXY'], ['Close']], 
                                                names=['Ticker', 'Field'])
        uvxy_data = np.random.randn(5, 1) * 0.02 + 50
        all_historical_data = pd.DataFrame(uvxy_data, index=dates, columns=uvxy_columns)
        
        # Empty benchmark and non-universe data
        benchmark_data = pd.DataFrame(index=dates)
        non_universe_data = pd.DataFrame(index=dates)
        
        # Generate signals
        signals = self.strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_data,
            current_date=current_date,
            non_universe_historical_data=non_universe_data
        )
        
        # Should generate empty signals (no position)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(len(signals), 1)
        self.assertIn('UVXY', signals.columns)
        self.assertEqual(signals.loc[current_date, 'UVXY'], 0.0)

    def test_create_empty_signals_range(self):
        """Test empty signals range creation"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        start_date = dates[0]
        end_date = dates[-1]
        
        universe_tickers = ['UVXY']
        
        empty_signals = self.strategy._create_empty_signals_range(universe_tickers, start_date, end_date)
        
        self.assertIsInstance(empty_signals, pd.DataFrame)
        self.assertEqual(len(empty_signals), 5)  # 5 days
        self.assertIn('UVXY', empty_signals.columns)
        self.assertTrue(all(empty_signals['UVXY'] == 0.0))


if __name__ == '__main__':
    unittest.main()