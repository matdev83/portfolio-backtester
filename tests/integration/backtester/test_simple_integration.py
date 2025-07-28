"""
Simplified integration tests that avoid complex mocking issues.
Focus on testing actual integration without problematic mock dependencies.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming


@pytest.mark.integration
class TestSimpleIntegration(unittest.TestCase):
    """Simple integration tests without complex mocking."""
    
    def setUp(self):
        """Set up simple test data."""
        # Create simple test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'MSFT']
        
        # Create MultiIndex DataFrame
        np.random.seed(42)
        data_dict = {}
        for ticker in tickers:
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
            data_dict[ticker] = {
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, len(dates))
            }
        
        # Convert to MultiIndex format
        columns = pd.MultiIndex.from_product([tickers, ['Close', 'Volume']], names=['Ticker', 'Field'])
        data_array = np.column_stack([
            data_dict[ticker][field] for ticker in tickers for field in ['Close', 'Volume']
        ])
        
        self.test_data = pd.DataFrame(data_array, index=dates, columns=columns)
        self.benchmark_data = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + np.random.normal(0.0008, 0.015, len(dates)))
        }, index=dates)
    
    def test_strategy_timing_integration(self):
        """Test strategy and timing integration."""
        # Test strategy initialization
        config = {
            'strategy_params': {
                'lookback_months': 1,
                'num_holdings': 1,
                'smoothing_lambda': 0.0
            }
        }
        
        strategy = MomentumStrategy(config)
        self.assertIsInstance(strategy, MomentumStrategy)
        
        # Test timing controller
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)
    
    def test_strategy_signal_generation(self):
        """Test strategy signal generation with real data."""
        config = {
            'strategy_params': {
                'lookback_months': 1,
                'num_holdings': 1,
                'smoothing_lambda': 0.0,
                'price_column_asset': 'Close',
                'price_column_benchmark': 'Close'
            }
        }
        
        strategy = MomentumStrategy(config)
        
        # Test signal generation
        current_date = self.test_data.index[-1]
        signals = strategy.generate_signals(
            self.test_data,
            self.benchmark_data,
            current_date=current_date
        )
        
        # Verify signals are generated
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertFalse(signals.empty)
    
    def test_data_structure_compatibility(self):
        """Test that data structures are compatible across components."""
        # Test MultiIndex structure
        self.assertIsInstance(self.test_data.columns, pd.MultiIndex)
        self.assertEqual(self.test_data.columns.names, ['Ticker', 'Field'])
        
        # Test data access patterns
        aapl_close = self.test_data[('AAPL', 'Close')]
        self.assertIsInstance(aapl_close, pd.Series)
        self.assertEqual(len(aapl_close), len(self.test_data))
    
    def test_component_initialization(self):
        """Test that components can be initialized without errors."""
        # Test strategy initialization with various configs
        configs = [
            {'strategy_params': {'lookback_months': 1}},
            {'strategy_params': {'lookback_months': 3, 'num_holdings': 5}},
            {'strategy_params': {'lookback_months': 6, 'smoothing_lambda': 0.5}}
        ]
        
        for config in configs:
            try:
                strategy = MomentumStrategy(config)
                self.assertIsNotNone(strategy)
            except Exception as e:
                self.fail(f"Strategy initialization failed with config {config}: {e}")


@pytest.mark.integration
@pytest.mark.fast
class TestQuickIntegration:
    """Quick integration tests for fast feedback."""
    
    def test_imports_work(self):
        """Test that all imports work correctly."""
        from src.portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
        from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
        from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
        
        # If we get here, imports work
        assert True
    
    def test_basic_strategy_creation(self):
        """Test basic strategy creation."""
        config = {'strategy_params': {'lookback_months': 1}}
        strategy = MomentumStrategy(config)
        assert strategy is not None
        assert hasattr(strategy, 'generate_signals')


if __name__ == '__main__':
    unittest.main()