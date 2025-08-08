"""
Base test classes for timing framework testing patterns.

This module provides base test classes that eliminate code duplication
in timing tests and provide common timing testing patterns.
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock

from tests.fixtures.timing_data import TimingDataFixture
from tests.fixtures.market_data import MarketDataFixture


class BaseTimingTest(unittest.TestCase):
    """
    Base test class for timing framework testing.
    
    Provides common timing setup patterns, standard assertions,
    and migration test patterns for timing framework tests.
    """
    
    def setUp(self):
        """Standard timing test setup."""
        # Set random seed for reproducible tests
        np.random.seed(42)
        
        # Default timing test configuration
        self.default_start_date = "2020-01-01"
        self.default_end_date = "2023-12-31"
        self.default_frequency = "M"
        self.default_assets = ["AAPL", "MSFT", "GOOGL"]
        
        # Generate test market data
        self.market_data = MarketDataFixture.create_basic_data(
            tickers=tuple(self.default_assets),
            start_date=self.default_start_date,
            end_date=self.default_end_date
        )
        
        # Create timing configurations
        self.time_based_config = TimingDataFixture.time_based_config(
            frequency=self.default_frequency,
            start_date=self.default_start_date,
            end_date=self.default_end_date
        )
        
        self.signal_based_config = TimingDataFixture.signal_based_config(
            signal_column="Close",
            threshold=0.0,
            start_date=self.default_start_date,
            end_date=self.default_end_date
        )
        
        # Mock strategy for timing tests
        self.mock_strategy = self.create_mock_strategy()
    
    def tearDown(self):
        """Standard timing test teardown."""
        pass
    
    def create_mock_strategy(self):
        """
        Create a mock strategy for timing tests.
        
        Returns:
            Mock strategy object with required methods
        """
        mock_strategy = Mock()
        
        # Mock generate_signals method
        def mock_generate_signals(*args, **kwargs):
            # Return equal weight signals for all assets
            weights = pd.Series(
                1.0 / len(self.default_assets),
                index=self.default_assets,
                name=kwargs.get('current_date', pd.Timestamp.now())
            )
            return pd.DataFrame([weights])
        
        mock_strategy.generate_signals = Mock(side_effect=mock_generate_signals)
        mock_strategy.get_non_universe_data_requirements = Mock(return_value=[])
        mock_strategy.strategy_config = {"strategy_params": {"leverage": 1.0}}
        
        return mock_strategy
    
    def create_timing_controller(self, timing_type: str = "time_based", **kwargs):
        """
        Create timing controller for testing.
        
        Args:
            timing_type: Type of timing controller ('time_based' or 'signal_based')
            **kwargs: Additional configuration parameters
            
        Returns:
            Timing controller instance
        """
        # Use factory pattern with interface integration
        from portfolio_backtester.timing.custom_timing_registry import TimingControllerFactory
        
        if timing_type == "time_based":
            config = self.time_based_config.copy()
            config.update(kwargs)
            config["mode"] = "time_based"
            return TimingControllerFactory.create_controller(config)
        elif timing_type == "signal_based":
            config = self.signal_based_config.copy()
            config.update(kwargs)
            config["mode"] = "signal_based"
            return TimingControllerFactory.create_controller(config)
        else:
            raise ValueError(f"Unknown timing type: {timing_type}")
    
    def assert_valid_rebalance_dates(
        self,
        rebalance_dates: List[pd.Timestamp],
        expected_frequency: str = None,
        start_date: str = None,
        end_date: str = None
    ):
        """
        Assert that rebalance dates are valid and follow expected patterns.
        
        Args:
            rebalance_dates: List of rebalance dates to validate
            expected_frequency: Expected rebalancing frequency
            start_date: Expected start date
            end_date: Expected end date
        """
        expected_frequency = expected_frequency or self.default_frequency
        start_date = pd.Timestamp(start_date or self.default_start_date)
        end_date = pd.Timestamp(end_date or self.default_end_date)
        
        # Check that dates are not empty
        self.assertGreater(len(rebalance_dates), 0, "Rebalance dates should not be empty")
        
        # Check that dates are sorted
        sorted_dates = sorted(rebalance_dates)
        self.assertEqual(
            rebalance_dates,
            sorted_dates,
            "Rebalance dates should be sorted"
        )
        
        # Check that dates are within expected range
        first_date = rebalance_dates[0]
        last_date = rebalance_dates[-1]
        
        self.assertGreaterEqual(
            first_date,
            start_date,
            f"First rebalance date {first_date} should be >= start date {start_date}"
        )
        
        self.assertLessEqual(
            last_date,
            end_date,
            f"Last rebalance date {last_date} should be <= end date {end_date}"
        )
        
        # Check frequency pattern for monthly rebalancing
        if expected_frequency == "M" and len(rebalance_dates) > 1:
            # Check that dates are roughly monthly apart
            intervals = [
                (rebalance_dates[i+1] - rebalance_dates[i]).days
                for i in range(len(rebalance_dates) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            
            # Monthly intervals should be between 28-31 days
            self.assertGreater(avg_interval, 25, "Monthly intervals too short")
            self.assertLess(avg_interval, 35, "Monthly intervals too long")
    
    def assert_timing_controller_initialization(self, timing_controller, expected_config: Dict[str, Any]):
        """
        Assert that timing controller is properly initialized.
        
        Args:
            timing_controller: Timing controller instance to validate
            expected_config: Expected configuration dictionary
        """
        # Check that controller is not None
        self.assertIsNotNone(timing_controller, "Timing controller should not be None")
        
        # Check that controller has required methods
        required_methods = ['get_rebalance_dates', 'should_rebalance']
        for method in required_methods:
            self.assertTrue(
                hasattr(timing_controller, method),
                f"Timing controller should have {method} method"
            )
            self.assertTrue(
                callable(getattr(timing_controller, method)),
                f"Timing controller {method} should be callable"
            )
        
        # Check configuration storage
        if hasattr(timing_controller, 'config'):
            for key, value in expected_config.items():
                if key in timing_controller.config:
                    self.assertEqual(
                        timing_controller.config[key],
                        value,
                        f"Configuration {key} should match expected value"
                    )
    
    def assert_migration_compatibility(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        migration_function: callable
    ):
        """
        Assert that migration function properly converts old config to new format.
        
        Args:
            old_config: Old configuration format
            new_config: Expected new configuration format
            migration_function: Function to perform migration
        """
        # Test migration function
        migrated_config = migration_function(old_config)
        
        # Check that migrated config matches expected format
        for key, expected_value in new_config.items():
            self.assertIn(key, migrated_config, f"Migrated config should contain {key}")
            self.assertEqual(
                migrated_config[key],
                expected_value,
                f"Migrated {key} should match expected value"
            )
        
        # Check that migration is idempotent
        double_migrated = migration_function(migrated_config)
        self.assertEqual(
            migrated_config,
            double_migrated,
            "Migration should be idempotent"
        )
    
    def create_migration_test_scenarios(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Create test scenarios for migration testing.
        
        Returns:
            List of (old_config, new_config) tuples for testing
        """
        scenarios = []
        
        # Monthly momentum strategy migration
        old_monthly = {
            "rebalance_frequency": "monthly",
            "strategy_type": "momentum",
            "lookback_months": 3
        }
        new_monthly = {
            "timing_type": "time_based",
            "rebalance_frequency": "M",
            "strategy_params": {
                "lookback_months": 3
            }
        }
        scenarios.append((old_monthly, new_monthly))
        
        # Daily UVXY strategy migration
        old_daily = {
            "rebalance_frequency": "daily",
            "strategy_type": "uvxy_rsi",
            "rsi_period": 2
        }
        new_daily = {
            "timing_type": "time_based",
            "rebalance_frequency": "D",
            "strategy_params": {
                "rsi_period": 2
            }
        }
        scenarios.append((old_daily, new_daily))
        
        return scenarios
    
    def test_timing_controller_initialization_smoke(self):
        """
        Standard smoke test for timing controller initialization.
        
        This test should be overridden in concrete timing test classes
        with timing-specific implementation.
        """
        self.skipTest("Timing initialization test should be implemented in concrete timing classes")
    
    
    
    def test_migration_compatibility_smoke(self):
        """
        Standard smoke test for migration compatibility.
        
        This test should be overridden in concrete timing test classes
        with migration-specific implementation.
        """
        self.skipTest("Migration test should be implemented in concrete timing classes")
    
    def run_parameterized_migration_test(self, migration_scenarios: List[Tuple[Dict, Dict]], migration_func: callable):
        """
        Run parameterized migration tests for multiple scenarios.
        
        Args:
            migration_scenarios: List of (old_config, new_config) tuples
            migration_func: Migration function to test
        """
        for i, (old_config, expected_new_config) in enumerate(migration_scenarios):
            with self.subTest(scenario=i):
                self.assert_migration_compatibility(
                    old_config,
                    expected_new_config,
                    migration_func
                )
    
    def assert_strategy_timing_integration(
        self,
        strategy,
        timing_controller,
        market_data: pd.DataFrame,
        expected_rebalance_count: int = None
    ):
        """
        Assert that strategy and timing controller integrate properly.
        
        Args:
            strategy: Strategy instance
            timing_controller: Timing controller instance
            market_data: Market data for testing
            expected_rebalance_count: Expected number of rebalances
        """
        # Get rebalance dates
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date=pd.Timestamp(self.default_start_date),
            end_date=pd.Timestamp(self.default_end_date)
        )
        
        # Validate rebalance dates
        self.assert_valid_rebalance_dates(rebalance_dates)
        
        # Check expected rebalance count if provided
        if expected_rebalance_count is not None:
            self.assertEqual(
                len(rebalance_dates),
                expected_rebalance_count,
                f"Expected {expected_rebalance_count} rebalances, got {len(rebalance_dates)}"
            )
        
        # Test signal generation at rebalance dates
        for rebalance_date in rebalance_dates[:3]:  # Test first 3 dates
            with self.subTest(date=rebalance_date):
                # Check if should rebalance
                should_rebalance = timing_controller.should_rebalance(
                    current_date=rebalance_date,
                    last_rebalance_date=None
                )
                self.assertTrue(
                    should_rebalance,
                    f"Should rebalance on {rebalance_date}"
                )
                
                # Generate signals
                signals = strategy.generate_signals(
                    all_historical_data=market_data,
                    benchmark_historical_data=market_data.iloc[:, :1],  # Use first asset as benchmark
                    current_date=rebalance_date
                )
                
                # Validate signals
                self.assertIsInstance(signals, pd.DataFrame, "Signals should be DataFrame")
                self.assertFalse(signals.empty, "Signals should not be empty")