"""
Base test classes for strategy testing patterns.

This module provides base test classes that eliminate code duplication
in strategy tests and provide common testing patterns.
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type
from unittest.mock import Mock, patch

from tests.fixtures.market_data import MarketDataFixture
from tests.fixtures.strategy_data import StrategyDataFixture


class BaseStrategyTest(unittest.TestCase):
    """
    Base test class for strategy testing with common patterns.
    
    Provides common setup/teardown, data generation methods, and standard
    assertion helpers that can be reused across strategy tests.
    """
    
    def setUp(self):
        """Standard test setup with common test data and configuration."""
        # Set random seed for reproducible tests
        np.random.seed(42)
        
        # Default test configuration
        self.default_assets = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        self.default_start_date = "2020-01-01"
        self.default_end_date = "2023-12-31"
        self.default_benchmark = "SPY"
        
        # Generate basic market data for tests
        self.market_data = MarketDataFixture.create_basic_data(
            tickers=tuple(self.default_assets),
            start_date=self.default_start_date,
            end_date=self.default_end_date
        )
        
        # Generate benchmark data
        self.benchmark_data = MarketDataFixture.create_basic_data(
            tickers=(self.default_benchmark,),
            start_date=self.default_start_date,
            end_date=self.default_end_date
        )
        
        # Default strategy configuration
        self.default_strategy_config = {
            "strategy_params": {
                "leverage": 1.0,
                "long_only": True,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close"
            }
        }
        
    def tearDown(self):
        """Standard test teardown."""
        # Clean up any temporary data or mocks
        pass
    
    def generate_test_data(
        self,
        tickers: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        pattern: str = "basic",
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate test market data with specified parameters.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data generation
            end_date: End date for data generation
            pattern: Data pattern ('basic', 'trending', 'volatile', 'stable')
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with MultiIndex columns (Ticker, Field) and datetime index
        """
        tickers = tickers or self.default_assets
        start_date = start_date or self.default_start_date
        end_date = end_date or self.default_end_date
        
        if pattern == "basic":
            return MarketDataFixture.create_basic_data(
                tickers=tuple(tickers),
                start_date=start_date,
                end_date=end_date,
                seed=seed
            )
        elif pattern == "trending":
            return MarketDataFixture.create_trending_data(
                tickers=tuple(tickers),
                start_date=start_date,
                end_date=end_date,
                seed=seed
            )
        elif pattern == "volatile":
            return MarketDataFixture.create_volatile_data(
                tickers=tuple(tickers),
                start_date=start_date,
                end_date=end_date,
                seed=seed
            )
        elif pattern == "stable":
            return MarketDataFixture.create_stable_data(
                tickers=tuple(tickers),
                start_date=start_date,
                end_date=end_date,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    def assert_valid_signals(self, signals: pd.DataFrame, expected_assets: List[str] = None):
        """
        Assert that generated signals are valid.
        
        Args:
            signals: DataFrame with signal data
            expected_assets: List of expected asset symbols
        """
        expected_assets = expected_assets or self.default_assets
        
        # Check that signals is a DataFrame
        self.assertIsInstance(signals, pd.DataFrame, "Signals should be a DataFrame")
        
        # Check that signals is not empty
        self.assertFalse(signals.empty, "Signals should not be empty")
        
        # Check that all expected assets are present
        if not signals.empty:
            signal_columns = signals.columns.tolist()
            for asset in expected_assets:
                self.assertIn(asset, signal_columns, f"Asset {asset} should be in signals")
        
        # Check that signal values are numeric
        for col in signals.columns:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(signals[col]),
                f"Signal values for {col} should be numeric"
            )
        
        # Check for NaN values (should be minimal)
        nan_count = signals.isna().sum().sum()
        total_values = signals.size
        nan_ratio = nan_count / total_values if total_values > 0 else 0
        self.assertLess(nan_ratio, 0.1, "Too many NaN values in signals")
    
    def assert_strategy_initialization(self, strategy, config: Dict[str, Any]):
        """
        Assert that strategy is properly initialized.
        
        Args:
            strategy: Strategy instance to validate
            config: Configuration dictionary used for initialization
        """
        # Check that strategy is not None
        self.assertIsNotNone(strategy, "Strategy should not be None")
        
        # Check that strategy has required methods
        required_methods = ['generate_signals', 'get_non_universe_data_requirements']
        for method in required_methods:
            self.assertTrue(
                hasattr(strategy, method),
                f"Strategy should have {method} method"
            )
            self.assertTrue(
                callable(getattr(strategy, method)),
                f"Strategy {method} should be callable"
            )
        
        # Check that strategy configuration is stored
        self.assertTrue(
            hasattr(strategy, 'strategy_config'),
            "Strategy should have strategy_config attribute"
        )
        
        # Validate configuration structure
        if 'strategy_params' in config:
            self.assertIn('strategy_params', strategy.strategy_config)
    
    def assert_signal_properties(
        self,
        signals: pd.DataFrame,
        long_only: bool = True,
        leverage: float = 1.0,
        tolerance: float = 1e-6
    ):
        """
        Assert that signals have expected properties.
        
        Args:
            signals: DataFrame with signal data
            long_only: Whether strategy should be long-only
            leverage: Expected leverage level
            tolerance: Numerical tolerance for comparisons
        """
        if signals.empty:
            return
        
        # Check leverage constraint
        total_leverage = signals.abs().sum(axis=1)
        max_leverage = total_leverage.max()
        self.assertLessEqual(
            max_leverage,
            leverage + tolerance,
            f"Total leverage {max_leverage} exceeds limit {leverage}"
        )
        
        # Check long-only constraint
        if long_only:
            min_signal = signals.min().min()
            self.assertGreaterEqual(
                min_signal,
                -tolerance,
                f"Found negative signal {min_signal} in long-only strategy"
            )
    
    def test_generate_signals_smoke(self):
        """
        Standard smoke test for signal generation.
        
        This test should be overridden in concrete strategy test classes
        with strategy-specific implementation.
        """
        self.skipTest("Smoke test should be implemented in concrete strategy classes")
    
    


class BaseMomentumStrategyTest(BaseStrategyTest):
    """
    Base test class for momentum strategy testing.
    
    Extends BaseStrategyTest with momentum-specific test patterns
    and calculations.
    """
    
    def setUp(self):
        """Setup with momentum-specific test data and configuration."""
        super().setUp()
        
        # Momentum-specific configuration
        self.momentum_config = StrategyDataFixture.momentum_config()
        
        # Generate trending data suitable for momentum testing
        self.trending_data = MarketDataFixture.create_trending_data(
            tickers=tuple(self.default_assets),
            start_date=self.default_start_date,
            end_date=self.default_end_date
        )
    
    def calculate_momentum_score(
        self,
        price_data: pd.DataFrame,
        lookback_months: int = 3,
        skip_months: int = 1
    ) -> pd.Series:
        """
        Calculate momentum scores for testing validation.
        
        Args:
            price_data: DataFrame with price data
            lookback_months: Number of months to look back
            skip_months: Number of months to skip
            
        Returns:
            Series with momentum scores
        """
        if price_data.empty:
            return pd.Series(dtype=float)
        
        # Calculate returns over the lookback period
        end_date = price_data.index[-1]
        skip_date = end_date - pd.DateOffset(months=skip_months)
        start_date = skip_date - pd.DateOffset(months=lookback_months)
        
        # Get prices at start and end of momentum period
        try:
            start_prices = price_data.loc[price_data.index <= start_date].iloc[-1]
            end_prices = price_data.loc[price_data.index <= skip_date].iloc[-1]
            
            # Calculate momentum as total return
            momentum = (end_prices / start_prices) - 1
            return momentum.fillna(0)
        except (IndexError, KeyError):
            # Return zeros if insufficient data
            return pd.Series(0.0, index=price_data.columns)
    
    def assert_momentum_ranking(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        lookback_months: int = 3,
        skip_months: int = 1,
        top_fraction: float = 0.5
    ):
        """
        Assert that momentum signals follow expected ranking patterns.
        
        Args:
            signals: DataFrame with signal data
            price_data: DataFrame with price data
            lookback_months: Momentum lookback period
            skip_months: Momentum skip period
            top_fraction: Fraction of top performers to select
        """
        if signals.empty or price_data.empty:
            return
        
        # Calculate expected momentum scores
        momentum_scores = self.calculate_momentum_score(
            price_data, lookback_months, skip_months
        )
        
        # Get the latest signals
        latest_signals = signals.iloc[-1]
        
        # Check that assets with positive signals have higher momentum
        positive_assets = latest_signals[latest_signals > 0].index
        zero_assets = latest_signals[latest_signals == 0].index
        
        if len(positive_assets) > 0 and len(zero_assets) > 0:
            avg_momentum_positive = momentum_scores[positive_assets].mean()
            avg_momentum_zero = momentum_scores[zero_assets].mean()
            
            self.assertGreaterEqual(
                avg_momentum_positive,
                avg_momentum_zero,
                "Assets with positive signals should have higher momentum on average"
            )
    
    def assert_momentum_concentration(
        self,
        signals: pd.DataFrame,
        expected_concentration: float = 0.5,
        tolerance: float = 0.1
    ):
        """
        Assert that momentum signals have expected concentration.
        
        Args:
            signals: DataFrame with signal data
            expected_concentration: Expected fraction of assets with positive signals
            tolerance: Tolerance for concentration check
        """
        if signals.empty:
            return
        
        # Calculate actual concentration
        latest_signals = signals.iloc[-1]
        positive_count = (latest_signals > 0).sum()
        total_count = len(latest_signals)
        actual_concentration = positive_count / total_count
        
        self.assertAlmostEqual(
            actual_concentration,
            expected_concentration,
            delta=tolerance,
            msg=f"Signal concentration {actual_concentration} differs from expected {expected_concentration}"
        )
    
    def test_momentum_signal_generation(self):
        """
        Test momentum signal generation with trending data.
        
        This test should be overridden in concrete momentum strategy classes
        with strategy-specific implementation.
        """
        self.skipTest("Momentum signal test should be implemented in concrete strategy classes")