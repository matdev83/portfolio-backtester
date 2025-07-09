"""
Tests for Asset Replacement Manager

This module contains comprehensive tests for the asset replacement functionality,
including random selection, synthetic data integration, and Monte Carlo dataset creation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import logging

from src.portfolio_backtester.monte_carlo.asset_replacement import (
    AssetReplacementManager,
    ReplacementInfo
)


class TestAssetReplacementManager:
    """Test suite for AssetReplacementManager class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'enable_synthetic_data': True,
            'replacement_percentage': 0.20,
            'min_historical_observations': 100,
            'garch_config': {
                'model_type': 'GARCH',
                'p': 1,
                'q': 1,
                'distribution': 'studentt',
                'bounds': {
                    'omega': [1e-6, 1.0],
                    'alpha': [0.01, 0.3],
                    'beta': [0.5, 0.99],
                    'nu': [2.1, 30.0]
                }
            },
            'generation_config': {
                'buffer_multiplier': 1.2,
                'max_attempts': 3,
                'validation_tolerance': 0.2
            },
            'validation_config': {
                'enable_validation': False  # Disable for faster testing
            },
            'random_seed': 42
        }
    
    @pytest.fixture
    def sample_universe(self):
        """Sample universe of assets."""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ORCL']
    
    @pytest.fixture
    def sample_asset_data(self, sample_universe):
        """Generate sample asset data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        asset_data = {}
        
        for i, asset in enumerate(sample_universe):
            # Generate different patterns for each asset
            np.random.seed(42 + i)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            asset_data[asset] = pd.DataFrame({
                'Open': prices * np.random.uniform(0.99, 1.01, len(prices)),
                'High': prices * np.random.uniform(1.0, 1.03, len(prices)),
                'Low': prices * np.random.uniform(0.97, 1.0, len(prices)),
                'Close': prices
            }, index=dates)
        
        return asset_data
    
    def test_initialization(self, sample_config):
        """Test manager initialization."""
        manager = AssetReplacementManager(sample_config)
        
        assert manager.config == sample_config
        assert manager.config.get('replacement_percentage') == 0.20
        assert manager.synthetic_generator is not None
        assert manager.replacement_history == []
        assert manager._asset_stats_cache == {}
    
    def test_select_assets_for_replacement(self, sample_config, sample_universe):
        """Test random asset selection."""
        manager = AssetReplacementManager(sample_config)
        
        # Test with fixed seed
        selected_assets = manager.select_assets_for_replacement(sample_universe, random_seed=42)
        
        # Should select 20% of assets (2 out of 10)
        expected_count = int(len(sample_universe) * 0.20)
        assert len(selected_assets) == expected_count
        assert all(asset in sample_universe for asset in selected_assets)
        
        # Should record replacement info
        assert len(manager.replacement_history) == 1
        assert manager.replacement_history[0].selected_assets == selected_assets
    
    def test_select_assets_reproducibility(self, sample_config, sample_universe):
        """Test that asset selection is reproducible with same seed."""
        manager1 = AssetReplacementManager(sample_config)
        manager2 = AssetReplacementManager(sample_config)
        
        selected1 = manager1.select_assets_for_replacement(sample_universe, random_seed=42)
        selected2 = manager2.select_assets_for_replacement(sample_universe, random_seed=42)
        
        assert selected1 == selected2
    
    def test_select_assets_different_seeds(self, sample_config, sample_universe):
        """Test that different seeds produce different selections."""
        manager = AssetReplacementManager(sample_config)
        
        selected1 = manager.select_assets_for_replacement(sample_universe, random_seed=42)
        selected2 = manager.select_assets_for_replacement(sample_universe, random_seed=123)
        
        # Should be different (with high probability)
        assert selected1 != selected2
    
    def test_select_assets_edge_cases(self, sample_config):
        """Test edge cases for asset selection."""
        manager = AssetReplacementManager(sample_config)
        
        # Test with single asset
        single_asset = ['AAPL']
        selected = manager.select_assets_for_replacement(single_asset, random_seed=42)
        assert len(selected) == 1
        assert 'AAPL' in selected
        
        # Test with small universe
        small_universe = ['AAPL', 'MSFT']
        selected = manager.select_assets_for_replacement(small_universe, random_seed=42)
        assert len(selected) == 1  # max(1, int(2 * 0.20)) = 1
    
    def test_replace_asset_data_basic(self, sample_config, sample_asset_data):
        """Test basic asset data replacement."""
        manager = AssetReplacementManager(sample_config)
        
        # Select assets to replace
        assets_to_replace = {'AAPL', 'MSFT'}
        
        # Define test period
        start_date = pd.Timestamp('2020-06-01')
        end_date = pd.Timestamp('2020-06-30')
        
        # Replace data
        modified_data = manager.replace_asset_data(
            original_data=sample_asset_data,
            assets_to_replace=assets_to_replace,
            start_date=start_date,
            end_date=end_date,
            phase="test"
        )
        
        # Check that data was modified
        assert len(modified_data) == len(sample_asset_data)
        assert all(asset in modified_data for asset in sample_asset_data)
        
        # Check that replaced assets have different data in test period
        for asset in assets_to_replace:
            original_period = sample_asset_data[asset].loc[start_date:end_date]
            modified_period = modified_data[asset].loc[start_date:end_date]
            
            # Should have same structure
            assert original_period.index.equals(modified_period.index)
            assert original_period.columns.equals(modified_period.columns)
            
            # Data should be different (synthetic)
            # Note: This might occasionally fail due to randomness, but very unlikely
            assert not original_period['Close'].equals(modified_period['Close'])
    
    def test_replace_asset_data_train_phase_warning(self, sample_config, sample_asset_data):
        """Test that replacement in train phase produces warning."""
        manager = AssetReplacementManager(sample_config)
        
        with patch('src.portfolio_backtester.monte_carlo.asset_replacement.logger') as mock_logger:
            result = manager.replace_asset_data(
                original_data=sample_asset_data,
                assets_to_replace={'AAPL'},
                start_date=pd.Timestamp('2020-06-01'),
                end_date=pd.Timestamp('2020-06-30'),
                phase="train"
            )
            
            # Should return original data unchanged
            assert result == sample_asset_data
            
            # Should log warning
            mock_logger.warning.assert_called_once()
    
    def test_replace_asset_data_missing_asset(self, sample_config, sample_asset_data):
        """Test handling of missing assets."""
        manager = AssetReplacementManager(sample_config)
        
        with patch('src.portfolio_backtester.monte_carlo.asset_replacement.logger') as mock_logger:
            modified_data = manager.replace_asset_data(
                original_data=sample_asset_data,
                assets_to_replace={'NONEXISTENT'},
                start_date=pd.Timestamp('2020-06-01'),
                end_date=pd.Timestamp('2020-06-30'),
                phase="test"
            )
            
            # Should return original data unchanged
            assert modified_data is sample_asset_data or all(
                modified_data[key].equals(sample_asset_data[key]) 
                for key in sample_asset_data.keys()
            )
            
            # Should log warning
            mock_logger.warning.assert_called()
    
    def test_create_monte_carlo_dataset(self, sample_config, sample_asset_data, sample_universe):
        """Test Monte Carlo dataset creation."""
        manager = AssetReplacementManager(sample_config)
        
        test_start = pd.Timestamp('2020-06-01')
        test_end = pd.Timestamp('2020-06-30')
        
        modified_data, replacement_info = manager.create_monte_carlo_dataset(
            original_data=sample_asset_data,
            universe=sample_universe,
            test_start=test_start,
            test_end=test_end,
            run_id="test_run",
            random_seed=42
        )
        
        # Check replacement info
        assert isinstance(replacement_info, ReplacementInfo)
        assert replacement_info.replacement_percentage == 0.20
        assert replacement_info.total_assets == len(sample_universe)
        assert len(replacement_info.selected_assets) == int(len(sample_universe) * 0.20)
        
        # Check modified data
        assert len(modified_data) == len(sample_asset_data)
        assert all(asset in modified_data for asset in sample_asset_data)
    
    def test_create_monte_carlo_dataset_disabled(self, sample_config, sample_asset_data, sample_universe):
        """Test dataset creation when synthetic data is disabled."""
        # Disable synthetic data
        sample_config['enable_synthetic_data'] = False
        manager = AssetReplacementManager(sample_config)
        
        modified_data, replacement_info = manager.create_monte_carlo_dataset(
            original_data=sample_asset_data,
            universe=sample_universe,
            test_start=pd.Timestamp('2020-06-01'),
            test_end=pd.Timestamp('2020-06-30')
        )
        
        # Should return original data unchanged
        assert modified_data == sample_asset_data
        assert len(replacement_info.selected_assets) == 0
        assert replacement_info.replacement_percentage == 0.0
    
    def test_generate_synthetic_data_for_period(self, sample_config, sample_asset_data):
        """Test synthetic data generation for specific period."""
        manager = AssetReplacementManager(sample_config)
        
        # Get historical data for one asset
        asset_data = sample_asset_data['AAPL']
        historical_data = asset_data.iloc[:200]  # First 200 days
        target_dates = asset_data.index[200:250]  # Next 50 days
        
        synthetic_data = manager._generate_synthetic_data_for_period(
            historical_data=historical_data,
            target_length=len(target_dates),
            target_dates=target_dates,
            asset_name='AAPL'
        )
        
        # Check structure
        assert len(synthetic_data) == len(target_dates)
        assert synthetic_data.index.equals(target_dates)
        assert all(col in synthetic_data.columns for col in ['Open', 'High', 'Low', 'Close'])
        
        # Check OHLC relationships
        assert (synthetic_data['High'] >= synthetic_data['Close']).all()
        assert (synthetic_data['High'] >= synthetic_data['Open']).all()
        assert (synthetic_data['Low'] <= synthetic_data['Close']).all()
        assert (synthetic_data['Low'] <= synthetic_data['Open']).all()
    
    def test_returns_to_prices_conversion(self, sample_config):
        """Test returns to prices conversion."""
        manager = AssetReplacementManager(sample_config)
        
        returns = np.array([0.01, -0.02, 0.015, -0.005])
        initial_price = 100.0
        
        prices = manager._returns_to_prices(returns, initial_price)
        
        # Check cumulative compounding
        expected_prices = [
            100.0 * 1.01,
            100.0 * 1.01 * 0.98,
            100.0 * 1.01 * 0.98 * 1.015,
            100.0 * 1.01 * 0.98 * 1.015 * 0.995
        ]
        
        np.testing.assert_array_almost_equal(prices, expected_prices)
    
    def test_ohlc_generation_from_prices(self, sample_config):
        """Test OHLC generation from price series."""
        manager = AssetReplacementManager(sample_config)
        
        prices = np.array([100, 102, 98, 101, 105])
        ohlc = manager._generate_ohlc_from_prices(prices)
        
        assert ohlc.shape == (5, 4)  # 5 periods, 4 OHLC fields
        
        # Check OHLC relationships for each period
        for i in range(len(ohlc)):
            open_price, high, low, close = ohlc[i]
            
            # Close should match input price
            assert close == prices[i]
            
            # OHLC relationships
            assert high >= max(open_price, close)
            assert low <= min(open_price, close)
            assert high >= low
            
            # All prices should be positive
            assert all(p > 0 for p in [open_price, high, low, close])
    
    def test_replacement_statistics(self, sample_config, sample_asset_data, sample_universe):
        """Test replacement statistics tracking."""
        manager = AssetReplacementManager(sample_config)
        
        # Create multiple datasets
        for i in range(3):
            manager.create_monte_carlo_dataset(
                original_data=sample_asset_data,
                universe=sample_universe,
                test_start=pd.Timestamp('2020-06-01'),
                test_end=pd.Timestamp('2020-06-30'),
                run_id=f"run_{i}",
                random_seed=42 + i
            )
        
        # Get statistics
        stats = manager.get_replacement_statistics()
        
        assert stats['total_runs'] == 3
        assert abs(stats['avg_replacement_percentage'] - 0.20) < 1e-10
        assert stats['total_assets_replaced'] > 0
        assert stats['avg_assets_per_run'] > 0
        assert 'asset_replacement_counts' in stats
        assert 'most_replaced_assets' in stats
    
    def test_cache_functionality(self, sample_config, sample_asset_data):
        """Test asset statistics caching."""
        manager = AssetReplacementManager(sample_config)
        
        # Generate data twice for same asset
        asset_data = sample_asset_data['AAPL']
        historical_data = asset_data.iloc[:200]
        target_dates = asset_data.index[200:250]
        
        # First generation
        synthetic_data1 = manager._generate_synthetic_data_for_period(
            historical_data=historical_data,
            target_length=len(target_dates),
            target_dates=target_dates,
            asset_name='AAPL'
        )
        
        # Second generation (should use cache)
        synthetic_data2 = manager._generate_synthetic_data_for_period(
            historical_data=historical_data,
            target_length=len(target_dates),
            target_dates=target_dates,
            asset_name='AAPL'
        )
        
        # Both should succeed
        assert len(synthetic_data1) == len(target_dates)
        assert len(synthetic_data2) == len(target_dates)
        
        # Clear cache
        manager.clear_cache()
        assert len(manager._asset_stats_cache) == 0
    
    def test_history_reset(self, sample_config, sample_asset_data, sample_universe):
        """Test replacement history reset."""
        manager = AssetReplacementManager(sample_config)
        
        # Create some replacement history
        manager.create_monte_carlo_dataset(
            original_data=sample_asset_data,
            universe=sample_universe,
            test_start=pd.Timestamp('2020-06-01'),
            test_end=pd.Timestamp('2020-06-30')
        )
        
        assert len(manager.replacement_history) > 0
        
        # Reset history
        manager.reset_history()
        assert len(manager.replacement_history) == 0


class TestReplacementInfo:
    """Test suite for ReplacementInfo dataclass."""
    
    def test_replacement_info_creation(self):
        """Test replacement info creation."""
        info = ReplacementInfo(
            selected_assets={'AAPL', 'MSFT'},
            replacement_percentage=0.20,
            random_seed=42,
            total_assets=10
        )
        
        assert info.selected_assets == {'AAPL', 'MSFT'}
        assert info.replacement_percentage == 0.20
        assert info.random_seed == 42
        assert info.total_assets == 10


# Integration tests
class TestAssetReplacementIntegration:
    """Integration tests for asset replacement with synthetic data generation."""
    
    @pytest.fixture
    def realistic_config(self):
        """Realistic configuration for integration testing."""
        return {
            'enable_synthetic_data': True,
            'replacement_percentage': 0.15,
            'min_historical_observations': 100,
            'garch_config': {
                'model_type': 'GARCH',
                'p': 1,
                'q': 1,
                'distribution': 'studentt',
                'bounds': {
                    'omega': [1e-6, 1.0],
                    'alpha': [0.01, 0.3],
                    'beta': [0.5, 0.99],
                    'nu': [2.1, 30.0]
                }
            },
            'generation_config': {
                'buffer_multiplier': 1.2,
                'max_attempts': 3,
                'validation_tolerance': 0.2
            },
            'validation_config': {
                'enable_validation': True,
                'ks_test_pvalue_threshold': 0.01,
                'autocorr_max_deviation': 0.2,
                'volatility_clustering_threshold': 0.03
            },
            'random_seed': None  # Test without fixed seed
        }
    
    @pytest.fixture
    def realistic_asset_data(self):
        """Generate realistic asset data with volatility clustering."""
        np.random.seed(42)
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        asset_data = {}
        
        for asset in assets:
            # Generate GARCH-like returns
            returns = []
            volatility = 0.02
            
            for i in range(len(dates)):
                if i > 0:
                    # Volatility clustering
                    volatility = 0.0001 + 0.05 * (returns[i-1]**2) + 0.9 * volatility
                
                daily_return = np.random.normal(0, np.sqrt(volatility))
                returns.append(daily_return)
            
            # Convert to prices
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create OHLC
            asset_data[asset] = pd.DataFrame({
                'Open': prices * np.random.uniform(0.995, 1.005, len(prices)),
                'High': np.maximum(prices * np.random.uniform(0.995, 1.005, len(prices)), 
                                 prices) * np.random.uniform(1.0, 1.02, len(prices)),
                'Low': np.minimum(prices * np.random.uniform(0.995, 1.005, len(prices)), 
                                prices) * np.random.uniform(0.98, 1.0, len(prices)),
                'Close': prices
            }, index=dates)
        
        return asset_data
    
    def test_end_to_end_monte_carlo_dataset(self, realistic_config, realistic_asset_data):
        """Test end-to-end Monte Carlo dataset creation."""
        manager = AssetReplacementManager(realistic_config)
        
        universe = list(realistic_asset_data.keys())
        test_start = pd.Timestamp('2020-09-01')
        test_end = pd.Timestamp('2020-09-30')
        
        # Create Monte Carlo dataset
        modified_data, replacement_info = manager.create_monte_carlo_dataset(
            original_data=realistic_asset_data,
            universe=universe,
            test_start=test_start,
            test_end=test_end,
            run_id="integration_test"
        )
        
        # Validate results
        assert len(modified_data) == len(realistic_asset_data)
        assert isinstance(replacement_info, ReplacementInfo)
        assert len(replacement_info.selected_assets) > 0
        
        # Check that synthetic data was actually generated
        for asset in replacement_info.selected_assets:
            original_period = realistic_asset_data[asset].loc[test_start:test_end]
            modified_period = modified_data[asset].loc[test_start:test_end]
            
            # Should have same structure
            assert original_period.index.equals(modified_period.index)
            assert original_period.columns.equals(modified_period.columns)
            
            # Check OHLC relationships in synthetic data
            assert (modified_period['High'] >= modified_period['Close']).all()
            assert (modified_period['High'] >= modified_period['Open']).all()
            assert (modified_period['Low'] <= modified_period['Close']).all()
            assert (modified_period['Low'] <= modified_period['Open']).all()
            
            # All prices should be positive
            assert (modified_period > 0).all().all()
    
    def test_multiple_runs_different_selections(self, realistic_config, realistic_asset_data):
        """Test that multiple runs produce different asset selections."""
        manager = AssetReplacementManager(realistic_config)
        
        universe = list(realistic_asset_data.keys())
        test_start = pd.Timestamp('2020-09-01')
        test_end = pd.Timestamp('2020-09-30')
        
        # Create multiple datasets
        selections = []
        for i in range(5):
            _, replacement_info = manager.create_monte_carlo_dataset(
                original_data=realistic_asset_data,
                universe=universe,
                test_start=test_start,
                test_end=test_end,
                run_id=f"run_{i}"
            )
            selections.append(replacement_info.selected_assets)
        
        # Should have some variation in selections
        unique_selections = [frozenset(s) for s in selections]
        assert len(set(unique_selections)) > 1  # At least some different selections
    
    def test_statistical_quality_of_synthetic_data(self, realistic_config, realistic_asset_data):
        """Test statistical quality of generated synthetic data."""
        manager = AssetReplacementManager(realistic_config)
        
        # Select one asset for detailed analysis
        asset_name = 'AAPL'
        asset_data = realistic_asset_data[asset_name]
        
        # Generate synthetic data
        historical_data = asset_data.iloc[:400]  # First 400 days
        target_dates = asset_data.index[400:450]  # Next 50 days
        
        synthetic_data = manager._generate_synthetic_data_for_period(
            historical_data=historical_data,
            target_length=len(target_dates),
            target_dates=target_dates,
            asset_name=asset_name
        )
        
        # Compare statistical properties
        original_returns = historical_data['Close'].pct_change(fill_method=None).dropna()
        synthetic_returns = synthetic_data['Close'].pct_change(fill_method=None).dropna()
        
        # Basic statistical tests
        assert len(synthetic_returns) > 0
        assert synthetic_returns.std() > 0
        
        # Volatility should be in reasonable range
        vol_ratio = synthetic_returns.std() / original_returns.std()
        assert 0.3 < vol_ratio < 3.0  # Within reasonable range
        
        # Should have some autocorrelation structure
        if len(synthetic_returns) > 10:
            autocorr = synthetic_returns.autocorr(lag=1)
            assert not pd.isna(autocorr)  # Should be computable 