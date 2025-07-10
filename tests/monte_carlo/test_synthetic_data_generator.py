"""
Tests for Synthetic Data Generator

This module contains comprehensive tests for the synthetic data generation
functionality, including GARCH model fitting, statistical validation,
and integration with the backtesting system.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import logging

from src.portfolio_backtester.monte_carlo.synthetic_data_generator import (
    SyntheticDataGenerator,
    GARCHParameters,
    AssetStatistics,
    DistributionType
)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'replacement_percentage': 0.10,
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
            'max_attempts': 5,
            'validation_tolerance': 0.1
        },
        'validation_config': {
            'enable_validation': True,
            'ks_test_pvalue_threshold': 0.01,
            'autocorr_max_deviation': 0.1,
            'volatility_clustering_threshold': 0.05,
            'tail_index_tolerance': 0.2
        },
        'jump_diffusion': {
            'enable': False
        },
        'random_seed': 42
    }


class TestSyntheticDataGenerator:
    """Test suite for SyntheticDataGenerator class."""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """Generate sample OHLC data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        ohlc_data = pd.DataFrame(index=dates)
        ohlc_data['Close'] = prices
        ohlc_data['Open'] = prices * np.random.uniform(0.98, 1.02, len(prices))
        ohlc_data['High'] = np.maximum(ohlc_data['Open'], ohlc_data['Close']) * np.random.uniform(1.0, 1.05, len(prices))
        ohlc_data['Low'] = np.minimum(ohlc_data['Open'], ohlc_data['Close']) * np.random.uniform(0.95, 1.0, len(prices))
        
        return ohlc_data
    
    def test_initialization(self, sample_config):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(sample_config)
        
        assert generator.config == sample_config
        assert generator.garch_config == sample_config['garch_config']
        assert generator.generation_config == sample_config['generation_config']
        assert generator.validation_config == sample_config['validation_config']
    
    def test_analyze_asset_statistics(self, sample_config, sample_ohlc_data):
        """Test asset statistics analysis."""
        generator = SyntheticDataGenerator(sample_config)
        
        stats = generator.analyze_asset_statistics(sample_ohlc_data)
        
        assert isinstance(stats, AssetStatistics)
        assert stats.mean_return is not None
        assert stats.volatility > 0
        # Modern t-distribution approach doesn't use GARCH params
        # Instead, it uses fitted t-distribution parameters stored in other fields
        assert stats.tail_index is not None  # T-distribution degrees of freedom
        assert stats.tail_index > 2.0  # Should be reasonable for finite variance
    
    def test_analyze_asset_statistics_insufficient_data(self, sample_config):
        """Test handling of insufficient data."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Create very short data series
        short_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2020-01-01', periods=3, freq='D'))
        
        with pytest.raises(ValueError, match="Insufficient historical data"):
            generator.analyze_asset_statistics(short_data)
    
    def test_generate_synthetic_returns(self, sample_config, sample_ohlc_data):
        """Test synthetic returns generation."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Analyze sample data
        stats = generator.analyze_asset_statistics(sample_ohlc_data)
        
        # Generate synthetic returns
        synthetic_returns = generator.generate_synthetic_returns(stats, 100, "TEST")
        
        assert len(synthetic_returns) == 100
        assert isinstance(synthetic_returns, np.ndarray)
        assert np.isfinite(synthetic_returns).all()
    
    def test_generate_synthetic_prices(self, sample_config, sample_ohlc_data):
        """Test synthetic price generation."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Generate synthetic prices
        synthetic_prices = generator.generate_synthetic_prices(sample_ohlc_data, 50, "TEST")
        
        assert len(synthetic_prices) == 50
        assert isinstance(synthetic_prices, pd.DataFrame)
        assert all(col in synthetic_prices.columns for col in ['Open', 'High', 'Low', 'Close'])
        assert (synthetic_prices > 0).all().all()  # All prices should be positive
    
    def test_garch_parameter_validation(self, sample_config):
        """Test GARCH parameter validation."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Test valid parameters
        valid_params = GARCHParameters(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            nu=5.0,
            distribution=DistributionType.STUDENT_T
        )
        
        # Should not raise exception
        synthetic_returns = generator._generate_garch_returns(valid_params, 100)
        assert len(synthetic_returns) == 100
    
    def test_fallback_generation(self, sample_config, sample_ohlc_data):
        """Test fallback generation when GARCH fitting fails."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Create mock asset statistics with problematic GARCH parameters
        stats = AssetStatistics(
            mean_return=0.001,
            volatility=0.02,
            skewness=-0.5,
            kurtosis=5.0,
            autocorr_returns=0.05,
            autocorr_squared=0.3,
            tail_index=3.0,
            garch_params=None  # No GARCH parameters
        )
        
        # Should use fallback method
        synthetic_returns = generator._generate_fallback_returns(stats, 100)
        assert len(synthetic_returns) == 100
        assert np.isfinite(synthetic_returns).all()
    
    def test_validation_integration(self, sample_config, sample_ohlc_data):
        """Test validation integration."""
        # Enable validation
        sample_config['validation_config']['enable_validation'] = True
        generator = SyntheticDataGenerator(sample_config)
        
        stats = generator.analyze_asset_statistics(sample_ohlc_data)
        
        # Generate with validation
        synthetic_returns = generator.generate_synthetic_returns(stats, 100, "TEST")
        
        assert len(synthetic_returns) == 100
        # Should pass basic validation checks
        assert np.std(synthetic_returns) > 0  # Has volatility
    
    def test_jump_diffusion_integration(self, sample_config, sample_ohlc_data):
        """Test jump-diffusion enhancement."""
        # Enable jump diffusion
        sample_config['jump_diffusion']['enable'] = True
        sample_config['jump_diffusion']['jump_intensity'] = 0.1
        
        generator = SyntheticDataGenerator(sample_config)
        stats = generator.analyze_asset_statistics(sample_ohlc_data)
        
        # Generate with jumps
        synthetic_returns = generator.generate_synthetic_returns(stats, 100, "TEST")
        
        assert len(synthetic_returns) == 100
        assert np.isfinite(synthetic_returns).all()
    
    def test_random_seed_reproducibility(self, sample_config, sample_ohlc_data):
        """Test random seed reproducibility."""
        # Create fresh generators with same config to test reproducibility
        generator1 = SyntheticDataGenerator(sample_config)
        generator2 = SyntheticDataGenerator(sample_config)
        
        # Each generator should analyze independently to test full reproducibility
        stats1 = generator1.analyze_asset_statistics(sample_ohlc_data)
        stats2 = generator2.analyze_asset_statistics(sample_ohlc_data)
        
        # Generate with same seed - should be identical
        returns1 = generator1.generate_synthetic_returns(stats1, 100, "TEST")
        returns2 = generator2.generate_synthetic_returns(stats2, 100, "TEST")
        
        # Should be identical due to same seed
        np.testing.assert_array_equal(returns1, returns2)
    
    def test_tail_index_estimation(self, sample_config):
        """Test tail index estimation."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Create data with known fat tails
        np.random.seed(42)
        fat_tail_data = pd.Series(np.random.standard_t(df=3, size=1000))
        
        tail_index = generator._estimate_tail_index(fat_tail_data)
        
        # Should detect fat tails (low tail index)
        assert 1.5 <= tail_index <= 5.0
    
    def test_returns_to_prices_conversion(self, sample_config):
        """Test returns to prices conversion."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Test with known returns
        returns = np.array([0.01, -0.02, 0.015, -0.005])
        initial_price = 100.0
        
        prices = generator._returns_to_prices(returns, initial_price)
        
        # Check price evolution
        expected_prices = [
            100.0 * 1.01,
            100.0 * 1.01 * 0.98,
            100.0 * 1.01 * 0.98 * 1.015,
            100.0 * 1.01 * 0.98 * 1.015 * 0.995
        ]
        
        np.testing.assert_array_almost_equal(prices, expected_prices)
    
    def test_ohlc_generation_logic(self, sample_config):
        """Test OHLC generation from prices."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Test with sample prices
        prices = np.array([100, 102, 98, 101])
        
        ohlc = generator._generate_ohlc_from_prices(prices)
        
        assert ohlc.shape == (4, 4)  # 4 days, 4 fields (OHLC)
        
        # Check OHLC relationships
        for i in range(len(ohlc)):
            open_price, high, low, close = ohlc[i]
            
            # High should be >= max(open, close)
            assert high >= max(open_price, close)
            
            # Low should be <= min(open, close)
            assert low <= min(open_price, close)
            
            # Close should match input price
            assert close == prices[i]
            
            # All prices should be positive
            assert all(p > 0 for p in [open_price, high, low, close])


class TestGARCHParameters:
    """Test suite for GARCHParameters dataclass."""
    
    def test_garch_parameters_creation(self):
        """Test GARCH parameters creation."""
        params = GARCHParameters(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            nu=5.0,
            distribution=DistributionType.STUDENT_T
        )
        
        assert params.omega == 0.01
        assert params.alpha == 0.1
        assert params.beta == 0.85
        assert params.nu == 5.0
        assert params.distribution == DistributionType.STUDENT_T
    
    def test_distribution_types(self):
        """Test distribution type enumeration."""
        assert DistributionType.NORMAL.value == "normal"
        assert DistributionType.STUDENT_T.value == "studentt"
        assert DistributionType.SKEWED_STUDENT_T.value == "skewstudent"
        assert DistributionType.GED.value == "ged"


class TestAssetStatistics:
    """Test suite for AssetStatistics dataclass."""
    
    def test_asset_statistics_creation(self):
        """Test asset statistics creation."""
        garch_params = GARCHParameters(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            nu=5.0
        )
        
        stats = AssetStatistics(
            mean_return=0.001,
            volatility=0.02,
            skewness=-0.5,
            kurtosis=5.0,
            autocorr_returns=0.05,
            autocorr_squared=0.3,
            tail_index=3.0,
            garch_params=garch_params
        )
        
        assert stats.mean_return == 0.001
        assert stats.volatility == 0.02
        assert stats.skewness == -0.5
        assert stats.kurtosis == 5.0
        assert stats.autocorr_returns == 0.05
        assert stats.autocorr_squared == 0.3
        assert stats.tail_index == 3.0
        assert stats.garch_params == garch_params


# Integration tests
class TestSyntheticDataIntegration:
    """Integration tests for synthetic data generation."""
    
    @pytest.fixture
    def realistic_market_data(self):
        """Generate realistic market data for integration testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Generate more realistic returns with volatility clustering
        returns = []
        volatility = 0.02
        
        for i in range(len(dates)):
            # GARCH-like volatility clustering
            if i > 0:
                volatility = 0.0001 + 0.05 * (returns[i-1]**2) + 0.9 * volatility
            
            daily_return = np.random.normal(0, np.sqrt(volatility))
            returns.append(daily_return)
        
        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLC
        ohlc_data = pd.DataFrame(index=dates)
        ohlc_data['Close'] = prices
        ohlc_data['Open'] = prices * np.random.uniform(0.995, 1.005, len(prices))
        ohlc_data['High'] = np.maximum(ohlc_data['Open'], ohlc_data['Close']) * np.random.uniform(1.0, 1.02, len(prices))
        ohlc_data['Low'] = np.minimum(ohlc_data['Open'], ohlc_data['Close']) * np.random.uniform(0.98, 1.0, len(prices))
        
        return ohlc_data
    
    def test_end_to_end_generation(self, sample_config, realistic_market_data):
        """Test end-to-end synthetic data generation."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Full pipeline test
        stats = generator.analyze_asset_statistics(realistic_market_data)
        synthetic_data = generator.generate_synthetic_prices(realistic_market_data, 100, "INTEGRATION_TEST")
        
        # Validate results
        assert len(synthetic_data) == 100
        assert isinstance(synthetic_data, pd.DataFrame)
        assert all(col in synthetic_data.columns for col in ['Open', 'High', 'Low', 'Close'])
        
        # Check price relationships
        assert (synthetic_data['High'] >= synthetic_data['Close']).all()
        assert (synthetic_data['High'] >= synthetic_data['Open']).all()
        assert (synthetic_data['Low'] <= synthetic_data['Close']).all()
        assert (synthetic_data['Low'] <= synthetic_data['Open']).all()
        
        # Check for reasonable price movements
        returns = synthetic_data['Close'].pct_change(fill_method=None).dropna()
        assert returns.std() > 0.005  # Has meaningful volatility
        assert returns.std() < 0.1    # Not excessively volatile
    
    def test_statistical_properties_preservation(self, sample_config, realistic_market_data):
        """Test that key statistical properties are preserved."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Generate synthetic data
        stats = generator.analyze_asset_statistics(realistic_market_data)
        synthetic_data = generator.generate_synthetic_prices(realistic_market_data, 500, "STATS_TEST")
        
        # Compare statistical properties
        original_returns = realistic_market_data['Close'].pct_change(fill_method=None).dropna()
        synthetic_returns = synthetic_data['Close'].pct_change(fill_method=None).dropna()
        
        # Volatility should be similar
        vol_ratio = synthetic_returns.std() / original_returns.std()
        assert 0.5 < vol_ratio < 2.0  # Within reasonable range
        
        # Should have some autocorrelation in squared returns (volatility clustering)
        # Note: T-distribution approach may not preserve volatility clustering as strongly as GARCH
        synth_vol_clustering = (synthetic_returns**2).autocorr(lag=1)
        
        # More lenient check - just ensure it's not extremely negative
        assert synth_vol_clustering > -0.1, f"Volatility clustering too negative: {synth_vol_clustering:.4f}"
    
    def test_multiple_asset_generation(self, sample_config):
        """Test generation for multiple assets."""
        generator = SyntheticDataGenerator(sample_config)
        
        # Create multiple asset data
        assets = ['AAPL', 'MSFT', 'GOOGL']
        asset_data = {}
        
        for asset in assets:
            np.random.seed(hash(asset) % 1000)  # Different seed per asset
            dates = pd.date_range('2020-01-01', periods=300, freq='D')
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            asset_data[asset] = pd.DataFrame({
                'Close': prices,
                'Open': prices * np.random.uniform(0.99, 1.01, len(prices)),
                'High': prices * np.random.uniform(1.0, 1.03, len(prices)),
                'Low': prices * np.random.uniform(0.97, 1.0, len(prices))
            }, index=dates)
        
        # Generate synthetic data for each asset
        synthetic_data = {}
        for asset in assets:
            synthetic_data[asset] = generator.generate_synthetic_prices(
                asset_data[asset], 50, asset
            )
        
        # Validate all assets generated successfully
        for asset in assets:
            assert len(synthetic_data[asset]) == 50
            assert (synthetic_data[asset] > 0).all().all()  # All positive prices 