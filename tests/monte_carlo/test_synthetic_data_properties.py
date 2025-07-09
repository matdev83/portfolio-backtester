"""
Tests for Synthetic Data Properties Preservation

This module contains CRITICAL tests that validate synthetic data preserves
the key statistical properties of the original asset it replaces. These tests
ensure the Monte Carlo system actually works as intended.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from src.portfolio_backtester.monte_carlo.synthetic_data_generator import (
    SyntheticDataGenerator,
    AssetStatistics
)
from src.portfolio_backtester.monte_carlo.asset_replacement import AssetReplacementManager


class TestSyntheticDataPropertiesPreservation:
    """
    CRITICAL TESTS: Validate that synthetic data preserves the statistical
    properties of the original asset it replaces.
    """
    
    @pytest.fixture
    def test_config(self):
        """Configuration for property preservation tests."""
        return {
            'enable_synthetic_data': True,
            'replacement_percentage': 1.0,  # Replace all for testing
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
                'validation_tolerance': 0.3
            },
            'validation_config': {
                'enable_validation': False  # We'll do manual validation
            },
            'random_seed': 42
        }
    
    def create_simple_asset_data(self, volatility=0.02, periods=500):
        """Create simple asset data for testing."""
        np.random.seed(42)
        
        # Generate simple returns
        returns = np.random.normal(0, volatility, periods)
        
        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range('2020-01-01', periods=periods, freq='D')
        
        # Create OHLC data
        asset_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.001,
            'Low': prices * 0.998,
            'Close': prices
        }, index=dates)
        
        return asset_data, returns
    
    def test_basic_volatility_preservation(self, test_config):
        """CRITICAL: Test that synthetic data preserves basic volatility."""
        generator = SyntheticDataGenerator(test_config)
        
        # Create simple test data
        asset_data, original_returns = self.create_simple_asset_data(volatility=0.02)
        
        # Analyze original data
        stats = generator.analyze_asset_statistics(asset_data)
        
        # Generate synthetic data
        synthetic_returns = generator.generate_synthetic_returns(
            stats, 200, "BASIC_VOL_TEST"
        )
        
        # Calculate volatilities
        original_volatility = np.std(original_returns)
        synthetic_volatility = np.std(synthetic_returns)
        
        # CRITICAL ASSERTION: Volatility should be preserved within reasonable bounds
        vol_ratio = synthetic_volatility / original_volatility
        assert 0.7 < vol_ratio < 1.4, (
            f"CRITICAL FAILURE: Basic volatility not preserved within acceptable range!\n"
            f"Original volatility: {original_volatility:.4f}\n"
            f"Synthetic volatility: {synthetic_volatility:.4f}\n"
            f"Ratio: {vol_ratio:.3f} (should be between 0.7 and 1.4)\n"
            f"This means the synthetic data does NOT preserve the volatility "
            f"characteristics of the original asset!"
        )
        
        print(f"✓ Basic volatility preserved: "
              f"Original={original_volatility:.4f}, Synthetic={synthetic_volatility:.4f}")
    
    def test_fat_tail_preservation(self, test_config):
        """CRITICAL: Test that synthetic data preserves fat-tail properties."""
        generator = SyntheticDataGenerator(test_config)
        
        # Create data with fat tails
        np.random.seed(42)
        fat_tail_returns = np.random.standard_t(df=3, size=500) * 0.02
        
        # Convert to prices
        prices = 100 * np.exp(np.cumsum(fat_tail_returns))
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        asset_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.001,
            'Low': prices * 0.998,
            'Close': prices
        }, index=dates)
        
        # Analyze and generate synthetic data
        stats = generator.analyze_asset_statistics(asset_data)
        synthetic_returns = generator.generate_synthetic_returns(
            stats, 300, "FAT_TAIL_TEST"
        )
        
        # Calculate kurtosis
        original_kurtosis = pd.Series(fat_tail_returns).kurtosis()
        synthetic_kurtosis = pd.Series(synthetic_returns).kurtosis()
        
        # CRITICAL ASSERTION: Fat tail properties should be preserved
        if original_kurtosis > 3:  # Excess kurtosis present
            assert synthetic_kurtosis > 2, (
                f"CRITICAL FAILURE: Fat tail properties NOT preserved!\n"
                f"Original kurtosis: {original_kurtosis:.2f} (indicates fat tails)\n"
                f"Synthetic kurtosis: {synthetic_kurtosis:.2f} (should be > 2 to preserve fat tails)\n"
                f"This means the synthetic data lost the fat-tail characteristics "
                f"of the original asset, which is crucial for risk modeling!"
            )
            
            # Additional check: kurtosis should be reasonably similar
            kurtosis_ratio = synthetic_kurtosis / original_kurtosis
            assert 0.3 < kurtosis_ratio < 3.0, (
                f"CRITICAL FAILURE: Fat tail magnitude not preserved!\n"
                f"Original kurtosis: {original_kurtosis:.2f}\n"
                f"Synthetic kurtosis: {synthetic_kurtosis:.2f}\n"
                f"Ratio: {kurtosis_ratio:.3f} (should be between 0.3 and 3.0)\n"
                f"The synthetic data's tail behavior is too different from the original!"
            )
        
        print(f"✓ Fat tails preserved: "
              f"Original kurtosis={original_kurtosis:.2f}, Synthetic={synthetic_kurtosis:.2f}")
    
    def test_return_mean_preservation(self, test_config):
        """CRITICAL: Test that synthetic data preserves return mean."""
        generator = SyntheticDataGenerator(test_config)
        
        # Create data with specific mean
        np.random.seed(42)
        target_mean = 0.001  # 0.1% daily return
        returns = np.random.normal(target_mean, 0.02, 400)
        
        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range('2020-01-01', periods=400, freq='D')
        
        asset_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.001,
            'Low': prices * 0.998,
            'Close': prices
        }, index=dates)
        
        # Analyze and generate synthetic data
        stats = generator.analyze_asset_statistics(asset_data)
        synthetic_returns = generator.generate_synthetic_returns(
            stats, 300, "MEAN_TEST"
        )
        
        # Calculate means
        original_mean = np.mean(returns)
        synthetic_mean = np.mean(synthetic_returns)
        
        # CRITICAL ASSERTION: Mean should be preserved within reasonable bounds
        mean_diff = abs(synthetic_mean - original_mean)
        assert mean_diff < 0.001, (  # Within 0.1% daily
            f"CRITICAL FAILURE: Mean return not preserved!\n"
            f"Original mean: {original_mean:.5f}\n"
            f"Synthetic mean: {synthetic_mean:.5f}\n"
            f"Difference: {mean_diff:.5f} (should be < 0.001)\n"
            f"This means the synthetic data has a significantly different "
            f"expected return than the original asset!"
        )
        
        print(f"✓ Mean preserved: "
              f"Original={original_mean:.5f}, Synthetic={synthetic_mean:.5f}")
    
    def test_comprehensive_properties_preservation(self, test_config):
        """CRITICAL: Comprehensive test of multiple properties together."""
        generator = SyntheticDataGenerator(test_config)
        
        # Create realistic asset data
        np.random.seed(42)
        
        # Generate returns with volatility clustering
        returns = []
        volatility = 0.02
        
        for i in range(600):
            # Simple volatility clustering
            if i > 0:
                volatility = 0.0001 + 0.05 * (returns[i-1]**2) + 0.9 * volatility
            
            # Fat-tailed innovation
            innovation = np.random.standard_t(df=4)
            daily_return = innovation * np.sqrt(volatility)
            returns.append(daily_return)
        
        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range('2020-01-01', periods=600, freq='D')
        
        asset_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.001,
            'Low': prices * 0.998,
            'Close': prices
        }, index=dates)
        
        # Analyze original data
        stats = generator.analyze_asset_statistics(asset_data)
        
        # Generate synthetic data
        synthetic_returns = generator.generate_synthetic_returns(
            stats, 400, "COMPREHENSIVE_TEST"
        )
        
        # Calculate properties
        original_series = pd.Series(returns)
        synthetic_series = pd.Series(synthetic_returns)
        
        original_props = {
            'mean': original_series.mean(),
            'volatility': original_series.std(),
            'skewness': original_series.skew(),
            'kurtosis': original_series.kurtosis(),
            'autocorr_returns': original_series.autocorr(lag=1),
            'autocorr_squared': (original_series**2).autocorr(lag=1)
        }
        
        synthetic_props = {
            'mean': synthetic_series.mean(),
            'volatility': synthetic_series.std(),
            'skewness': synthetic_series.skew(),
            'kurtosis': synthetic_series.kurtosis(),
            'autocorr_returns': synthetic_series.autocorr(lag=1),
            'autocorr_squared': (synthetic_series**2).autocorr(lag=1)
        }
        
        # CRITICAL ASSERTIONS
        tolerances = {
            'mean': 0.001,
            'volatility': 0.5,  # 50% relative tolerance
            'skewness': 1.0,
            'kurtosis': 3.0,
            'autocorr_returns': 0.2,
            'autocorr_squared': 0.5
        }
        
        print("\n" + "="*60)
        print("COMPREHENSIVE PROPERTIES PRESERVATION TEST")
        print("="*60)
        
        failed_properties = []
        
        for prop in original_props:
            orig_val = original_props[prop]
            synth_val = synthetic_props[prop]
            
            if prop == 'volatility' and abs(orig_val) > 1e-6:
                # Relative tolerance for volatility
                rel_diff = abs(synth_val - orig_val) / abs(orig_val)
                passed = rel_diff < tolerances[prop]
                print(f"{prop:20} {'✓ PASS' if passed else '✗ FAIL':8} "
                      f"Original: {orig_val:8.4f} Synthetic: {synth_val:8.4f} "
                      f"Rel diff: {rel_diff:.3f}")
                
                if not passed:
                    failed_properties.append(prop)
                    print(f"  CRITICAL: {prop} relative difference {rel_diff:.3f} "
                          f"exceeds tolerance {tolerances[prop]:.3f}")
            else:
                # Absolute tolerance
                abs_diff = abs(synth_val - orig_val)
                passed = abs_diff < tolerances[prop]
                print(f"{prop:20} {'✓ PASS' if passed else '✗ FAIL':8} "
                      f"Original: {orig_val:8.4f} Synthetic: {synth_val:8.4f} "
                      f"Abs diff: {abs_diff:.3f}")
                
                if not passed:
                    failed_properties.append(prop)
                    print(f"  CRITICAL: {prop} absolute difference {abs_diff:.3f} "
                          f"exceeds tolerance {tolerances[prop]:.3f}")
        
        print("="*60)
        
        # CRITICAL ASSERTION: All properties must be preserved within tolerance
        assert len(failed_properties) == 0, (
            f"CRITICAL FAILURE: Properties not preserved within tolerance: {failed_properties}\n"
            f"This means the synthetic data does NOT preserve the statistical properties "
            f"of the original asset, which defeats the purpose of the Monte Carlo system!"
        )
        
        print("✓ ALL COMPREHENSIVE PROPERTIES SUCCESSFULLY PRESERVED!")
        print("="*60)
    
    def test_asset_replacement_integration(self, test_config):
        """CRITICAL: Test property preservation in asset replacement context."""
        
        # Create multiple assets
        assets = ['ASSET_A', 'ASSET_B', 'ASSET_C']
        asset_data = {}
        
        for asset in assets:
            data, _ = self.create_simple_asset_data(
                volatility=0.02 + 0.01 * hash(asset) % 3  # Different volatilities
            )
            asset_data[asset] = data
        
        # Create asset replacement manager
        manager = AssetReplacementManager(test_config)
        
        # Define test periods
        test_start = pd.Timestamp('2020-08-01')
        test_end = pd.Timestamp('2020-12-31')
        
        # Create Monte Carlo dataset
        modified_data, replacement_info = manager.create_monte_carlo_dataset(
            original_data=asset_data,
            universe=assets,
            test_start=test_start,
            test_end=test_end,
            run_id="property_preservation_test"
        )
        
        # Validate property preservation for replaced assets
        for asset in replacement_info.selected_assets:
            # Get test period data
            test_mask = (asset_data[asset].index >= test_start) & (asset_data[asset].index <= test_end)
            original_test_data = asset_data[asset].loc[test_mask]
            synthetic_test_data = modified_data[asset].loc[test_mask]
            
            # Calculate returns
            original_returns = original_test_data['Close'].pct_change(fill_method=None).dropna()
            synthetic_returns = synthetic_test_data['Close'].pct_change(fill_method=None).dropna()
            
            if len(original_returns) > 20 and len(synthetic_returns) > 20:
                # CRITICAL ASSERTION: Volatility should be preserved
                original_vol = original_returns.std()
                synthetic_vol = synthetic_returns.std()
                
                if original_vol > 1e-6:
                    vol_ratio = synthetic_vol / original_vol
                    assert 0.5 < vol_ratio < 2.0, (
                        f"CRITICAL FAILURE: Volatility not preserved for {asset} in asset replacement!\n"
                        f"Original volatility: {original_vol:.4f}\n"
                        f"Synthetic volatility: {synthetic_vol:.4f}\n"
                        f"Ratio: {vol_ratio:.3f} (should be between 0.5 and 2.0)\n"
                        f"This means the Monte Carlo replacement system is not working correctly!"
                    )
                    
                    print(f"✓ {asset} volatility preserved in replacement: "
                          f"Original={original_vol:.4f}, Synthetic={synthetic_vol:.4f}")
        
        print(f"✓ Property preservation validated for {len(replacement_info.selected_assets)} replaced assets")


if __name__ == "__main__":
    # Run critical tests
    print("Running CRITICAL synthetic data property preservation tests...")
    pytest.main([__file__, "-v", "-s"]) 