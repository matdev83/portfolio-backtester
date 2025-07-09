"""
End-to-End Property Preservation Tests

This module contains CRITICAL tests that validate synthetic data property
preservation in the actual backtesting context, ensuring the Monte Carlo
system works correctly when integrated with the full optimization pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.monte_carlo.asset_replacement import AssetReplacementManager
from src.portfolio_backtester.monte_carlo.synthetic_data_generator import SyntheticDataGenerator
from src.portfolio_backtester.monte_carlo.validation_metrics import SyntheticDataValidator


class TestEndToEndPropertyPreservation:
    """
    CRITICAL END-TO-END TESTS: Validate property preservation in actual backtesting.
    """
    
    @pytest.fixture
    def monte_carlo_config(self):
        """Full Monte Carlo configuration for end-to-end testing."""
        return {
            'enable_synthetic_data': True,
            'replacement_percentage': 0.3,  # Replace 30% of assets
            'min_historical_observations': 150,
            'random_seed': 42,
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
                'enable_validation': True,
                'basic_stats_tolerance': 0.3,
                'ks_test_pvalue_threshold': 0.01,
                'autocorr_max_deviation': 0.15,
                'volatility_clustering_threshold': 0.03,
                'fat_tail_threshold': 3.0,
                'extreme_value_threshold': 0.05
            }
        }
    
    def create_realistic_market_data(self, assets, start_date='2020-01-01', periods=1000):
        """
        Create realistic market data for multiple assets with different characteristics.
        """
        market_data = {}
        
        # Define asset characteristics
        asset_profiles = {
            'STABLE_LARGE_CAP': {'vol': 0.015, 'tail_df': 8.0, 'autocorr': 0.02, 'clustering': 0.03},
            'VOLATILE_TECH': {'vol': 0.035, 'tail_df': 4.0, 'autocorr': 0.05, 'clustering': 0.08},
            'FINANCIAL_STOCK': {'vol': 0.025, 'tail_df': 5.0, 'autocorr': 0.03, 'clustering': 0.05},
            'UTILITY_STOCK': {'vol': 0.012, 'tail_df': 10.0, 'autocorr': 0.01, 'clustering': 0.02},
            'COMMODITY_ETF': {'vol': 0.028, 'tail_df': 3.5, 'autocorr': 0.08, 'clustering': 0.10},
            'BOND_ETF': {'vol': 0.008, 'tail_df': 12.0, 'autocorr': 0.05, 'clustering': 0.01},
            'REIT_STOCK': {'vol': 0.022, 'tail_df': 6.0, 'autocorr': 0.04, 'clustering': 0.06},
            'EMERGING_MARKET': {'vol': 0.045, 'tail_df': 3.0, 'autocorr': 0.10, 'clustering': 0.12}
        }
        
        dates = pd.date_range(start_date, periods=periods, freq='D')
        
        for i, asset in enumerate(assets):
            # Use asset profile or default
            profile_key = list(asset_profiles.keys())[i % len(asset_profiles)]
            profile = asset_profiles[profile_key]
            
            # Set seed for reproducibility
            np.random.seed(hash(asset) % 1000)
            
            # Generate returns with asset-specific characteristics
            returns = []
            volatility = profile['vol']
            
            for t in range(periods):
                # Volatility clustering (GARCH-like)
                if t > 0:
                    volatility = (0.0001 + 
                                profile['clustering'] * (returns[t-1]**2) + 
                                (0.95 - profile['clustering']) * volatility)
                
                # Fat-tailed innovations
                innovation = np.random.standard_t(df=profile['tail_df'])
                
                # Add autocorrelation
                if t > 0:
                    base_return = innovation * np.sqrt(volatility)
                    autocorr_component = profile['autocorr'] * returns[t-1]
                    daily_return = base_return + autocorr_component
                else:
                    daily_return = innovation * np.sqrt(volatility)
                
                returns.append(daily_return)
            
            # Convert to prices
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create realistic OHLC data
            market_data[asset] = pd.DataFrame({
                'Open': prices * np.random.uniform(0.995, 1.005, len(prices)),
                'High': np.maximum(prices * np.random.uniform(0.995, 1.005, len(prices)), 
                                 prices) * np.random.uniform(1.0, 1.02, len(prices)),
                'Low': np.minimum(prices * np.random.uniform(0.995, 1.005, len(prices)), 
                                prices) * np.random.uniform(0.98, 1.0, len(prices)),
                'Close': prices,
                'Volume': np.random.lognormal(15, 0.5, len(prices))
            }, index=dates)
        
        return market_data
    
    def test_property_preservation_in_evaluation_logic(self, monte_carlo_config):
        """CRITICAL: Test property preservation in EvaluationLogic integration."""
        
        # Create test assets
        assets = ['STABLE_LARGE_CAP', 'VOLATILE_TECH', 'FINANCIAL_STOCK', 'UTILITY_STOCK', 
                 'COMMODITY_ETF', 'BOND_ETF', 'REIT_STOCK', 'EMERGING_MARKET']
        
        # Create market data
        market_data = self.create_realistic_market_data(assets, periods=800)
        
        # Create Backtester instance
        from unittest.mock import Mock
        args = Mock()
        # Create a dummy scenario to avoid index error
        dummy_scenario = {'strategy': 'momentum', 'strategy_params': {}}
        evaluator = Backtester(monte_carlo_config, [dummy_scenario], args)
        
        # Define test periods
        train_start = pd.Timestamp('2020-01-01')
        train_end = pd.Timestamp('2020-08-31')
        test_start = pd.Timestamp('2020-09-01')
        test_end = pd.Timestamp('2020-12-31')
        
        # Apply Monte Carlo replacement
        modified_data = evaluator.apply_monte_carlo_replacement(
            original_data=market_data,
            universe=assets,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            trial_number=1
        )
        
        # Get replacement info
        replacement_info = evaluator.asset_replacement_manager.get_replacement_info()
        
        # CRITICAL VALIDATIONS
        print("\n" + "="*70)
        print("END-TO-END PROPERTY PRESERVATION VALIDATION")
        print("="*70)
        
        # 1. Validate that training data is unchanged
        for asset in assets:
            train_mask = (market_data[asset].index >= train_start) & (market_data[asset].index <= train_end)
            original_train = market_data[asset].loc[train_mask]
            modified_train = modified_data[asset].loc[train_mask]
            
            # CRITICAL ASSERTION: Training data should be identical
            pd.testing.assert_frame_equal(original_train, modified_train, 
                                        check_dtype=False, check_exact=False, rtol=1e-10)
            
        print("✓ Training data unchanged (as required)")
        
        # 2. Validate that test data is modified only for selected assets
        unmodified_count = 0
        modified_count = 0
        
        for asset in assets:
            test_mask = (market_data[asset].index >= test_start) & (market_data[asset].index <= test_end)
            original_test = market_data[asset].loc[test_mask]
            modified_test = modified_data[asset].loc[test_mask]
            
            if asset in replacement_info.selected_assets:
                # Should be modified
                try:
                    pd.testing.assert_frame_equal(original_test, modified_test, 
                                                check_dtype=False, check_exact=False, rtol=1e-10)
                    pytest.fail(f"Asset {asset} should have been modified but wasn't")
                except AssertionError:
                    # Expected - data should be different
                    modified_count += 1
            else:
                # Should be unchanged
                pd.testing.assert_frame_equal(original_test, modified_test, 
                                            check_dtype=False, check_exact=False, rtol=1e-10)
                unmodified_count += 1
        
        print(f"✓ Test data correctly modified: {modified_count} assets modified, {unmodified_count} unchanged")
        
        # 3. Validate replacement percentage
        expected_replacements = int(len(assets) * monte_carlo_config['replacement_percentage'])
        actual_replacements = len(replacement_info.selected_assets)
        
        assert abs(actual_replacements - expected_replacements) <= 1, (
            f"Replacement percentage not respected: Expected ~{expected_replacements}, "
            f"Got {actual_replacements}"
        )
        
        print(f"✓ Replacement percentage respected: {actual_replacements}/{len(assets)} assets replaced")
        
        # 4. CRITICAL: Validate property preservation for each replaced asset
        validator = SyntheticDataValidator(monte_carlo_config['validation_config'])
        
        for asset in replacement_info.selected_assets:
            print(f"\nValidating properties for {asset}:")
            
            # Get original and synthetic test data
            test_mask = (market_data[asset].index >= test_start) & (market_data[asset].index <= test_end)
            original_test_data = market_data[asset].loc[test_mask]
            synthetic_test_data = modified_data[asset].loc[test_mask]
            
            # Calculate returns
            original_returns = original_test_data['Close'].pct_change(fill_method=None).dropna()
            synthetic_returns = synthetic_test_data['Close'].pct_change(fill_method=None).dropna()
            
            if len(original_returns) > 50 and len(synthetic_returns) > 50:
                # Validate properties
                validation_result = validator.validate_synthetic_data(
                    original_returns, synthetic_returns, f"EndToEnd_{asset}"
                )
                
                # CRITICAL ASSERTIONS based on validation results
                assert validation_result.overall_quality_score >= 0.6, (
                    f"Property preservation failed for {asset}: "
                    f"Quality score={validation_result.overall_quality_score:.3f}"
                )
                
                # Check specific critical properties
                assert validation_result.basic_stats_test['passed'], (
                    f"Basic statistics not preserved for {asset}: "
                    f"{validation_result.basic_stats_test['details']}"
                )
                
                assert validation_result.fat_tail_test['passed'], (
                    f"Fat tail properties not preserved for {asset}: "
                    f"{validation_result.fat_tail_test['details']}"
                )
                
                print(f"  ✓ Quality score: {validation_result.overall_quality_score:.3f}")
                print(f"  ✓ Basic stats: {validation_result.basic_stats_test['details']['volatility_ratio']:.3f}")
                print(f"  ✓ Fat tails: {validation_result.fat_tail_test['details']['kurtosis_ratio']:.3f}")
        
        print("\n" + "="*70)
        print("✓ ALL END-TO-END PROPERTY PRESERVATION TESTS PASSED!")
        print("="*70)
    
    def test_multiple_optimization_runs_consistency(self, monte_carlo_config):
        """CRITICAL: Test property preservation across multiple optimization runs."""
        
        # Create test assets
        assets = ['ASSET_A', 'ASSET_B', 'ASSET_C', 'ASSET_D', 'ASSET_E']
        market_data = self.create_realistic_market_data(assets, periods=600)
        
        # Run multiple optimization runs
        num_runs = 5
        results = []
        
        for run_id in range(num_runs):
            # Create fresh Backtester for each run
            from unittest.mock import Mock
            args = Mock()
            # Create a dummy scenario to avoid index error
            dummy_scenario = {'strategy': 'momentum', 'strategy_params': {}}
            evaluator = Backtester(monte_carlo_config, [dummy_scenario], args)
            
            # Define test periods
            train_start = pd.Timestamp('2020-01-01')
            train_end = pd.Timestamp('2020-06-30')
            test_start = pd.Timestamp('2020-07-01')
            test_end = pd.Timestamp('2020-12-31')
            
            # Apply Monte Carlo replacement
            modified_data = evaluator.apply_monte_carlo_replacement(
                original_data=market_data,
                universe=assets,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                trial_number=run_id
            )
            
            # Collect results
            replacement_info = evaluator.asset_replacement_manager.get_replacement_info()
            run_result = {
                'run_id': run_id,
                'selected_assets': replacement_info.selected_assets.copy(),
                'replacement_count': len(replacement_info.selected_assets),
                'modified_data': modified_data
            }
            results.append(run_result)
        
        # CRITICAL VALIDATIONS across runs
        print("\n" + "="*70)
        print("MULTI-RUN CONSISTENCY VALIDATION")
        print("="*70)
        
        # 1. Validate replacement count consistency
        replacement_counts = [r['replacement_count'] for r in results]
        expected_count = int(len(assets) * monte_carlo_config['replacement_percentage'])
        
        for count in replacement_counts:
            assert abs(count - expected_count) <= 1, (
                f"Inconsistent replacement count across runs: {replacement_counts}"
            )
        
        print(f"✓ Consistent replacement counts: {replacement_counts}")
        
        # 2. Validate that different runs select different assets (randomness)
        all_selected = set()
        for result in results:
            all_selected.update(result['selected_assets'])
        
        # Should have good coverage of assets across runs
        coverage_ratio = len(all_selected) / len(assets)
        assert coverage_ratio >= 0.6, (
            f"Poor asset coverage across runs: {coverage_ratio:.2f}"
        )
        
        print(f"✓ Good asset coverage across runs: {coverage_ratio:.2f}")
        
        # 3. CRITICAL: Validate property preservation in each run
        validator = SyntheticDataValidator(monte_carlo_config['validation_config'])
        
        for run_result in results:
            run_id = run_result['run_id']
            modified_data = run_result['modified_data']
            
            for asset in run_result['selected_assets']:
                # Get test period data
                test_mask = (market_data[asset].index >= pd.Timestamp('2020-07-01')) & \
                           (market_data[asset].index <= pd.Timestamp('2020-12-31'))
                
                original_test = market_data[asset].loc[test_mask]
                synthetic_test = modified_data[asset].loc[test_mask]
                
                # Calculate returns
                original_returns = original_test['Close'].pct_change(fill_method=None).dropna()
                synthetic_returns = synthetic_test['Close'].pct_change(fill_method=None).dropna()
                
                if len(original_returns) > 30 and len(synthetic_returns) > 30:
                    # Validate properties
                    validation_result = validator.validate_synthetic_data(
                        original_returns, synthetic_returns, f"Run{run_id}_{asset}"
                    )
                    
                    # CRITICAL ASSERTION: Properties should be preserved in each run
                    assert validation_result.overall_quality_score >= 0.5, (
                        f"Property preservation failed in run {run_id} for {asset}: "
                        f"Quality score={validation_result.overall_quality_score:.3f}"
                    )
        
        print(f"✓ Properties preserved across all {num_runs} runs")
        
        print("\n" + "="*70)
        print("✓ ALL MULTI-RUN CONSISTENCY TESTS PASSED!")
        print("="*70)
    
    def test_extreme_market_conditions_property_preservation(self, monte_carlo_config):
        """CRITICAL: Test property preservation under extreme market conditions."""
        
        # Create assets with extreme characteristics
        extreme_assets = ['CRASH_PRONE', 'BUBBLE_STOCK', 'ULTRA_VOLATILE', 'DEAD_CAT_BOUNCE']
        
        # Generate extreme market data
        market_data = {}
        dates = pd.date_range('2020-01-01', periods=400, freq='D')
        
        for asset in extreme_assets:
            np.random.seed(hash(asset) % 1000)
            
            if asset == 'CRASH_PRONE':
                # Stock prone to sudden crashes
                returns = np.random.normal(0.001, 0.02, len(dates))
                # Add crashes
                crash_days = np.random.choice(len(dates), 5, replace=False)
                for day in crash_days:
                    returns[day] = -0.15  # 15% crash
                
            elif asset == 'BUBBLE_STOCK':
                # Stock with bubble-like behavior
                returns = np.random.normal(0.003, 0.04, len(dates))
                # Add occasional huge gains
                bubble_days = np.random.choice(len(dates), 8, replace=False)
                for day in bubble_days:
                    returns[day] = 0.12  # 12% gain
                
            elif asset == 'ULTRA_VOLATILE':
                # Extremely volatile stock
                returns = np.random.standard_t(df=2.5, size=len(dates)) * 0.06
                
            elif asset == 'DEAD_CAT_BOUNCE':
                # Stock with periods of no movement then sudden jumps
                returns = np.zeros(len(dates))
                for i in range(0, len(dates), 20):
                    # 5% jump every 20 days
                    if i < len(dates):
                        returns[i] = np.random.choice([-0.05, 0.05])
            
            # Convert to prices
            prices = 100 * np.exp(np.cumsum(returns))
            
            market_data[asset] = pd.DataFrame({
                'Open': prices * 0.999,
                'High': prices * 1.002,
                'Low': prices * 0.998,
                'Close': prices
            }, index=dates)
        
        # Test property preservation under extreme conditions
        from unittest.mock import Mock
        args = Mock()
        # Create a dummy scenario to avoid index error
        dummy_scenario = {'strategy': 'momentum', 'strategy_params': {}}
        evaluator = Backtester(monte_carlo_config, [dummy_scenario], args)
        
        # Define periods
        train_start = pd.Timestamp('2020-01-01')
        train_end = pd.Timestamp('2020-06-30')
        test_start = pd.Timestamp('2020-07-01')
        test_end = pd.Timestamp('2020-12-31')
        
        # Apply Monte Carlo replacement
        modified_data = evaluator.apply_monte_carlo_replacement(
            original_data=market_data,
            universe=extreme_assets,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            trial_number=99
        )
        
        # Get replacement info
        replacement_info = evaluator.asset_replacement_manager.get_replacement_info()
        
        # CRITICAL VALIDATIONS for extreme conditions
        print("\n" + "="*70)
        print("EXTREME CONDITIONS PROPERTY PRESERVATION")
        print("="*70)
        
        validator = SyntheticDataValidator(monte_carlo_config['validation_config'])
        
        for asset in replacement_info.selected_assets:
            print(f"\nValidating {asset} under extreme conditions:")
            
            # Get test period data
            test_mask = (market_data[asset].index >= test_start) & (market_data[asset].index <= test_end)
            original_test = market_data[asset].loc[test_mask]
            synthetic_test = modified_data[asset].loc[test_mask]
            
            # Calculate returns
            original_returns = original_test['Close'].pct_change(fill_method=None).dropna()
            synthetic_returns = synthetic_test['Close'].pct_change(fill_method=None).dropna()
            
            if len(original_returns) > 20 and len(synthetic_returns) > 20:
                # Calculate extreme statistics
                original_stats = {
                    'volatility': original_returns.std(),
                    'min_return': original_returns.min(),
                    'max_return': original_returns.max(),
                    'skewness': original_returns.skew(),
                    'kurtosis': original_returns.kurtosis()
                }
                
                synthetic_stats = {
                    'volatility': synthetic_returns.std(),
                    'min_return': synthetic_returns.min(),
                    'max_return': synthetic_returns.max(),
                    'skewness': synthetic_returns.skew(),
                    'kurtosis': synthetic_returns.kurtosis()
                }
                
                # CRITICAL ASSERTIONS for extreme conditions
                
                # 1. Volatility should be preserved within reasonable bounds
                vol_ratio = synthetic_stats['volatility'] / original_stats['volatility']
                assert 0.5 < vol_ratio < 2.0, (
                    f"Extreme volatility not preserved for {asset}: "
                    f"Original={original_stats['volatility']:.4f}, "
                    f"Synthetic={synthetic_stats['volatility']:.4f}"
                )
                
                # 2. Extreme values should be possible
                if original_stats['min_return'] < -0.05:  # Had crashes
                    assert synthetic_stats['min_return'] < -0.02, (
                        f"Crash potential not preserved for {asset}: "
                        f"Original min={original_stats['min_return']:.4f}, "
                        f"Synthetic min={synthetic_stats['min_return']:.4f}"
                    )
                
                if original_stats['max_return'] > 0.05:  # Had big gains
                    assert synthetic_stats['max_return'] > 0.02, (
                        f"Gain potential not preserved for {asset}: "
                        f"Original max={original_stats['max_return']:.4f}, "
                        f"Synthetic max={synthetic_stats['max_return']:.4f}"
                    )
                
                # 3. Fat tail properties should be preserved
                if original_stats['kurtosis'] > 5:  # Very fat tails
                    assert synthetic_stats['kurtosis'] > 2, (
                        f"Extreme fat tails not preserved for {asset}: "
                        f"Original kurtosis={original_stats['kurtosis']:.2f}, "
                        f"Synthetic kurtosis={synthetic_stats['kurtosis']:.2f}"
                    )
                
                print(f"  ✓ Volatility preserved: {vol_ratio:.3f}")
                print(f"  ✓ Min return: {original_stats['min_return']:.4f} → {synthetic_stats['min_return']:.4f}")
                print(f"  ✓ Max return: {original_stats['max_return']:.4f} → {synthetic_stats['max_return']:.4f}")
                print(f"  ✓ Kurtosis: {original_stats['kurtosis']:.2f} → {synthetic_stats['kurtosis']:.2f}")
        
        print("\n" + "="*70)
        print("✓ ALL EXTREME CONDITIONS TESTS PASSED!")
        print("✓ PROPERTY PRESERVATION ROBUST TO EXTREME MARKET CONDITIONS!")
        print("="*70)


if __name__ == "__main__":
    # Run end-to-end property preservation tests
    print("Running CRITICAL end-to-end property preservation tests...")
    pytest.main([__file__, "-v", "-s"]) 