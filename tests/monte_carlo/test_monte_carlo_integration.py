"""
Monte Carlo Integration Tests

This module contains comprehensive integration tests for the entire Monte Carlo
synthetic data generation system, testing the integration between all components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import logging
import tempfile
import os

from src.portfolio_backtester.monte_carlo import (
    SyntheticDataGenerator,
    AssetReplacementManager,
    MonteCarloSimulator
)
from src.portfolio_backtester.monte_carlo.validation_metrics import SyntheticDataValidator


class TestMonteCarloSystemIntegration:
    """Integration tests for the complete Monte Carlo system."""
    
    @pytest.fixture
    def system_config(self):
        """Complete system configuration for integration testing."""
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
                'validation_tolerance': 0.3
            },
            'validation_config': {
                'enable_validation': True,
                'basic_stats_tolerance': 0.3,
                'ks_test_pvalue_threshold': 0.01,
                'autocorr_max_deviation': 0.2,
                'volatility_clustering_threshold': 0.03,
                'tail_index_tolerance': 0.7,
                'extreme_value_tolerance': 0.7,
                'overall_quality_threshold': 0.5
            },
            'jump_diffusion': {
                'enable': False
            },
            'random_seed': 42
        }
    
    @pytest.fixture
    def market_universe(self):
        """Market universe for testing."""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    @pytest.fixture
    def realistic_market_data(self, market_universe):
        """Generate realistic market data for multiple assets."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        market_data = {}
        
        for i, asset in enumerate(market_universe):
            # Use different seed for each asset
            np.random.seed(42 + i)
            
            # Generate GARCH-like returns with asset-specific characteristics
            returns = []
            volatility = 0.02 + (i * 0.005)  # Different base volatility per asset
            
            for j in range(len(dates)):
                if j > 0:
                    # Volatility clustering with asset-specific parameters
                    alpha = 0.05 + (i * 0.01)
                    beta = 0.9 - (i * 0.02)
                    volatility = 0.0001 + alpha * (returns[j-1]**2) + beta * volatility
                
                # Use Student-t for fat tails with asset-specific df
                df = 5 + (i % 3)  # Different tail fatness
                daily_return = np.random.standard_t(df=df) * np.sqrt(volatility)
                returns.append(daily_return)
            
            # Convert to prices
            prices = 100 * (1 + i * 0.5) * np.exp(np.cumsum(returns))  # Different starting prices
            
            # Create realistic OHLC data
            market_data[asset] = pd.DataFrame({
                'Open': prices * np.random.uniform(0.995, 1.005, len(prices)),
                'High': np.maximum(prices * np.random.uniform(0.995, 1.005, len(prices)), 
                                 prices) * np.random.uniform(1.0, 1.03, len(prices)),
                'Low': np.minimum(prices * np.random.uniform(0.995, 1.005, len(prices)), 
                                prices) * np.random.uniform(0.97, 1.0, len(prices)),
                'Close': prices,
                'Volume': np.random.lognormal(15, 0.5, len(prices))  # Realistic volume
            }, index=dates)
        
        return market_data
    
    def test_end_to_end_monte_carlo_workflow(self, system_config, market_universe, realistic_market_data):
        """Test complete end-to-end Monte Carlo workflow."""
        # Initialize components
        asset_manager = AssetReplacementManager(system_config)
        validator = SyntheticDataValidator(system_config)
        
        # Define test period
        test_start = pd.Timestamp('2020-09-01')
        test_end = pd.Timestamp('2020-12-31')
        
        # Step 1: Create Monte Carlo dataset
        modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
            original_data=realistic_market_data,
            universe=market_universe,
            test_start=test_start,
            test_end=test_end,
            run_id="integration_test"
        )
        
        # Validate the dataset creation
        assert len(modified_data) == len(realistic_market_data)
        assert len(replacement_info.selected_assets) > 0
        assert replacement_info.replacement_percentage == 0.15
        
        # Step 2: Validate synthetic data quality for replaced assets
        validation_results = {}
        for asset in replacement_info.selected_assets:
            if asset in realistic_market_data:
                # Get original and synthetic data for validation
                original_returns = realistic_market_data[asset]['Close'].pct_change(fill_method=None).dropna()
                synthetic_returns = modified_data[asset]['Close'].pct_change(fill_method=None).dropna()
                
                # Focus on test period for validation
                test_mask = (original_returns.index >= test_start) & (original_returns.index <= test_end)
                original_test = original_returns.loc[test_mask]
                synthetic_test = synthetic_returns.loc[test_mask]
                
                if len(original_test) > 50 and len(synthetic_test) > 50:  # Sufficient data
                    validation_results[asset] = validator.validate_synthetic_data(
                        original_test, synthetic_test, asset
                    )
        
        # Step 3: Analyze validation results
        assert len(validation_results) > 0
        
        for asset, results in validation_results.items():
            # Check that all validation tests were performed
            expected_tests = ['basic_stats', 'distribution', 'autocorrelation', 
                            'volatility_clustering', 'fat_tails', 'extreme_values', 
                            'overall_quality']
            
            for test_name in expected_tests:
                assert test_name in results
            
            # Generate validation report
            report = validator.generate_validation_report(results, asset)
            assert len(report) > 0
            assert asset in report
        
        # Step 4: Check replacement statistics
        replacement_stats = asset_manager.get_replacement_statistics()
        assert replacement_stats['total_runs'] == 1
        assert replacement_stats['avg_replacement_percentage'] == 0.15
        assert len(replacement_stats['asset_replacement_counts']) > 0
    
    def test_multiple_optimization_runs_simulation(self, system_config, market_universe, realistic_market_data):
        """Test multiple optimization runs with different random seeds."""
        asset_manager = AssetReplacementManager(system_config)
        
        # Simulate multiple optimization runs
        num_runs = 5
        test_start = pd.Timestamp('2020-09-01')
        test_end = pd.Timestamp('2020-10-31')
        
        run_results = []
        
        for run_id in range(num_runs):
            # Each run gets different random seed
            run_seed = 42 + run_id
            
            modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
                original_data=realistic_market_data,
                universe=market_universe,
                test_start=test_start,
                test_end=test_end,
                run_id=f"run_{run_id}",
                random_seed=run_seed
            )
            
            run_results.append({
                'run_id': run_id,
                'selected_assets': replacement_info.selected_assets,
                'modified_data': modified_data,
                'replacement_info': replacement_info
            })
        
        # Analyze results across runs
        assert len(run_results) == num_runs
        
        # Check that different runs selected different assets (with high probability)
        selected_assets_per_run = [result['selected_assets'] for result in run_results]
        unique_selections = set(frozenset(assets) for assets in selected_assets_per_run)
        assert len(unique_selections) > 1  # Should have some variation
        
        # Check replacement statistics
        replacement_stats = asset_manager.get_replacement_statistics()
        assert replacement_stats['total_runs'] == num_runs
        assert replacement_stats['total_assets_replaced'] > 0
        
        # Check that all assets were replaced at least once across runs
        all_replaced_assets = set()
        for result in run_results:
            all_replaced_assets.update(result['selected_assets'])
        
        assert len(all_replaced_assets) >= len(market_universe) * 0.15  # At least some coverage
    
    def test_walk_forward_optimization_integration(self, system_config, market_universe, realistic_market_data):
        """Test integration with walk-forward optimization windows."""
        asset_manager = AssetReplacementManager(system_config)
        
        # Simulate multiple WFO windows
        wfo_windows = [
            ('2020-01-01', '2020-06-30', '2020-07-01', '2020-09-30'),
            ('2020-03-01', '2020-08-31', '2020-09-01', '2020-11-30'),
            ('2020-06-01', '2020-11-30', '2020-12-01', '2021-02-28')
        ]
        
        window_results = []
        
        for window_idx, (train_start, train_end, test_start, test_end) in enumerate(wfo_windows):
            # Convert to timestamps
            test_start_ts = pd.Timestamp(test_start)
            test_end_ts = pd.Timestamp(test_end)
            
            # Create Monte Carlo dataset for this window
            modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
                original_data=realistic_market_data,
                universe=market_universe,
                test_start=test_start_ts,
                test_end=test_end_ts,
                run_id=f"window_{window_idx}",
                random_seed=42 + window_idx
            )
            
            window_results.append({
                'window_idx': window_idx,
                'test_period': (test_start_ts, test_end_ts),
                'selected_assets': replacement_info.selected_assets,
                'modified_data': modified_data
            })
        
        # Validate results
        assert len(window_results) == len(wfo_windows)
        
        # Check that each window has replacements
        for result in window_results:
            assert len(result['selected_assets']) > 0
            assert len(result['modified_data']) == len(realistic_market_data)
        
        # Check replacement statistics across windows
        replacement_stats = asset_manager.get_replacement_statistics()
        assert replacement_stats['total_runs'] == len(wfo_windows)
    
    def test_data_structure_consistency(self, system_config, market_universe, realistic_market_data):
        """Test that data structure consistency is maintained throughout the process."""
        asset_manager = AssetReplacementManager(system_config)
        
        test_start = pd.Timestamp('2020-09-01')
        test_end = pd.Timestamp('2020-09-30')
        
        modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
            original_data=realistic_market_data,
            universe=market_universe,
            test_start=test_start,
            test_end=test_end
        )
        
        # Check data structure consistency
        for asset in market_universe:
            original_asset_data = realistic_market_data[asset]
            modified_asset_data = modified_data[asset]
            
            # Same index
            assert original_asset_data.index.equals(modified_asset_data.index)
            
            # Same columns
            assert original_asset_data.columns.equals(modified_asset_data.columns)
            
            # Same data types
            for col in original_asset_data.columns:
                assert original_asset_data[col].dtype == modified_asset_data[col].dtype
            
            # Check OHLC relationships in modified data
            test_mask = (modified_asset_data.index >= test_start) & (modified_asset_data.index <= test_end)
            test_data = modified_asset_data.loc[test_mask]
            
            if len(test_data) > 0:
                # OHLC relationships should be maintained
                assert (test_data['High'] >= test_data['Close']).all()
                assert (test_data['High'] >= test_data['Open']).all()
                assert (test_data['Low'] <= test_data['Close']).all()
                assert (test_data['Low'] <= test_data['Open']).all()
                
                # All prices should be positive
                price_cols = ['Open', 'High', 'Low', 'Close']
                for col in price_cols:
                    assert (test_data[col] > 0).all()
    
    def test_error_handling_and_robustness(self, system_config, market_universe):
        """Test error handling and system robustness."""
        asset_manager = AssetReplacementManager(system_config)
        
        # Test 1: Missing asset data
        incomplete_data = {
            'AAPL': pd.DataFrame({
                'Close': [100, 101, 102],
                'Open': [99, 100, 101],
                'High': [101, 102, 103],
                'Low': [98, 99, 100]
            }, index=pd.date_range('2020-01-01', periods=3, freq='D'))
        }
        
        # Should handle gracefully
        modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
            original_data=incomplete_data,
            universe=['AAPL', 'MSFT'],  # MSFT missing
            test_start=pd.Timestamp('2020-01-02'),
            test_end=pd.Timestamp('2020-01-03')
        )
        
        assert len(modified_data) == 1  # Only AAPL
        assert 'AAPL' in modified_data
        
        # Test 2: Insufficient historical data
        short_data = {
            'AAPL': pd.DataFrame({
                'Close': [100, 101],
                'Open': [99, 100],
                'High': [101, 102],
                'Low': [98, 99]
            }, index=pd.date_range('2020-01-01', periods=2, freq='D'))
        }
        
        # Should handle gracefully (may use fallback methods)
        modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
            original_data=short_data,
            universe=['AAPL'],
            test_start=pd.Timestamp('2020-01-02'),
            test_end=pd.Timestamp('2020-01-02')
        )
        
        assert len(modified_data) == 1
        assert 'AAPL' in modified_data
    
    def test_performance_and_scalability(self, system_config, market_universe, realistic_market_data):
        """Test performance and scalability with larger datasets."""
        asset_manager = AssetReplacementManager(system_config)
        
        # Test with larger universe
        large_universe = market_universe + [f'STOCK_{i}' for i in range(10)]
        
        # Create additional data for new stocks
        extended_data = realistic_market_data.copy()
        for i in range(10):
            stock_name = f'STOCK_{i}'
            # Simple data generation for performance test
            dates = realistic_market_data['AAPL'].index
            prices = 100 + np.cumsum(np.random.normal(0, 0.02, len(dates)))
            
            extended_data[stock_name] = pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.lognormal(15, 0.5, len(dates))
            }, index=dates)
        
        # Test performance
        import time
        start_time = time.time()
        
        modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
            original_data=extended_data,
            universe=large_universe,
            test_start=pd.Timestamp('2020-09-01'),
            test_end=pd.Timestamp('2020-09-30')
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 30  # 30 seconds max
        
        # Validate results
        assert len(modified_data) == len(extended_data)
        assert len(replacement_info.selected_assets) > 0
    
    def test_configuration_flexibility(self, market_universe, realistic_market_data):
        """Test system flexibility with different configurations."""
        
        # Test 1: High replacement percentage
        high_replacement_config = {
            'enable_synthetic_data': True,
            'replacement_percentage': 0.8,  # 80% replacement
            'min_historical_observations': 50,
            'garch_config': {
                'model_type': 'GARCH',
                'p': 1,
                'q': 1,
                'distribution': 'normal',  # Different distribution
                'bounds': {
                    'omega': [1e-6, 1.0],
                    'alpha': [0.01, 0.3],
                    'beta': [0.5, 0.99],
                    'nu': [2.1, 30.0]
                }
            },
            'generation_config': {
                'buffer_multiplier': 1.5,
                'max_attempts': 2,
                'validation_tolerance': 0.5
            },
            'validation_config': {
                'enable_validation': False  # Disabled for speed
            },
            'random_seed': 123
        }
        
        manager = AssetReplacementManager(high_replacement_config)
        
        modified_data, replacement_info = manager.create_monte_carlo_dataset(
            original_data=realistic_market_data,
            universe=market_universe,
            test_start=pd.Timestamp('2020-09-01'),
            test_end=pd.Timestamp('2020-09-30')
        )
        
        # Should replace 80% of assets
        expected_replacements = int(len(market_universe) * 0.8)
        assert len(replacement_info.selected_assets) == expected_replacements
        
        # Test 2: Disabled synthetic data
        disabled_config = {
            'enable_synthetic_data': False,
            'replacement_percentage': 0.2
        }
        
        manager_disabled = AssetReplacementManager(disabled_config)
        
        modified_data_disabled, replacement_info_disabled = manager_disabled.create_monte_carlo_dataset(
            original_data=realistic_market_data,
            universe=market_universe,
            test_start=pd.Timestamp('2020-09-01'),
            test_end=pd.Timestamp('2020-09-30')
        )
        
        # Should return original data unchanged
        assert modified_data_disabled == realistic_market_data
        assert len(replacement_info_disabled.selected_assets) == 0
    
    def test_logging_and_monitoring(self, system_config, market_universe, realistic_market_data):
        """Test logging and monitoring capabilities."""
        
        with patch('src.portfolio_backtester.monte_carlo.asset_replacement.logger') as mock_logger:
            asset_manager = AssetReplacementManager(system_config)
            
            modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
                original_data=realistic_market_data,
                universe=market_universe,
                test_start=pd.Timestamp('2020-09-01'),
                test_end=pd.Timestamp('2020-09-30'),
                run_id="logging_test"
            )
            
            # Check that appropriate logging occurred
            assert mock_logger.info.called
            
            # Check that replacement info was logged
            log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Selected" in call and "assets for replacement" in call for call in log_calls)
    
    def test_memory_management(self, system_config, market_universe, realistic_market_data):
        """Test memory management and cleanup."""
        asset_manager = AssetReplacementManager(system_config)
        
        # Create multiple datasets to test memory usage
        for i in range(3):
            modified_data, replacement_info = asset_manager.create_monte_carlo_dataset(
                original_data=realistic_market_data,
                universe=market_universe,
                test_start=pd.Timestamp('2020-09-01'),
                test_end=pd.Timestamp('2020-09-30'),
                run_id=f"memory_test_{i}"
            )
            
            # Force garbage collection would be here in real scenario
            # del modified_data  # Cleanup
        
        # Test cache management
        initial_cache_size = len(asset_manager._asset_stats_cache)
        
        # Clear cache
        asset_manager.clear_cache()
        assert len(asset_manager._asset_stats_cache) == 0
        
        # Reset history
        asset_manager.reset_history()
        assert len(asset_manager.replacement_history) == 0 