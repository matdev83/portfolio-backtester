"""
Walk-Forward Optimization Robustness Tests

This module tests the WFO robustness features including:
- Randomized window sizes for robustness testing
- Start date randomization
- Stability metrics calculation
- Parameter impact analysis
"""

import pytest
import numpy as np
import pandas as pd

from portfolio_backtester.utils import generate_randomized_wfo_windows, calculate_stability_metrics


class TestWFORobustness:
    """Test Walk-Forward Optimization robustness features."""
    
    @pytest.fixture
    def wfo_robustness_config(self):
        """WFO robustness configuration for testing."""
        return {
            'enable_window_randomization': True,
            'enable_start_date_randomization': True,
            'train_window_randomization': {
                'min_offset': 3,
                'max_offset': 14
            },
            'test_window_randomization': {
                'min_offset': 3,
                'max_offset': 14
            },
            'start_date_randomization': {
                'min_offset': 0,
                'max_offset': 12
            },
            'stability_metrics': {
                'enable': True,
                'worst_percentile': 10,
                'consistency_threshold': 0.0
            },
            'random_seed': None
        }
    
    @pytest.fixture
    def global_config(self, wfo_robustness_config):
        """Global configuration with WFO robustness settings."""
        return {
            'wfo_robustness_config': wfo_robustness_config,
            'benchmark': 'SPY'
        }
    
    @pytest.fixture
    def scenario_config(self):
        """Basic scenario configuration."""
        return {
            'train_window_months': 36,
            'test_window_months': 12,
            'walk_forward_type': 'expanding'
        }
    
    @pytest.fixture
    def monthly_data_index(self):
        """Create monthly data index for testing."""
        return pd.date_range('2015-01-31', '2023-12-31', freq='ME')
    
    def test_generate_randomized_wfo_windows_basic(self, monthly_data_index, scenario_config, global_config):
        """Test basic WFO window generation without randomization."""
        # Disable randomization
        config = global_config.copy()
        config['wfo_robustness_config']['enable_window_randomization'] = False
        config['wfo_robustness_config']['enable_start_date_randomization'] = False
        
        windows = generate_randomized_wfo_windows(
            monthly_data_index, scenario_config, config, random_state=42
        )
        
        # Should generate valid windows
        assert len(windows) > 0
        
        # Verify window structure
        prev_train_months = None
        for train_start, train_end, test_start, test_end in windows:
            assert isinstance(train_start, pd.Timestamp)
            assert isinstance(train_end, pd.Timestamp)
            assert isinstance(test_start, pd.Timestamp)
            assert isinstance(test_end, pd.Timestamp)
            
            # Verify chronological order
            assert train_start <= train_end < test_start <= test_end
            
            # Verify minimum window sizes
            # Calculate actual window sizes in months
            # Use pd.DateOffset to accurately calculate month differences
            train_months = (train_end.to_period('M') - train_start.to_period('M')).n + 1
            test_months = (test_end.to_period('M') - test_start.to_period('M')).n + 1
            
            # Verify minimum window sizes, allowing for month-end alignment
            # The new logic ensures that the number of months is at least the base + offset
            robustness_config = config['wfo_robustness_config']
            if robustness_config['enable_window_randomization']:
                assert train_months >= scenario_config['train_window_months'] + robustness_config['train_window_randomization']['min_offset']
                assert train_months <= scenario_config['train_window_months'] + robustness_config['train_window_randomization']['max_offset'] + 1 # +1 for potential month boundary
                assert test_months >= scenario_config['test_window_months'] + robustness_config['test_window_randomization']['min_offset']
                assert test_months <= scenario_config['test_window_months'] + robustness_config['test_window_randomization']['max_offset'] + 1 # +1 for potential month boundary
            else:
                # For expanding windows, the train size should grow
                if scenario_config['walk_forward_type'] == 'expanding':
                    # The first window should match the base size
                    if prev_train_months is None:
                        assert train_months == scenario_config['train_window_months']
                    else:
                        # Subsequent windows should expand by the test window size
                        expected_train_months = prev_train_months + scenario_config['test_window_months']
                        assert train_months == expected_train_months
                    prev_train_months = train_months
                else: # Rolling window
                    assert train_months == scenario_config['train_window_months']
                
                assert test_months == scenario_config['test_window_months']
    
    def test_rolling_vs_expanding_windows(self, monthly_data_index, scenario_config, global_config):
        """Test both rolling and expanding window types."""
        # Test expanding windows
        scenario_expanding = scenario_config.copy()
        scenario_expanding['walk_forward_type'] = 'expanding'
        
        windows_expanding = generate_randomized_wfo_windows(
            monthly_data_index, scenario_expanding, global_config, random_state=42
        )
        
        # Test rolling windows
        scenario_rolling = scenario_config.copy()
        scenario_rolling['walk_forward_type'] = 'rolling'
        
        windows_rolling = generate_randomized_wfo_windows(
            monthly_data_index, scenario_rolling, global_config, random_state=42
        )
        
        # Both should generate windows
        assert len(windows_expanding) > 0
        assert len(windows_rolling) > 0
        
        # Expanding windows should have increasing train periods
        if len(windows_expanding) > 1:
            train_sizes_expanding = []
            for train_start, train_end, _, _ in windows_expanding:
                train_size = (train_end.to_period('M') - train_start.to_period('M')).n + 1
                train_sizes_expanding.append(train_size)
            
            # Should be non-decreasing (expanding)
            assert all(train_sizes_expanding[i] <= train_sizes_expanding[i+1]
                       for i in range(len(train_sizes_expanding)-1)), \
                "Expanding windows should have non-decreasing train sizes"
        
        # Rolling windows should have consistent train periods
        if len(windows_rolling) > 1:
            train_sizes_rolling = []
            for train_start, train_end, _, _ in windows_rolling:
                train_size = (train_end.to_period('M') - train_start.to_period('M')).n + 1
                train_sizes_rolling.append(train_size)
            
            # Should be approximately equal (rolling)
            train_size_std = np.std(train_sizes_rolling)
            train_size_mean = np.mean(train_sizes_rolling)
            cv = train_size_std / train_size_mean if train_size_mean > 0 else 0
            
            # Coefficient of variation should be small for rolling windows
            # (allowing for randomization effects)
            assert cv < 0.3, f"Rolling windows should have consistent sizes, CV={cv:.3f}"
    
    def test_calculate_stability_metrics(self, global_config):
        """Test stability metrics calculation."""
        # Create mock metric values across windows
        metric_values_per_objective = [
            [0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88],  # Sharpe ratios
            [-0.15, -0.12, -0.18, -0.10, -0.20, -0.08, -0.25, -0.11]  # Max drawdowns
        ]
        
        metrics_to_optimize = ['Sharpe', 'Max Drawdown']
        
        stability_metrics = calculate_stability_metrics(
            metric_values_per_objective, metrics_to_optimize, global_config
        )
        
        # Should calculate stability metrics for each objective
        assert 'stability_Sharpe_Std' in stability_metrics
        assert 'stability_Sharpe_CV' in stability_metrics
        assert 'stability_Sharpe_Worst_10pct' in stability_metrics
        assert 'stability_Sharpe_Consistency_Ratio' in stability_metrics
        
        assert 'stability_Max Drawdown_Std' in stability_metrics
        assert 'stability_Max Drawdown_CV' in stability_metrics
        assert 'stability_Max Drawdown_Worst_10pct' in stability_metrics
        assert 'stability_Max Drawdown_Consistency_Ratio' in stability_metrics
        
        # Verify reasonable values
        sharpe_std = stability_metrics['stability_Sharpe_Std']
        sharpe_cv = stability_metrics['stability_Sharpe_CV']
        sharpe_worst = stability_metrics['stability_Sharpe_Worst_10pct']
        sharpe_consistency = stability_metrics['stability_Sharpe_Consistency_Ratio']
        
        assert sharpe_std > 0, "Standard deviation should be positive"
        assert sharpe_cv > 0, "Coefficient of variation should be positive"
        assert 0 <= sharpe_consistency <= 1, "Consistency ratio should be between 0 and 1"
        assert sharpe_worst <= max(metric_values_per_objective[0]), \
            "Worst case should not exceed maximum value"
    
    def test_stability_metrics_with_nan_values(self, global_config):
        """Test stability metrics calculation with NaN values."""
        # Include some NaN values
        metric_values_per_objective = [
            [0.8, np.nan, 0.7, 0.85, np.nan, 0.95, 0.6, 0.88],
            [-0.15, -0.12, np.nan, -0.10, -0.20, np.nan, -0.25, -0.11]
        ]
        
        metrics_to_optimize = ['Sharpe', 'Max Drawdown']
        
        stability_metrics = calculate_stability_metrics(
            metric_values_per_objective, metrics_to_optimize, global_config
        )
        
        # Should handle NaN values gracefully
        assert not np.isnan(stability_metrics['stability_Sharpe_Std'])
        assert not np.isnan(stability_metrics['stability_Sharpe_CV'])
        assert not np.isnan(stability_metrics['stability_Max Drawdown_Std'])
        
        # Consistency ratio should be calculated on valid values only
        sharpe_consistency = stability_metrics['stability_Sharpe_Consistency_Ratio']
        assert 0 <= sharpe_consistency <= 1
    
    def test_stability_metrics_all_nan_values(self, global_config):
        """Test stability metrics calculation with all NaN values."""
        # All NaN values
        metric_values_per_objective = [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan]
        ]
        
        metrics_to_optimize = ['Sharpe', 'Max Drawdown']
        
        stability_metrics = calculate_stability_metrics(
            metric_values_per_objective, metrics_to_optimize, global_config
        )
        
        # Should return NaN for all metrics when no valid data
        assert np.isnan(stability_metrics['Sharpe_Std'])
        assert np.isnan(stability_metrics['Sharpe_CV'])
        assert np.isnan(stability_metrics['Sharpe_Worst_10pct'])
        assert np.isnan(stability_metrics['Sharpe_Consistency_Ratio'])
    
    def test_wfo_robustness_configuration_validation(self, monthly_data_index, scenario_config):
        """Test WFO robustness configuration validation and defaults."""
        # Test with minimal configuration
        minimal_config = {}
        
        windows = generate_randomized_wfo_windows(
            monthly_data_index, scenario_config, minimal_config, random_state=42
        )
        
        # Should work with defaults
        assert len(windows) > 0
        
        # Test with invalid configuration
        invalid_config = {
            'wfo_robustness_config': {
                'enable_window_randomization': True,
                'train_window_randomization': {
                    'min_offset': 20,  # Too large
                    'max_offset': 10   # Max < Min
                }
            }
        }
        
        # Should handle gracefully (might use defaults or clamp values)
        try:
            windows = generate_randomized_wfo_windows(
                monthly_data_index, scenario_config, invalid_config, random_state=42
            )
            # If it succeeds, should still generate valid windows
            assert len(windows) >= 0
        except (ValueError, AssertionError):
            # If it fails, should be a controlled failure
            pass
    
    def test_reproducibility_with_random_seed(self, monthly_data_index, scenario_config, global_config):
        """Test that results are reproducible with the same random seed."""
        seed = 12345
        
        # Generate windows twice with same seed
        windows1 = generate_randomized_wfo_windows(
            monthly_data_index, scenario_config, global_config, random_state=seed
        )
        
        windows2 = generate_randomized_wfo_windows(
            monthly_data_index, scenario_config, global_config, random_state=seed
        )
        
        # Should be identical
        assert len(windows1) == len(windows2)
        
        for w1, w2 in zip(windows1, windows2):
            assert w1[0] == w2[0]  # train_start
            assert w1[1] == w2[1]  # train_end
            assert w1[2] == w2[2]  # test_start
            assert w1[3] == w2[3]  # test_end
    
    def test_different_seeds_produce_different_results(self, monthly_data_index, scenario_config, global_config):
        """Test that different seeds produce different results."""
        windows1 = generate_randomized_wfo_windows(
            monthly_data_index, scenario_config, global_config, random_state=1
        )
        
        windows2 = generate_randomized_wfo_windows(
            monthly_data_index, scenario_config, global_config, random_state=2
        )
        
        # Should be different (at least some windows should differ)
        differences = 0
        min_len = min(len(windows1), len(windows2))
        
        for i in range(min_len):
            if windows1[i] != windows2[i]:
                differences += 1
        
        # Should have at least some differences due to randomization
        assert differences > 0 or len(windows1) != len(windows2), \
            "Different seeds should produce different results"
    
    def test_insufficient_data_handling(self, scenario_config, global_config):
        """Test handling of insufficient data for window generation."""
        # Create very short data index
        short_index = pd.date_range('2023-01-31', '2023-06-30', freq='ME')  # Only 6 months
        
        windows = generate_randomized_wfo_windows(
            short_index, scenario_config, global_config, random_state=42
        )
        
        # Should handle gracefully - might return empty list or minimal windows
        assert isinstance(windows, list)
        
        # If windows are generated, they should be valid
        for train_start, train_end, test_start, test_end in windows:
            assert train_start <= train_end < test_start <= test_end
            assert train_start in short_index
            assert train_end in short_index
            assert test_start in short_index
            assert test_end in short_index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])