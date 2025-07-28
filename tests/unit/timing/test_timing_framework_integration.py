"""
Tests for integration of new features with existing timing framework.
Split from test_configuration_extensibility.py for better organization.
"""

import pytest
import os
import yaml
from src.portfolio_backtester.timing.config_schema import validate_timing_config


class TestIntegrationWithExistingFramework:
    """Test integration of new features with existing timing framework."""
    
    def test_enhanced_backward_compatibility(self):
        """Test enhanced backward compatibility with validation."""
        from src.portfolio_backtester.timing.backward_compatibility import ensure_backward_compatibility
        
        # Test valid legacy config
        legacy_config = {
            'strategy_params': {'lookback_period': 252},
            'rebalance_frequency': 'M'
        }
        
        migrated = ensure_backward_compatibility(legacy_config)
        assert 'timing_config' in migrated
        assert migrated['timing_config']['mode'] == 'time_based'
        assert migrated['timing_config']['rebalance_frequency'] == 'M'
        assert 'rebalance_frequency' not in migrated  # Should be removed
    
    def test_configuration_examples_validation(self):
        """Test that configuration examples are valid."""
        # Load and validate examples
        examples_file = 'config/timing_examples.yaml'
        
        if os.path.exists(examples_file):
            with open(examples_file, 'r') as f:
                examples = yaml.safe_load(f)
            
            # Test a few key examples
            test_examples = [
                'basic_monthly_strategy',
                'daily_signal_strategy',
                'adaptive_strategy'
            ]
            
            for example_name in test_examples:
                if example_name in examples:
                    config = examples[example_name]
                    if 'timing_config' in config:
                        errors = validate_timing_config(config, raise_on_error=False)
                        error_count = sum(1 for e in errors if e.severity == 'error')
                        assert error_count == 0, f"Example {example_name} has validation errors: {errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])