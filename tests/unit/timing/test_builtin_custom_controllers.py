"""
Tests for built-in custom timing controllers.
Split from test_configuration_extensibility.py for better organization.
"""

import pytest
import pandas as pd
from unittest.mock import Mock
from src.portfolio_backtester.timing.custom_timing_registry import TimingControllerFactory


class TestBuiltinCustomControllers:
    """Test built-in custom timing controllers."""
    
    def test_adaptive_timing_controller(self):
        """Test adaptive timing controller."""
        config = {
            'mode': 'custom',
            'custom_controller_class': 'adaptive_timing',
            'custom_controller_params': {
                'volatility_threshold': 0.03,
                'base_frequency': 'M'
            }
        }
        
        controller = TimingControllerFactory.create_controller(config)
        assert controller.__class__.__name__ == 'AdaptiveTimingController'
        assert controller.volatility_threshold == 0.03
        assert controller.base_frequency == 'M'
    
    def test_momentum_timing_controller(self):
        """Test momentum timing controller."""
        config = {
            'mode': 'custom',
            'custom_controller_class': 'momentum_timing',
            'custom_controller_params': {
                'momentum_period': 30
            }
        }
        
        controller = TimingControllerFactory.create_controller(config)
        assert controller.__class__.__name__ == 'MomentumTimingController'
        assert controller.momentum_period == 30
    
    def test_adaptive_controller_signal_generation(self):
        """Test adaptive controller signal generation."""
        config = {
            'base_frequency': 'M',
            'high_vol_frequency': 'W',
            'low_vol_frequency': 'Q'
        }
        
        from src.portfolio_backtester.timing.custom_timing_registry import AdaptiveTimingController
        controller = AdaptiveTimingController(config, volatility_threshold=0.02)
        
        # Test with mock strategy
        mock_strategy = Mock()
        test_date = pd.Timestamp('2023-01-02')  # Monday
        
        # Test without volatility method (should use base frequency)
        result = controller.should_generate_signal(test_date, mock_strategy)
        assert isinstance(result, bool)
    
    def test_momentum_controller_signal_generation(self):
        """Test momentum controller signal generation."""
        config = {}
        
        from src.portfolio_backtester.timing.custom_timing_registry import MomentumTimingController
        controller = MomentumTimingController(config, momentum_period=20)
        
        # Test signal generation
        mock_strategy = Mock()
        test_date = pd.Timestamp('2023-01-02')  # Monday
        
        result = controller.should_generate_signal(test_date, mock_strategy)
        assert result == True  # Should be True for Monday


if __name__ == '__main__':
    pytest.main([__file__, '-v'])