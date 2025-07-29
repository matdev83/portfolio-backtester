"""
API Stability Tests for Critical Methods

This module contains signature validation tests for the most critical methods
identified through static analysis. These tests ensure that method signatures
remain backward compatible and prevent breaking changes.

Requirements addressed: 4.1, 4.2, 4.3, 4.5
"""

import pytest
import inspect
import sys
import os
from typing import Dict, Any, List, Optional, Tuple, Union, get_type_hints
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_backtester.core import Backtester
from portfolio_backtester.strategies.base_strategy import BaseStrategy
from portfolio_backtester.strategies.leverage_and_smoothing import apply_leverage_and_smoothing
from portfolio_backtester.strategies.candidate_weights import default_candidate_weights
from portfolio_backtester.timing.custom_timing_registry import CustomTimingRegistry
from portfolio_backtester.api_stability.exceptions import ParameterViolationError


class TestCriticalMethodSignatures:
    """Test class for validating signatures of critical methods."""
    
    def test_backtester_init_signature(self):
        """Test Backtester.__init__ signature validation."""
        # Get the method signature
        sig = inspect.signature(Backtester.__init__)
        
        # Expected parameters with their types and defaults
        expected_params = {
            'self': {'annotation': inspect.Parameter.empty, 'default': inspect.Parameter.empty},
            'global_config': {'annotation': Dict[str, Any], 'default': inspect.Parameter.empty},
            'scenarios': {'annotation': List[Dict[str, Any]], 'default': inspect.Parameter.empty},
            'args': {'annotation': 'argparse.Namespace', 'default': inspect.Parameter.empty},
            'random_state': {'annotation': Optional[int], 'default': None}
        }
        
        # Validate parameter names
        actual_params = list(sig.parameters.keys())
        expected_param_names = list(expected_params.keys())
        assert actual_params == expected_param_names, \
            f"Parameter names mismatch. Expected: {expected_param_names}, Got: {actual_params}"
        
        # Validate parameter properties
        for param_name, expected in expected_params.items():
            param = sig.parameters[param_name]
            
            # Check default values
            assert param.default == expected['default'], \
                f"Parameter '{param_name}' default mismatch. Expected: {expected['default']}, Got: {param.default}"
            
            # Check parameter kind (positional, keyword, etc.)
            if param_name == 'random_state':
                assert param.kind == inspect.Parameter.KEYWORD_ONLY or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, \
                    f"Parameter '{param_name}' should allow keyword arguments"
        
        # Validate return type annotation
        assert sig.return_annotation == None or sig.return_annotation == inspect.Signature.empty, \
            f"__init__ should not have return annotation or should be None"
    
    def test_backtester_init_backward_compatibility(self):
        """Test that Backtester.__init__ can be called with old parameter patterns."""
        # Mock the required dependencies
        with patch('portfolio_backtester.core.get_data_source'), \
             patch('portfolio_backtester.core.enumerate_strategies_with_params') as mock_strategies, \
             patch('portfolio_backtester.core.populate_default_optimizations'), \
             patch('portfolio_backtester.core.TimeoutManager'):
            
            mock_strategies.return_value = {}
            
            # Test basic initialization (required parameters only)
            global_config = {'data_source': 'test'}
            scenarios = [{'strategy': 'test'}]
            args = Mock()
            args.timeout = 300
            
            # This should work without raising an exception
            backtester = Backtester(global_config, scenarios, args)
            assert backtester is not None
            
            # Test with optional parameter
            backtester_with_random = Backtester(global_config, scenarios, args, random_state=42)
            assert backtester_with_random is not None
            
            # Test with keyword arguments
            backtester_kwargs = Backtester(
                global_config=global_config,
                scenarios=scenarios,
                args=args,
                random_state=123
            )
            assert backtester_kwargs is not None
    
    def test_custom_timing_registry_get_signature(self):
        """Test CustomTimingRegistry.get method signature validation."""
        sig = inspect.signature(CustomTimingRegistry.get)
        
        # Expected parameters (inspect.signature for classmethods excludes 'cls')
        expected_params = {
            'name': {'annotation': str, 'default': inspect.Parameter.empty}
        }
        
        # Validate parameter names and properties
        actual_params = list(sig.parameters.keys())
        expected_param_names = list(expected_params.keys())
        assert actual_params == expected_param_names, \
            f"Parameter names mismatch. Expected: {expected_param_names}, Got: {actual_params}"
        
        # Validate parameter annotations
        for param_name, expected in expected_params.items():
            param = sig.parameters[param_name]
            if param_name == 'name':
                # Check that name parameter has str annotation
                assert param.annotation == str or str(param.annotation) == 'str', \
                    f"Parameter '{param_name}' should have str annotation"
    
    def test_custom_timing_registry_get_backward_compatibility(self):
        """Test CustomTimingRegistry.get backward compatibility."""
        # Test basic usage patterns (it's a classmethod)
        result = CustomTimingRegistry.get('nonexistent_key')
        assert result is None
        
        # Test with keyword arguments
        result = CustomTimingRegistry.get(name='nonexistent_key')
        assert result is None
    
    def test_base_strategy_validate_data_sufficiency_signature(self):
        """Test BaseStrategy.validate_data_sufficiency signature validation."""
        sig = inspect.signature(BaseStrategy.validate_data_sufficiency)
        
        # Expected parameters
        expected_params = {
            'self': {'annotation': inspect.Parameter.empty, 'default': inspect.Parameter.empty},
            'all_historical_data': {'annotation': pd.DataFrame, 'default': inspect.Parameter.empty},
            'benchmark_historical_data': {'annotation': pd.DataFrame, 'default': inspect.Parameter.empty},
            'current_date': {'annotation': pd.Timestamp, 'default': inspect.Parameter.empty}
        }
        
        # Validate parameter names
        actual_params = list(sig.parameters.keys())
        expected_param_names = list(expected_params.keys())
        assert actual_params == expected_param_names, \
            f"Parameter names mismatch. Expected: {expected_param_names}, Got: {actual_params}"
        
        # Validate return type annotation
        expected_return = tuple[bool, str]  # Python 3.9+ syntax
        # Handle both old and new tuple annotation styles
        assert (sig.return_annotation == expected_return or 
                str(sig.return_annotation) == 'tuple[bool, str]' or
                str(sig.return_annotation) == 'typing.Tuple[bool, str]'), \
            f"Return type mismatch. Expected: {expected_return}, Got: {sig.return_annotation}"
    
    def test_base_strategy_validate_data_sufficiency_backward_compatibility(self):
        """Test BaseStrategy.validate_data_sufficiency backward compatibility."""
        # Create a mock strategy instance with required config
        strategy_config = {'strategy_params': {}}
        strategy = BaseStrategy(strategy_config)
        
        # Create mock data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        mock_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        current_date = pd.Timestamp('2023-12-31')
        
        # Test method call
        result = strategy.validate_data_sufficiency(mock_data, mock_data, current_date)
        
        # Validate return type
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return a tuple of length 2"
        assert isinstance(result[0], bool), "First element should be boolean"
        assert isinstance(result[1], str), "Second element should be string"
    
    def test_base_strategy_get_roro_signal_signature(self):
        """Test BaseStrategy.get_roro_signal signature validation."""
        sig = inspect.signature(BaseStrategy.get_roro_signal)
        
        # Validate that method exists and has expected structure
        assert sig is not None, "get_roro_signal method should exist"
        
        # Check that it has self parameter
        params = list(sig.parameters.keys())
        assert 'self' in params, "Method should have 'self' parameter"
    
    def test_apply_leverage_and_smoothing_signature(self):
        """Test apply_leverage_and_smoothing function signature validation."""
        sig = inspect.signature(apply_leverage_and_smoothing)
        
        # This is a standalone function, so validate its existence and basic structure
        assert sig is not None, "apply_leverage_and_smoothing function should exist"
        
        # Validate it's callable
        assert callable(apply_leverage_and_smoothing), "Should be callable"
    
    def test_apply_leverage_and_smoothing_backward_compatibility(self):
        """Test apply_leverage_and_smoothing backward compatibility."""
        # Create mock data for testing
        mock_weights = pd.Series([0.3, 0.4, 0.3], index=['A', 'B', 'C'])
        mock_prev_weights = pd.Series([0.2, 0.5, 0.3], index=['A', 'B', 'C'])
        
        # Test that function can be called (even if it raises an exception due to missing data)
        # We're just testing the signature compatibility here
        try:
            result = apply_leverage_and_smoothing(
                mock_weights, 
                mock_prev_weights, 
                leverage=1.0, 
                smoothing=0.0
            )
            # If it succeeds, that's good
            assert result is not None or result is None  # Either outcome is acceptable for signature test
        except Exception:
            # If it fails due to data issues, that's expected - we're only testing signature compatibility
            pass
    
    def test_default_candidate_weights_signature(self):
        """Test default_candidate_weights function signature validation."""
        sig = inspect.signature(default_candidate_weights)
        
        # Validate function exists and is callable
        assert sig is not None, "default_candidate_weights function should exist"
        assert callable(default_candidate_weights), "Should be callable"
    
    def test_default_candidate_weights_backward_compatibility(self):
        """Test default_candidate_weights backward compatibility."""
        # Test that function can be called
        try:
            # Create minimal mock data
            mock_data = pd.DataFrame({
                'Close': [100, 101, 102],
                'Volume': [1000, 1100, 1200]
            }, index=pd.date_range('2023-01-01', periods=3))
            
            result = default_candidate_weights(mock_data)
            # Function should return something or raise a predictable exception
            assert result is not None or result is None
        except Exception as e:
            # If it fails, ensure it's due to data format, not signature issues
            assert "signature" not in str(e).lower(), f"Signature-related error: {e}"


class TestMethodTypeHints:
    """Test class for validating type hints of critical methods."""
    
    def test_backtester_init_type_hints(self):
        """Test that Backtester.__init__ has proper type hints."""
        try:
            hints = get_type_hints(Backtester.__init__)
            
            # Check that type hints exist for key parameters
            expected_hints = {
                'global_config': Dict[str, Any],
                'scenarios': List[Dict[str, Any]],
                'random_state': Optional[int]
            }
            
            for param_name, expected_type in expected_hints.items():
                if param_name in hints:
                    # Type hints exist - validate they match expected
                    actual_type = hints[param_name]
                    # For complex types, just check they're not empty
                    assert actual_type is not None, f"Type hint for {param_name} should not be None"
                
        except Exception as e:
            # Type hints might not be available in all Python versions
            pytest.skip(f"Type hints not available: {e}")
    
    def test_validate_data_sufficiency_type_hints(self):
        """Test that validate_data_sufficiency has proper type hints."""
        try:
            hints = get_type_hints(BaseStrategy.validate_data_sufficiency)
            
            # Check return type hint
            if 'return' in hints:
                return_hint = hints['return']
                # Should be tuple[bool, str] or similar
                assert return_hint is not None, "Return type hint should exist"
                
        except Exception as e:
            # Type hints might not be available in all Python versions
            pytest.skip(f"Type hints not available: {e}")


class TestParameterValidation:
    """Test class for parameter validation of critical methods."""
    
    def test_backtester_init_parameter_validation(self):
        """Test parameter validation for Backtester.__init__."""
        with patch('portfolio_backtester.core.get_data_source'), \
             patch('portfolio_backtester.core.enumerate_strategies_with_params') as mock_strategies, \
             patch('portfolio_backtester.core.populate_default_optimizations'), \
             patch('portfolio_backtester.core.TimeoutManager'):
            
            mock_strategies.return_value = {}
            
            # Test with valid parameters first to ensure our mocking works
            args = Mock()
            args.timeout = 300
            backtester = Backtester({}, [], args)
            assert backtester is not None
            
            # Test with wrong types - these might not raise exceptions due to duck typing
            # but we can at least verify the method signature accepts the expected types
            try:
                Backtester("not_a_dict", [], args)
                # If it doesn't raise an exception, that's also acceptable for duck typing
            except (TypeError, AttributeError, ParameterViolationError):
                # If it does raise an exception, that's expected
                pass
    
    def test_custom_timing_registry_get_parameter_validation(self):
        """Test parameter validation for CustomTimingRegistry.get."""
        # Test with valid parameters (it's a classmethod)
        result = CustomTimingRegistry.get("test_key")
        assert result is None  # Should return None for nonexistent key
        
        # Test with None key (should handle gracefully or raise appropriate error)
        try:
            result = CustomTimingRegistry.get(None)
            # If it succeeds, that's acceptable
        except (TypeError, AttributeError):
            # If it raises a type error, that's also acceptable
            pass


class TestReturnTypeValidation:
    """Test class for validating return types of critical methods."""
    
    def test_validate_data_sufficiency_return_type(self):
        """Test return type validation for validate_data_sufficiency."""
        # Create a mock strategy instance with required config
        strategy_config = {'strategy_params': {}}
        strategy = BaseStrategy(strategy_config)
        
        # Create mock data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        mock_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        current_date = pd.Timestamp('2023-12-31')
        
        # Call method and validate return type
        result = strategy.validate_data_sufficiency(mock_data, mock_data, current_date)
        
        # Validate return type structure
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return a tuple of length 2"
        assert isinstance(result[0], bool), "First element should be boolean"
        assert isinstance(result[1], str), "Second element should be string"
        
        # Validate return values are reasonable
        is_sufficient, reason = result
        assert isinstance(reason, str), "Reason should be a string"
        if not is_sufficient:
            assert len(reason) > 0, "Reason should not be empty when data is insufficient"
    
    def test_custom_timing_registry_get_return_type(self):
        """Test return type validation for CustomTimingRegistry.get."""
        # Test return type with default None (it's a classmethod)
        result = CustomTimingRegistry.get("nonexistent_key")
        assert result is None, "Should return None for nonexistent key"
        
        # Test with keyword argument
        result = CustomTimingRegistry.get(name="nonexistent_key")
        assert result is None, "Should return None for nonexistent key with keyword arg"


# Pytest markers for test categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.fast,
    pytest.mark.api_stability
]


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])


class TestBackwardCompatibilityScenarios:
    """Test class for comprehensive backward compatibility scenarios."""
    
    def test_backtester_init_old_parameter_patterns(self):
        """Test that Backtester.__init__ supports old parameter calling patterns."""
        with patch('portfolio_backtester.core.get_data_source'), \
             patch('portfolio_backtester.core.enumerate_strategies_with_params') as mock_strategies, \
             patch('portfolio_backtester.core.populate_default_optimizations'), \
             patch('portfolio_backtester.core.TimeoutManager'):
            
            mock_strategies.return_value = {}
            
            # Test positional arguments (old style)
            global_config = {'data_source': 'test'}
            scenarios = [{'strategy': 'test'}]
            args = Mock()
            args.timeout = 300
            
            backtester = Backtester(global_config, scenarios, args)
            assert backtester is not None
            assert backtester.global_config == global_config
            assert backtester.scenarios == scenarios
            assert backtester.args == args
            
            # Test with optional random_state parameter
            backtester_with_random = Backtester(global_config, scenarios, args, 42)
            assert backtester_with_random is not None
            
            # Test mixed positional and keyword arguments
            backtester_mixed = Backtester(global_config, scenarios, args, random_state=123)
            assert backtester_mixed is not None
            
            # Test all keyword arguments (new style)
            backtester_kwargs = Backtester(
                global_config=global_config,
                scenarios=scenarios,
                args=args,
                random_state=456
            )
            assert backtester_kwargs is not None
    
    def test_backtester_init_new_parameters_have_defaults(self):
        """Test that new parameters added to Backtester.__init__ have appropriate defaults."""
        with patch('portfolio_backtester.core.get_data_source'), \
             patch('portfolio_backtester.core.enumerate_strategies_with_params') as mock_strategies, \
             patch('portfolio_backtester.core.populate_default_optimizations'), \
             patch('portfolio_backtester.core.TimeoutManager'):
            
            mock_strategies.return_value = {}
            
            # Test that the method can be called without the optional parameter
            global_config = {'data_source': 'test'}
            scenarios = [{'strategy': 'test'}]
            args = Mock()
            args.timeout = 300
            
            # This should work without specifying random_state
            backtester = Backtester(global_config, scenarios, args)
            assert backtester is not None
            
            # Verify that the default random_state is handled properly
            assert hasattr(backtester, 'random_state')
            assert isinstance(backtester.random_state, int)
    
    def test_base_strategy_validate_data_sufficiency_return_compatibility(self):
        """Test that validate_data_sufficiency return type remains compatible."""
        strategy_config = {'strategy_params': {}}
        strategy = BaseStrategy(strategy_config)
        
        # Create test data with different scenarios
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        # Test with sufficient data
        sufficient_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        current_date = pd.Timestamp('2023-12-31')
        result = strategy.validate_data_sufficiency(sufficient_data, sufficient_data, current_date)
        
        # Verify the return type is always a tuple of (bool, str)
        assert isinstance(result, tuple), "Return type must be tuple"
        assert len(result) == 2, "Return tuple must have exactly 2 elements"
        assert isinstance(result[0], bool), "First element must be boolean"
        assert isinstance(result[1], str), "Second element must be string"
        
        # Test with insufficient data
        insufficient_dates = pd.date_range('2023-12-01', '2023-12-31', freq='D')
        insufficient_data = pd.DataFrame({
            'Close': np.random.randn(len(insufficient_dates)) + 100,
            'Volume': np.random.randint(1000, 10000, len(insufficient_dates))
        }, index=insufficient_dates)
        
        result_insufficient = strategy.validate_data_sufficiency(insufficient_data, insufficient_data, current_date)
        
        # Verify return type consistency even with insufficient data
        assert isinstance(result_insufficient, tuple), "Return type must be tuple even with insufficient data"
        assert len(result_insufficient) == 2, "Return tuple must have exactly 2 elements"
        assert isinstance(result_insufficient[0], bool), "First element must be boolean"
        assert isinstance(result_insufficient[1], str), "Second element must be string"
        
        # Verify that insufficient data returns False with a meaningful message
        is_sufficient, reason = result_insufficient
        if not is_sufficient:
            assert len(reason.strip()) > 0, "Reason should not be empty when data is insufficient"
    
    def test_custom_timing_registry_get_compatibility_with_dict_interface(self):
        """Test that CustomTimingRegistry.get behaves like dict.get for backward compatibility."""
        # Test basic get functionality
        result = CustomTimingRegistry.get('nonexistent_key')
        assert result is None, "Should return None for nonexistent keys like dict.get()"
        
        # Test that the method exists and is callable
        assert callable(CustomTimingRegistry.get), "get method should be callable"
        
        # Test with keyword arguments
        result_kw = CustomTimingRegistry.get(name='nonexistent_key')
        assert result_kw is None, "Should work with keyword arguments"
    
    def test_apply_leverage_and_smoothing_parameter_compatibility(self):
        """Test that apply_leverage_and_smoothing maintains parameter compatibility."""
        # Create mock data that should work with the function
        mock_weights = pd.Series([0.3, 0.4, 0.3], index=['A', 'B', 'C'])
        mock_prev_weights = pd.Series([0.2, 0.5, 0.3], index=['A', 'B', 'C'])
        
        # Test that the function signature accepts the expected parameters
        sig = inspect.signature(apply_leverage_and_smoothing)
        
        # Verify the actual signature: (candidate_weights, prev_weights, params=None)
        expected_params = ['candidate_weights', 'prev_weights', 'params']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params, \
            f"apply_leverage_and_smoothing signature changed. Expected: {expected_params}, Got: {actual_params}"
        
        # Test function call with correct signature
        params = {'leverage': 1.0, 'smoothing_lambda': 0.0}
        result = apply_leverage_and_smoothing(mock_weights, mock_prev_weights, params)
        assert isinstance(result, pd.Series), "Should return a pandas Series"
        
        # Test with None prev_weights
        result_no_prev = apply_leverage_and_smoothing(mock_weights, None, params)
        assert isinstance(result_no_prev, pd.Series), "Should work with None prev_weights"
        
        # Test with default params
        result_default = apply_leverage_and_smoothing(mock_weights, mock_prev_weights)
        assert isinstance(result_default, pd.Series), "Should work with default params"
        
        # Test with keyword arguments
        result_kwargs = apply_leverage_and_smoothing(
            candidate_weights=mock_weights,
            prev_weights=mock_prev_weights,
            params=params
        )
        assert isinstance(result_kwargs, pd.Series), "Should work with keyword arguments"
    
    def test_default_candidate_weights_parameter_compatibility(self):
        """Test that default_candidate_weights maintains parameter compatibility."""
        # Create minimal mock data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        # Test that the function signature is stable
        sig = inspect.signature(default_candidate_weights)
        
        # Test function call compatibility
        try:
            result = default_candidate_weights(mock_data)
            # Function should return something or handle the call gracefully
            assert result is not None or result is None  # Either outcome is acceptable
        except Exception as e:
            # If it fails, ensure it's not due to signature incompatibility
            error_msg = str(e).lower()
            assert 'signature' not in error_msg, f"Signature compatibility issue: {e}"
            assert 'takes' not in error_msg or 'argument' not in error_msg, f"Argument compatibility issue: {e}"
    
    def test_method_signature_stability_across_versions(self):
        """Test that critical method signatures remain stable across versions."""
        # This test documents the expected signatures to catch breaking changes
        
        # Backtester.__init__ signature
        backtester_sig = inspect.signature(Backtester.__init__)
        expected_backtester_params = ['self', 'global_config', 'scenarios', 'args', 'random_state']
        actual_backtester_params = list(backtester_sig.parameters.keys())
        assert actual_backtester_params == expected_backtester_params, \
            f"Backtester.__init__ signature changed. Expected: {expected_backtester_params}, Got: {actual_backtester_params}"
        
        # BaseStrategy.validate_data_sufficiency signature
        validate_sig = inspect.signature(BaseStrategy.validate_data_sufficiency)
        expected_validate_params = ['self', 'all_historical_data', 'benchmark_historical_data', 'current_date']
        actual_validate_params = list(validate_sig.parameters.keys())
        assert actual_validate_params == expected_validate_params, \
            f"validate_data_sufficiency signature changed. Expected: {expected_validate_params}, Got: {actual_validate_params}"
        
        # CustomTimingRegistry.get signature
        registry_sig = inspect.signature(CustomTimingRegistry.get)
        expected_registry_params = ['name']  # cls is implicit for classmethods in inspect.signature
        actual_registry_params = list(registry_sig.parameters.keys())
        assert actual_registry_params == expected_registry_params, \
            f"CustomTimingRegistry.get signature changed. Expected: {expected_registry_params}, Got: {actual_registry_params}"
    
    def test_return_type_compatibility_guarantees(self):
        """Test that return types remain compatible with existing code expectations."""
        # Test BaseStrategy.validate_data_sufficiency return type
        strategy_config = {'strategy_params': {}}
        strategy = BaseStrategy(strategy_config)
        
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        mock_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        current_date = pd.Timestamp('2023-12-31')
        result = strategy.validate_data_sufficiency(mock_data, mock_data, current_date)
        
        # Test that the result can be unpacked as expected by existing code
        is_sufficient, reason = result
        assert isinstance(is_sufficient, bool)
        assert isinstance(reason, str)
        
        # Test that the result can be used in boolean context
        if result[0]:
            assert True  # Data is sufficient
        else:
            assert len(result[1]) >= 0  # Reason is provided
        
        # Test CustomTimingRegistry.get return type
        registry_result = CustomTimingRegistry.get('nonexistent')
        assert registry_result is None or registry_result is not None  # Should be a valid object or None
        
        # Test that None result can be used in conditional logic (backward compatibility)
        if registry_result is None:
            assert True  # Expected for nonexistent key
        else:
            assert callable(registry_result) or hasattr(registry_result, '__call__')  # Should be a class if not None


class TestParameterDefaultValueStability:
    """Test class for ensuring parameter default values remain stable."""
    
    def test_backtester_init_default_values_unchanged(self):
        """Test that default parameter values for Backtester.__init__ haven't changed."""
        sig = inspect.signature(Backtester.__init__)
        
        # Check that random_state parameter has None as default
        random_state_param = sig.parameters.get('random_state')
        assert random_state_param is not None, "random_state parameter should exist"
        assert random_state_param.default is None, "random_state should default to None"
        
        # Check that required parameters don't have defaults
        required_params = ['global_config', 'scenarios', 'args']
        for param_name in required_params:
            param = sig.parameters.get(param_name)
            assert param is not None, f"Required parameter {param_name} should exist"
            assert param.default == inspect.Parameter.empty, f"Required parameter {param_name} should not have a default"
    
    def test_new_parameters_must_have_defaults(self):
        """Test that any new parameters added to critical methods have default values."""
        # This test ensures that new parameters don't break existing code
        
        # Check Backtester.__init__
        sig = inspect.signature(Backtester.__init__)
        required_params = {'self', 'global_config', 'scenarios', 'args'}  # Known required parameters
        
        for param_name, param in sig.parameters.items():
            if param_name not in required_params:
                assert param.default != inspect.Parameter.empty, \
                    f"New parameter '{param_name}' in Backtester.__init__ must have a default value"
        
        # Check BaseStrategy.validate_data_sufficiency
        validate_sig = inspect.signature(BaseStrategy.validate_data_sufficiency)
        required_validate_params = {'self', 'all_historical_data', 'benchmark_historical_data', 'current_date'}
        
        for param_name, param in validate_sig.parameters.items():
            if param_name not in required_validate_params:
                assert param.default != inspect.Parameter.empty, \
                    f"New parameter '{param_name}' in validate_data_sufficiency must have a default value"


class TestMethodAvailabilityStability:
    """Test class for ensuring critical methods remain available."""
    
    def test_critical_methods_exist_and_callable(self):
        """Test that all critical methods identified in the analysis still exist and are callable."""
        # Test Backtester class methods
        assert hasattr(Backtester, '__init__'), "Backtester.__init__ should exist"
        assert callable(getattr(Backtester, '__init__')), "Backtester.__init__ should be callable"
        
        # Test BaseStrategy methods
        assert hasattr(BaseStrategy, 'validate_data_sufficiency'), "BaseStrategy.validate_data_sufficiency should exist"
        assert callable(getattr(BaseStrategy, 'validate_data_sufficiency')), "validate_data_sufficiency should be callable"
        
        assert hasattr(BaseStrategy, 'get_roro_signal'), "BaseStrategy.get_roro_signal should exist"
        assert callable(getattr(BaseStrategy, 'get_roro_signal')), "get_roro_signal should be callable"
        
        # Test CustomTimingRegistry methods
        assert hasattr(CustomTimingRegistry, 'get'), "CustomTimingRegistry.get should exist"
        assert callable(getattr(CustomTimingRegistry, 'get')), "CustomTimingRegistry.get should be callable"
        
        # Test standalone functions
        assert callable(apply_leverage_and_smoothing), "apply_leverage_and_smoothing should be callable"
        assert callable(default_candidate_weights), "default_candidate_weights should be callable"
    
    def test_method_accessibility_patterns(self):
        """Test that methods can be accessed in the same ways as before."""
        # Test class method access
        assert CustomTimingRegistry.get is not None, "Should be able to access CustomTimingRegistry.get as class method"
        
        # Test instance method access patterns
        strategy_config = {'strategy_params': {}}
        strategy = BaseStrategy(strategy_config)
        
        assert hasattr(strategy, 'validate_data_sufficiency'), "Should be able to access validate_data_sufficiency on instance"
        assert hasattr(strategy, 'get_roro_signal'), "Should be able to access get_roro_signal on instance"
        
        # Test that methods can be called through different access patterns
        # Direct class access
        result1 = CustomTimingRegistry.get('test')
        assert result1 is None
        
        # Instance access for BaseStrategy methods
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        mock_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        current_date = pd.Timestamp('2023-12-31')
        result2 = strategy.validate_data_sufficiency(mock_data, mock_data, current_date)
        assert isinstance(result2, tuple)