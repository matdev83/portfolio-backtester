"""
Tests for the ParameterGenerator interface and related functionality.

This module tests the ParameterGenerator abstract base class, validation
functions, and the factory pattern implementation.
"""

import pytest

from portfolio_backtester.optimization.parameter_generator import (
    ParameterGenerator,
    ParameterGeneratorError,
    InvalidParameterSpaceError,
    validate_parameter_space,
    validate_optimization_config
)
from portfolio_backtester.optimization.factory import (
    create_parameter_generator,
    UnknownOptimizerError,
    get_available_optimizers,
    validate_optimizer_type
)


class TestParameterGeneratorInterface:
    """Test the ParameterGenerator abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that ParameterGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ParameterGenerator()
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that subclasses must implement all abstract methods."""
        
        class IncompleteGenerator(ParameterGenerator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteGenerator()
    
    def test_concrete_implementation_works(self):
        """Test that a complete implementation can be instantiated."""
        
        class ConcreteGenerator(ParameterGenerator):
            def suggest_parameters(self, optimization_spec):
                return {"param1": 1}
            
            def report_result(self, parameters, result):
                pass
            
            def is_finished(self):
                return False
            
            def get_best_parameters(self):
                return {"param1": 1}
        
        generator = ConcreteGenerator()
        assert generator is not None


class TestParameterSpaceValidation:
    """Test parameter space validation functions."""
    
    def test_valid_parameter_space(self):
        """Test validation of valid parameter spaces."""
        valid_space = {
            'param1': {'type': 'int', 'low': 1, 'high': 10},
            'param2': {'type': 'float', 'low': 0.1, 'high': 1.0},
            'param3': {'type': 'categorical', 'choices': ['a', 'b', 'c']}
        }
        assert validate_parameter_space(valid_space) is True
    
    def test_invalid_parameter_space_type(self):
        """Test validation with invalid parameter space type."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space("not_a_dict")
    
    def test_empty_parameter_space(self):
        """Test validation with empty parameter space."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({})
    
    def test_invalid_parameter_name(self):
        """Test validation with invalid parameter name."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({123: {'type': 'int', 'low': 1, 'high': 10}})
    
    def test_invalid_parameter_config_type(self):
        """Test validation with invalid parameter config type."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({'param1': 'not_a_dict'})
    
    def test_invalid_parameter_type(self):
        """Test validation with invalid parameter type."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({'param1': {'type': 'invalid_type'}})
    
    def test_missing_bounds_for_numeric_types(self):
        """Test validation with missing bounds for numeric types."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({'param1': {'type': 'int'}})
    
    def test_invalid_bounds_type(self):
        """Test validation with invalid bounds type."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({'param1': {'type': 'int', 'low': 'not_a_number', 'high': 10}})
    
    def test_invalid_bounds_order(self):
        """Test validation with invalid bounds order."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({'param1': {'type': 'int', 'low': 10, 'high': 5}})
    
    def test_missing_choices_for_categorical(self):
        """Test validation with missing choices for categorical type."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({'param1': {'type': 'categorical'}})
    
    def test_invalid_choices_type(self):
        """Test validation with invalid choices type."""
        with pytest.raises(InvalidParameterSpaceError):
            validate_parameter_space({'param1': {'type': 'categorical', 'choices': 'not_a_list'}})


class TestOptimizationConfigValidation:
    """Test optimization configuration validation functions."""
    
    def test_valid_optimization_config(self):
        """Test validation of valid optimization configs."""
        valid_config = {
            'metrics_to_optimize': ['sharpe_ratio'],
            'max_evaluations': 100
        }
        assert validate_optimization_config(valid_config) is True
    
    def test_invalid_config_type(self):
        """Test validation with invalid config type."""
        with pytest.raises(ParameterGeneratorError):
            validate_optimization_config("not_a_dict")
    
    def test_invalid_metrics_type(self):
        """Test validation with invalid metrics type."""
        with pytest.raises(ParameterGeneratorError):
            validate_optimization_config({'metrics_to_optimize': 'not_a_list'})
    
    def test_empty_metrics(self):
        """Test validation with empty metrics list."""
        with pytest.raises(ParameterGeneratorError):
            validate_optimization_config({'metrics_to_optimize': []})
    
    def test_invalid_metric_name_type(self):
        """Test validation with invalid metric name type."""
        with pytest.raises(ParameterGeneratorError):
            validate_optimization_config({'metrics_to_optimize': [123]})
    
    def test_invalid_max_evaluations(self):
        """Test validation with invalid max_evaluations."""
        with pytest.raises(ParameterGeneratorError):
            validate_optimization_config({
                'metrics_to_optimize': ['sharpe_ratio'],
                'max_evaluations': -1
            })


class TestParameterGeneratorFactory:
    """Test the parameter generator factory functionality."""
    
    def test_create_mock_generator_removed(self):
        """Test that mock generator has been removed."""
        with pytest.raises(UnknownOptimizerError, match="Unknown optimizer type: 'mock'"):
            create_parameter_generator("mock", random_state=42)
    
    def test_create_optuna_generator(self):
        """Test creating Optuna generator."""
        try:
            generator = create_parameter_generator("optuna", random_state=42)
            assert generator is not None
        except ImportError:
            # Optuna not available, skip test
            pytest.skip("Optuna not available")
    
    def test_create_genetic_generator(self):
        """Test creating genetic generator."""
        try:
            generator = create_parameter_generator("genetic", random_state=42)
            assert generator is not None
        except ImportError:
            # PyGAD not available, skip test
            pytest.skip("PyGAD not available")
    
    def test_unknown_optimizer_type(self):
        """Test creating an unknown generator type."""
        with pytest.raises(UnknownOptimizerError):
            create_parameter_generator("unknown_type")
    
    def test_invalid_optimizer_type(self):
        """Test creating generator with invalid type."""
        with pytest.raises(ParameterGeneratorError):
            create_parameter_generator(None)
        
        with pytest.raises(ParameterGeneratorError):
            create_parameter_generator(123)
    
    def test_get_available_optimizers(self):
        """Test getting available optimizer information."""
        optimizers = get_available_optimizers()
        assert isinstance(optimizers, dict)
        assert "optuna" in optimizers or "genetic" in optimizers
        assert "mock" not in optimizers  # Mock generator removed
    
    def test_validate_optimizer_type(self):
        """Test optimizer type validation."""
        assert validate_optimizer_type("optuna") is True
        assert validate_optimizer_type("genetic") is True
        assert validate_optimizer_type("unknown") is False
        assert validate_optimizer_type("mock") is False  # Mock generator removed


if __name__ == "__main__":
    pytest.main([__file__])