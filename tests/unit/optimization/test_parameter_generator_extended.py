import pytest
from src.portfolio_backtester.optimization.parameter_generator import (
    validate_parameter_space,
    validate_optimization_config,
    InvalidParameterSpaceError,
    ParameterGeneratorError,
    ParameterGenerator
)

class TestParameterGeneratorValidation:
    def test_validate_parameter_space_valid(self):
        space = {
            "param1": {"type": "int", "low": 1, "high": 10},
            "param2": {"type": "float", "low": 0.1, "high": 0.9},
            "param3": {"type": "categorical", "choices": ["A", "B", "C"]}
        }
        assert validate_parameter_space(space) is True

    def test_validate_parameter_space_invalid_structure(self):
        with pytest.raises(InvalidParameterSpaceError, match="must be a dictionary"):
            validate_parameter_space("not a dict")
            
        with pytest.raises(InvalidParameterSpaceError, match="cannot be empty"):
            validate_parameter_space({})

    def test_validate_parameter_space_invalid_types(self):
        space = {"param1": {"type": "unknown"}}
        with pytest.raises(InvalidParameterSpaceError, match="invalid type"):
            validate_parameter_space(space)

    def test_validate_parameter_space_int_float_bounds(self):
        # Missing bounds
        with pytest.raises(InvalidParameterSpaceError, match="must have 'low' and 'high'"):
            validate_parameter_space({"p": {"type": "int"}})
            
        # Invalid bound types
        with pytest.raises(InvalidParameterSpaceError, match="must be integers"):
            validate_parameter_space({"p": {"type": "int", "low": 1.5, "high": 2}})
            
        # Low >= High
        with pytest.raises(InvalidParameterSpaceError, match="low bound .* must be less than high"):
            validate_parameter_space({"p": {"type": "int", "low": 10, "high": 5}})

    def test_validate_parameter_space_categorical(self):
        # Missing choices
        with pytest.raises(InvalidParameterSpaceError, match="must have 'choices'"):
            validate_parameter_space({"p": {"type": "categorical"}})
            
        # Empty choices
        with pytest.raises(InvalidParameterSpaceError, match="must be a non-empty list"):
            validate_parameter_space({"p": {"type": "categorical", "choices": []}})

    def test_validate_optimization_config_valid(self):
        config = {
            "parameter_space": {"p": {"type": "int", "low": 1, "high": 10}},
            "metrics_to_optimize": ["sharpe"],
            "max_evaluations": 100
        }
        assert validate_optimization_config(config) is True

    def test_validate_optimization_config_errors(self):
        # Missing metrics
        with pytest.raises(ParameterGeneratorError, match="metric must be specified"):
            validate_optimization_config({"metrics_to_optimize": []})
            
        # Invalid max_evals
        with pytest.raises(ParameterGeneratorError, match="max_evaluations must be a positive integer"):
            validate_optimization_config({"metrics_to_optimize": ["sharpe"], "max_evaluations": -1})


class TestParameterGeneratorAbstract:
    def test_abstract_methods(self):
        # Ensure we can't instantiate the base class
        with pytest.raises(TypeError):
            ParameterGenerator()
            
    def test_default_implementations(self):
        # Create a dummy concrete class
        class ConcreteGenerator(ParameterGenerator):
            def suggest_parameters(self): return {}
            def report_result(self, p, r): pass
            def is_finished(self): return True
            def get_best_parameters(self): return {}
        
        gen = ConcreteGenerator()
        
        # Test default optional methods
        assert gen.supports_multi_objective() is False
        assert gen.supports_pruning() is False
        assert gen.get_optimization_history() == []
        assert gen.get_parameter_importance() is None
        assert gen.get_current_evaluation_count() == 0
        
        # Test initialize base (empty)
        gen.initialize({}, {})
        
        # Test default get_best_result raises NotImplemented
        with pytest.raises(NotImplementedError):
            gen.get_best_result()
