import pytest
from src.portfolio_backtester.optimization.parameter_generator import (
    validate_parameter_space,
    validate_optimization_config,
    InvalidParameterSpaceError,
    ParameterGeneratorError
)

def test_validate_parameter_space_valid():
    space = {
        "p1": {"type": "int", "low": 0, "high": 10},
        "p2": {"type": "float", "low": 0.0, "high": 1.0},
        "p3": {"type": "categorical", "choices": ["a", "b"]}
    }
    assert validate_parameter_space(space) is True

def test_validate_parameter_space_invalid_structure():
    with pytest.raises(InvalidParameterSpaceError, match="must be a dictionary"):
        validate_parameter_space([]) # type: ignore

    with pytest.raises(InvalidParameterSpaceError, match="cannot be empty"):
        validate_parameter_space({})

def test_validate_parameter_space_invalid_param_config():
    with pytest.raises(InvalidParameterSpaceError, match="must be a dictionary"):
        validate_parameter_space({"p1": []}) # type: ignore

    with pytest.raises(InvalidParameterSpaceError, match="invalid type"):
        validate_parameter_space({"p1": {"type": "unknown"}})

def test_validate_parameter_space_numeric_bounds():
    # Int bad bounds
    with pytest.raises(InvalidParameterSpaceError, match="must have 'low' and 'high'"):
        validate_parameter_space({"p1": {"type": "int"}})
        
    with pytest.raises(InvalidParameterSpaceError, match="must be integers"):
        validate_parameter_space({"p1": {"type": "int", "low": 0.5, "high": 1}})

    with pytest.raises(InvalidParameterSpaceError, match="low bound .* must be less than high"):
        validate_parameter_space({"p1": {"type": "int", "low": 10, "high": 5}})

def test_validate_parameter_space_categorical():
    with pytest.raises(InvalidParameterSpaceError, match="must have 'choices'"):
        validate_parameter_space({"p1": {"type": "categorical"}})
        
    with pytest.raises(InvalidParameterSpaceError, match="must be a non-empty list"):
        validate_parameter_space({"p1": {"type": "categorical", "choices": []}})

def test_validate_optimization_config_valid():
    config = {
        "parameter_space": {"p": {"type": "int", "low": 0, "high": 1}},
        "metrics_to_optimize": ["sharpe"],
        "max_evaluations": 10
    }
    assert validate_optimization_config(config) is True

def test_validate_optimization_config_invalid():
    with pytest.raises(ParameterGeneratorError, match="must be a dictionary"):
        validate_optimization_config(None) # type: ignore
        
    with pytest.raises(ParameterGeneratorError, match="metrics_to_optimize must be a list"):
        validate_optimization_config({"metrics_to_optimize": "sharpe"})
        
    with pytest.raises(ParameterGeneratorError, match="At least one metric"):
        validate_optimization_config({"metrics_to_optimize": []})