"""
Property-based tests for parameter generator components.

This module uses Hypothesis to test invariants and properties of the parameter generators,
which are responsible for suggesting parameter sets to evaluate during optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, cast
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st, assume, example
from hypothesis.extra import numpy as hnp

from portfolio_backtester.optimization.parameter_generator import (
    validate_parameter_space,
    InvalidParameterSpaceError,
    PopulationBasedParameterGenerator,
    ParameterGenerator
)
from portfolio_backtester.optimization.results import EvaluationResult, OptimizationResult

from tests.strategies.optimization_strategies import (
    parameter_spaces,
    parameter_values,
    populations,
    evaluation_results,
    optimization_configs
)


@given(parameter_spaces())
@settings(deadline=None)
def test_validate_parameter_space_valid(param_space):
    """Test that valid parameter spaces pass validation."""
    # This should not raise an exception
    validate_parameter_space(param_space)


@given(st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.booleans()
    ),
    min_size=1,
    max_size=5
))
@settings(deadline=None)
def test_validate_parameter_space_invalid_format(invalid_param_space):
    """Test that invalid parameter spaces fail validation."""
    # Parameter values should be dictionaries with specific keys
    assume(not all(isinstance(v, dict) and "type" in v for v in invalid_param_space.values()))
    
    # This should raise an exception
    try:
        validate_parameter_space(invalid_param_space)
        assert False, "Expected InvalidParameterSpaceError but no exception was raised"
    except InvalidParameterSpaceError:
        # Expected exception
        pass


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.dictionaries(
            keys=st.just("type"),
            values=st.text(min_size=1, max_size=10).filter(
                lambda x: x not in ["int", "float", "categorical", "multi-categorical"]
            ),
            min_size=1,
            max_size=1
        ),
        min_size=1,
        max_size=5
    )
)
@settings(deadline=None)
def test_validate_parameter_space_invalid_type(invalid_type_space):
    """Test that parameter spaces with invalid types fail validation."""
    # This should raise an exception
    try:
        validate_parameter_space(invalid_type_space)
        assert False, "Expected InvalidParameterSpaceError but no exception was raised"
    except InvalidParameterSpaceError:
        # Expected exception
        pass


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.fixed_dictionaries({
            "type": st.just("int"),
            # Missing 'low' and 'high'
        }),
        min_size=1,
        max_size=5
    )
)
@settings(deadline=None)
def test_validate_parameter_space_missing_bounds(missing_bounds_space):
    """Test that numeric parameter spaces without bounds fail validation."""
    # This should raise an exception
    try:
        validate_parameter_space(missing_bounds_space)
        assert False, "Expected InvalidParameterSpaceError but no exception was raised"
    except InvalidParameterSpaceError:
        # Expected exception
        pass


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.fixed_dictionaries({
            "type": st.just("categorical"),
            # Missing 'choices'
        }),
        min_size=1,
        max_size=5
    )
)
@settings(deadline=None)
def test_validate_parameter_space_missing_choices(missing_choices_space):
    """Test that categorical parameter spaces without choices fail validation."""
    # This should raise an exception
    try:
        validate_parameter_space(missing_choices_space)
        assert False, "Expected InvalidParameterSpaceError but no exception was raised"
    except InvalidParameterSpaceError:
        # Expected exception
        pass


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.fixed_dictionaries({
            "type": st.just("int"),
            "low": st.integers(min_value=0, max_value=100),
            "high": st.integers(min_value=0, max_value=100)
        }),
        min_size=1,
        max_size=5
    )
)
@settings(deadline=None)
def test_validate_parameter_space_invalid_bounds(invalid_bounds_space):
    """Test that numeric parameter spaces with invalid bounds fail validation."""
    # Ensure that at least one parameter has low >= high
    has_invalid_bounds = False
    for param_config in invalid_bounds_space.values():
        if param_config["low"] >= param_config["high"]:
            has_invalid_bounds = True
            break
    
    assume(has_invalid_bounds)
    
    # This should raise an exception
    try:
        validate_parameter_space(invalid_bounds_space)
        assert False, "Expected InvalidParameterSpaceError but no exception was raised"
    except InvalidParameterSpaceError:
        # Expected exception
        pass


@given(st.data())
@settings(deadline=None)
def test_parameter_value_within_space(data):
    """Test that parameter values generated from a space are valid within that space."""
    # Generate a parameter space and values from it
    param_space = data.draw(parameter_spaces())
    param_values = data.draw(parameter_values(param_space))
    
    # First validate the parameter space
    validate_parameter_space(param_space)
    
    # Check that the parameter values match the parameter space
    for param_name, param_config in param_space.items():
        assert param_name in param_values, f"Parameter {param_name} missing from values"
        
        param_type = param_config["type"]
        param_value = param_values[param_name]
        
        if param_type == "int":
            assert isinstance(param_value, (int, np.integer)), f"Parameter {param_name} should be an integer"
            assert param_config["low"] <= param_value <= param_config["high"], f"Parameter {param_name} out of bounds"
            
        elif param_type == "float":
            assert isinstance(param_value, (float, np.floating)), f"Parameter {param_name} should be a float"
            assert param_config["low"] <= param_value <= param_config["high"], f"Parameter {param_name} out of bounds"
            
        elif param_type == "categorical":
            assert param_value in param_config["choices"], f"Parameter {param_name} not in choices"
    """Test that parameter values generated from a space are valid within that space."""
    # First validate the parameter space
    validate_parameter_space(param_space)
    
    # Check that the parameter values match the parameter space
    for param_name, param_config in param_space.items():
        assert param_name in param_values, f"Parameter {param_name} missing from values"
        
        param_type = param_config["type"]
        param_value = param_values[param_name]
        
        if param_type == "int":
            assert isinstance(param_value, (int, np.integer)), f"Parameter {param_name} should be an integer"
            assert param_config["low"] <= param_value <= param_config["high"], f"Parameter {param_name} out of bounds"
            
        elif param_type == "float":
            assert isinstance(param_value, (float, np.floating)), f"Parameter {param_name} should be a float"
            assert param_config["low"] <= param_value <= param_config["high"], f"Parameter {param_name} out of bounds"
            
        elif param_type == "categorical":
            assert param_value in param_config["choices"], f"Parameter {param_name} not in choices"


# Mock implementation of ParameterGenerator for testing
class MockParameterGenerator(ParameterGenerator):
    """Mock parameter generator for testing."""
    
    def __init__(self):
        self.initialized = False
        self.parameter_space = {}
        self.suggested_params = []
        self.reported_results = []
    
    def initialize(self, scenario_config, optimization_config):
        self.initialized = True
        self.parameter_space = scenario_config.get("parameter_space", {})
    
    def suggest_parameters(self):
        if not self.initialized:
            raise RuntimeError("Not initialized")
        
        # Generate a simple parameter set
        params = {}
        for name, config in self.parameter_space.items():
            param_type = config["type"]
            if param_type == "int":
                params[name] = config["low"]
            elif param_type == "float":
                params[name] = config["low"]
            elif param_type == "categorical":
                params[name] = config["choices"][0] if config["choices"] else None
        
        self.suggested_params.append(params)
        return params
    
    def report_result(self, parameters, result):
        self.reported_results.append((parameters, result))
    
    def is_finished(self):
        return len(self.reported_results) >= 5
    
    def get_best_parameters(self):
        if not self.reported_results:
            return {}
        
        # Return the parameters with the highest objective value
        best_idx = np.argmax([r[1].objective_value for r in self.reported_results])
        return self.reported_results[best_idx][0]
    
    def get_best_result(self):
        if not self.reported_results:
            return OptimizationResult(
                best_parameters={},
                best_value=-1e9,
                n_evaluations=0,
                optimization_history=[]
            )
        
        best_idx = np.argmax([r[1].objective_value for r in self.reported_results])
        best_params = self.reported_results[best_idx][0]
        best_result = self.reported_results[best_idx][1]
        
        return OptimizationResult(
            best_parameters=best_params,
            best_value=best_result.objective_value,
            n_evaluations=len(self.reported_results),
            optimization_history=[]
        )


@given(parameter_spaces())
@settings(deadline=None)
def test_parameter_generator_lifecycle(param_space):
    """Test the lifecycle of a parameter generator."""
    # Create a parameter generator
    generator = MockParameterGenerator()
    
    # Initialize the generator
    scenario_config = {"parameter_space": param_space}
    generator.initialize(scenario_config, {})
    
    assert generator.initialized
    assert generator.parameter_space == param_space
    
    # Suggest parameters and report results
    for i in range(5):
        params = generator.suggest_parameters()
        
        # Check that the parameters are valid
        for name, value in params.items():
            assert name in param_space
        
        # Report a result
        result = EvaluationResult(
            objective_value=float(i),
            metrics={"metric": float(i)},
            window_results=[]
        )
        generator.report_result(params, result)
    
    # Check that the generator is finished
    assert generator.is_finished()
    
    # Get the best parameters and result
    best_params = generator.get_best_parameters()
    best_result = generator.get_best_result()
    
    # The best result should have the highest objective value (4.0)
    assert best_result.best_value == 4.0
    assert best_result.n_evaluations == 5
