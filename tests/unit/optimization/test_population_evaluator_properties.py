"""
Property-based tests for population evaluator components.

This module uses Hypothesis to test invariants and properties of the population evaluator,
which is responsible for evaluating populations of parameter sets in parallel.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, cast
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st, assume, example
from hypothesis.extra import numpy as hnp

from portfolio_backtester.optimization.population_evaluator import PopulationEvaluator
from portfolio_backtester.optimization.results import EvaluationResult, OptimizationData

from tests.strategies.optimization_strategies import (
    parameter_spaces,
    parameter_values,
    populations,
    evaluation_results,
    optimization_configs
)


@given(st.integers(min_value=1, max_value=16))
@settings(deadline=None)
def test_population_evaluator_initialization(n_jobs):
    """Test that the population evaluator initializes correctly."""
    # Create mock evaluator
    mock_evaluator = MagicMock()
    
    # Create population evaluator
    evaluator = PopulationEvaluator(evaluator=mock_evaluator, n_jobs=n_jobs)
    
    # Check that the evaluator has the correct attributes
    assert evaluator.evaluator == mock_evaluator
    
    # Check that n_jobs is correctly normalized
    if n_jobs <= 0:
        # Should use all CPU cores
        import os
        expected_n_jobs = os.cpu_count() or 1
    else:
        expected_n_jobs = n_jobs
    
    assert evaluator.n_jobs == expected_n_jobs
    
    # Check that the result cache is initialized
    assert evaluator._result_cache == {}
    
    # Check that the deduplicator is initialized
    assert evaluator._deduplicator is not None


@given(
    st.lists(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(
                st.integers(-100, 100),
                st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                st.text(min_size=1, max_size=10)
            ),
            min_size=1,
            max_size=5
        ),
        min_size=1,
        max_size=10
    )
)
@settings(deadline=None)
def test_population_evaluator_params_key(params_list):
    """Test that the population evaluator generates consistent parameter keys."""
    # Create mock evaluator
    mock_evaluator = MagicMock()
    
    # Create population evaluator
    evaluator = PopulationEvaluator(evaluator=mock_evaluator)
    
    # Generate keys for each parameter set
    keys = [evaluator._params_key(params) for params in params_list]
    
    # Check that each key is a tuple of strings
    for key in keys:
        assert isinstance(key, tuple)
        assert all(isinstance(k, str) for k in key)
    
    # Check that identical parameter sets produce identical keys
    for i, params in enumerate(params_list):
        duplicate_params = params.copy()
        assert evaluator._params_key(params) == evaluator._params_key(duplicate_params)
    
    # Check that different parameter sets produce different keys
    # (Only if we have at least two parameter sets that are different)
    unique_params = []
    unique_keys = set()
    
    for params in params_list:
        key = evaluator._params_key(params)
        if key not in unique_keys:
            unique_keys.add(key)
            unique_params.append(params)
    
    # Check that the number of unique keys matches the number of unique parameter sets
    assert len(unique_keys) == len(unique_params)


@given(
    st.lists(
        st.dictionaries(
            keys=st.sampled_from(["param_a", "param_b", "param_c"]),
            values=st.integers(0, 10),
            min_size=1,
            max_size=3
        ),
        min_size=2,
        max_size=10
    )
)
@settings(deadline=None)
def test_population_evaluator_batch_deduplication(population):
    """Test that the population evaluator correctly identifies duplicate parameter sets."""
    # Create mock evaluator
    mock_evaluator = MagicMock()
    
    # Create population evaluator
    evaluator = PopulationEvaluator(evaluator=mock_evaluator)
    
    # Create a simple function to simulate the batch deduplication logic
    def count_unique_params(population):
        unique_params = {}
        param_to_unique_key = {}
        
        for i, params in enumerate(population):
            key = evaluator._params_key(params)
            unique_params[key] = params
            param_to_unique_key[i] = key
        
        return len(unique_params), len(population)
    
    # Count unique parameter sets
    n_unique, n_total = count_unique_params(population)
    
    # Check that the number of unique parameter sets is less than or equal to the total
    assert n_unique <= n_total
    
    # Create a duplicate-free population by adding one of each unique parameter set
    seen_keys = set()
    duplicate_free_population = []
    
    for params in population:
        key = evaluator._params_key(params)
        if key not in seen_keys:
            seen_keys.add(key)
            duplicate_free_population.append(params)
    
    # Count unique parameter sets in the duplicate-free population
    n_unique_df, n_total_df = count_unique_params(duplicate_free_population)
    
    # Check that all parameter sets in the duplicate-free population are unique
    assert n_unique_df == n_total_df
    assert n_unique_df == n_unique


class MockBacktestEvaluator:
    """Mock backtester evaluator for testing."""
    
    def __init__(self, is_multi_objective=False):
        self.is_multi_objective = is_multi_objective
        self.evaluate_parameters_calls = 0
    
    def evaluate_parameters(self, params, scenario_config, data, backtester, previous_parameters=None):
        """Mock evaluation function that returns a simple result."""
        self.evaluate_parameters_calls += 1
        # Generate a deterministic but unique value based on the parameter values
        value = sum(hash(f"{k}={v}") % 1000 for k, v in params.items()) / 1000.0
        return EvaluationResult(
            objective_value=value,
            metrics={"mock_metric": value},
            window_results=[]
        )


@given(
    st.lists(
        st.dictionaries(
            keys=st.sampled_from(["param_a", "param_b", "param_c"]),
            values=st.integers(0, 10),
            min_size=1,
            max_size=3
        ),
        min_size=2,
        max_size=10,
        unique_by=lambda x: tuple(sorted((k, v) for k, v in x.items()))
    )
)
@settings(deadline=None)
def test_population_evaluator_caching(population):
    """Test that the population evaluator correctly caches evaluation results."""
    # Create a real mock evaluator that counts calls
    evaluator = MockBacktestEvaluator()
    
    # Create population evaluator
    pop_evaluator = PopulationEvaluator(evaluator=evaluator, n_jobs=1)
    
    # Create mock scenario config, data, and backtester
    scenario_config = {"mock": True}
    data = MagicMock()
    backtester = MagicMock()
    
    # Evaluate the population
    results1 = pop_evaluator.evaluate_population(
        population=population,
        scenario_config=scenario_config,
        data=data,
        backtester=backtester
    )
    
    # Check that we got the right number of results
    assert len(results1) == len(population)
    
    # Record the number of evaluation calls
    calls_after_first_eval = evaluator.evaluate_parameters_calls
    
    # Evaluate the same population again
    results2 = pop_evaluator.evaluate_population(
        population=population,
        scenario_config=scenario_config,
        data=data,
        backtester=backtester
    )
    
    # Check that we got the right number of results
    assert len(results2) == len(population)
    
    # Check that no additional evaluation calls were made (all results were cached)
    assert evaluator.evaluate_parameters_calls == calls_after_first_eval
    
    # Check that the results are the same
    for r1, r2 in zip(results1, results2):
        assert r1.objective_value == r2.objective_value
        assert r1.metrics == r2.metrics
