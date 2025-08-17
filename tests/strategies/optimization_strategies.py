"""
Hypothesis strategies for property-based testing of optimization components.

This module provides reusable strategies for generating test data for optimization
components, including parameter spaces, populations, and evaluation results.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# Define common parameter types for optimization
PARAM_TYPES = ["int", "float", "categorical", "boolean"]


@st.composite
def parameter_spaces(draw, min_params=1, max_params=5):
    """
    Generate a valid parameter space dictionary for optimization.
    
    Args:
        min_params: Minimum number of parameters
        max_params: Maximum number of parameters
    
    Returns:
        A dictionary representing a parameter space
    """
    n_params = draw(st.integers(min_value=min_params, max_value=max_params))
    param_space = {}
    
    for i in range(n_params):
        param_name = f"param_{i}"
        param_type = draw(st.sampled_from(PARAM_TYPES))
        
        if param_type == "int":
            min_val = draw(st.integers(min_value=-100, max_value=100))
            max_val = draw(st.integers(min_value=min_val + 1, max_value=min_val + 200))
            param_space[param_name] = {
                "type": "int",
                "low": min_val,
                "high": max_val
            }
            
        elif param_type == "float":
            min_val = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
            max_val = min_val + draw(st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False))
            param_space[param_name] = {
                "type": "float",
                "low": min_val,
                "high": max_val
            }
            
        elif param_type == "categorical":
            n_choices = draw(st.integers(min_value=2, max_value=5))
            choices = [f"choice_{j}" for j in range(n_choices)]
            param_space[param_name] = {
                "type": "categorical",
                "choices": choices
            }
            
        elif param_type == "boolean":
            # Convert boolean to categorical with [True, False] choices
            param_space[param_name] = {
                "type": "categorical",
                "choices": [True, False]
            }
    
    return param_space


@st.composite
def parameter_values(draw, param_space):
    """
    Generate a valid parameter value dictionary based on a parameter space.
    
    Args:
        param_space: A parameter space dictionary
    
    Returns:
        A dictionary of parameter values
    """
    params = {}
    
    for param_name, param_config in param_space.items():
        param_type = param_config["type"]
        
        if param_type == "int":
            params[param_name] = draw(st.integers(min_value=param_config["low"], max_value=param_config["high"]))
            
        elif param_type == "float":
            params[param_name] = draw(st.floats(
                min_value=param_config["low"],
                max_value=param_config["high"],
                allow_nan=False,
                allow_infinity=False
            ))
            
        elif param_type == "categorical":
            params[param_name] = draw(st.sampled_from(param_config["choices"]))
            
        elif param_type == "boolean":
            params[param_name] = draw(st.booleans())
    
    return params


@st.composite
def populations(draw, param_space=None, population_size=None):
    """
    Generate a population of parameter dictionaries based on a parameter space.
    
    Args:
        param_space: A parameter space dictionary (if None, will be randomly chosen)
        population_size: Size of the population (if None, will be randomly chosen)
    
    Returns:
        A list of parameter dictionaries
    """
    if param_space is None:
        param_space = draw(parameter_spaces())
    
    if population_size is None:
        population_size = draw(st.integers(min_value=5, max_value=20))
    
    return [draw(parameter_values(param_space)) for _ in range(population_size)]


@st.composite
def evaluation_results(draw, params=None, min_metrics=1, max_metrics=3):
    """
    Generate evaluation results for parameter sets.
    
    Args:
        params: Optional parameter dictionary
        min_metrics: Minimum number of metrics
        max_metrics: Maximum number of metrics
    
    Returns:
        A tuple of (params, metrics, is_valid)
    """
    if params is None:
        param_space = draw(parameter_spaces())
        params = draw(parameter_values(param_space))
    
    n_metrics = draw(st.integers(min_value=min_metrics, max_value=max_metrics))
    metrics = {}
    
    for i in range(n_metrics):
        metric_name = f"metric_{i}"
        # Generate a metric value (sometimes NaN or inf to test robustness)
        metric_value = draw(
            st.one_of(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                st.just(float('nan')),
                st.just(float('inf')),
                st.just(float('-inf'))
            )
        )
        metrics[metric_name] = metric_value
    
    # Sometimes generate invalid results to test error handling
    is_valid = draw(st.booleans())
    
    return params, metrics, is_valid


@st.composite
def optimization_configs(draw):
    """
    Generate optimization configuration dictionaries.
    
    Returns:
        A dictionary with optimization configuration
    """
    param_space = draw(parameter_spaces())
    
    config = {
        "parameter_space": param_space,
        "metrics_to_optimize": draw(st.lists(
            st.sampled_from(["sharpe_ratio", "total_return", "max_drawdown", "calmar_ratio", "sortino_ratio"]),
            min_size=1,
            max_size=3,
            unique=True
        )),
        "optimization_direction": draw(st.sampled_from(["maximize", "minimize"])),
        "n_trials": draw(st.integers(min_value=10, max_value=100)),
        "timeout_seconds": draw(st.one_of(
            st.none(),
            st.integers(min_value=10, max_value=300)
        )),
        "early_stop_patience": draw(st.one_of(
            st.none(),
            st.integers(min_value=5, max_value=20)
        )),
    }
    
    # Add GA-specific settings
    ga_settings = {
        "population_size": draw(st.integers(min_value=10, max_value=50)),
        "max_generations": draw(st.integers(min_value=5, max_value=20)),
        "mutation_rate": draw(st.floats(min_value=0.01, max_value=0.3, allow_nan=False, allow_infinity=False)),
        "crossover_rate": draw(st.floats(min_value=0.5, max_value=0.9, allow_nan=False, allow_infinity=False)),
    }
    
    config["ga_settings"] = ga_settings
    
    return config


@st.composite
def ga_worker_contexts(draw):
    """
    Generate GA worker context configurations.
    
    Returns:
        A dictionary with worker context configuration
    """
    return {
        "run_id": draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=8, max_size=8)),
        "worker_id": draw(st.integers(min_value=0, max_value=16)),
        "n_jobs": draw(st.integers(min_value=1, max_value=16)),
        "use_vectorized_tracking": draw(st.booleans()),
        "use_memmap": draw(st.booleans()),
    }


@st.composite
def individuals_from_space(draw, param_space):
    """
    Generate an individual based on a parameter space.
    
    Args:
        param_space: A parameter space dictionary
    
    Returns:
        A dictionary of parameter values
    """
    return draw(parameter_values(param_space))


@st.composite
def populations_with_results(draw, param_space=None, population_size=None):
    """
    Generate a population of parameter dictionaries with corresponding evaluation results.
    
    Args:
        param_space: Optional parameter space dictionary
        population_size: Size of the population
    
    Returns:
        A tuple of (population, results)
    """
    if param_space is None:
        param_space = draw(parameter_spaces())
    
    if population_size is None:
        population_size = draw(st.integers(min_value=5, max_value=20))
    
    population = draw(populations(param_space, population_size))
    
    # Create results with fitness values that increase with index
    results = []
    for i, params in enumerate(population):
        fitness = float(i) / len(population)  # Normalized fitness between 0 and 1
        from portfolio_backtester.optimization.results import EvaluationResult
        result = EvaluationResult(
            objective_value=fitness,
            metrics={"fitness": fitness},
            window_results=[]
        )
        results.append(result)
    
    return population, results


@st.composite
def ga_settings(draw):
    """
    Generate GA settings dictionaries.
    
    Returns:
        A dictionary with GA settings
    """
    return {
        "population_size": draw(st.integers(min_value=10, max_value=50)),
        "max_generations": draw(st.integers(min_value=5, max_value=20)),
        "mutation_rate": draw(st.floats(min_value=0.01, max_value=0.3, allow_nan=False, allow_infinity=False)),
        "crossover_rate": draw(st.floats(min_value=0.5, max_value=0.9, allow_nan=False, allow_infinity=False)),
    }