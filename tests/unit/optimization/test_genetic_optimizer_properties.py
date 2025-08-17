"""
Property-based tests for genetic algorithm optimizer components.

This module uses Hypothesis to test invariants and properties of the genetic algorithm
optimization components, including parameter generation, population evolution, and selection.
"""

import numpy as np
import pandas as pd
from hypothesis import assume
from typing import Dict, Any, List, Tuple, Optional

from hypothesis import given, settings, strategies as st, assume, example
from hypothesis.extra import numpy as hnp

from portfolio_backtester.optimization.generators.fixed_genetic_generator import (
    FixedGeneticParameterGenerator
)
from portfolio_backtester.optimization.performance.genetic_optimizer import (
    GeneticPerformanceOptimizer
)
from portfolio_backtester.optimization.results import EvaluationResult

from tests.strategies.optimization_strategies import (
    parameter_spaces,
    parameter_values,
    populations,
    evaluation_results,
    optimization_configs
)


@given(parameter_spaces())
@settings(deadline=None)
def test_genetic_generator_initialization(param_space):
    """Test that the genetic parameter generator initializes correctly."""
    # Create a genetic parameter generator
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": 20,
            "max_generations": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Initialize the generator
    generator.initialize(scenario_config, {})
    
    # Check that the population was created with the correct size
    assert generator.population is not None
    assert len(generator.population) == 20
    
    # Check that each individual in the population has all parameters
    for individual in generator.population:
        assert set(individual.keys()) == set(param_space.keys())
        
        # Check that each parameter value is within bounds
        for param_name, param_value in individual.items():
            param_config = param_space[param_name]
            param_type = param_config["type"]
            
            if param_type == "int":
                assert isinstance(param_value, (int, np.integer))
                assert param_config["low"] <= param_value <= param_config["high"]
                
            elif param_type == "float":
                assert isinstance(param_value, (float, np.floating))
                assert param_config["low"] <= param_value <= param_config["high"]
                
            elif param_type == "categorical":
                assert param_value in param_config["choices"]
                
            elif param_type == "boolean":
                assert isinstance(param_value, bool)


@given(parameter_spaces())
@settings(deadline=None)
def test_genetic_generator_suggest_population(param_space):
    """Test that the genetic parameter generator suggests valid populations."""
    # Create a genetic parameter generator
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": 15,
            "max_generations": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Initialize the generator
    generator.initialize(scenario_config, {})
    
    # Get the initial population
    population = generator.suggest_population()
    
    # Check that the population has the correct size
    assert len(population) == 15
    
    # Check that each individual in the population has all parameters
    for individual in population:
        assert set(individual.keys()) == set(param_space.keys())


@given(parameter_spaces(), st.integers(min_value=10, max_value=30))
@settings(deadline=None)
def test_genetic_generator_evolution(param_space, population_size):
    """Test that the genetic parameter generator evolves the population correctly."""
    # Create a genetic parameter generator
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": population_size,
            "max_generations": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Initialize the generator
    generator.initialize(scenario_config, {})
    
    # Get the initial population
    initial_population = generator.suggest_population()
    
    # Create evaluation results for the population
    # Make some individuals perform better than others to drive selection
    results = []
    for i, individual in enumerate(initial_population):
        fitness = float(i) / len(initial_population)  # Fitness increases with index
        result = EvaluationResult(
            objective_value=fitness,
            metrics={"fitness": fitness},
            window_results=[]
        )
        results.append(result)
    
    # Report the results
    generator.report_population_results(initial_population, results)
    
    # Get the next generation
    next_generation = generator.suggest_population()
    
    # Check that the next generation has the correct size
    assert len(next_generation) == population_size
    
    # Check that the generation count was incremented
    assert generator.generation_count == 1
    
    # Check that the best individual was preserved (elitism)
    best_individual = initial_population[-1]  # Last individual had highest fitness
    assert any(
        all(ind[k] == best_individual[k] for k in ind.keys())
        for ind in next_generation
    ), "Best individual not preserved in next generation"


@given(optimization_configs())
@settings(deadline=None)
def test_genetic_performance_optimizer_creation(config):
    """Test that the genetic performance optimizer is created correctly."""
    # Create a genetic performance optimizer
    optimizer = GeneticPerformanceOptimizer(
        enable_vectorized_tracking=True,
        enable_deduplication=True,
        enable_parallel_execution=True,
        n_jobs=4
    )
    
    # Check that the optimizer has the correct attributes
    assert optimizer.enable_vectorized_tracking is True
    assert optimizer.enable_deduplication is True
    assert optimizer.enable_parallel_execution is True
    assert optimizer.n_jobs == 4
    
    # Check that the optimizer components are initialized
    assert optimizer._trade_tracker is None  # Lazy initialization
    assert optimizer._deduplicator is None  # Lazy initialization
    assert optimizer._parallel_runner is None  # Lazy initialization


@given(parameter_spaces(), st.integers(min_value=10, max_value=30))
@settings(deadline=None)
def test_genetic_generator_determinism(param_space, population_size):
    """Test that the genetic parameter generator is deterministic with the same seed."""
    # Create two genetic parameter generators with the same seed
    generator1 = FixedGeneticParameterGenerator(random_seed=42)
    generator2 = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": population_size,
            "max_generations": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Initialize both generators
    generator1.initialize(scenario_config, {})
    generator2.initialize(scenario_config, {})
    
    # Get the initial populations
    population1 = generator1.suggest_population()
    population2 = generator2.suggest_population()
    
    # Check that the populations are identical
    assert len(population1) == len(population2)
    
    for ind1, ind2 in zip(population1, population2):
        assert set(ind1.keys()) == set(ind2.keys())
        for key in ind1.keys():
            assert ind1[key] == ind2[key], f"Different values for {key}: {ind1[key]} vs {ind2[key]}"


@given(parameter_spaces())
@settings(deadline=None)
def test_genetic_generator_termination(param_space):
    """Test that the genetic parameter generator terminates after max_generations."""
    # Create a genetic parameter generator
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": 10,
            "max_generations": 3,  # Small number for testing
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Initialize the generator
    generator.initialize(scenario_config, {})
    
    # Run through all generations
    for _ in range(3):
        assert not generator.is_finished()
        population = generator.suggest_population()
        
        # Create evaluation results
        results = []
        for i, individual in enumerate(population):
            result = EvaluationResult(
                objective_value=float(i),
                metrics={"fitness": float(i)},
                window_results=[]
            )
            results.append(result)
        
        generator.report_population_results(population, results)
    
    # Check that the generator is now finished
    assert generator.is_finished()
    assert generator.generation_count == 3
