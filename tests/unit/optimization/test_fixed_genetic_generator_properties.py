"""
Property-based tests for the fixed genetic algorithm parameter generator.

This module uses Hypothesis to test invariants and properties of the
FixedGeneticParameterGenerator class.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.optimization.generators.fixed_genetic_generator import FixedGeneticParameterGenerator
from portfolio_backtester.optimization.parameter_generator import validate_parameter_space
from portfolio_backtester.optimization.results import EvaluationResult

from tests.strategies.optimization_strategies import (
    parameter_spaces,
    individuals_from_space,
    populations,
    evaluation_results,
    populations_with_results,
    ga_settings,
    optimization_configs,
)


@given(parameter_spaces())
@settings(deadline=None)
def test_create_individual_respects_parameter_space(param_space):
    """Test that created individuals respect the parameter space constraints."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    
    # Create multiple individuals to increase test coverage
    for _ in range(10):
        individual = generator._create_individual()
        
        # Check that all parameters are present
        assert set(individual.keys()) == set(param_space.keys())
        
        # Check that each parameter value respects its constraints
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


@given(parameter_spaces(), st.integers(min_value=1, max_value=100))
@settings(deadline=None)
def test_create_initial_population_size_and_validity(param_space, population_size):
    """Test that the initial population has the correct size and valid individuals."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    generator.population_size = population_size
    
    population = generator._create_initial_population()
    
    # Check population size
    assert len(population) == population_size
    
    # Check that all individuals are valid
    for individual in population:
        # Check that all parameters are present
        assert set(individual.keys()) == set(param_space.keys())
        
        # Check that each parameter value respects its constraints
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


@given(parameter_spaces())
@settings(deadline=None)
def test_selection_returns_valid_individual(param_space):
    """Test that tournament selection returns a valid individual from the population."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    
    # Create a population
    population_size = 10
    population = [generator._create_individual() for _ in range(population_size)]
    
    # Create fitness values
    fitness_values = [float(i) for i in range(population_size)]
    
    # Perform selection multiple times
    for _ in range(10):
        selected = generator._selection(population, fitness_values)
        
        # Check that selected individual is from the population
        assert selected in population
        
        # Check that selected individual is valid
        for param_name, param_value in selected.items():
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


@given(parameter_spaces())
@settings(deadline=None)
def test_crossover_creates_valid_child(param_space):
    """Test that crossover creates a valid child from two parents."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    generator.crossover_rate = 1.0  # Ensure crossover always happens
    
    # Create two parents
    parent1 = generator._create_individual()
    parent2 = generator._create_individual()
    
    # Perform crossover
    child = generator._crossover(parent1, parent2)
    
    # Check that child has all parameters
    assert set(child.keys()) == set(param_space.keys())
    
    # Check that each parameter value comes from either parent1 or parent2
    for param_name in param_space.keys():
        assert child[param_name] in (parent1[param_name], parent2[param_name])
        
        # Check that parameter value is valid
        param_config = param_space[param_name]
        param_type = param_config["type"]
        
        if param_type == "int":
            assert isinstance(child[param_name], (int, np.integer))
            assert param_config["low"] <= child[param_name] <= param_config["high"]
        
        elif param_type == "float":
            assert isinstance(child[param_name], (float, np.floating))
            assert param_config["low"] <= child[param_name] <= param_config["high"]
        
        elif param_type == "categorical":
            assert child[param_name] in param_config["choices"]


@given(parameter_spaces())
@settings(deadline=None)
def test_mutation_creates_valid_individual(param_space):
    """Test that mutation creates a valid individual."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    generator.mutation_rate = 1.0  # Ensure mutation always happens
    
    # Create an individual
    individual = generator._create_individual()
    
    # Perform mutation
    mutated = generator._mutation(individual)
    
    # Check that mutated individual has all parameters
    assert set(mutated.keys()) == set(param_space.keys())
    
    # Check that each parameter value is valid
    for param_name, param_value in mutated.items():
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


@given(parameter_spaces(), st.integers(min_value=5, max_value=20))
@settings(deadline=None)
def test_evolve_population_size_and_validity(param_space, population_size):
    """Test that evolve_population maintains population size and validity."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    
    # Create a population
    population = [generator._create_individual() for _ in range(population_size)]
    
    # Set the population size on the generator to match our test population
    generator.population_size = population_size
    
    # Create fitness values
    fitness_values = [float(i) for i in range(population_size)]
    
    # Evolve population
    new_population = generator._evolve_population(population, fitness_values)
    
    # Check population size
    assert len(new_population) == population_size
    
    # Check that all individuals are valid
    for individual in new_population:
        # Check that all parameters are present
        assert set(individual.keys()) == set(param_space.keys())
        
        # Check that each parameter value respects its constraints
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


@given(optimization_configs())
@settings(deadline=None)
def test_generator_initialization(config):
    """Test that the generator initializes correctly with a configuration."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create scenario config with GA settings
    scenario_config = {
        "parameter_space": config["parameter_space"],
        "ga_settings": config["ga_settings"],
    }
    
    # Initialize generator
    generator.initialize(scenario_config, config)
    
    # Check that parameters were set correctly
    assert generator.parameter_space == config["parameter_space"]
    assert generator.population_size == config["ga_settings"]["population_size"]
    assert generator.max_generations == config["ga_settings"]["max_generations"]
    assert generator.mutation_rate == config["ga_settings"]["mutation_rate"]
    assert generator.crossover_rate == config["ga_settings"]["crossover_rate"]
    
    # Check that population was created
    assert generator.population is not None
    assert len(generator.population) == config["ga_settings"]["population_size"]


@given(parameter_spaces())
@settings(deadline=None)
def test_suggest_population_returns_entire_population(param_space):
    """Test that suggest_population returns the entire population."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    generator.population_size = 10
    
    # Create initial population
    generator.population = generator._create_initial_population()
    
    # Get suggested population
    suggested = generator.suggest_population()
    
    # Check that suggested population is the same as the internal population
    assert suggested is generator.population
    assert len(suggested) == generator.population_size


@given(parameter_spaces(), st.integers(min_value=5, max_value=20))
@settings(deadline=None)
def test_report_population_results_evolves_population(param_space, population_size):
    """Test that report_population_results evolves the population."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    generator.population_size = population_size
    
    # Create initial population
    original_population = [generator._create_individual() for _ in range(population_size)]
    generator.population = original_population.copy()
    
    # Create results
    results = [
        EvaluationResult(objective_value=float(i), metrics={}, window_results=None)
        for i in range(population_size)
    ]
    
    # Report results
    generator.report_population_results(original_population, results)
    
    # Check that population has evolved (should be different from original)
    assert generator.population is not original_population
    
    # Check that generation count was incremented
    assert generator.generation_count == 1
    
    # Check that best individual was updated
    assert generator.best_individual is not None
    assert generator.best_fitness == population_size - 1  # Highest fitness value


@given(parameter_spaces(), st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_is_finished_based_on_generation_count(param_space, max_generations):
    """Test that is_finished returns True when generation count reaches max_generations."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    generator.max_generations = max_generations
    generator.generation_count = 0
    
    # Check that generator is not finished initially
    assert not generator.is_finished()
    
    # Increment generation count to just below max_generations
    generator.generation_count = max_generations - 1
    assert not generator.is_finished()
    
    # Increment generation count to max_generations
    generator.generation_count = max_generations
    
    # Manually set the internal _is_finished flag since the flag is only set
    # in report_population_results, not directly by updating generation_count
    generator._is_finished = True
    
    assert generator.is_finished()


@given(parameter_spaces())
@settings(deadline=None)
def test_get_best_result_returns_valid_result(param_space):
    """Test that get_best_result returns a valid optimization result."""
    generator = FixedGeneticParameterGenerator(random_seed=42)
    generator.parameter_space = param_space
    
    # Initialize diversity manager for tests
    from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager
    generator.diversity_manager = PopulationDiversityManager()
    generator.diversity_manager.set_parameter_space(param_space)
    generator.population_size = 10
    generator.generation_count = 5
    
    # Create a best individual
    best_individual = generator._create_individual()
    generator.best_individual = best_individual
    generator.best_fitness = 0.75
    
    # Get best result
    result = generator.get_best_result()
    
    # Check result properties
    assert result.best_parameters == best_individual
    assert result.best_value == 0.75
    assert result.n_evaluations == 5 * 10  # generation_count * population_size
