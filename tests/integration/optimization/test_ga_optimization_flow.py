"""
Integration tests for the full genetic algorithm optimization flow.

This module tests the end-to-end flow of genetic algorithm optimization,
including population generation, evaluation, evolution, and result reporting.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, cast
from unittest.mock import MagicMock, patch

from portfolio_backtester.optimization.generators.fixed_genetic_generator import (
    FixedGeneticParameterGenerator
)
from portfolio_backtester.optimization.population_evaluator import PopulationEvaluator
from portfolio_backtester.optimization.population_orchestrator import PopulationOrchestrator
from portfolio_backtester.optimization.results import OptimizationData, OptimizationResult
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester


class MockBacktestEvaluator:
    """Mock backtester evaluator for testing."""
    
    def __init__(self, is_multi_objective=False):
        self.is_multi_objective = is_multi_objective
        self.evaluate_parameters_calls = 0
    
    def evaluate_parameters(self, params, scenario_config, data, backtester, previous_parameters=None):
        """Mock evaluation function that returns a simple result based on parameter values."""
        self.evaluate_parameters_calls += 1
        
        # Generate a deterministic score based on the parameters
        # For testing purposes, we'll make param_a more important than param_b
        param_a = params.get("param_a", 0)
        param_b = params.get("param_b", 0)
        value = param_a * 2 + param_b * 0.5
        
        # Create a result with this value
        from portfolio_backtester.optimization.results import EvaluationResult
        return EvaluationResult(
            objective_value=value,
            metrics={"score": value},
            window_results=[]
        )


def test_ga_optimization_flow():
    """Test the full genetic algorithm optimization flow."""
    # Create a simple parameter space
    param_space = {
        "param_a": {"type": "int", "low": 0, "high": 10},
        "param_b": {"type": "int", "low": 0, "high": 10}
    }
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": 10,
            "max_generations": 3,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Create a mock backtester
    backtester = MagicMock(spec=StrategyBacktester)
    
    # Create a mock optimization data
    data = MagicMock(spec=OptimizationData)
    
    # Create a mock evaluator
    evaluator = MockBacktestEvaluator()
    
    # Create the genetic parameter generator
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create the population evaluator
    pop_evaluator = PopulationEvaluator(evaluator=evaluator, n_jobs=1)
    
    # Create the population orchestrator
    orchestrator = PopulationOrchestrator(
        parameter_generator=generator,
        population_evaluator=pop_evaluator,
        timeout_seconds=None,
        early_stop_patience=None
    )
    
    # Run the optimization
    result = orchestrator.optimize(
        scenario_config=scenario_config,
        optimization_config={},
        data=data,
        backtester=backtester
    )
    
    # Check that the result is an OptimizationResult
    assert isinstance(result, OptimizationResult)
    
    # Check that the best parameters were found
    assert isinstance(result.best_parameters, dict)
    assert "param_a" in result.best_parameters
    assert "param_b" in result.best_parameters
    
    # Check that the best value is reasonable
    # The optimal value should be close to param_a=10, param_b=10 (value = 25)
    # But with random initialization and limited generations, we might not reach the absolute optimum
    assert result.best_value >= 20.0
    
    # Check that the correct number of evaluations were performed
    # 3 generations of 10 individuals = 30 evaluations
    # But with deduplication, it might be less
    assert evaluator.evaluate_parameters_calls <= 30
    
    # The optimization history might be empty in our mock implementation
    # Just check that it's a list
    assert isinstance(result.optimization_history, list)


def test_ga_optimization_with_early_stopping():
    """Test the genetic algorithm optimization with early stopping."""
    # Create a simple parameter space
    param_space = {
        "param_a": {"type": "int", "low": 0, "high": 10},
        "param_b": {"type": "int", "low": 0, "high": 10}
    }
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": 10,
            "max_generations": 10,  # More generations than we expect to run
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Create a mock backtester
    backtester = MagicMock(spec=StrategyBacktester)
    
    # Create a mock optimization data
    data = MagicMock(spec=OptimizationData)
    
    # Create a mock evaluator
    evaluator = MockBacktestEvaluator()
    
    # Create the genetic parameter generator
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create the population evaluator
    pop_evaluator = PopulationEvaluator(evaluator=evaluator, n_jobs=1)
    
    # Create the population orchestrator with early stopping
    orchestrator = PopulationOrchestrator(
        parameter_generator=generator,
        population_evaluator=pop_evaluator,
        timeout_seconds=1,  # 1 second timeout
        early_stop_patience=1  # Stop after 1 generation with no improvement
    )
    
    # Run the optimization
    result = orchestrator.optimize(
        scenario_config=scenario_config,
        optimization_config={},
        data=data,
        backtester=backtester
    )
    
    # Check that the result is an OptimizationResult
    assert isinstance(result, OptimizationResult)
    
    # Check that we didn't run all 10 generations
    assert evaluator.evaluate_parameters_calls < 10 * 10


def test_ga_optimization_with_deduplication():
    """Test the genetic algorithm optimization with deduplication."""
    # Create a very small parameter space to encourage duplicates
    param_space = {
        "param_a": {"type": "int", "low": 0, "high": 3},
        "param_b": {"type": "int", "low": 0, "high": 3}
    }
    
    # Create a scenario config with the parameter space
    scenario_config = {
        "parameter_space": param_space,
        "ga_settings": {
            "population_size": 10,
            "max_generations": 3,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
        }
    }
    
    # Create a mock backtester
    backtester = MagicMock(spec=StrategyBacktester)
    
    # Create a mock optimization data
    data = MagicMock(spec=OptimizationData)
    
    # Create a mock evaluator that counts unique parameter sets
    evaluator = MockBacktestEvaluator()
    
    # Create the genetic parameter generator
    generator = FixedGeneticParameterGenerator(random_seed=42)
    
    # Create the population evaluator
    pop_evaluator = PopulationEvaluator(evaluator=evaluator, n_jobs=1)
    
    # Create the population orchestrator
    orchestrator = PopulationOrchestrator(
        parameter_generator=generator,
        population_evaluator=pop_evaluator,
        timeout_seconds=None,
        early_stop_patience=None
    )
    
    # Run the optimization
    result = orchestrator.optimize(
        scenario_config=scenario_config,
        optimization_config={},
        data=data,
        backtester=backtester
    )
    
    # Check that the result is an OptimizationResult
    assert isinstance(result, OptimizationResult)
    
    # Check that the number of evaluations is less than the total population size
    # due to deduplication (with such a small parameter space, we expect duplicates)
    # 3 generations of 10 individuals = 30 evaluations
    # But with deduplication and a small parameter space (4x4=16 possible combinations),
    # we expect significantly fewer evaluations
    assert evaluator.evaluate_parameters_calls < 30
    
    # Check that the best parameters were found
    assert isinstance(result.best_parameters, dict)
    assert "param_a" in result.best_parameters
    assert "param_b" in result.best_parameters
    
    # The optimal value should be param_a=3, param_b=3 (value = 7.5)
    # But with random initialization and limited generations, we might not reach the absolute optimum
    assert result.best_value >= 6.0
