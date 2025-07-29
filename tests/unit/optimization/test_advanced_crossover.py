"""Tests for advanced crossover operators."""

import numpy as np
import pytest
from src.portfolio_backtester.optimization.advanced_crossover import (
    simulated_binary_crossover,
    multi_point_crossover,
    uniform_crossover_variant,
    arithmetic_crossover,
    get_crossover_operator,
    CROSSOVER_OPERATORS
)

class MockGAInstance:
    """Mock GA instance for testing."""
    def __init__(self):
        self.gene_space = [
            {'low': 0, 'high': 10},
            {'low': -5, 'high': 5},
            {'low': 0, 'high': 1}
        ]
        self.num_generations = 100
        self.sbx_distribution_index = 20.0
        self.num_crossover_points = 3
        self.uniform_crossover_bias = 0.5

def test_simulated_binary_crossover():
    """Test Simulated Binary Crossover operator."""
    # Create mock parents
    parents = np.array([
        [1.0, 2.0, 0.5],
        [3.0, 4.0, 0.8],
        [5.0, 6.0, 0.3],
        [7.0, 8.0, 0.9]
    ])
    
    offspring_size = (4, 3)
    ga_instance = MockGAInstance()
    
    # Test SBX
    offspring = simulated_binary_crossover(parents, offspring_size, ga_instance)
    
    # Check output shape
    assert offspring.shape == offspring_size
    
    # Check that offspring are within bounds
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            gene_space = ga_instance.gene_space[j]
            assert gene_space['low'] <= offspring[i, j] <= gene_space['high']

def test_multi_point_crossover():
    """Test Multi-point Crossover operator."""
    # Create mock parents
    parents = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, 20.0]
    ])
    
    offspring_size = (4, 5)
    ga_instance = MockGAInstance()
    ga_instance.num_crossover_points = 2
    
    # Test multi-point crossover
    offspring = multi_point_crossover(parents, offspring_size, ga_instance)
    
    # Check output shape
    assert offspring.shape == offspring_size
    
    # Check that all values come from parents (no new values created)
    parent_values = set(parents.flatten())
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            assert offspring[i, j] in parent_values

def test_uniform_crossover_variant():
    """Test Uniform Crossover Variant operator."""
    # Create mock parents
    parents = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    
    offspring_size = (2, 3)
    ga_instance = MockGAInstance()
    ga_instance.uniform_crossover_bias = 0.7
    
    # Test uniform crossover variant
    offspring = uniform_crossover_variant(parents, offspring_size, ga_instance)
    
    # Check output shape
    assert offspring.shape == offspring_size

def test_arithmetic_crossover():
    """Test Arithmetic Crossover operator."""
    # Create mock parents
    parents = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    
    offspring_size = (2, 3)
    ga_instance = MockGAInstance()
    
    # Test arithmetic crossover
    offspring = arithmetic_crossover(parents, offspring_size, ga_instance)
    
    # Check output shape
    assert offspring.shape == offspring_size
    
    # Check that offspring are within bounds
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            gene_space = ga_instance.gene_space[j]
            assert gene_space['low'] <= offspring[i, j] <= gene_space['high']

def test_get_crossover_operator():
    """Test getting crossover operators by name."""
    # Test that all operators can be retrieved
    for name in CROSSOVER_OPERATORS.keys():
        operator = get_crossover_operator(name)
        assert operator is not None
        assert callable(operator)
    
    # Test that invalid name returns None
    operator = get_crossover_operator("invalid_operator")
    assert operator is None

def test_crossover_operators_registry():
    """Test that all expected crossover operators are registered."""
    expected_operators = {
        "simulated_binary",
        "multi_point",
        "uniform_variant",
        "arithmetic"
    }
    
    assert set(CROSSOVER_OPERATORS.keys()) == expected_operators