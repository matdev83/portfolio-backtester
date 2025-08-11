
import unittest
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pytest

from portfolio_backtester.optimization.generators.fixed_genetic_generator import (
    FixedGeneticParameterGenerator,
)
from portfolio_backtester.optimization.parameter_generator import validate_parameter_space
from portfolio_backtester.optimization.results import EvaluationResult, OptimizationResult


class TestFixedGeneticParameterGenerator:
    """Test suite for FixedGeneticParameterGenerator."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.scenario_config = {
            "parameter_space": {
                "param1": {"type": "int", "low": 1, "high": 10},
                "param2": {"type": "float", "low": 0.1, "high": 1.0},
            }
        }
        self.optimization_config = {
            "ga_settings": {"population_size": 10, "max_generations": 5}
        }

    def test_initialization_and_population_generation(self):
        """Test that the generator initializes and creates a population."""
        # Arrange
        generator = FixedGeneticParameterGenerator(random_seed=42)

        # Act
        generator.initialize(self.scenario_config, self.optimization_config)
        population = generator.suggest_population()

        # Assert
        assert len(population) == 10
        for individual in population:
            assert "param1" in individual
            assert "param2" in individual
            assert 1 <= individual["param1"] <= 10
            assert 0.1 <= individual["param2"] <= 1.0

    def test_evolution(self):
        """Test a single step of evolution."""
        # Arrange
        generator = FixedGeneticParameterGenerator(random_seed=42)
        generator.initialize(self.scenario_config, self.optimization_config)
        initial_population = generator.suggest_population()

        # Dummy results
        results = [
            EvaluationResult(objective_value=float(i), metrics={}, window_results=[])
            for i in range(len(initial_population))
        ]

        # Act
        generator.report_population_results(initial_population, results)
        new_population = generator.suggest_population()

        # Assert
        assert len(new_population) == 10
        # Check that the new population is different from the old one
        assert initial_population != new_population


if __name__ == "__main__":
    unittest.main()
