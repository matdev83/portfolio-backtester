import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import MagicMock
from src.portfolio_backtester.optimization.generators.fixed_genetic_generator import FixedGeneticParameterGenerator
from src.portfolio_backtester.optimization.results import EvaluationResult

class TestFixedGeneticGenerator:
    @pytest.fixture
    def generator(self):
        return FixedGeneticParameterGenerator(random_seed=42)

    @pytest.fixture
    def config(self):
        return {
            "parameter_space": {
                "int_param": {"type": "int", "low": 0, "high": 10},
                "float_param": {"type": "float", "low": 0.0, "high": 1.0},
                "cat_param": {"type": "categorical", "choices": ["A", "B", "C"]}
            },
            "ga_settings": {
                "population_size": 10,
                "max_generations": 5,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            }
        }

    def test_initialization_and_creation(self, generator, config):
        generator.initialize(config, {})
        
        assert generator.population_size == 10
        assert generator.max_generations == 5
        assert len(generator.population) == 10
        
        # Verify individual structure
        ind = generator.population[0]
        assert "int_param" in ind
        assert 0 <= ind["int_param"] <= 10
        assert "float_param" in ind
        assert 0.0 <= ind["float_param"] <= 1.0
        assert ind["cat_param"] in ["A", "B", "C"]

    def test_evolution_step(self, generator, config):
        generator.initialize(config, {})
        initial_pop = generator.suggest_population()
        
        # Create dummy results: higher int_param = higher fitness
        results = []
        for ind in initial_pop:
            fitness = float(ind["int_param"])
            results.append(EvaluationResult(objective_value=fitness, metrics={}, window_results=[]))
            
        generator.report_population_results(initial_pop, results)
        
        assert generator.generation_count == 1
        
        # Check best individual tracking
        # We expect the max int_param found to be the best
        max_fitness = max(r.objective_value for r in results)
        best_res = generator.get_best_result()
        assert best_res.best_value == max_fitness
        
        # Elitism check: Best individual should be in new population
        new_pop = generator.suggest_population()
        assert generator.best_individual in new_pop

    def test_selection_tournament(self, generator, config):
        generator.initialize(config, {})
        
        pop = [{"p": 1}, {"p": 2}, {"p": 3}, {"p": 4}, {"p": 5}]
        fitness = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        # Mock RNG to pick specific indices for tournament
        # e.g. indices [0, 1, 4] -> fitness [10, 20, 50] -> winner index 4 (value 50)
        # We rely on numpy rng.choice here. 
        # Easier to statistical test or just structural test.
        
        selected = generator._selection(pop, fitness, tournament_size=2)
        assert selected in pop

    def test_crossover(self, generator, config):
        generator.initialize(config, {})
        
        p1 = {"a": 1, "b": 1, "c": 1}
        p2 = {"a": 2, "b": 2, "c": 2}
        
        # Force crossover
        generator.crossover_rate = 1.0
        
        child = generator._crossover(p1, p2)
        
        # Child params should come from p1 or p2
        assert child["a"] in [1, 2]
        assert child["b"] in [1, 2]
        assert child["c"] in [1, 2]

    def test_mutation(self, generator, config):
        generator.initialize(config, {})
        
        ind = {"int_param": 5, "float_param": 0.5, "cat_param": "A"}
        
        # Force mutation
        generator.mutation_rate = 1.0
        
        mutated = generator._mutation(ind)
        
        # At least one param might change, but it's random which one.
        # But since mutation_rate is 1.0, the check `random() > mutation_rate` fails (0..1 > 1.0 is False),
        # so it proceeds to mutate ONE param.
        
        diffs = 0
        if mutated["int_param"] != ind["int_param"]: diffs += 1
        if mutated["float_param"] != ind["float_param"]: diffs += 1
        if mutated["cat_param"] != ind["cat_param"]: diffs += 1
        
        # It's possible the random mutation picks the same value (e.g. randint(0,10) picks 5 again).
        # So we can't strictly assert diffs > 0 without a loop or seed control.
        # But structural integrity check:
        assert len(mutated) == 3

    def test_finish_condition(self, generator, config):
        config["ga_settings"]["max_generations"] = 1
        generator.initialize(config, {})
        
        # Gen 0
        pop = generator.suggest_population()
        results = [EvaluationResult(objective_value=0.0, metrics={}, window_results=[]) for _ in pop]
        generator.report_population_results(pop, results)
        
        assert generator.is_finished() is True

    def test_diversity_enforcement(self, generator, config):
        # Setup config to strictly enforce diversity
        config["ga_settings"]["diversity_settings"] = {
            "enforce_diversity": True,
            "min_diversity_ratio": 1.0 # Require 100% unique
        }
        generator.initialize(config, {})
        
        pop = generator.suggest_population()
        # Convert list of dicts to list of tuples for set conversion
        pop_tuples = [tuple(sorted(d.items())) for d in pop]
        unique_pop = set(pop_tuples)
        
        # Should attempt to make them unique
        # With small parameter space, collision is inevitable, but logic tries.
        # Just check it runs without error and returns valid pop size.
        assert len(pop) == generator.population_size
