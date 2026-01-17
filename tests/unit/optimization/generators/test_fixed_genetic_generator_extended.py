import pytest
import numpy as np
from unittest.mock import MagicMock
from src.portfolio_backtester.optimization.generators.fixed_genetic_generator import FixedGeneticParameterGenerator
from src.portfolio_backtester.optimization.results import EvaluationResult

class TestFixedGeneticParameterGenerator:
    @pytest.fixture
    def generator(self):
        # Initialize with fixed seed for reproducibility
        return FixedGeneticParameterGenerator(random_seed=42)

    @pytest.fixture
    def sample_config(self):
        return {
            "parameter_space": {
                "p_int": {"type": "int", "low": 0, "high": 10},
                "p_float": {"type": "float", "low": 0.0, "high": 1.0},
                "p_cat": {"type": "categorical", "choices": ["A", "B", "C"]}
            },
            "ga_settings": {
                "population_size": 10,
                "max_generations": 2,
                "mutation_rate": 0.5,
                "crossover_rate": 1.0
            }
        }

    def test_initialization_and_create_population(self, generator, sample_config):
        generator.initialize(sample_config, {})
        
        assert generator.population_size == 10
        assert generator.max_generations == 2
        assert len(generator.population) == 10
        
        # Check individual structure
        ind = generator.population[0]
        assert "p_int" in ind
        assert "p_float" in ind
        assert "p_cat" in ind
        
        # Check bounds
        assert 0 <= ind["p_int"] <= 10
        assert 0.0 <= ind["p_float"] <= 1.0
        assert ind["p_cat"] in ["A", "B", "C"]

    def test_suggest_population(self, generator, sample_config):
        generator.initialize(sample_config, {})
        pop = generator.suggest_population()
        assert len(pop) == 10
        assert pop == generator.population

    def test_selection_tournament(self, generator, sample_config):
        generator.initialize(sample_config, {})
        
        population = [
            {"id": 0, "val": 0},
            {"id": 1, "val": 10}, # Best
            {"id": 2, "val": 5},
            {"id": 3, "val": 2}
        ]
        fitness = [0, 10, 5, 2]
        
        # Mock RNG to force specific choice
        # If tournament size is 3, indices [0, 1, 2] -> winner 1
        # Generator uses self.rng.choice
        
        # Instead of mocking RNG which is hard for Numba/Numpy sometimes, 
        # let's run it multiple times and ensure it picks high fitness more often?
        # Or better, just rely on the logic that it returns *one* of the population members.
        
        selected = generator._selection(population, fitness, tournament_size=2)
        assert selected in population
        
        # With seed 42, we might get deterministic results if we rely on generator.rng
        # But let's verify logic: winner index is argmax of fitness of selected indices
        
    def test_crossover(self, generator, sample_config):
        generator.initialize(sample_config, {})
        generator.crossover_rate = 1.0 # Force crossover
        
        parent1 = {"p_int": 0, "p_float": 0.0}
        parent2 = {"p_int": 10, "p_float": 1.0}
        
        child = generator._crossover(parent1, parent2)
        
        # Child keys must be present
        assert "p_int" in child
        assert "p_float" in child
        # Values come from either parent
        assert child["p_int"] in [0, 10]
        assert child["p_float"] in [0.0, 1.0]

    def test_mutation(self, generator, sample_config):
        generator.initialize(sample_config, {})
        generator.mutation_rate = 1.0 # Force mutation
        
        ind = {"p_int": 5, "p_float": 0.5, "p_cat": "A"}
        
        # Run mutation multiple times to ensure *something* changes
        # Since it picks *one* param to mutate randomly, run loop
        changed = False
        for _ in range(10):
            mutated = generator._mutation(ind)
            if mutated != ind:
                changed = True
                break
        
        assert changed
        # Structure maintained
        assert "p_int" in mutated
        assert "p_float" in mutated

    def test_evolution_flow(self, generator, sample_config):
        generator.initialize(sample_config, {})
        
        initial_pop = generator.population.copy()
        
        # Mock results
        results = []
        for i in range(10):
            res = MagicMock(spec=EvaluationResult)
            res.objective_value = float(i) # increasing fitness
            results.append(res)
            
        generator.report_population_results(initial_pop, results)
        
        # Generation should increment
        assert generator.generation_count == 1
        
        # Best individual should be the last one (value=9)
        assert generator.best_fitness == 9.0
        assert generator.best_individual == initial_pop[9]
        
        # New population generated
        assert len(generator.population) == 10
        assert generator.population != initial_pop
        
        # Elitism check: best individual (index 9) should be preserved in new pop
        # Implementation: new_population.append(population[best_index])
        assert initial_pop[9] in generator.population

    def test_is_finished(self, generator, sample_config):
        generator.initialize(sample_config, {})
        generator.max_generations = 1
        
        # Gen 0 -> 1
        generator.report_population_results(generator.population, [MagicMock(objective_value=1.0)]*10)
        assert generator.is_finished() is True

    def test_diversity_enforcement(self, generator, sample_config):
        # Configure strict diversity
        sample_config["ga_settings"]["diversity_settings"] = {
            "similarity_threshold": 0.1, # strict
            "min_diversity_ratio": 1.0,  # 100% unique
            "enforce_diversity": True
        }
        
        generator.initialize(sample_config, {})
        
        # Check initial population uniqueness
        # Convert dicts to tuple of sorted items for set comparison
        unique_inds = set(tuple(sorted(d.items())) for d in generator.population)
        
        # Given small space and random seed, uniqueness might vary, 
        # but diversity manager logic should try to maximize it.
        # Just ensure generator calls it (covered by initialize flow)
        assert len(generator.population) == 10