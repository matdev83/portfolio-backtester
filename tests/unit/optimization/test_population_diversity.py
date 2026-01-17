import pytest
import numpy as np
from portfolio_backtester.optimization.population_diversity import PopulationDiversityManager

class TestPopulationDiversityManager:
    @pytest.fixture
    def param_space(self):
        return {
            "int_param": {"type": "int", "low": 0, "high": 10},
            "float_param": {"type": "float", "low": 0.0, "high": 1.0},
            "cat_param": {"type": "categorical", "choices": ["a", "b", "c"]}
        }

    @pytest.fixture
    def manager(self, param_space):
        manager = PopulationDiversityManager(
            similarity_threshold=0.9,
            min_diversity_ratio=0.5,
            enforce_diversity=True
        )
        manager.set_parameter_space(param_space)
        return manager

    def test_compute_similarity(self, manager):
        """Test similarity computation between individuals."""
        # Identical individuals
        ind1 = {"int_param": 5, "float_param": 0.5, "cat_param": "a"}
        ind2 = {"int_param": 5, "float_param": 0.5, "cat_param": "a"}
        assert manager.compute_similarity(ind1, ind2) == 1.0
        
        # Completely different
        ind3 = {"int_param": 0, "float_param": 0.0, "cat_param": "b"}
        # int: |5-0|/10 = 0.5 diff -> 0.5 sim
        # float: |0.5-0.0|/1.0 = 0.5 diff -> 0.5 sim
        # cat: different -> 0.0 sim
        # avg: (0.5 + 0.5 + 0.0) / 3 = 1.0 / 3 = 0.333...
        assert np.isclose(manager.compute_similarity(ind1, ind3), 1.0/3.0)

    def test_is_too_similar(self, manager):
        """Test detection of overly similar individuals."""
        pop = [
            {"int_param": 5, "float_param": 0.5, "cat_param": "a"}
        ]
        
        # Identical -> too similar
        ind1 = {"int_param": 5, "float_param": 0.5, "cat_param": "a"}
        assert manager.is_too_similar(ind1, pop)
        
        # Very close -> too similar (threshold 0.9)
        # int: 5 vs 5 -> 1.0
        # float: 0.5 vs 0.51 -> diff 0.01 -> sim 0.99
        # cat: a vs a -> 1.0
        # avg: (1+0.99+1)/3 = 2.99/3 = 0.996 > 0.9
        ind2 = {"int_param": 5, "float_param": 0.51, "cat_param": "a"}
        assert manager.is_too_similar(ind2, pop)
        
        # Different -> not too similar
        ind3 = {"int_param": 0, "float_param": 0.0, "cat_param": "b"}
        assert not manager.is_too_similar(ind3, pop)

    def test_analyze_population_diversity(self, manager):
        """Test population diversity metrics."""
        # 3 identical individuals
        pop = [
            {"int_param": 5, "float_param": 0.5, "cat_param": "a"},
            {"int_param": 5, "float_param": 0.5, "cat_param": "a"},
            {"int_param": 5, "float_param": 0.5, "cat_param": "a"}
        ]
        
        metrics = manager.analyze_population_diversity(pop)
        assert metrics["duplicate_count"] == 2
        assert metrics["unique_count"] == 1
        assert metrics["diversity_ratio"] == 1.0/3.0
        assert not metrics["is_diverse_enough"] # < 0.5

    def test_diversify_individual(self, manager):
        """Test individual diversification."""
        rng = np.random.default_rng(42)
        ind = {"int_param": 5, "float_param": 0.5, "cat_param": "a"}
        
        # Diversify
        div_ind = manager.diversify_individual(ind, rng)
        
        # Should be different but valid
        assert div_ind != ind
        assert 0 <= div_ind["int_param"] <= 10
        assert 0.0 <= div_ind["float_param"] <= 1.0
        assert div_ind["cat_param"] in ["a", "b", "c"]

    def test_diversify_population(self, manager):
        """Test population diversification."""
        rng = np.random.default_rng(42)
        # 5 identical individuals
        pop = [{"int_param": 5, "float_param": 0.5, "cat_param": "a"} for _ in range(5)]
        
        # Should diversify
        new_pop = manager.diversify_population(pop, rng)
        
        assert len(new_pop) == 5
        metrics = manager.analyze_population_diversity(new_pop)
        
        # Should have higher diversity
        assert metrics["diversity_ratio"] > 0.2  # at least better than 1/5
        assert metrics["unique_count"] > 1

    def test_create_random_individual(self, manager):
        """Test random individual creation."""
        ind = manager._create_random_individual()
        
        assert "int_param" in ind
        assert "float_param" in ind
        assert "cat_param" in ind
        
        assert 0 <= ind["int_param"] <= 10
        assert 0.0 <= ind["float_param"] <= 1.0
        assert ind["cat_param"] in ["a", "b", "c"]
