import pytest
import numpy as np
from portfolio_backtester.optimization.adaptive_batch_sizing import AdaptiveBatchSizer

class TestAdaptiveBatchSizer:
    def test_initialization(self):
        """Test initialization with various parameters."""
        # Default initialization
        sizer = AdaptiveBatchSizer()
        assert sizer.min_batch_size == 1
        assert sizer.max_batch_size == 50
        assert sizer.target_cpu_utilization == 0.85
        assert sizer.n_jobs == 1
        
        # Custom initialization
        sizer_custom = AdaptiveBatchSizer(
            min_batch_size=5,
            max_batch_size=100,
            target_cpu_utilization=0.9,
            adaptation_rate=0.5,
            n_jobs=4
        )
        assert sizer_custom.min_batch_size == 5
        assert sizer_custom.max_batch_size == 100
        assert sizer_custom.target_cpu_utilization == 0.9
        assert sizer_custom.adaptation_rate == 0.5
        assert sizer_custom.n_jobs == 4
        
        # Auto n_jobs
        sizer_auto = AdaptiveBatchSizer(n_jobs=-1)
        assert sizer_auto.n_jobs == 1  # max(1, -1)
        
    def test_estimate_parameter_space_complexity(self):
        """Test complexity estimation for different parameter spaces."""
        sizer = AdaptiveBatchSizer()
        
        # Empty space
        assert sizer.estimate_parameter_space_complexity({}) == 1.0
        
        # Simple integer space
        simple_space = {
            "p1": {"type": "int", "low": 0, "high": 10, "step": 1}
        }
        # 10 possible values -> score 10
        # normalized: 10 / 1 -> 10.0 (capped at 10.0)
        assert sizer.estimate_parameter_space_complexity(simple_space) >= 1.0
        
        # Complex float space
        complex_space = {
            "p1": {"type": "float", "low": 0.0, "high": 1.0},
            "p2": {"type": "float", "low": 0.0, "high": 100.0},
            "p3": {"type": "categorical", "choices": ["a", "b", "c"]}
        }
        # Float: ~100 points -> 100 score (capped at 200)
        # Categorical: 3 choices -> 3 score
        # Total: ~203 / 3 => ~67. Normalized to 10.0 max.
        assert sizer.estimate_parameter_space_complexity(complex_space) == 10.0

    def test_analyze_population_diversity(self):
        """Test population diversity analysis."""
        sizer = AdaptiveBatchSizer()
        
        # Empty population
        assert sizer.analyze_population_diversity([]) == 0.0
        
        # Identical population (zero diversity)
        pop_identical = [
            {"p1": 1, "p2": 0.5},
            {"p1": 1, "p2": 0.5},
            {"p1": 1, "p2": 0.5}
        ]
        assert sizer.analyze_population_diversity(pop_identical) == 0.0
        
        # Diverse population
        pop_diverse = [
            {"p1": 1, "p2": 0.0},
            {"p1": 5, "p2": 0.5},
            {"p1": 10, "p2": 1.0}
        ]
        diversity = sizer.analyze_population_diversity(pop_diverse)
        assert diversity > 0.0
        assert diversity <= 1.0
        
        # Mixed types
        pop_mixed = [
            {"p1": "a", "p2": 1},
            {"p1": "b", "p2": 2},
            {"p1": "c", "p2": 3}
        ]
        diversity_mixed = sizer.analyze_population_diversity(pop_mixed)
        assert diversity_mixed > 0.0

    def test_update_batch_size(self):
        """Test batch size updating logic."""
        sizer = AdaptiveBatchSizer(
            min_batch_size=2,
            max_batch_size=20,
            n_jobs=2
        )
        
        param_space = {"p1": {"type": "int", "low": 0, "high": 100}}
        population = [{"p1": i} for i in range(10)]
        
        # Initial update
        result = sizer.update_batch_size(param_space, population)
        
        assert "batch_size" in result
        assert "batch_count" in result
        assert result["batch_size"] >= 2
        assert result["batch_size"] <= 20
        assert result["population_size"] == 10
        
        # Update with slow execution time -> should decrease batch size (or factor)
        result_slow = sizer.update_batch_size(param_space, population, execution_time_ms=1000.0)
        result_slower = sizer.update_batch_size(param_space, population, execution_time_ms=2000.0)
        
        assert result_slower["time_factor"] < 1.0
        
        # Update with fast execution time -> should increase batch size
        result_fast = sizer.update_batch_size(param_space, population, execution_time_ms=100.0)
        # We need history to trend down
        result_faster = sizer.update_batch_size(param_space, population, execution_time_ms=50.0)
        
        assert result_faster["time_factor"] > 1.0

    def test_get_batch_size_as_joblib_param(self):
        """Test joblib parameter formatting."""
        sizer = AdaptiveBatchSizer()
        
        # Force low complexity history
        sizer._complexity_history = [0.1] * 10
        assert sizer.get_batch_size_as_joblib_param() == "auto"
        
        # Force high complexity history
        sizer._complexity_history = [10.0] * 10
        assert isinstance(sizer.get_batch_size_as_joblib_param(), int)

    def test_reset(self):
        """Test reset functionality."""
        sizer = AdaptiveBatchSizer()
        sizer._complexity_history = [1.0, 2.0]
        sizer._batch_history = [5, 6]
        sizer._performance_history = [100.0]
        
        sizer.reset()
        
        assert len(sizer._complexity_history) == 0
        assert len(sizer._batch_history) == 0
        assert len(sizer._performance_history) == 0
