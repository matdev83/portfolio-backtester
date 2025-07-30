"""Integration tests for advanced crossover operators with GeneticOptimizer."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.portfolio_backtester.optimization.genetic_optimizer import GeneticOptimizer


class TestAdvancedCrossoverIntegration:
    """Test advanced crossover operators integration with GeneticOptimizer."""

    @pytest.fixture
    def mock_backtester(self):
        """Create a mock backtester instance."""
        backtester = Mock()
        backtester.args = Mock()
        backtester.args.early_stop_patience = 0
        backtester.args.timeout = None
        return backtester

    @pytest.fixture
    def mock_global_config(self):
        """Create a mock global configuration."""
        return {
            "optimizer_parameter_defaults": {
                "ga_num_generations": {"default": 5, "type": "int", "low": 1, "high": 100},
                "ga_sol_per_pop": {"default": 4, "type": "int", "low": 4, "high": 100},
                "ga_num_parents_mating": {"default": 2, "type": "int", "low": 1, "high": 50},
            }
        }

    @pytest.fixture
    def mock_scenario_config(self):
        """Create a mock scenario configuration."""
        return {
            "name": "test_advanced_crossover",
            "strategy_params": {
                "param1": 5,
                "param2": 0.5
            },
            "optimization_metric": "Sharpe",
            "genetic_algorithm_params": {
                "ga_num_generations": 2,
                "ga_sol_per_pop": 4,
                "ga_num_parents_mating": 2,
                "advanced_crossover_type": "simulated_binary",
                "sbx_distribution_index": 15.0
            },
            "optimize": [
                {"parameter": "param1", "type": "int", "min_value": 1, "max_value": 10, "step": 1},
                {"parameter": "param2", "type": "float", "min_value": 0.1, "max_value": 1.0}
            ]
        }

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        # Create simple mock data
        monthly_index = Mock()
        daily_index = Mock()
        rets_full = Mock()
        
        return monthly_index, daily_index, rets_full

    @patch('src.portfolio_backtester.optimization.genetic_optimizer.TrialEvaluator')
    @patch('src.portfolio_backtester.optimization.genetic_optimizer.pygad.GA')
    def test_simulated_binary_crossover_integration(self, mock_pygad_ga, mock_trial_evaluator, 
                                                   mock_backtester, mock_global_config, 
                                                   mock_scenario_config, mock_data):
        """Test that Simulated Binary Crossover is properly integrated."""
        monthly_index, daily_index, rets_full = mock_data
        
        # Mock the trial evaluator to return fixed fitness values
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.evaluate.return_value = [0.5]  # Simple fitness value
        mock_trial_evaluator.return_value = mock_evaluator_instance
        
        # Mock the PyGAD GA instance
        mock_ga_instance = Mock()
        mock_ga_instance.generations_completed = 2
        mock_ga_instance.best_solution_generation = 1
        mock_ga_instance.best_solution.return_value = (
            np.array([3.0, 0.7]),  # solution
            0.8,  # fitness
            0     # solution_idx
        )
        mock_ga_instance.last_generation_fitness = [0.5, 0.6, 0.7, 0.8]
        mock_pygad_ga.return_value = mock_ga_instance
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            scenario_config=mock_scenario_config,
            backtester_instance=mock_backtester,
            global_config=mock_global_config,
            monthly_data=monthly_index,
            daily_data=daily_index,
            rets_full=rets_full
        )
        
        # Run optimization
        best_params, num_evaluations, _ = optimizer.optimize(save_plot=False)
        
        # Verify that PyGAD was called with the correct crossover function
        call_kwargs = mock_pygad_ga.call_args[1]
        assert 'crossover_type' in call_kwargs
        assert callable(call_kwargs['crossover_type'])
        
        # Verify that SBX specific parameters were passed
        assert 'sbx_distribution_index' in call_kwargs
        assert call_kwargs['sbx_distribution_index'] == 15.0
        
        # Verify that the advanced crossover type was used instead of standard
        # This is a bit tricky to test directly, but we can check that the
        # crossover_type is a function (our custom one) rather than a string
        assert not isinstance(call_kwargs['crossover_type'], str)

    def test_get_ga_optimizer_parameter_defaults_includes_advanced_crossover(self):
        """Test that GA parameter defaults include advanced crossover options."""
        from src.portfolio_backtester.optimization.genetic_optimizer import get_ga_optimizer_parameter_defaults
        
        defaults = get_ga_optimizer_parameter_defaults()
        
        # Check that advanced crossover parameters are included
        assert "ga_advanced_crossover_type" in defaults
        assert "ga_sbx_distribution_index" in defaults
        assert "ga_num_crossover_points" in defaults
        assert "ga_uniform_crossover_bias" in defaults
        
        # Check that the advanced crossover type has the expected values
        advanced_crossover_defaults = defaults["ga_advanced_crossover_type"]
        assert advanced_crossover_defaults["type"] == "categorical"
        assert None in advanced_crossover_defaults["values"]
        assert "simulated_binary" in advanced_crossover_defaults["values"]
        assert "multi_point" in advanced_crossover_defaults["values"]
        assert "uniform_variant" in advanced_crossover_defaults["values"]
        assert "arithmetic" in advanced_crossover_defaults["values"]