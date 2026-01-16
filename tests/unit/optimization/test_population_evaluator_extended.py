import pytest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
from src.portfolio_backtester.optimization.population_evaluator import PopulationEvaluator
from src.portfolio_backtester.optimization.results import EvaluationResult

class TestPopulationEvaluator:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = MagicMock()
        return evaluator

    @pytest.fixture
    def population_evaluator(self, mock_evaluator):
        # Disable fancy features for basic testing to avoid complex dependencies
        return PopulationEvaluator(
            evaluator=mock_evaluator,
            n_jobs=1,
            enable_adaptive_batch_sizing=False,
            enable_hybrid_parallelism=False,
            enable_incremental_evaluation=False,
            enable_gpu_acceleration=False
        )

    def test_evaluate_population_sequential(self, population_evaluator, mock_evaluator):
        population = [{"a": 1}, {"a": 2}]
        scenario_config = {}
        data = MagicMock()
        backtester = MagicMock()
        
        # Mock single evaluation result
        mock_result1 = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
        mock_result2 = EvaluationResult(objective_value=2.0, metrics={}, window_results=[])
        
        mock_evaluator.evaluate_parameters.side_effect = [mock_result1, mock_result2]
        
        results = population_evaluator.evaluate_population(population, scenario_config, data, backtester)
        
        assert len(results) == 2
        assert results[0].objective_value == 1.0
        assert results[1].objective_value == 2.0
        assert mock_evaluator.evaluate_parameters.call_count == 2

    def test_evaluate_population_caching(self, population_evaluator, mock_evaluator):
        population = [{"a": 1}, {"a": 1}] # Duplicate
        
        mock_result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
        mock_evaluator.evaluate_parameters.return_value = mock_result
        
        results = population_evaluator.evaluate_population(population, {}, MagicMock(), MagicMock())
        
        assert len(results) == 2
        # Should only call evaluate ONCE due to caching
        assert mock_evaluator.evaluate_parameters.call_count == 1
        assert results[0] == results[1]

    @patch('src.portfolio_backtester.optimization.population_evaluator.Parallel')
    @patch('src.portfolio_backtester.optimization.population_evaluator.delayed')
    def test_evaluate_parallel_fallback(self, mock_delayed, mock_parallel, mock_evaluator):
        # Test that we fall back to sequential if joblib fails or if n_jobs > 1 logic is triggered
        # Here we manually trigger parallel path but mock it to fail or return nothing to verify behavior
        
        # Force n_jobs > 1
        pop_eval = PopulationEvaluator(
            evaluator=mock_evaluator,
            n_jobs=2,
            enable_adaptive_batch_sizing=False,
            enable_hybrid_parallelism=False,
            enable_incremental_evaluation=False,
            enable_gpu_acceleration=False
        )
        
        # Mock Parallel to return results directly (mocking success case first)
        mock_parallel_instance = MagicMock()
        mock_parallel.return_value = mock_parallel_instance
        
        mock_result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
        # Parallel returns a list of results
        mock_parallel_instance.side_effect = [[mock_result]] 
        
        population = [{"a": 1}]
        
        # We need to mock evaluate_with_context or whatever is delayed
        # Since we can't easily import the worker function here (circular imports/complexity),
        # we rely on the fact that delayed() wraps something.
        
        results = pop_eval.evaluate_population(population, {}, MagicMock(), MagicMock())
        
        assert len(results) == 1
        assert results[0] == mock_result
        mock_parallel.assert_called()

    def test_dedup_stats(self, population_evaluator):
        stats = population_evaluator.get_dedup_stats()
        assert isinstance(stats, dict)
        assert "local_cache_size" in stats
        assert "batch_dedup_enabled" in stats

    def test_incremental_evaluation_logic(self, mock_evaluator):
        pop_eval = PopulationEvaluator(
            evaluator=mock_evaluator,
            n_jobs=1,
            enable_incremental_evaluation=True
        )
        
        population = [{"a": 1}, {"a": 2}]
        mock_result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
        mock_evaluator.evaluate_parameters.return_value = mock_result
        
        pop_eval.evaluate_population(population, {}, MagicMock(), MagicMock())
        
        # Check that previous_parameters was passed
        # First call: previous_parameters=None
        # Second call: previous_parameters={"a": 1}
        
        call_args_list = mock_evaluator.evaluate_parameters.call_args_list
        assert len(call_args_list) == 2
        
        # Check kwargs of calls
        assert call_args_list[0].kwargs.get('previous_parameters') is None
        assert call_args_list[1].kwargs.get('previous_parameters') == {"a": 1}
