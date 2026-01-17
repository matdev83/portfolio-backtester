import pytest
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.optimization.population_evaluator import PopulationEvaluator
from src.portfolio_backtester.optimization.results import EvaluationResult

class TestPopulationEvaluator:
    @pytest.fixture
    def mock_backtest_evaluator(self):
        return MagicMock()

    @pytest.fixture
    def population_evaluator(self, mock_backtest_evaluator):
        return PopulationEvaluator(
            evaluator=mock_backtest_evaluator,
            n_jobs=1, # Sequential for simple tests
            enable_adaptive_batch_sizing=False,
            enable_hybrid_parallelism=False,
            enable_incremental_evaluation=False,
            enable_gpu_acceleration=False
        )

    def test_evaluate_population_sequential(self, population_evaluator, mock_backtest_evaluator):
        population = [{"p": 1}, {"p": 2}]
        config = {}
        data = MagicMock()
        backtester = MagicMock()
        
        # Mock results
        res1 = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
        res2 = EvaluationResult(objective_value=2.0, metrics={}, window_results=[])
        
        # Setup mock side effects for sequential evaluation
        mock_backtest_evaluator.evaluate_parameters.side_effect = [res1, res2]
        
        results = population_evaluator.evaluate_population(population, config, data, backtester)
        
        assert len(results) == 2
        assert results[0] == res1
        assert results[1] == res2
        assert mock_backtest_evaluator.evaluate_parameters.call_count == 2

    def test_evaluate_population_deduplication(self, population_evaluator, mock_backtest_evaluator):
        # Population with duplicates
        population = [{"p": 1}, {"p": 1}, {"p": 2}]
        
        res1 = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
        res2 = EvaluationResult(objective_value=2.0, metrics={}, window_results=[])
        
        # Should only evaluate unique params: p=1 and p=2
        mock_backtest_evaluator.evaluate_parameters.side_effect = [res1, res2]
        
        results = population_evaluator.evaluate_population(population, {}, MagicMock(), MagicMock())
        
        assert len(results) == 3
        # First and second result should be the same object (cached)
        assert results[0] == res1
        assert results[1] == res1
        assert results[2] == res2
        
        # Only 2 actual evaluations
        assert mock_backtest_evaluator.evaluate_parameters.call_count == 2

    def test_get_dedup_stats(self, population_evaluator):
        # By default deduplicator is mocked or minimal?
        # The constructor calls DeduplicationFactory.create_deduplicator
        # We can inspect the stats dict structure
        stats = population_evaluator.get_dedup_stats()
        assert isinstance(stats, dict)
        assert "local_cache_size" in stats
        assert "vectorized_tracking_enabled" in stats

    @patch("src.portfolio_backtester.optimization.population_evaluator.Parallel")
    def test_evaluate_parallel_logic(self, mock_parallel, mock_backtest_evaluator):
        # Initialize parallel evaluator
        evaluator = PopulationEvaluator(
            evaluator=mock_backtest_evaluator,
            n_jobs=2,
            enable_hybrid_parallelism=False
        )
        
        # Mock Parallel return
        res = EvaluationResult(objective_value=0.0, metrics={}, window_results=[])
        mock_parallel.return_value = lambda iterable: [res] * 2 # Returns callable that consumes generator
        
        population = [{"p": 1}, {"p": 2}]
        results = evaluator.evaluate_population(population, {}, MagicMock(), MagicMock())
        
        assert len(results) == 2
        # Should invoke Parallel
        mock_parallel.assert_called()