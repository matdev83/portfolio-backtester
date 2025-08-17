
import unittest
import time
from unittest.mock import MagicMock

from portfolio_backtester.optimization.population_evaluator import PopulationEvaluator
from portfolio_backtester.optimization.results import EvaluationResult
from portfolio_backtester.optimization.evaluator import BacktestEvaluator


# A simple function that can be pickled
def simple_evaluation_function(params, *args, **kwargs):
    return EvaluationResult(
        objective_value=params["param1"], metrics={}, window_results=[]
    )


def error_evaluation_function(params, *args, **kwargs):
    if params["param1"] == 5:
        raise ValueError("Test error")
    return EvaluationResult(
        objective_value=params["param1"], metrics={}, window_results=[]
    )


class TestPopulationEvaluation(unittest.TestCase):
    def setUp(self):
        self.mock_evaluator = MagicMock(spec=BacktestEvaluator)
        self.mock_evaluator.evaluate_parameters.side_effect = simple_evaluation_function
        self.scenario_config = {}
        self.data = MagicMock()
        self.backtester = MagicMock()
        self.population = [{"param1": i} for i in range(10)]

    def test_sequential_vs_parallel_equivalence(self):
        # Arrange
        # Using a real evaluator with a mock side effect
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        evaluator.evaluate_parameters = MagicMock(side_effect=simple_evaluation_function)


        # Act
        sequential_evaluator = PopulationEvaluator(evaluator, n_jobs=1)
        sequential_results = sequential_evaluator.evaluate_population(
            self.population, self.scenario_config, self.data, self.backtester
        )

        parallel_evaluator = PopulationEvaluator(evaluator, n_jobs=2)
        parallel_results = parallel_evaluator.evaluate_population(
            self.population, self.scenario_config, self.data, self.backtester
        )

        # Assert
        self.assertEqual(
            sorted([r.objective_value for r in sequential_results if r is not None]),
            sorted([r.objective_value for r in parallel_results if r is not None]),
        )

    def test_error_handling_in_parallel(self):
        # Arrange
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe"], is_multi_objective=False)
        evaluator.evaluate_parameters = MagicMock(side_effect=error_evaluation_function)

        # Act & Assert
        parallel_evaluator = PopulationEvaluator(evaluator, n_jobs=2)
        
        # The parallel execution should fail, and it should fall back to sequential,
        # which will then raise the error.
        with self.assertRaises(ValueError):
            parallel_evaluator.evaluate_population(
                self.population, self.scenario_config, self.data, self.backtester
            )


if __name__ == "__main__":
    unittest.main()
