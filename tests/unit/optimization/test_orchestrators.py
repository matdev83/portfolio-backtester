
import unittest
from unittest.mock import MagicMock, patch

from portfolio_backtester.optimization.sequential_orchestrator import SequentialOrchestrator
from portfolio_backtester.optimization.population_orchestrator import PopulationOrchestrator
from portfolio_backtester.optimization.results import EvaluationResult, OptimizationResult


class TestOrchestrators(unittest.TestCase):
    def setUp(self):
        self.mock_param_generator = MagicMock()
        self.mock_evaluator = MagicMock()
        self.mock_population_evaluator = MagicMock()
        self.scenario_config = {}
        self.optimization_config = {}
        self.data = MagicMock()
        self.backtester = MagicMock()

    def test_sequential_orchestrator(self):
        # Arrange
        self.mock_param_generator.is_finished.side_effect = [False, False, True]
        self.mock_param_generator.suggest_parameters.return_value = {"param1": 1}
        self.mock_evaluator.evaluate_parameters.return_value = EvaluationResult(
            objective_value=1.0, metrics={}, window_results=[]
        )
        self.mock_param_generator.get_best_result.return_value = OptimizationResult(
            best_parameters={"param1": 1},
            best_value=1.0,
            n_evaluations=2,
            optimization_history=[],
        )

        orchestrator = SequentialOrchestrator(
            parameter_generator=self.mock_param_generator, evaluator=self.mock_evaluator
        )

        # Act
        result = orchestrator.optimize(
            self.scenario_config, self.optimization_config, self.data, self.backtester
        )

        # Assert
        self.assertEqual(self.mock_param_generator.suggest_parameters.call_count, 2)
        self.assertEqual(self.mock_evaluator.evaluate_parameters.call_count, 2)
        self.assertEqual(self.mock_param_generator.report_result.call_count, 2)
        self.assertEqual(result.best_value, 1.0)

    def test_population_orchestrator(self):
        # Arrange
        self.mock_param_generator.is_finished.side_effect = [False, False, True]
        self.mock_param_generator.suggest_population.return_value = [{"param1": 1}, {"param1": 2}]
        self.mock_population_evaluator.evaluate_population.return_value = [
            EvaluationResult(objective_value=1.0, metrics={}, window_results=[]),
            EvaluationResult(objective_value=2.0, metrics={}, window_results=[]),
        ]
        self.mock_param_generator.get_best_result.return_value = OptimizationResult(
            best_parameters={"param1": 2},
            best_value=2.0,
            n_evaluations=4,
            optimization_history=[],
        )

        # Mock the evaluator inside PopulationEvaluator
        self.mock_population_evaluator.evaluator = self.mock_evaluator
        self.mock_evaluator.is_multi_objective = False


        orchestrator = PopulationOrchestrator(
            parameter_generator=self.mock_param_generator,
            population_evaluator=self.mock_population_evaluator,
        )

        # Act
        result = orchestrator.optimize(
            self.scenario_config, self.optimization_config, self.data, self.backtester
        )

        # Assert
        self.assertEqual(self.mock_param_generator.suggest_population.call_count, 2)
        self.assertEqual(self.mock_population_evaluator.evaluate_population.call_count, 2)
        self.assertEqual(self.mock_param_generator.report_population_results.call_count, 2)
        self.assertEqual(result.best_value, 2.0)


if __name__ == "__main__":
    unittest.main()
