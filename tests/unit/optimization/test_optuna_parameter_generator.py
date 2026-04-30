from typing import List, Tuple, cast

import pytest

import optuna

from portfolio_backtester.optimization.generators.optuna_generator import OptunaParameterGenerator
from portfolio_backtester.optimization.results import EvaluationResult
from portfolio_backtester.optimization.study_utils import StudyNameGenerator, ensure_study_cleanup


# A simple objective function for testing
def objective_function(trial):
    x = trial.suggest_float("param1", 0.1, 1.0)
    y = trial.suggest_int("param2", 1, 10)
    return float((x - 0.5) ** 2 + (y - 5) ** 2)


class TestOptunaParameterGeneratorIntegration:
    """
    Integration tests for the OptunaParameterGenerator using a real in-memory study.
    """

    def setup_method(self):
        """
        Set up the test environment before each test.
        """
        # Generate unique study name for this test instance
        self.unique_study_name = StudyNameGenerator.generate_test_study_name("optuna_generator")
        self.scenario_config = {"name": self.unique_study_name}
        self.optimization_config = {
            "parameter_space": {
                "param1": {"type": "float", "low": 0.1, "high": 1.0},
                "param2": {"type": "int", "low": 1, "high": 10},
            },
            "optimization_targets": [{"name": "value", "direction": "minimize"}],
            "max_evaluations": 15,
        }

        # Create a real in-memory study for testing
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective_function, n_trials=3)  # Populate with some trials

    def teardown_method(self):
        """
        Clean up after each test.
        """
        # Clean up any studies created during testing
        if hasattr(self, "unique_study_name"):
            ensure_study_cleanup("sqlite:///:memory:", self.unique_study_name)

    def test_initialization_with_real_study(self):
        """
        Test that the generator initializes correctly with a real study.
        """
        generator = OptunaParameterGenerator(random_state=42)
        # We can't easily inject a real study, so we test initialization logic
        # by ensuring it runs without error and sets up internal state.
        generator.initialize(self.scenario_config, self.optimization_config)
        assert generator._initialized
        assert generator.study is not None
        assert generator.study.study_name.startswith(f"{self.unique_study_name}_optuna")

    def test_suggest_parameters_from_real_study(self):
        """
        Test that suggest_parameters returns a valid set of parameters from a real study.
        """
        generator = OptunaParameterGenerator(random_state=42)
        generator.initialize(self.scenario_config, self.optimization_config)

        # Run a few suggestion cycles
        for _ in range(cast(int, self.optimization_config["max_evaluations"])):
            params = generator.suggest_parameters()
            assert "param1" in params
            assert "param2" in params
            assert 0.1 <= params["param1"] <= 1.0
            assert 1 <= params["param2"] <= 10

            # Simulate reporting a result to advance the generator
            result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
            generator.report_result(params, result)

    def test_report_and_get_best_result_from_real_study(self):
        """
        Test reporting results and getting the best result from a real study.
        """
        generator = OptunaParameterGenerator(random_state=42)
        generator.initialize(self.scenario_config, self.optimization_config)

        # Manually run a few trials
        for i in range(cast(int, self.optimization_config["max_evaluations"])):
            params = generator.suggest_parameters()
            value = (params["param1"] - 0.5) ** 2 + (params["param2"] - 5) ** 2
            result = EvaluationResult(
                objective_value=value, metrics={"value": value}, window_results=[]
            )
            generator.report_result(params, result)

        best_result = generator.get_best_result()

        assert best_result.best_parameters is not None
        assert "param1" in best_result.best_parameters
        assert "param2" in best_result.best_parameters
        assert best_result.best_value is not None
        assert best_result.n_evaluations == self.optimization_config["max_evaluations"]

    def test_get_parameter_importance_from_real_study(self):
        """
        Test that get_parameter_importance returns importance scores from a real study.
        """
        generator = OptunaParameterGenerator(random_state=42)
        generator.initialize(self.scenario_config, self.optimization_config)

        # Run optimization to populate the study
        for _ in range(10):  # More trials for better importance calculation
            params = generator.suggest_parameters()
            value = (params["param1"] - 0.5) ** 2 + (params["param2"] - 5) ** 2
            result = EvaluationResult(
                objective_value=value, metrics={"value": value}, window_results=[]
            )
            generator.report_result(params, result)

        importance = generator.get_parameter_importance()

        assert importance is not None
        assert "param1" in importance
        assert "param2" in importance
        assert importance["param1"] + importance["param2"] == pytest.approx(1.0)

    def test_is_finished_logic(self):
        """
        Test the is_finished logic.
        """
        generator = OptunaParameterGenerator(random_state=42)
        generator.initialize(self.scenario_config, self.optimization_config)

        for i in range(cast(int, self.optimization_config["max_evaluations"])):
            assert not generator.is_finished()
            params = generator.suggest_parameters()
            result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
            generator.report_result(params, result)

        assert generator.is_finished()

    def test_completed_optuna_trials_after_twenty_evaluations(self):
        unique_study_name = StudyNameGenerator.generate_test_study_name("optuna_gt20")
        scenario_config = {"name": unique_study_name}
        optimization_config = {
            "parameter_space": {
                "param1": {"type": "float", "low": 0.1, "high": 1.0},
                "param2": {"type": "int", "low": 1, "high": 10},
            },
            "optimization_targets": [{"name": "value", "direction": "minimize"}],
            "max_evaluations": 25,
        }
        generator = OptunaParameterGenerator(random_state=42)
        generator.initialize(scenario_config, optimization_config)
        study = generator.study
        assert study is not None
        for _ in range(25):
            params = generator.suggest_parameters()
            value = (params["param1"] - 0.5) ** 2 + (params["param2"] - 5) ** 2
            generator.report_result(
                params,
                EvaluationResult(objective_value=float(value), metrics={}, window_results=[]),
            )
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed) == 25
        assert len(study.trials) == 25

    def test_integer_seed_deterministic_across_twenty_plus_trials(self):
        def run_once() -> List[Tuple[float, int]]:
            unique_study_name = StudyNameGenerator.generate_test_study_name("optuna_det42")
            scenario_config = {"name": unique_study_name}
            optimization_config = {
                "parameter_space": {
                    "param1": {"type": "float", "low": 0.1, "high": 1.0},
                    "param2": {"type": "int", "low": 1, "high": 10},
                },
                "optimization_targets": [{"name": "value", "direction": "minimize"}],
                "max_evaluations": 25,
            }
            generator = OptunaParameterGenerator(random_state=42)
            generator.initialize(scenario_config, optimization_config)
            sequence: List[Tuple[float, int]] = []
            for _ in range(25):
                params = generator.suggest_parameters()
                sequence.append((params["param1"], int(params["param2"])))
                generator.report_result(
                    params, EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
                )
            return sequence

        assert run_once() == run_once()
