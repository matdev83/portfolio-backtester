"""
Integration test to verify that Optuna study name conflicts are resolved.

This test demonstrates that multiple optimization runs can execute
without study name conflicts.
"""

import pytest

from portfolio_backtester.optimization.study_utils import StudyNameGenerator
from portfolio_backtester.optimization.generators.optuna_generator import OptunaParameterGenerator


@pytest.mark.skipif(
    not pytest.importorskip("optuna", reason="Optuna not available"), reason="Optuna not available"
)
class TestStudyNameConflictResolution:
    """Test that study name conflicts are resolved."""

    def test_multiple_generators_use_unique_study_names(self):
        """Test that multiple generators create unique study names."""
        scenario_config = {"name": "test_scenario"}
        optimization_config = {
            "parameter_space": {
                "param1": {"type": "float", "low": 0.1, "high": 1.0},
            },
            "optimization_targets": [{"name": "value", "direction": "minimize"}],
            "max_evaluations": 5,
        }

        # Create multiple generators
        generators = []
        study_names = []

        for i in range(3):
            generator = OptunaParameterGenerator(random_state=42 + i)
            generator.initialize(scenario_config, optimization_config)
            generators.append(generator)
            study_names.append(generator.get_study_name())

        # All study names should be unique
        assert len(study_names) == len(
            set(study_names)
        ), f"Study names are not unique: {study_names}"

        # All should start with the base name
        for name in study_names:
            assert name.startswith(
                "test_scenario_optuna"
            ), f"Study name {name} doesn't start with expected prefix"

    def test_study_name_generator_produces_unique_names(self):
        """Test that the StudyNameGenerator produces unique names."""
        base_name = "test_optimization"

        # Generate multiple names using UUID for guaranteed uniqueness
        names = []
        for i in range(10):
            name = StudyNameGenerator.generate_unique_name(
                base_name, use_timestamp=False, use_uuid=True
            )
            names.append(name)

        # All names should be unique
        assert len(names) == len(set(names)), f"Generated names are not unique: {names}"

        # All should start with base name
        for name in names:
            assert name.startswith(base_name), f"Name {name} doesn't start with {base_name}"

    def test_test_study_names_are_unique(self):
        """Test that test study names are unique."""
        test_names = []

        for i in range(5):
            name = StudyNameGenerator.generate_test_study_name("optimization_test", f"method_{i}")
            test_names.append(name)

        # All names should be unique
        assert len(test_names) == len(
            set(test_names)
        ), f"Test study names are not unique: {test_names}"

        # All should follow the expected pattern
        for name in test_names:
            assert name.startswith(
                "test_optimization_test_method_"
            ), f"Test name {name} doesn't follow expected pattern"

    def test_parallel_study_creation_simulation(self):
        """Simulate parallel study creation to ensure no conflicts."""
        import concurrent.futures

        def create_study_name(thread_id):
            """Create a study name in a thread."""
            base_name = f"parallel_test_{thread_id}"
            return StudyNameGenerator.generate_unique_name(base_name)

        # Simulate parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_study_name, i) for i in range(10)]
            study_names = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All names should be unique even when created in parallel
        assert len(study_names) == len(
            set(study_names)
        ), f"Parallel study names are not unique: {study_names}"
