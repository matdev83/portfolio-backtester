"""
Tests for the automated, context-aware study management system.
"""

import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
import optuna

from portfolio_backtester.optimization.parallel_optimization_runner import (
    ParallelOptimizationRunner,
)
from portfolio_backtester.optimization.results import OptimizationData
from portfolio_backtester.strategies._core.registry import get_strategy_registry
from tests.unit.optimization.dummy_strategy_for_context_test import DummyStrategyForContextTest


@pytest.fixture
def test_workspace(tmp_path: Path):
    """Creates a temporary workspace for tests."""
    workspace_dir = tmp_path / "test_workspace"
    workspace_dir.mkdir()

    # Create dummy strategy file by copying
    strategy_source_path = Path(__file__).parent / "dummy_strategy_for_context_test.py"
    strategy_dest_dir = workspace_dir / "strategies"
    strategy_dest_dir.mkdir()
    strategy_dest_file = strategy_dest_dir / "dummy_strategy_for_context_test.py"
    shutil.copy(strategy_source_path, strategy_dest_file)

    # Create scenario file
    scenario_dir = workspace_dir / "scenarios"
    scenario_dir.mkdir()
    scenario_file = scenario_dir / "test_scenario.yaml"
    scenario_file.write_text(
        """
name: test_context_scenario
strategy: DummyStrategyForContextTest
optimize:
  - parameter: param1
    min_value: 0.1
    max_value: 0.5
    step: 0.1
"""
    )

    # Create data directory
    (workspace_dir / "data" / "optuna" / "studies").mkdir(parents=True)

    yield {
        "workspace": workspace_dir,
        "strategy_file": strategy_dest_file,
        "scenario_file": scenario_file,
    }

    # Teardown
    try:
        shutil.rmtree(workspace_dir)
    except PermissionError:
        # On Windows, the DB file might be locked. The OS will clean it up.
        pass


@pytest.fixture
def mock_runner_dependencies():
    """Mock external dependencies for the runner."""

    # Simulate a persistent database for studies
    fake_db = {}

    with patch(
        "portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker"
    ) as mock_worker, patch(
        "portfolio_backtester.optimization.parallel_optimization_runner.get_strategy_registry"
    ) as mock_get_registry, patch(
        "portfolio_backtester.optimization.parallel_optimization_runner.os.path.exists"
    ) as mock_exists, patch(
        "portfolio_backtester.optimization.parallel_optimization_runner.os.remove"
    ) as mock_remove:

        mock_registry = Mock()
        mock_registry.get_strategy_class.return_value = DummyStrategyForContextTest
        mock_get_registry.return_value = mock_registry

        # Mocks that use the fake_db to simulate persistence
        def mock_create_study_func(study_name, storage, direction, load_if_exists):
            if load_if_exists and study_name in fake_db:
                return fake_db[study_name]

            study = Mock(spec=optuna.Study)
            study.trials = [Mock()]
            study.best_value = 1.0
            study.best_params = {"param1": 0.5}
            study.user_attrs = {}
            study.set_user_attr = lambda key, value: study.user_attrs.update({key: value})
            fake_db[study_name] = study
            return study

        def mock_load_study_func(study_name, storage):
            if study_name in fake_db:
                return fake_db[study_name]
            raise KeyError("Study not found")

        def mock_remove_func(path):
            # The study name is the key in our fake_db
            study_name_to_remove = "fixed_test_study"
            if study_name_to_remove in fake_db:
                del fake_db[study_name_to_remove]
            # Also call the mock to track the call
            mock_remove.return_value = None

        # os.path.exists should be true if the study is in our fake db
        mock_exists.side_effect = lambda path: "fixed_test_study" in fake_db
        mock_remove.side_effect = mock_remove_func

        with patch(
            "optuna.create_study", side_effect=mock_create_study_func
        ) as mock_create_study, patch(
            "optuna.load_study", side_effect=mock_load_study_func
        ) as mock_load_study:

            yield {
                "worker": mock_worker,
                "registry": mock_registry,
                "create_study": mock_create_study,
                "load_study": mock_load_study,
                "fake_db": fake_db,
                "remove": mock_remove,
            }


def _run_optimizer(workspace_data, fresh_study=False):
    """Helper function to run the optimization."""
    # Load scenario from the temporary file to detect changes
    import yaml
    with open(workspace_data["scenario_file"], "r") as f:
        scenario_config = yaml.safe_load(f)

    optimization_config = {
        "optuna_trials": 1,
        "parameter_space": {"param1": {"type": "float", "low": 0.1, "high": 0.5}},
    }

    # Mock data
    mock_data = Mock(spec=OptimizationData)

    runner = ParallelOptimizationRunner(
        scenario_config=scenario_config,
        optimization_config=optimization_config,
        data=mock_data,
        n_jobs=1,
        fresh_study=fresh_study,
        study_name="fixed_test_study",  # Use a fixed name for predictability
        storage_url=f"sqlite:///{workspace_data['workspace']}/data/optuna/studies/test_context_scenario.db",
    )
    runner.run()
    return runner


def test_resume_on_no_change(test_workspace, mock_runner_dependencies):
    """Test that the optimizer resumes the study when nothing has changed."""

    # Run once to create the study
    _run_optimizer(test_workspace)

    # Run again and check that the study is reused
    _run_optimizer(test_workspace)
    mock_runner_dependencies["remove"].assert_not_called()


def test_fresh_study_on_scenario_change(test_workspace, mock_runner_dependencies):
    """Test that a new study is created when the scenario file changes."""
    import yaml

    # Run once
    _run_optimizer(test_workspace)

    # Modify the scenario
    with open(test_workspace["scenario_file"], "r") as f:
        scenario = yaml.safe_load(f)
    scenario["optimize"][0]["max_value"] = 0.9 # Change a value
    with open(test_workspace["scenario_file"], "w") as f:
        yaml.dump(scenario, f)


    # Run again and check that the old study was removed
    _run_optimizer(test_workspace)
    mock_runner_dependencies["remove"].assert_called_once()


def test_fresh_study_on_strategy_change(test_workspace, mock_runner_dependencies):
    """Test that a new study is created when the strategy source code changes."""
    import importlib
    from tests.unit.optimization import dummy_strategy_for_context_test

    # Run once
    _run_optimizer(test_workspace)

    # Modify the strategy code by appending a comment
    with open(test_workspace["strategy_file"], "a") as f:
        f.write("\n# A change to trigger re-hash\n")

    # Force a reload of the module to defeat caching
    importlib.reload(dummy_strategy_for_context_test)

    # Run again and check that the old study was removed
    with patch(
        "portfolio_backtester.optimization.parallel_optimization_runner.get_strategy_source_path",
        return_value=test_workspace["strategy_file"],
    ):
        _run_optimizer(test_workspace)
    mock_runner_dependencies["remove"].assert_called_once()


def test_fresh_study_flag_override(test_workspace, mock_runner_dependencies):
    """Test that the --fresh-study flag forces a new study."""

    # Run once
    _run_optimizer(test_workspace)

    # Run again with the flag
    _run_optimizer(test_workspace, fresh_study=True)
    mock_runner_dependencies["remove"].assert_called_once()


# Test cases will go here...
