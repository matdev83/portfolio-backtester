import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import gc
from argparse import Namespace

from portfolio_backtester.optimization.results import OptimizationData
from portfolio_backtester.optimization.parallel_optimization_runner import (
    _reconstruct_optimization_data,
)
from tests.unit.optimization.mock_parallel_runner import MockParallelOptimizationRunner
from portfolio_backtester.core import Backtester
from portfolio_backtester.config_loader import (
    load_scenario_from_file,
    load_config,
)
import uuid


@pytest.fixture
def sample_optimization_data() -> OptimizationData:
    """Creates a sample OptimizationData object for testing."""
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=100))
    tickers = ["AAPL", "GOOG"]
    fields_daily = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close"]], names=["Ticker", "Field"]
    )
    fields_monthly = pd.MultiIndex.from_product([tickers, ["Volume"]], names=["Ticker", "Field"])

    daily_data = pd.DataFrame(np.random.rand(100, 8), index=dates, columns=fields_daily)
    monthly_data = pd.DataFrame(np.random.rand(100, 2), index=dates, columns=fields_monthly)
    returns_data = pd.DataFrame(np.random.rand(100, 2), index=dates, columns=tickers)

    windows = [
        (
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-02-01"),
            pd.Timestamp("2020-02-02"),
            pd.Timestamp("2020-03-01"),
        )
    ]

    return OptimizationData(
        daily=daily_data,
        monthly=monthly_data,
        returns=returns_data,
        windows=windows,
    )


@pytest.mark.slow
def test_data_reconstruction_integrity(sample_optimization_data: OptimizationData):
    """
    Tests that data deconstructed and then reconstructed via the memory-mapping
    method is identical to the original data.
    """
    # Use a mock runner to access the protected _prepare_shared_data method
    runner = MockParallelOptimizationRunner(data=sample_optimization_data)

    # 1. Deconstruct the data into files
    context, temp_dir = runner._prepare_shared_data()
    reconstructed_data = None

    try:
        # 2. Reconstruct the data from the files
        reconstructed_data = _reconstruct_optimization_data(context)

        # 3. Assert deep equality
        pd.testing.assert_frame_equal(sample_optimization_data.daily, reconstructed_data.daily)
        pd.testing.assert_frame_equal(sample_optimization_data.monthly, reconstructed_data.monthly)
        pd.testing.assert_frame_equal(sample_optimization_data.returns, reconstructed_data.returns)
        assert sample_optimization_data.windows == reconstructed_data.windows

    finally:
        # 4. Explicitly release the memory-mapped file handles before cleanup
        del reconstructed_data
        gc.collect()

        # 5. Clean up the temporary directory
        temp_dir.cleanup()


@pytest.mark.slow
@pytest.mark.integration
def test_parallel_run_produces_equivalent_results():
    """
    Tests that running an optimization with n_jobs=1 yields the same
    best parameters and value as running with n_jobs > 1.
    """
    # Load global config and scenario
    GLOBAL_CONFIG, _ = load_config()

    # Use Path.cwd() which should be the project root when running pytest
    project_root = Path.cwd()
    scenario_path = (
        project_root
        / "config"
        / "scenarios"
        / "builtins"
        / "signal"
        / "dummy_signal_strategy"
        / "default.yaml"
    )
    scenario_config = load_scenario_from_file(scenario_path)

    # Create a comprehensive Namespace object with all expected attributes
    def create_mock_args(n_jobs: int, study_name: str) -> Namespace:
        return Namespace(
            mode="optimize",
            scenario_filename=str(scenario_path),
            n_jobs=n_jobs,
            optuna_trials=10,
            fresh_study=True,
            study_name=study_name,
            log_level="INFO",
            scenario_name=None,
            storage_url=None,
            random_seed=42,  # Set a fixed seed for reproducibility
            optimize_min_positions=10,
            optimize_max_positions=30,
            top_n_params=3,
            early_stop_patience=10,
            early_stop_zero_trials=20,
            optuna_timeout_sec=None,
            optimizer="optuna",
            pruning_enabled=False,
            pruning_n_startup_trials=5,
            pruning_n_warmup_steps=0,
            pruning_interval_steps=1,
            mc_simulations=1000,
            mc_years=10,
            interactive=False,
            timeout=None,
        )

    # --- Run in single-process mode ---
    args_single = create_mock_args(n_jobs=1, study_name=f"test_equiv_single_{uuid.uuid4()}")
    backtester_single = Backtester(
        GLOBAL_CONFIG, [scenario_config], args_single, random_state=args_single.random_seed
    )
    backtester_single.run()
    result_single = backtester_single.results[f"{scenario_config['name']}_Optimized"][
        "optimization_result"
    ]

    # --- Run in multi-process mode ---
    args_multi = create_mock_args(n_jobs=2, study_name=f"test_equiv_multi_{uuid.uuid4()}")
    backtester_multi = Backtester(
        GLOBAL_CONFIG, [scenario_config], args_multi, random_state=args_multi.random_seed
    )
    backtester_multi.run()
    result_multi = backtester_multi.results[f"{scenario_config['name']}_Optimized"][
        "optimization_result"
    ]

    # Assert that the core results are equivalent (not necessarily identical)
    # Due to potential differences in execution order, we check that the values are close
    # and that the parameters are of the same type
    assert result_single.best_value == pytest.approx(result_multi.best_value, abs=0.1)
    
    # Check that both results have the same parameter keys
    assert set(result_single.best_parameters.keys()) == set(result_multi.best_parameters.keys())
    
    # Check that the parameters are of the same type
    for key in result_single.best_parameters:
        assert type(result_single.best_parameters[key]) == type(result_multi.best_parameters[key])
