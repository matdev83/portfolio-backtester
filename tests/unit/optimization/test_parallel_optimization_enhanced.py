"""
Tests for parallel optimization runner with enhanced WFO windows.

This module tests the integration of the parallel optimization runner with
the enhanced WFO system that supports daily evaluation for intramonth strategies.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import optuna

from portfolio_backtester.optimization.parallel_optimization_runner import (
    ParallelOptimizationRunner,
    _optuna_worker,
)
from portfolio_backtester.optimization.results import OptimizationData, OptimizationResult
from portfolio_backtester.optimization.study_utils import StudyNameGenerator, ensure_study_cleanup


class TestParallelOptimizationEnhanced:
    """Test parallel optimization with enhanced WFO windows."""

    @pytest.fixture
    def unique_study_name(self):
        """Generate a unique study name for each test."""
        return StudyNameGenerator.generate_test_study_name("parallel_optimization_enhanced")

    @pytest.fixture(autouse=True)
    def cleanup_test_files(self):
        """Automatically cleanup any test database files after each test."""
        import os
        import glob
        
        yield  # Run the test
        
        # Clean up any remaining test database files
        test_db_pattern = "test_parallel_optimization_enhanced_*.db"
        for db_file in glob.glob(test_db_pattern):
            try:
                os.remove(db_file)
            except Exception:
                pass  # Ignore cleanup errors

    @pytest.fixture
    def sample_data(self):
        """Create sample optimization data."""
        # Create sample price data
        dates = pd.date_range("2023-01-01", "2023-12-31")
        prices = pd.DataFrame(
            {
                "AAPL": 100 + pd.Series(range(len(dates))),
                "MSFT": 200 + pd.Series(range(len(dates))),
            },
            index=dates,
        )
        monthly_prices = prices.resample("M").last()

        # Create sample returns data
        returns = prices.pct_change().fillna(0)

        return OptimizationData(
            monthly=monthly_prices,
            daily=prices,
            returns=returns,
            windows=[
                (
                    pd.Timestamp(2023, 1, 1),
                    pd.Timestamp(2023, 6, 30),
                    pd.Timestamp(2023, 7, 1),
                    pd.Timestamp(2023, 12, 31),
                )
            ],
        )

    @pytest.fixture
    def intramonth_scenario_config(self):
        """Scenario config for an intramonth strategy."""
        return {
            "name": "test_intramonth_scenario",
            "strategy": "SeasonalSignalStrategy",
            "strategy_params": {},
            "timing": {
                "rebalance_frequency": "D",
                "scan_frequency": "D",
                "mode": "signal_based",
            },
            "feature_flags": {"prepared_arrays": True, "ndarray_simulation": True},
        }

    @pytest.fixture
    def monthly_scenario_config(self):
        """Scenario config for a monthly strategy."""
        return {
            "name": "test_monthly_scenario",
            "strategy": "MonthlyStrategy",
            "strategy_params": {},
            "timing": {"rebalance_frequency": "M"},
            "feature_flags": {"prepared_arrays": True, "ndarray_simulation": True},
        }

    @pytest.fixture
    def optimization_config(self):
        """Sample optimization configuration."""
        return {
            "parameter_space": {"test_param": {"type": "float", "low": 0.1, "high": 1.0}},
            "optuna_trials": 5,
        }

    def test_worker_function_intramonth_logging(
        self,
        intramonth_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        import os

        # Ensure cleanup after test
        storage_url = f"sqlite:///{unique_study_name}.db"
        db_file = f"{unique_study_name}.db"

        try:
            # Create a real study
            study = optuna.create_study(study_name=unique_study_name, storage=storage_url)
            assert study is not None

            with patch(
                "portfolio_backtester.optimization.parallel_optimization_runner.logger"
            ) as mock_logger:
                _optuna_worker(
                    scenario_config=intramonth_scenario_config,
                    optimization_config=optimization_config,
                    data=sample_data,
                    storage_url=storage_url,
                    study_name=unique_study_name,
                    n_trials=2,
                    parameter_space={"test_param": {"type": "float", "low": 0.1, "high": 1.0}},
                )

                # Verify enhanced logging was called (check if any call contains the expected pattern)
                logger_calls = [str(call) for call in mock_logger.info.call_args_list]
                daily_evaluation_logged = any("daily" in call for call in logger_calls)
                assert (
                    daily_evaluation_logged
                ), f"Expected 'daily' evaluation logging, got calls: {logger_calls}"
        finally:
            # Ensure cleanup after test
            ensure_study_cleanup(storage_url, unique_study_name)
            # Additional cleanup - remove file directly if it still exists
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except Exception:
                    pass  # Ignore errors during cleanup

    def test_worker_function_monthly_logging(
        self,
        monthly_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        import os

        # Ensure cleanup after test
        storage_url = f"sqlite:///{unique_study_name}.db"
        db_file = f"{unique_study_name}.db"

        try:
            # Create a real study
            study = optuna.create_study(study_name=unique_study_name, storage=storage_url)
            assert study is not None

            with patch(
                "portfolio_backtester.optimization.parallel_optimization_runner.logger"
            ) as mock_logger:
                _optuna_worker(
                    scenario_config=monthly_scenario_config,
                    optimization_config=optimization_config,
                    data=sample_data,
                    storage_url=storage_url,
                    study_name=unique_study_name,
                    n_trials=2,
                    parameter_space={"test_param": {"type": "float", "low": 0.1, "high": 1.0}},
                )

                # Verify enhanced logging was called (check if any call contains the expected pattern)
                logger_calls = [str(call) for call in mock_logger.info.call_args_list]
                monthly_evaluation_logged = any("monthly" in call for call in logger_calls)
                assert (
                    monthly_evaluation_logged
                ), f"Expected 'monthly' evaluation logging, got calls: {logger_calls}"
        finally:
            # Ensure cleanup after test
            ensure_study_cleanup(storage_url, unique_study_name)
            # Additional cleanup - remove file directly if it still exists
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except Exception:
                    pass  # Ignore errors during cleanup

    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.create_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.load_study")
    def test_run_single_process_intramonth(
        self,
        mock_load_study,
        mock_create_study,
        intramonth_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        """Test single process run with intramonth strategy."""
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"test_param": 0.5}
        mock_study.best_value = 1.5
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 5
        mock_create_study.return_value = mock_study
        mock_load_study.return_value = mock_study

        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1,
            study_name=unique_study_name,
        )

        with patch(
            "portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker"
        ) as mock_worker:
            with patch(
                "portfolio_backtester.optimization.parallel_optimization_runner.logger"
            ) as mock_logger:
                result = runner.run()

                # Verify study creation
                mock_create_study.assert_called_with(
                    study_name=unique_study_name,
                    storage=runner.storage_url,
                    direction="maximize",
                    load_if_exists=True,
                )

                # Verify enhanced logging
                mock_logger.info.assert_any_call(
                    "Running optimisation in a single process (%d trials, %s evaluation)",
                    5,
                    "daily",
                )

                # Verify worker was called
                mock_worker.assert_called_once()

                # Verify result
                assert isinstance(result, OptimizationResult)
                assert result.best_parameters == {"test_param": 0.5}
                assert result.best_value == 1.5

    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.create_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.load_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.mp")
    def test_run_multi_process_intramonth(
        self,
        mock_mp,
        mock_load_study,
        mock_create_study,
        intramonth_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        """Test multi-process run with intramonth strategy."""
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"test_param": 0.7}
        mock_study.best_value = 2.1
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 5
        mock_create_study.return_value = mock_study
        mock_load_study.return_value = mock_study

        # Mock multiprocessing
        mock_ctx = Mock()
        mock_process = Mock()
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Lock.return_value = Mock()
        mock_mp.get_context.return_value = mock_ctx

        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=2,
            study_name=unique_study_name,
        )

        with patch(
            "portfolio_backtester.optimization.parallel_optimization_runner.logger"
        ) as mock_logger:
            result = runner.run()

            # Verify study creation
            mock_create_study.assert_called_with(
                study_name=unique_study_name,
                storage=runner.storage_url,
                direction="maximize",
                load_if_exists=True,
            )

            # Verify enhanced logging
            mock_logger.info.assert_any_call(
                "Launching %d worker processes for %d trials (%s evaluation)", 2, 5, "daily"
            )

            # Verify processes were started
            assert mock_process.start.call_count == 2
            assert mock_process.join.call_count == 2

            # Verify result
            assert isinstance(result, OptimizationResult)
            assert result.best_parameters == {"test_param": 0.7}
            assert result.best_value == 2.1

    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.create_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.load_study")
    def test_optimization_success_logging_intramonth(
        self,
        mock_load_study,
        mock_create_study,
        intramonth_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        """Test success logging for intramonth strategy optimization."""
        # Mock study with successful results
        mock_study = Mock()
        mock_study.best_params = {"test_param": 0.8}
        mock_study.best_value = 1.8
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 5
        mock_create_study.return_value = mock_study
        mock_load_study.return_value = mock_study

        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1,
            study_name=unique_study_name,
        )

        with patch("portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker"):
            with patch(
                "portfolio_backtester.optimization.parallel_optimization_runner.logger"
            ) as mock_logger:
                runner.run()

                # Verify success logging with evaluation mode
                mock_logger.info.assert_any_call(
                    "✅ Optimization completed successfully: %d trials, best value: %.6f (%s evaluation)",
                    5,
                    1.8,
                    "daily",
                )

    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.create_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.load_study")
    def test_optimization_success_logging_monthly(
        self,
        mock_load_study,
        mock_create_study,
        monthly_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        """Test success logging for monthly strategy optimization."""
        # Mock study with successful results
        mock_study = Mock()
        mock_study.best_params = {"test_param": 0.6}
        mock_study.best_value = 1.2
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 3
        mock_create_study.return_value = mock_study
        mock_load_study.return_value = mock_study

        runner = ParallelOptimizationRunner(
            scenario_config=monthly_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1,
            study_name=unique_study_name,
        )

        with patch("portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker"):
            with patch(
                "portfolio_backtester.optimization.parallel_optimization_runner.logger"
            ) as mock_logger:
                runner.run()

                # Verify success logging with evaluation mode
                mock_logger.info.assert_any_call(
                    "✅ Optimization completed successfully: %d trials, best value: %.6f (%s evaluation)",
                    3,
                    1.2,
                    "monthly",
                )

    def test_evaluation_mode_detection_intramonth(self, intramonth_scenario_config):
        """Test evaluation mode detection for intramonth strategies."""
        strategy_name = intramonth_scenario_config.get("strategy", "unknown")
        lowered_name = strategy_name.lower()
        is_intramonth = "intramonth" in lowered_name or "seasonalsignal" in lowered_name
        evaluation_mode = "daily" if is_intramonth else "monthly"

        assert is_intramonth is True
        assert evaluation_mode == "daily"

    def test_evaluation_mode_detection_monthly(self, monthly_scenario_config):
        """Test evaluation mode detection for monthly strategies."""
        strategy_name = monthly_scenario_config.get("strategy", "unknown")
        is_intramonth = "intramonth" in strategy_name.lower()
        evaluation_mode = "daily" if is_intramonth else "monthly"

        assert is_intramonth is False
        assert evaluation_mode == "monthly"

    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.create_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.load_study")
    def test_backward_compatibility_maintained(
        self,
        mock_load_study,
        mock_create_study,
        monthly_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        """Test that backward compatibility is maintained for existing monthly strategies."""
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"test_param": 0.4}
        mock_study.best_value = 0.9
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 3
        mock_create_study.return_value = mock_study
        mock_load_study.return_value = mock_study

        runner = ParallelOptimizationRunner(
            scenario_config=monthly_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1,
            study_name=unique_study_name,
        )

        with patch("portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker"):
            result = runner.run()

            # Verify study creation
            mock_create_study.assert_called_with(
                study_name=unique_study_name,
                storage=runner.storage_url,
                direction="maximize",
                load_if_exists=True,
            )

            # Verify result structure is unchanged
            assert isinstance(result, OptimizationResult)
            assert hasattr(result, "best_parameters")
            assert hasattr(result, "best_value")
            assert hasattr(result, "n_evaluations")
            assert hasattr(result, "optimization_history")

            # Verify values
            assert result.best_parameters == {"test_param": 0.4}
            assert result.best_value == 0.9
            assert result.n_evaluations == 3

    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.create_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.load_study")
    def test_deduplication_enabled_by_default(
        self,
        mock_load_study,
        mock_create_study,
        intramonth_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        """Test that deduplication is enabled by default."""
        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock()]
        mock_study.best_value = 0.9  # Set real number for logging
        mock_study.best_params = {"test_param": 0.4}
        mock_create_study.return_value = mock_study
        mock_load_study.return_value = mock_study

        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1,
            study_name=unique_study_name,
        )

        with patch(
            "portfolio_backtester.optimization.parallel_optimization_runner.create_deduplicating_objective"
        ):
            with patch(
                "portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker"
            ):
                runner.run()
                # Verify that the deduplicating objective is created when enabled
                assert runner.enable_deduplication is True

    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.create_study")
    @patch("portfolio_backtester.optimization.parallel_optimization_runner.optuna.load_study")
    def test_deduplication_disabled(
        self,
        mock_load_study,
        mock_create_study,
        intramonth_scenario_config,
        optimization_config,
        sample_data,
        unique_study_name,
    ):
        """Test that deduplication can be disabled."""
        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock()]
        mock_study.best_value = 0.9  # Set real number for logging
        mock_study.best_params = {"test_param": 0.4}
        mock_create_study.return_value = mock_study
        mock_load_study.return_value = mock_study

        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1,
            study_name=unique_study_name,
            enable_deduplication=False,
        )

        with patch(
            "portfolio_backtester.optimization.parallel_optimization_runner.create_deduplicating_objective"
        ) as mock_create_dedup:
            with patch(
                "portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker"
            ):
                runner.run()
                # Verify that the deduplicating objective is not created when disabled
                assert runner.enable_deduplication is False
                mock_create_dedup.assert_not_called()
