"""
Unit tests for the EvaluationEngine class.
"""

import pathlib

import pandas as pd
from typing import Any, cast
from unittest.mock import Mock, patch

from portfolio_backtester.backtester_logic.evaluation_engine import EvaluationEngine


_PKG_ROOT = pathlib.Path(__file__).resolve().parents[3] / "src" / "portfolio_backtester"


class TestEvaluationEngine:
    """Test suite for EvaluationEngine class."""

    def setup_method(self):
        """Set up the test environment."""
        self.global_config = {
            "data_source": {"type": "memory", "data": {}},
            "benchmark": "SPY",
        }
        self.data_source = Mock()
        self.strategy_manager = Mock()
        self.evaluation_engine = EvaluationEngine(
            self.global_config, self.data_source, self.strategy_manager
        )
        cast(Any, self.evaluation_engine).metrics_to_optimize = ["sharpe"]

    def test_initialization(self):
        """Test that the EvaluationEngine is initialized correctly."""
        assert self.evaluation_engine.global_config == self.global_config
        assert self.evaluation_engine.data_source == self.data_source
        assert self.evaluation_engine.strategy_manager == self.strategy_manager

    @patch("portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester")
    @patch("portfolio_backtester.optimization.evaluator.BacktestEvaluator")
    def test_evaluate_walk_forward_fast_single_objective(self, mock_evaluator, mock_backtester):
        """Test evaluate_walk_forward_fast with a single objective."""
        mock_evaluator.return_value.evaluate_parameters.return_value.objective_value = 0.5
        trial = Mock()
        trial.params = {}
        result = self.evaluation_engine.evaluate_walk_forward_fast(
            trial,
            {"strategy": "SimpleMomentumPortfolioStrategy"},
            [],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            ["sharpe"],
            False,
        )
        assert result == 0.5

    @patch("portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester")
    @patch("portfolio_backtester.optimization.evaluator.BacktestEvaluator")
    def test_evaluate_walk_forward_fast_multi_objective(self, mock_evaluator, mock_backtester):
        """Test evaluate_walk_forward_fast with multiple objectives."""
        mock_evaluator.return_value.evaluate_parameters.return_value.objective_value = [
            0.5,
            0.6,
        ]
        trial = Mock()
        trial.params = {}
        result = self.evaluation_engine.evaluate_walk_forward_fast(
            trial,
            {"strategy": "SimpleMomentumPortfolioStrategy"},
            [],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            ["sharpe", "calmar"],
            True,
        )
        assert result == (0.5, 0.6)

    def test_evaluation_engine_has_no_evaluate_fast_numba(self):
        assert not hasattr(EvaluationEngine, "evaluate_fast_numba")

    def test_backtester_has_no_evaluate_fast_numba(self):
        from portfolio_backtester.backtester_logic.backtester_facade import Backtester

        assert not hasattr(Backtester, "evaluate_fast_numba")

    def test_package_source_has_no_numba_walkforward_env_switches(self):
        for path in _PKG_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert (
                "ENABLE_NUMBA_WALKFORWARD" not in text
            ), f"{path} must not reference ENABLE_NUMBA_WALKFORWARD"
            assert (
                "DISABLE_NUMBA_WALKFORWARD" not in text
            ), f"{path} must not reference DISABLE_NUMBA_WALKFORWARD"

    def test_package_source_has_no_legacy_window_return_numba_helpers(self):
        for path in _PKG_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert (
                "run_backtest_numba" not in text
            ), f"{path} must not reference removed run_backtest_numba"
            assert (
                "_calculate_window_return_numba" not in text
            ), f"{path} must not reference removed _calculate_window_return_numba"

    @patch("portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester")
    @patch("portfolio_backtester.optimization.evaluator.BacktestEvaluator")
    def test_evaluate_fast_delegates_to_backtest_evaluator(self, mock_evaluator, mock_backtester):
        mock_evaluator.return_value.evaluate_parameters.return_value.objective_value = 0.5
        trial = Mock()
        trial.params = {}
        trial.user_attrs = {}
        result, _ = self.evaluation_engine.evaluate_fast(
            trial,
            {"strategy": "SimpleMomentumPortfolioStrategy"},
            [],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            ["sharpe"],
            False,
        )
        assert result == 0.5
        mock_evaluator.return_value.evaluate_parameters.assert_called_once()

    @patch("portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester")
    @patch("portfolio_backtester.optimization.evaluator.BacktestEvaluator")
    def test_evaluate_fast_returns_full_pnl_from_trial_user_attrs(
        self, mock_evaluator, mock_backtester
    ):
        mock_evaluator.return_value.evaluate_parameters.return_value.objective_value = 0.5
        trial = Mock()
        trial.params = {}
        trial.user_attrs = {
            "full_pnl_returns": {"2020-01-01": 0.1, "2020-01-02": 0.2},
        }
        _, pnl = self.evaluation_engine.evaluate_fast(
            trial,
            {"strategy": "SimpleMomentumPortfolioStrategy"},
            [],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            ["sharpe"],
            False,
        )
        assert len(pnl) == 2

    def test_evaluate_trial_parameters_returns_none(self):
        """Test evaluate_trial_parameters when returns is None."""
        run_scenario_func = Mock(return_value=None)
        result = self.evaluation_engine.evaluate_trial_parameters(
            {"strategy": "SimpleMomentumPortfolioStrategy"},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            run_scenario_func,
        )
        assert result == {"sharpe": 0.0}

    def test_evaluate_trial_parameters_returns_empty(self):
        """Test evaluate_trial_parameters when returns is empty."""
        run_scenario_func = Mock(return_value=pd.Series(dtype=float))
        result = self.evaluation_engine.evaluate_trial_parameters(
            {"strategy": "SimpleMomentumPortfolioStrategy"},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            run_scenario_func,
        )
        assert result == {"sharpe": 0.0}

    def test_evaluate_trial_parameters_passes_exposure_when_run_returns_tuple(self):
        idx = pd.bdate_range("2023-01-03", periods=3)
        returns = pd.Series([0.01, -0.005, 0.002], index=idx)
        weights = pd.DataFrame({"AAA": [1.0, 1.0, 0.5]}, index=idx)
        daily = pd.DataFrame({"SPY": [400.0, 401.0, 402.0]}, index=idx)
        monthly = pd.DataFrame({"SPY": [400.0]}, index=[idx[0]])
        run_scenario_func = Mock(return_value=(returns, weights))
        with patch(
            "portfolio_backtester.reporting.performance_metrics.calculate_metrics"
        ) as mock_cm:
            mock_cm.return_value = pd.Series({"Sharpe": 0.5})
            self.evaluation_engine.evaluate_trial_parameters(
                {"strategy": "SimpleMomentumPortfolioStrategy", "name": "trial_exp"},
                {},
                monthly,
                daily,
                pd.DataFrame(),
                run_scenario_func,
            )
        mock_cm.assert_called_once()
        pd.testing.assert_frame_equal(
            mock_cm.call_args.kwargs["exposure"],
            weights.reindex(returns.index),
        )
