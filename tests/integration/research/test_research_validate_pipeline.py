"""Integration-style pipeline test for ``research_validate`` via ``Backtester.run``."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
from portfolio_backtester.backtester_logic.backtester_facade import Backtester
from portfolio_backtester.backtester_logic.optimization_orchestrator import (
    OptimizationOrchestrator,
)
from portfolio_backtester.optimization.results import OptimizationResult


def _research_scenario_dict() -> dict:
    inner = {
        "enabled": True,
        "type": "double_oos_wfo",
        "global_train_period": {"start_date": "2020-01-01", "end_date": "2022-12-31"},
        "unseen_test_period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "wfo_window_grid": {
            "train_window_months": [24],
            "test_window_months": [6],
            "wfo_step_months": [3],
            "walk_forward_type": ["rolling"],
        },
        "selection": {"top_n": 2, "metric": "Sharpe"},
        "final_unseen_mode": "fixed_selected_params",
        "lock": {"enabled": True, "refuse_overwrite": False},
        "reporting": {"enabled": True},
    }
    return {
        "name": "pipeline_scen",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {"x": 1},
        "extras": {"research_protocol": inner},
    }


def _research_scenario_dict_with_cost_sensitivity() -> dict:
    inner = {
        "enabled": True,
        "type": "double_oos_wfo",
        "global_train_period": {"start_date": "2020-01-01", "end_date": "2022-12-31"},
        "unseen_test_period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "wfo_window_grid": {
            "train_window_months": [24],
            "test_window_months": [6],
            "wfo_step_months": [3],
            "walk_forward_type": ["rolling"],
        },
        "selection": {"top_n": 2, "metric": "Sharpe"},
        "final_unseen_mode": "fixed_selected_params",
        "lock": {"enabled": True, "refuse_overwrite": False},
        "reporting": {"enabled": True},
        "cost_sensitivity": {
            "enabled": True,
            "slippage_bps_grid": [0.0],
            "commission_multiplier_grid": [1.0],
            "run_on": "unseen",
        },
    }
    return {
        "name": "pipeline_scen_cs",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {"x": 1},
        "extras": {"research_protocol": inner},
    }


def _research_scenario_dict_with_bootstrap() -> dict:
    inner = {
        "enabled": True,
        "type": "double_oos_wfo",
        "global_train_period": {"start_date": "2020-01-01", "end_date": "2022-12-31"},
        "unseen_test_period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "wfo_window_grid": {
            "train_window_months": [24],
            "test_window_months": [6],
            "wfo_step_months": [3],
            "walk_forward_type": ["rolling"],
        },
        "selection": {"top_n": 2, "metric": "Sharpe"},
        "final_unseen_mode": "fixed_selected_params",
        "lock": {"enabled": True, "refuse_overwrite": False},
        "reporting": {"enabled": True},
        "bootstrap": {
            "enabled": True,
            "n_samples": 12,
            "random_seed": 99,
            "random_wfo_architecture": {"enabled": True},
            "block_shuffled_returns": {"enabled": True, "block_size_days": 10},
        },
    }
    return {
        "name": "pipeline_scen_bs",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {"x": 1},
        "extras": {"research_protocol": inner},
    }


@pytest.fixture
def sample_global_config() -> dict:
    return {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "benchmark": "SPY",
        "universe": ["AAPL", "MSFT", "GOOGL"],
    }


def test_research_validate_pipeline_writes_heatmaps_when_enabled(
    tmp_path: Path,
    sample_global_config: dict,
) -> None:
    inner = {
        "enabled": True,
        "type": "double_oos_wfo",
        "global_train_period": {"start_date": "2020-01-01", "end_date": "2022-12-31"},
        "unseen_test_period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "wfo_window_grid": {
            "train_window_months": [24],
            "test_window_months": [6],
            "wfo_step_months": [3],
            "walk_forward_type": ["rolling"],
        },
        "selection": {"top_n": 2, "metric": "Sharpe"},
        "final_unseen_mode": "fixed_selected_params",
        "lock": {"enabled": True, "refuse_overwrite": False},
        "reporting": {
            "enabled": True,
            "generate_heatmaps": True,
            "heatmap_metrics": ["score"],
        },
    }
    scenarios = [
        {
            "name": "pipeline_scen_hm",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"x": 1},
            "extras": {"research_protocol": inner},
        },
    ]
    args = argparse.Namespace(
        mode="research_validate",
        scenario_name=None,
        timeout=None,
        n_jobs=1,
        early_stop_patience=10,
        study_name=None,
        random_seed=42,
        protocol="double_oos_wfo",
        force_new_research_run=False,
        research_skip_unseen=False,
        test_fast_optimize=False,
        research_artifact_base_dir=str(tmp_path),
    )

    idx = pd.bdate_range("2020-01-01", "2023-12-31")
    n = len(idx)
    daily_ohlc = pd.DataFrame({"SPY": 100.0 + np.arange(n, dtype=float) * 0.01}, index=idx)
    monthly_data = pd.DataFrame()
    daily_closes = daily_ohlc.copy()
    rets_df = daily_closes["SPY"].pct_change(fill_method=None).dropna().to_frame()
    rets_df.columns = ["SPY"]

    mock_cache = Mock()
    mock_cache.get_cached_returns.return_value = rets_df

    mock_timeout = Mock()
    mock_timeout.check_timeout.return_value = False

    def fake_run_optimization(
        _cell: object,
        _mo: object,
        daily_tr: pd.DataFrame,
        _rets_tr: pd.DataFrame,
        _args: object,
    ) -> OptimizationResult:
        stitched = pd.Series(0.001, index=daily_tr.index, dtype=float)
        return OptimizationResult(
            best_parameters={"a": 1.0},
            best_value=1.0,
            n_evaluations=4,
            optimization_history=[],
            stitched_returns=stitched,
        )

    def fake_run_backtest(
        _canon: object,
        _mo: object,
        daily_ut: pd.DataFrame,
        _rets_ut: pd.DataFrame,
        study_name: str | None = None,
    ) -> dict[str, pd.Series]:
        stitched = pd.Series(0.002, index=daily_ut.index, dtype=float)
        return {"returns": stitched}

    with (
        patch("portfolio_backtester.backtester_logic.backtester_facade.create_data_source"),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_cache_manager",
            return_value=mock_cache,
        ),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_timeout_manager",
            return_value=mock_timeout,
        ),
        patch.object(BacktestRunner, "run_scenario"),
        patch(
            "portfolio_backtester.backtester_logic.data_fetcher.DataFetcher.prepare_data_for_backtesting",
            return_value=(daily_ohlc, monthly_data, daily_closes),
        ),
        patch.object(Backtester, "_run_backtest_mode"),
        patch.object(Backtester, "_run_optimize_mode"),
        patch.object(Backtester, "_display_results"),
        patch(
            "portfolio_backtester.backtester_logic.execution.generate_deferred_report",
        ),
        patch.object(
            OptimizationOrchestrator,
            "run_optimization",
            side_effect=fake_run_optimization,
        ),
        patch.object(
            BacktestRunner,
            "run_backtest_mode",
            side_effect=fake_run_backtest,
        ),
    ):
        facade = Backtester(sample_global_config, scenarios, args, random_state=42)
        facade.run()

    key = "pipeline_scen_hm_ResearchValidation"
    result = facade.results[key]
    run_dir = Path(result.artifact_dir)
    assert (run_dir / "wfo_heatmap_score_step_3_rolling.png").is_file()


def test_research_validate_pipeline_artifacts_and_results(
    tmp_path: Path,
    sample_global_config: dict,
) -> None:
    scenarios = [_research_scenario_dict()]
    args = argparse.Namespace(
        mode="research_validate",
        scenario_name=None,
        timeout=None,
        n_jobs=1,
        early_stop_patience=10,
        study_name=None,
        random_seed=42,
        protocol="double_oos_wfo",
        force_new_research_run=False,
        research_skip_unseen=False,
        test_fast_optimize=False,
        research_artifact_base_dir=str(tmp_path),
    )

    idx = pd.bdate_range("2020-01-01", "2023-12-31")
    n = len(idx)
    daily_ohlc = pd.DataFrame({"SPY": 100.0 + np.arange(n, dtype=float) * 0.01}, index=idx)
    monthly_data = pd.DataFrame()
    daily_closes = daily_ohlc.copy()
    rets_df = daily_closes["SPY"].pct_change(fill_method=None).dropna().to_frame()
    rets_df.columns = ["SPY"]

    mock_cache = Mock()
    mock_cache.get_cached_returns.return_value = rets_df

    mock_timeout = Mock()
    mock_timeout.check_timeout.return_value = False

    def fake_run_optimization(
        _cell: object,
        _mo: object,
        daily_tr: pd.DataFrame,
        _rets_tr: pd.DataFrame,
        _args: object,
    ) -> OptimizationResult:
        stitched = pd.Series(0.001, index=daily_tr.index, dtype=float)
        return OptimizationResult(
            best_parameters={"a": 1.0},
            best_value=1.0,
            n_evaluations=4,
            optimization_history=[],
            stitched_returns=stitched,
        )

    def fake_run_backtest(
        _canon: object,
        _mo: object,
        daily_ut: pd.DataFrame,
        _rets_ut: pd.DataFrame,
        study_name: str | None = None,
    ) -> dict[str, pd.Series]:
        stitched = pd.Series(0.002, index=daily_ut.index, dtype=float)
        return {"returns": stitched}

    with (
        patch("portfolio_backtester.backtester_logic.backtester_facade.create_data_source"),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_cache_manager",
            return_value=mock_cache,
        ),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_timeout_manager",
            return_value=mock_timeout,
        ),
        patch.object(BacktestRunner, "run_scenario"),
        patch(
            "portfolio_backtester.backtester_logic.data_fetcher.DataFetcher.prepare_data_for_backtesting",
            return_value=(daily_ohlc, monthly_data, daily_closes),
        ) as mock_prepare,
        patch.object(Backtester, "_run_backtest_mode") as mock_backtest,
        patch.object(Backtester, "_run_optimize_mode") as mock_opt,
        patch.object(Backtester, "_display_results"),
        patch(
            "portfolio_backtester.backtester_logic.execution.generate_deferred_report",
        ),
        patch.object(
            OptimizationOrchestrator,
            "run_optimization",
            side_effect=fake_run_optimization,
        ),
        patch.object(
            BacktestRunner,
            "run_backtest_mode",
            side_effect=fake_run_backtest,
        ),
    ):
        facade = Backtester(sample_global_config, scenarios, args, random_state=42)
        facade.run()

    mock_prepare.assert_called_once()
    mock_backtest.assert_not_called()
    mock_opt.assert_not_called()

    key = "pipeline_scen_ResearchValidation"
    assert key in facade.results
    result = facade.results[key]
    assert result.scenario_name == "pipeline_scen"

    run_dir = Path(result.artifact_dir)
    assert tmp_path.resolve() in [p.resolve() for p in run_dir.parents]

    reg_yaml = tmp_path / "pipeline_scen" / "research_protocol" / "registry.yaml"
    assert reg_yaml.is_file()

    assert (run_dir / "wfo_architecture_grid.csv").is_file()
    assert (run_dir / "selected_protocols.yaml").is_file()
    assert (run_dir / "protocol_lock.yaml").is_file()
    assert (run_dir / "unseen_test_returns.csv").is_file()
    assert (run_dir / "unseen_test_metrics.yaml").is_file()
    assert (run_dir / "research_validation_report.md").is_file()


def test_research_validate_pipeline_writes_html_report_when_enabled(
    tmp_path: Path,
    sample_global_config: dict,
) -> None:
    inner = {
        "enabled": True,
        "type": "double_oos_wfo",
        "global_train_period": {"start_date": "2020-01-01", "end_date": "2022-12-31"},
        "unseen_test_period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        "wfo_window_grid": {
            "train_window_months": [24],
            "test_window_months": [6],
            "wfo_step_months": [3],
            "walk_forward_type": ["rolling"],
        },
        "selection": {"top_n": 2, "metric": "Sharpe"},
        "final_unseen_mode": "fixed_selected_params",
        "lock": {"enabled": True, "refuse_overwrite": False},
        "reporting": {"enabled": True, "generate_html": True},
    }
    scenarios = [
        {
            "name": "pipeline_scen_html",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {"x": 1},
            "extras": {"research_protocol": inner},
        },
    ]
    args = argparse.Namespace(
        mode="research_validate",
        scenario_name=None,
        timeout=None,
        n_jobs=1,
        early_stop_patience=10,
        study_name=None,
        random_seed=42,
        protocol="double_oos_wfo",
        force_new_research_run=False,
        research_skip_unseen=False,
        test_fast_optimize=False,
        research_artifact_base_dir=str(tmp_path),
    )

    idx = pd.bdate_range("2020-01-01", "2023-12-31")
    n = len(idx)
    daily_ohlc = pd.DataFrame({"SPY": 100.0 + np.arange(n, dtype=float) * 0.01}, index=idx)
    monthly_data = pd.DataFrame()
    daily_closes = daily_ohlc.copy()
    rets_df = daily_closes["SPY"].pct_change(fill_method=None).dropna().to_frame()
    rets_df.columns = ["SPY"]

    mock_cache = Mock()
    mock_cache.get_cached_returns.return_value = rets_df

    mock_timeout = Mock()
    mock_timeout.check_timeout.return_value = False

    def fake_run_optimization(
        _cell: object,
        _mo: object,
        daily_tr: pd.DataFrame,
        _rets_tr: pd.DataFrame,
        _args: object,
    ) -> OptimizationResult:
        stitched = pd.Series(0.001, index=daily_tr.index, dtype=float)
        return OptimizationResult(
            best_parameters={"a": 1.0},
            best_value=1.0,
            n_evaluations=4,
            optimization_history=[],
            stitched_returns=stitched,
        )

    def fake_run_backtest(
        _canon: object,
        _mo: object,
        daily_ut: pd.DataFrame,
        _rets_ut: pd.DataFrame,
        study_name: str | None = None,
    ) -> dict[str, pd.Series]:
        stitched = pd.Series(0.002, index=daily_ut.index, dtype=float)
        return {"returns": stitched}

    with (
        patch("portfolio_backtester.backtester_logic.backtester_facade.create_data_source"),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_cache_manager",
            return_value=mock_cache,
        ),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_timeout_manager",
            return_value=mock_timeout,
        ),
        patch.object(BacktestRunner, "run_scenario"),
        patch(
            "portfolio_backtester.backtester_logic.data_fetcher.DataFetcher.prepare_data_for_backtesting",
            return_value=(daily_ohlc, monthly_data, daily_closes),
        ),
        patch.object(Backtester, "_run_backtest_mode"),
        patch.object(Backtester, "_run_optimize_mode"),
        patch.object(Backtester, "_display_results"),
        patch(
            "portfolio_backtester.backtester_logic.execution.generate_deferred_report",
        ),
        patch.object(
            OptimizationOrchestrator,
            "run_optimization",
            side_effect=fake_run_optimization,
        ),
        patch.object(
            BacktestRunner,
            "run_backtest_mode",
            side_effect=fake_run_backtest,
        ),
    ):
        facade = Backtester(sample_global_config, scenarios, args, random_state=42)
        facade.run()

    key = "pipeline_scen_html_ResearchValidation"
    assert key in facade.results
    run_dir = Path(facade.results[key].artifact_dir)
    html_path = run_dir / "research_validation_report.html"
    assert html_path.is_file()
    text = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert "pipeline_scen_html" in text


def test_research_validate_pipeline_writes_cost_sensitivity_artifacts(
    tmp_path: Path,
    sample_global_config: dict,
) -> None:
    scenarios = [_research_scenario_dict_with_cost_sensitivity()]
    sample_gc = {
        **sample_global_config,
        "slippage_bps": 1.0,
        "commission_per_share": 0.005,
    }
    args = argparse.Namespace(
        mode="research_validate",
        scenario_name=None,
        timeout=None,
        n_jobs=1,
        early_stop_patience=10,
        study_name=None,
        random_seed=42,
        protocol="double_oos_wfo",
        force_new_research_run=False,
        research_skip_unseen=False,
        test_fast_optimize=False,
        research_artifact_base_dir=str(tmp_path),
    )

    idx = pd.bdate_range("2020-01-01", "2023-12-31")
    n = len(idx)
    daily_ohlc = pd.DataFrame({"SPY": 100.0 + np.arange(n, dtype=float) * 0.01}, index=idx)
    monthly_data = pd.DataFrame()
    daily_closes = daily_ohlc.copy()
    rets_df = daily_closes["SPY"].pct_change(fill_method=None).dropna().to_frame()
    rets_df.columns = ["SPY"]

    mock_cache = Mock()
    mock_cache.get_cached_returns.return_value = rets_df

    mock_timeout = Mock()
    mock_timeout.check_timeout.return_value = False

    def fake_run_optimization(
        _cell: object,
        _mo: object,
        daily_tr: pd.DataFrame,
        _rets_tr: pd.DataFrame,
        _args: object,
    ) -> OptimizationResult:
        stitched = pd.Series(0.001, index=daily_tr.index, dtype=float)
        return OptimizationResult(
            best_parameters={"a": 1.0},
            best_value=1.0,
            n_evaluations=4,
            optimization_history=[],
            stitched_returns=stitched,
        )

    def fake_run_backtest(
        _canon: object,
        _mo: object,
        daily_ut: pd.DataFrame,
        _rets_ut: pd.DataFrame,
        study_name: str | None = None,
    ) -> dict[str, pd.Series]:
        stitched = pd.Series(0.002, index=daily_ut.index, dtype=float)
        return {"returns": stitched}

    with (
        patch("portfolio_backtester.backtester_logic.backtester_facade.create_data_source"),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_cache_manager",
            return_value=mock_cache,
        ),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_timeout_manager",
            return_value=mock_timeout,
        ),
        patch.object(BacktestRunner, "run_scenario"),
        patch(
            "portfolio_backtester.backtester_logic.data_fetcher.DataFetcher.prepare_data_for_backtesting",
            return_value=(daily_ohlc, monthly_data, daily_closes),
        ),
        patch.object(Backtester, "_run_backtest_mode"),
        patch.object(Backtester, "_run_optimize_mode"),
        patch.object(Backtester, "_display_results"),
        patch(
            "portfolio_backtester.backtester_logic.execution.generate_deferred_report",
        ),
        patch.object(
            OptimizationOrchestrator,
            "run_optimization",
            side_effect=fake_run_optimization,
        ),
        patch.object(
            BacktestRunner,
            "run_backtest_mode",
            side_effect=fake_run_backtest,
        ),
    ):
        facade = Backtester(sample_gc, scenarios, args, random_state=42)
        facade.run()

    key = "pipeline_scen_cs_ResearchValidation"
    assert key in facade.results
    run_dir = Path(facade.results[key].artifact_dir)
    assert (run_dir / "cost_sensitivity.csv").is_file()
    assert (run_dir / "cost_sensitivity_summary.yaml").is_file()


def test_research_validate_pipeline_writes_bootstrap_artifacts(
    tmp_path: Path,
    sample_global_config: dict,
) -> None:
    scenarios = [_research_scenario_dict_with_bootstrap()]
    args = argparse.Namespace(
        mode="research_validate",
        scenario_name=None,
        timeout=None,
        n_jobs=1,
        early_stop_patience=10,
        study_name=None,
        random_seed=42,
        protocol="double_oos_wfo",
        force_new_research_run=False,
        research_skip_unseen=False,
        test_fast_optimize=False,
        research_artifact_base_dir=str(tmp_path),
    )

    idx = pd.bdate_range("2020-01-01", "2023-12-31")
    n = len(idx)
    daily_ohlc = pd.DataFrame({"SPY": 100.0 + np.arange(n, dtype=float) * 0.01}, index=idx)
    monthly_data = pd.DataFrame()
    daily_closes = daily_ohlc.copy()
    rets_df = daily_closes["SPY"].pct_change(fill_method=None).dropna().to_frame()
    rets_df.columns = ["SPY"]

    mock_cache = Mock()
    mock_cache.get_cached_returns.return_value = rets_df

    mock_timeout = Mock()
    mock_timeout.check_timeout.return_value = False

    def fake_run_optimization(
        _cell: object,
        _mo: object,
        daily_tr: pd.DataFrame,
        _rets_tr: pd.DataFrame,
        _args: object,
    ) -> OptimizationResult:
        stitched = pd.Series(0.001, index=daily_tr.index, dtype=float)
        return OptimizationResult(
            best_parameters={"a": 1.0},
            best_value=1.0,
            n_evaluations=4,
            optimization_history=[],
            stitched_returns=stitched,
        )

    def fake_run_backtest(
        _canon: object,
        _mo: object,
        daily_ut: pd.DataFrame,
        _rets_ut: pd.DataFrame,
        study_name: str | None = None,
    ) -> dict[str, pd.Series]:
        stitched = pd.Series(0.002, index=daily_ut.index, dtype=float)
        return {"returns": stitched}

    with (
        patch("portfolio_backtester.backtester_logic.backtester_facade.create_data_source"),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_cache_manager",
            return_value=mock_cache,
        ),
        patch(
            "portfolio_backtester.backtester_logic.backtester_facade.create_timeout_manager",
            return_value=mock_timeout,
        ),
        patch.object(BacktestRunner, "run_scenario"),
        patch(
            "portfolio_backtester.backtester_logic.data_fetcher.DataFetcher.prepare_data_for_backtesting",
            return_value=(daily_ohlc, monthly_data, daily_closes),
        ),
        patch.object(Backtester, "_run_backtest_mode"),
        patch.object(Backtester, "_run_optimize_mode"),
        patch.object(Backtester, "_display_results"),
        patch(
            "portfolio_backtester.backtester_logic.execution.generate_deferred_report",
        ),
        patch.object(
            OptimizationOrchestrator,
            "run_optimization",
            side_effect=fake_run_optimization,
        ),
        patch.object(
            BacktestRunner,
            "run_backtest_mode",
            side_effect=fake_run_backtest,
        ),
    ):
        facade = Backtester(sample_global_config, scenarios, args, random_state=42)
        facade.run()

    key = "pipeline_scen_bs_ResearchValidation"
    assert key in facade.results
    run_dir = Path(facade.results[key].artifact_dir)
    assert (run_dir / "bootstrap_significance.csv").is_file()
    assert (run_dir / "bootstrap_summary.yaml").is_file()
