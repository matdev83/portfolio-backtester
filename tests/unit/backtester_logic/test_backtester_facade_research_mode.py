import argparse
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
from portfolio_backtester.backtester_logic.backtester_facade import Backtester


@pytest.fixture
def sample_global_config() -> dict:
    return {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "benchmark": "SPY",
        "universe": ["AAPL", "MSFT", "GOOGL"],
    }


def test_run_research_validate_does_not_call_deferred_report_or_display(
    sample_global_config: dict,
) -> None:
    scenarios = [
        {
            "name": "alpha",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
        }
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
    )

    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    daily_ohlc = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=idx)
    monthly_data = pd.DataFrame()
    daily_closes = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=idx)
    rets_df = pd.DataFrame({"SPY": [0.01, 0.0]}, index=idx[:2])

    mock_cache = Mock()
    mock_cache.get_cached_returns.return_value = rets_df

    mock_timeout = Mock()
    mock_timeout.check_timeout.return_value = False

    orchestrator_instance = MagicMock()
    orchestrator_instance.run.return_value = {"status": "stub"}

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
        patch.object(Backtester, "_run_backtest_mode") as mock_backtest,
        patch.object(Backtester, "_run_optimize_mode") as mock_opt,
        patch.object(Backtester, "_display_results") as mock_display,
        patch(
            "portfolio_backtester.backtester_logic.execution.generate_deferred_report",
        ) as mock_deferred,
        patch(
            "portfolio_backtester.research.protocol_orchestrator.ResearchProtocolOrchestrator",
            return_value=orchestrator_instance,
        ),
    ):
        facade = Backtester(sample_global_config, scenarios, args, random_state=42)
        facade.run()

    mock_deferred.assert_not_called()
    mock_display.assert_not_called()
    mock_backtest.assert_not_called()
    mock_opt.assert_not_called()


def test_run_research_validate_prepares_data_once_and_dispatches_orchestrator(
    sample_global_config: dict,
) -> None:
    scenarios = [
        {
            "name": "alpha",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
        }
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
    )

    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    daily_ohlc = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=idx)
    monthly_data = pd.DataFrame()
    daily_closes = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=idx)
    rets_df = pd.DataFrame({"SPY": [0.01, 0.0]}, index=idx[:2])

    mock_cache = Mock()
    mock_cache.get_cached_returns.return_value = rets_df

    mock_timeout = Mock()
    mock_timeout.check_timeout.return_value = False

    orchestrator_instance = MagicMock()
    orchestrator_instance.run.return_value = {"status": "stub"}

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
        patch(
            "portfolio_backtester.research.protocol_orchestrator.ResearchProtocolOrchestrator",
            return_value=orchestrator_instance,
        ) as mock_orch_cls,
    ):
        facade = Backtester(sample_global_config, scenarios, args, random_state=42)
        facade.run()

    mock_prepare.assert_called_once()
    mock_orch_cls.assert_called_once()
    oc_args, oc_kwargs = mock_orch_cls.call_args
    assert oc_args == (facade.optimization_orchestrator, facade.backtest_runner, None)
    assert callable(oc_kwargs.get("optimization_orchestrator_factory"))
    orchestrator_instance.run.assert_called_once()
    mock_backtest.assert_not_called()
    mock_opt.assert_not_called()
    assert "alpha_ResearchValidation" in facade.results
    assert facade.results["alpha_ResearchValidation"] == {"status": "stub"}
