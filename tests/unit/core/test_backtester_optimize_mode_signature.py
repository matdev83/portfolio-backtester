import inspect
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Module under test
from src.portfolio_backtester.backtester_logic import execution as exec_mod
from src.portfolio_backtester.core import Backtester
import src.portfolio_backtester.core as core_mod
import pandas as pd


def test_run_optimize_mode_signature_stable():
    """Ensure the run_optimize_mode helper keeps its expected 4-argument signature.

    (self, scenario_config, monthly_data, daily_data, rets_full)
    If a developer changes this signature, the test will fail, prompting
    a corresponding update in the Backtester.run() call-site.
    """
    sig = inspect.signature(exec_mod.run_optimize_mode)

    # «self» plus exactly four additional parameters → 5 total
    assert len(sig.parameters) == 5, (
        "run_optimize_mode signature changed – adjust Backtester.run() call-site and tests accordingly"
    )


class _DummyDataSource:
    """Very lightweight replacement for the real data source used in tests."""

    def get_data(self, tickers, start_date, end_date):
        # Create a tiny DataFrame with business-day index and a single close column.
        idx = pd.date_range(start=start_date, end=end_date, freq="B")
        return pd.DataFrame({"TLT": np.ones(len(idx))}, index=idx)


class _DummyCache:
    """Stub cache that returns a zero matrix of matching shape."""

    def get_cached_returns(self, daily_closes, key):  # noqa: D401 – simple stub
        return pd.DataFrame(
            np.zeros_like(daily_closes), index=daily_closes.index, columns=daily_closes.columns
        )


def _create_minimal_backtester(tmp_path, mode: str = "optimize") -> Backtester:
    """Create a Backtester instance with minimal config and heavy components stubbed out."""

    # Minimal global configuration required by Backtester
    global_config = {
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
        "benchmark": "TLT",
        "universe": ["TLT"],
        "enable_synthetic_data": False,
    }

    # Single dummy scenario
    scenario = {
        "name": "dummy",
        "strategy": "intramonth_seasonal",
        "strategy_params": {"direction": "long"},
        "universe": ["TLT"],
        "rebalance_frequency": "M",  # required by portfolio logic
    }

    # Mimic the argparse Namespace that Backtester expects
    args = SimpleNamespace(
        mode=mode,
        log_level="INFO",
        scenario_name="dummy",
        timeout=None,
        n_jobs=1,
        early_stop_patience=10,
    )

    bt = Backtester(global_config, [scenario], args)

    # Replace heavy dependencies with lightweight stubs
    bt.data_source = _DummyDataSource()
    bt.data_cache = _DummyCache()

    # Patch _get_strategy so that it returns a dummy object with minimal API
    dummy_strategy = mock.Mock()
    # Strategy helper stubs so Backtester.run() does not crash on iterables
    dummy_strategy.get_non_universe_data_requirements.return_value = []
    dummy_strategy.get_universe.return_value = [("TLT", 1.0)]
    # Provide a timing controller stub returning an empty rebalance list (iterable)
    _timing = mock.Mock()
    _timing.get_rebalance_dates.return_value = []
    _timing.reset_state.return_value = None
    dummy_strategy.get_timing_controller.return_value = _timing
    bt._get_strategy = mock.Mock(return_value=dummy_strategy)

    # Skip the heavy optimisation logic itself – it’s enough that the call happens
    bt.run_optimization = mock.Mock(return_value=({}, 0, None))

    return bt


def test_backtester_run_optimize_mode_binding(tmp_path):
    """Ensure run_optimize_mode function is accessible and has the expected signature (no longer dynamically bound due to multiprocessing fix)."""
    bt = _create_minimal_backtester(tmp_path, mode="optimize")
    
    # After the multiprocessing fix, run_optimize_mode is no longer dynamically bound
    # Instead, it's called directly from the module. Test that the function exists and is callable.
    from src.portfolio_backtester.backtester_logic.execution import run_optimize_mode
    
    assert callable(run_optimize_mode)
    
    # Check the signature
    sig_original = inspect.signature(run_optimize_mode)
    
    # Should have 5 parameters: self, scenario_config, monthly_data, daily_data, rets_full
    assert len(sig_original.parameters) == 5
    
    # Check parameter names
    expected_params = ['self', 'scenario_config', 'monthly_data', 'daily_data', 'rets_full']
    actual_params = list(sig_original.parameters.keys())
    assert actual_params == expected_params


@mock.patch("src.portfolio_backtester.backtester_logic.strategy_logic.generate_signals")
def test_run_scenario_passes_callable_timeout(mock_gen_signals, tmp_path):
    """run_scenario should provide a *callable* timeout checker, not a boolean."""
    # Prepare minimal BT
    bt = _create_minimal_backtester(tmp_path, mode="backtest")
    # Stub get_timing_controller on strategy to avoid actual timing logic
    import pandas as pd
    dummy_timing = mock.Mock()
    dummy_timing.get_rebalance_dates.return_value = [pd.Timestamp("2024-01-31")]
    dummy_timing.reset_state.return_value = None
    dummy_strategy = bt._get_strategy.return_value
    dummy_strategy.get_timing_controller.return_value = dummy_timing

    # Ensure generate_signals returns empty DF to skip deeper logic
    mock_gen_signals.return_value = pd.DataFrame(
        {"TLT": [0.0]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )

    # Run scenario directly (public helper) – should NOT error
    scenario_cfg = bt.scenarios[0]
    bt.monthly_data = pd.DataFrame()
    bt.daily_data_ohlc = bt.data_source.get_data(["TLT"], "2024-01-01", "2024-01-05")

    # Provide monthly closes with at least the 'TLT' column so scenario is not skipped
    monthly_closes = bt.daily_data_ohlc.resample("BME").last()

    from src.portfolio_backtester.backtester_logic import portfolio_logic as _pl
    with mock.patch.object(_pl, "calculate_portfolio_returns", return_value=pd.Series(dtype=float)), \
         mock.patch.object(core_mod, "calculate_portfolio_returns", return_value=pd.Series(dtype=float)), \
         mock.patch.object(core_mod, "generate_signals", wraps=mock_gen_signals):
        bt.run_scenario(
            scenario_cfg,
            monthly_closes,
            bt.daily_data_ohlc,
            pd.DataFrame(),
        )

    # Validate call arg 5 is callable
    assert mock_gen_signals.call_args is not None, "generate_signals was not called"
    timeout_checker = mock_gen_signals.call_args[0][5]
    assert callable(timeout_checker), "Expected callable timeout checker, got non-callable type" 