from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.strategy_logic import (
    LegacyGenerateSignalsAdapter,
    StrategyTargetGenerationError,
    generate_signals,
)
from portfolio_backtester.canonical_config import CanonicalScenarioConfig


def _minimal_timing_controller(dates: pd.DatetimeIndex) -> MagicMock:
    tc = MagicMock()
    tc.reset_state.return_value = None
    tc.get_rebalance_dates.return_value = dates
    tc.should_generate_signal.return_value = True
    tc.update_signal_state.return_value = None
    tc.update_position_state.return_value = None
    return tc


class _LegacySignalStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def get_timing_controller(self) -> MagicMock:
        return _minimal_timing_controller(pd.DatetimeIndex([]))

    def get_non_universe_data_requirements(self) -> list:
        return []

    def generate_signals(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        tickers = list(kwargs["all_historical_data"].columns)
        w = pd.Series(1.0 / max(len(tickers), 1), index=tickers, dtype=float)
        return pd.DataFrame([w])


def test_missing_generate_target_weights_raises():
    idx = pd.date_range("2024-01-02", periods=5, freq="B")

    class Broken:
        def get_timing_controller(self):
            tc = _minimal_timing_controller(idx)
            return tc

        def get_non_universe_data_requirements(self):
            return []

    ohlc = pd.DataFrame({"A": range(100, 105), "B": range(200, 205)}, index=idx)
    cfg = CanonicalScenarioConfig.from_dict(
        {
            "name": "x",
            "strategy": "Dummy",
            "benchmark_ticker": "A",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }
    )
    with pytest.raises(StrategyTargetGenerationError, match="generate_target_weights"):
        generate_signals(Broken(), cfg, ohlc, ["A"], "A", lambda: False)


def test_generate_target_weights_non_dataframe_raises():
    idx = pd.date_range("2024-01-02", periods=3, freq="B")

    class BadFrame:
        def get_timing_controller(self):
            return _minimal_timing_controller(idx)

        def get_non_universe_data_requirements(self):
            return []

        def generate_target_weights(self, ctx):  # noqa: ARG002
            return []

    ohlc = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=idx)
    cfg = CanonicalScenarioConfig.from_dict(
        {
            "name": "y",
            "strategy": "Dummy",
            "benchmark_ticker": "A",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }
    )
    with pytest.raises(StrategyTargetGenerationError, match="pandas.DataFrame"):
        generate_signals(BadFrame(), cfg, ohlc, ["A"], "A", lambda: False)


def test_generate_target_weights_none_raises():
    idx = pd.date_range("2024-01-02", periods=3, freq="B")

    class NoneWeights:
        def get_timing_controller(self):
            return _minimal_timing_controller(idx)

        def get_non_universe_data_requirements(self):
            return []

        def generate_target_weights(self, ctx):  # noqa: ARG002
            return None

    ohlc = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=idx)
    cfg = CanonicalScenarioConfig.from_dict(
        {
            "name": "z",
            "strategy": "Dummy",
            "benchmark_ticker": "A",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }
    )
    with pytest.raises(StrategyTargetGenerationError, match="returned None"):
        generate_signals(NoneWeights(), cfg, ohlc, ["A"], "A", lambda: False)


def test_legacy_adapter_invokes_generate_signals_per_date():
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    inner = _LegacySignalStrategy()
    inner.get_timing_controller = lambda: _minimal_timing_controller(idx)  # type: ignore[method-assign]
    wrapped = LegacyGenerateSignalsAdapter(inner)
    ohlc = pd.DataFrame({"A": range(10, 14), "B": range(20, 24)}, index=idx)
    cfg = CanonicalScenarioConfig.from_dict(
        {
            "name": "leg",
            "strategy": "Dummy",
            "benchmark_ticker": "A",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }
    )
    out = generate_signals(wrapped, cfg, ohlc, ["A", "B"], "A", lambda: False)
    assert isinstance(out, pd.DataFrame)
    assert inner.calls == len(idx)


def test_generate_target_weights_failure_has_exception_chain():
    idx = pd.date_range("2024-01-02", periods=2, freq="B")

    class Boom:
        def get_timing_controller(self):
            return _minimal_timing_controller(idx)

        def get_non_universe_data_requirements(self):
            return []

        def generate_target_weights(self, ctx):  # noqa: ARG002
            raise ZeroDivisionError("boom")

    ohlc = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=idx)
    cfg = CanonicalScenarioConfig.from_dict(
        {
            "name": "e",
            "strategy": "Dummy",
            "benchmark_ticker": "A",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }
    )
    with pytest.raises(StrategyTargetGenerationError) as excinfo:
        generate_signals(Boom(), cfg, ohlc, ["A"], "A", lambda: False)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)
