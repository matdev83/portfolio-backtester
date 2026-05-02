"""Tests for :mod:`portfolio_backtester.optimization.signal_cache`."""

from __future__ import annotations

from typing import Any

import pandas as pd

from portfolio_backtester.optimization.signal_cache import (
    SignalCache,
    compute_signal_matrix_cache_digest,
    index_fingerprint,
    signal_affecting_param_subset,
    strategy_allows_signal_matrix_cache,
)


def _digest(**overrides: Any) -> str:
    values: dict[str, Any] = {
        "strategy_module_qualname": "m.S",
        "universe_tickers": ("A",),
        "benchmark_ticker": "B",
        "non_universe_tickers": tuple(),
        "rebalance_dates_ns": (1, 2),
        "use_sparse_nan_for_inactive_rows": False,
        "timing_mode": "time_based",
        "timing_config": {"mode": "time_based"},
        "scenario_bounds": {"start_date": None},
        "strategy_params_slice": {},
        "feature_flags": {},
        "index_fp": (10, 0, "0:1"),
    }
    values.update(overrides)
    return compute_signal_matrix_cache_digest(
        strategy_module_qualname=str(values["strategy_module_qualname"]),
        universe_tickers=values["universe_tickers"],
        benchmark_ticker=str(values["benchmark_ticker"]),
        non_universe_tickers=values["non_universe_tickers"],
        rebalance_dates_ns=values["rebalance_dates_ns"],
        use_sparse_nan_for_inactive_rows=bool(values["use_sparse_nan_for_inactive_rows"]),
        timing_mode=str(values["timing_mode"]),
        timing_config=values["timing_config"],
        scenario_bounds=values["scenario_bounds"],
        strategy_params_slice=values["strategy_params_slice"],
        feature_flags=values["feature_flags"],
        index_fp=values["index_fp"],
    )


def test_digest_changes_with_strategy_params() -> None:
    d1 = _digest(strategy_params_slice={"x": 1})
    d2 = _digest(strategy_params_slice={"x": 2})
    assert d1 != d2


def test_digest_differs_timing_config() -> None:
    d1 = _digest(timing_config={"trade_execution_timing": "bar_close"})
    d2 = _digest(timing_config={"trade_execution_timing": "next_bar_open"})
    assert d1 != d2


def test_digest_differs_feature_flags() -> None:
    d1 = _digest(feature_flags={"strategy_data_context": False})
    d2 = _digest(feature_flags={"strategy_data_context": True})
    assert d1 != d2


def test_digest_differs_strategy_qualname() -> None:
    d1 = _digest(strategy_module_qualname="m.S1")
    d2 = _digest(strategy_module_qualname="m.S2")
    assert d1 != d2


def test_signal_affecting_param_subset_all_keys_without_metadata() -> None:
    class S:
        pass

    sub = signal_affecting_param_subset(S(), {"p": 1, "q": 2})
    assert set(sub.keys()) == {"p", "q"}


def test_signal_affecting_param_subset_declared_only() -> None:
    class S2:
        @classmethod
        def signal_affecting_parameter_names(cls) -> frozenset[str]:
            return frozenset({"p"})

    sub = signal_affecting_param_subset(S2(), {"p": 1, "q": 2})
    assert sub == {"p": 1}


def test_signal_affecting_param_subset_none_means_all() -> None:
    class S3:
        @classmethod
        def signal_affecting_parameter_names(cls) -> None:
            return None

    sub = signal_affecting_param_subset(S3(), {"p": 1})
    assert sub == {"p": 1}


def test_strategy_allows_cache_class_attr() -> None:
    class C:
        signal_matrix_cache_deterministic = True

    assert strategy_allows_signal_matrix_cache(C()) is True


def test_strategy_allows_cache_classmethod() -> None:
    class D:
        @classmethod
        def is_signal_matrix_cache_deterministic(cls) -> bool:
            return True

    assert strategy_allows_signal_matrix_cache(D()) is True


def test_signal_cache_stats_and_copy_put() -> None:
    c = SignalCache()
    df = pd.DataFrame([[1.0]], columns=["X"])
    c.put("k", df.copy(deep=True))
    g = c.get("k")
    assert g is not None
    assert c.stats()["hits"] == 1
    c.get("missing")
    assert c.stats()["misses"] == 1


def test_index_fingerprint() -> None:
    ix = pd.date_range("2020-01-01", periods=3, freq="B")
    fp = index_fingerprint(ix)
    assert fp[0] == 3
