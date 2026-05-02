"""Tests for research bootstrap significance helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.research.bootstrap import (
    block_shuffle_returns_preserving_blocks,
    bootstrap_block_shuffled_positions_p_value,
    bootstrap_block_shuffled_returns_p_value,
    bootstrap_random_strategy_parameters_p_value,
    bootstrap_random_wfo_architecture_p_value,
    run_research_bootstrap,
)
from portfolio_backtester.research.protocol_config import (
    BlockShuffledPositionsBootstrapConfig,
    BlockShuffledReturnsBootstrapConfig,
    BootstrapConfig,
    RandomStrategyParametersBootstrapConfig,
    RandomWfoArchitectureBootstrapConfig,
)
from portfolio_backtester.research.results import (
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitecture,
    WFOArchitectureResult,
)


def test_block_shuffle_preserves_length_and_values() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    s = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], index=idx)
    rng = np.random.default_rng(99)
    out = block_shuffle_returns_preserving_blocks(s, block_size_days=2, rng=rng)
    assert len(out) == len(s)
    assert set(out.values) == set(s.values)
    assert out.index.equals(s.index)


def test_random_wfo_p_value_deterministic_seed() -> None:
    pool = [0.1, 0.2, 0.3, 0.4, 0.5]
    p1 = bootstrap_random_wfo_architecture_p_value(
        eligible_scores=pool,
        selected_score=0.45,
        n_samples=500,
        rng=np.random.default_rng(42),
    )
    p2 = bootstrap_random_wfo_architecture_p_value(
        eligible_scores=pool,
        selected_score=0.45,
        n_samples=500,
        rng=np.random.default_rng(42),
    )
    assert p1 == p2


def test_block_shuffle_p_value_deterministic_seed() -> None:
    idx = pd.bdate_range("2024-01-01", periods=40)
    rets = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx)

    def metrics_fn(_x: pd.Series) -> dict[str, float]:
        return {"Total Return": float(_x.sum())}

    p1 = bootstrap_block_shuffled_returns_p_value(
        returns=rets,
        n_samples=100,
        block_size_days=10,
        survival_metric="Total Return",
        observed_value=0.02,
        metrics_from_returns=metrics_fn,
        rng=np.random.default_rng(42),
    )
    p2 = bootstrap_block_shuffled_returns_p_value(
        returns=rets,
        n_samples=100,
        block_size_days=10,
        survival_metric="Total Return",
        observed_value=0.02,
        metrics_from_returns=metrics_fn,
        rng=np.random.default_rng(42),
    )
    assert p1 == p2


def test_bootstrap_config_dataclass_fields() -> None:
    cfg = BootstrapConfig(
        enabled=True,
        n_samples=10,
        random_seed=7,
        random_wfo_architecture=RandomWfoArchitectureBootstrapConfig(enabled=True),
        block_shuffled_returns=BlockShuffledReturnsBootstrapConfig(
            enabled=True,
            block_size_days=20,
        ),
        block_shuffled_positions=BlockShuffledPositionsBootstrapConfig(
            enabled=False,
            block_size_days=20,
        ),
        random_strategy_parameters=RandomStrategyParametersBootstrapConfig(
            enabled=False,
            sample_size=100,
        ),
    )
    assert cfg.n_samples == 10
    assert cfg.block_shuffled_returns.block_size_days == 20
    assert cfg.random_strategy_parameters.sample_size == 100


def test_random_strategy_parameters_p_value_deterministic_with_seed() -> None:
    space: dict[str, list[Any]] = {"a": [1, 2, 3], "b": [0.1, 0.2]}

    def run_fn(sampled: Mapping[str, Any]) -> dict[str, float]:
        return {"Calmar": float(sampled["a"]) * float(sampled["b"])}

    p1 = bootstrap_random_strategy_parameters_p_value(
        selected_unseen_survival_value=0.25,
        n_samples=50,
        sample_size=30,
        param_space=space,
        run_with_params_fn=run_fn,
        survival_metric="Calmar",
        rng=np.random.default_rng(123),
    )
    p2 = bootstrap_random_strategy_parameters_p_value(
        selected_unseen_survival_value=0.25,
        n_samples=50,
        sample_size=30,
        param_space=space,
        run_with_params_fn=run_fn,
        survival_metric="Calmar",
        rng=np.random.default_rng(123),
    )
    assert p1 == p2


def test_random_strategy_parameters_p_value_skips_when_param_space_none() -> None:
    def run_fn(_sampled: Mapping[str, Any]) -> dict[str, float]:
        return {"Calmar": 1.0}

    p = bootstrap_random_strategy_parameters_p_value(
        selected_unseen_survival_value=0.5,
        n_samples=10,
        sample_size=5,
        param_space=None,
        run_with_params_fn=run_fn,
        survival_metric="Calmar",
        rng=np.random.default_rng(0),
    )
    assert np.isnan(p)


def test_random_strategy_parameters_p_value_counts_fraction_gte_selected() -> None:
    values = [0.2, 0.6, 0.6, 0.2, 0.7]
    idx = {"i": 0}

    def run_fn(_sampled: Mapping[str, Any]) -> dict[str, float]:
        i = idx["i"]
        idx["i"] = i + 1
        return {"Calmar": float(values[i])}

    p = bootstrap_random_strategy_parameters_p_value(
        selected_unseen_survival_value=0.5,
        n_samples=10,
        sample_size=5,
        param_space={"x": [1]},
        run_with_params_fn=run_fn,
        survival_metric="Calmar",
        rng=np.random.default_rng(0),
    )
    assert p == pytest.approx(0.6)


def test_run_research_bootstrap_includes_rsp_when_enabled() -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=3,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Calmar": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    idx = pd.date_range("2023-01-01", periods=2, freq="D")
    unseen = UnseenValidationResult(
        selected_protocol=sp,
        metrics={"Calmar": 0.4},
        returns=pd.Series([0.0, 0.01], index=idx),
        mode="fixed_selected_params",
        trade_history=None,
    )
    cfg = BootstrapConfig(
        enabled=True,
        n_samples=100,
        random_seed=9,
        random_wfo_architecture=RandomWfoArchitectureBootstrapConfig(enabled=False),
        block_shuffled_returns=BlockShuffledReturnsBootstrapConfig(
            enabled=False,
            block_size_days=20,
        ),
        block_shuffled_positions=BlockShuffledPositionsBootstrapConfig(
            enabled=False,
            block_size_days=20,
        ),
        random_strategy_parameters=RandomStrategyParametersBootstrapConfig(
            enabled=True,
            sample_size=4,
        ),
    )

    def metrics_from_returns(_s: pd.Series) -> dict[str, float]:
        return {}

    calls = {"i": 0}

    def run_with_params(_m: Mapping[str, Any]) -> dict[str, float]:
        calls["i"] += 1
        return {"Calmar": 0.3}

    out = run_research_bootstrap(
        cfg=cfg,
        grid_results=[war],
        selected=sp,
        unseen_result=unseen,
        selection_metric="Calmar",
        metrics_from_returns=metrics_from_returns,
        param_space={"k": [1, 2]},
        run_with_params_fn=run_with_params,
    )
    assert out is not None
    summary, rows = out
    rsp = summary["random_strategy_parameters"]
    assert rsp["enabled"] is True
    assert isinstance(rsp["p_value"], float)
    assert rsp["sample_size"] == 4
    rsp_rows = [r for r in rows if r.get("test") == "random_strategy_parameters"]
    assert len(rsp_rows) == 1
    assert rsp_rows[0]["n_samples"] == 4


def test_block_shuffled_positions_p_value_skips_when_trade_history_empty() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    asset = pd.DataFrame({"A": [0.01] * 4}, index=idx)

    def metrics_fn(s: pd.Series) -> dict[str, float]:
        return {"Total Return": float(s.sum())}

    p = bootstrap_block_shuffled_positions_p_value(
        trade_history=pd.DataFrame(),
        asset_returns=asset,
        returns_fallback=None,
        n_samples=10,
        block_size_days=2,
        survival_metric="Total Return",
        observed_value=0.0,
        metrics_from_returns=metrics_fn,
        rng=np.random.default_rng(0),
    )
    assert np.isnan(p)


def test_block_shuffled_positions_p_value_deterministic_with_seed() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    asset = pd.DataFrame({"A": np.linspace(0.001, -0.001, len(idx))}, index=idx)
    th = pd.DataFrame({"date": [idx[0]], "ticker": ["A"], "position": [1.0]})

    def metrics_fn(s: pd.Series) -> dict[str, float]:
        return {"Total Return": float(s.sum())}

    kwargs = dict(
        trade_history=th,
        asset_returns=asset,
        returns_fallback=None,
        n_samples=80,
        block_size_days=100,
        survival_metric="Total Return",
        observed_value=0.001,
        metrics_from_returns=metrics_fn,
    )
    p1 = bootstrap_block_shuffled_positions_p_value(**kwargs, rng=np.random.default_rng(42))
    p2 = bootstrap_block_shuffled_positions_p_value(**kwargs, rng=np.random.default_rng(42))
    assert p1 == p2


def test_block_shuffled_positions_p_value_counts_fraction_gte_selected() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    asset = pd.DataFrame(
        {"A": [0.1, 0.1, 0.0, 0.0], "B": [0.0, 0.0, 0.2, 0.2]},
        index=idx,
    )
    th = pd.DataFrame(
        {
            "date": [idx[0], idx[2]],
            "ticker": ["A", "B"],
            "position": [1.0, 1.0],
        }
    )

    def metrics_fn(s: pd.Series) -> dict[str, float]:
        return {"Total Return": float(s.sum())}

    class _AltPermRng:
        def __init__(self) -> None:
            self._i = 0

        def permutation(self, _x: object) -> np.ndarray:
            self._i += 1
            return np.array([1, 0]) if self._i % 2 == 1 else np.array([0, 1])

    p = bootstrap_block_shuffled_positions_p_value(
        trade_history=th,
        asset_returns=asset,
        returns_fallback=None,
        n_samples=10,
        block_size_days=2,
        survival_metric="Total Return",
        observed_value=0.25,
        metrics_from_returns=metrics_fn,
        rng=_AltPermRng(),
    )
    assert p == pytest.approx(0.5)


def test_run_research_bootstrap_includes_positions_when_enabled() -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    war = WFOArchitectureResult(
        architecture=arch,
        metrics={"Total Return": 1.0},
        score=1.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=3,
    )
    sp = SelectedProtocol(
        rank=1,
        architecture=arch,
        metrics={"Total Return": 1.0},
        score=1.0,
        robust_score=None,
        selected_parameters={"p": 1},
    )
    idx = pd.bdate_range("2023-01-01", periods=4)
    asset = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01]}, index=idx)
    th = pd.DataFrame({"date": [idx[0]], "ticker": ["A"], "position": [1.0]})
    tot = float(asset["A"].sum())
    unseen = UnseenValidationResult(
        selected_protocol=sp,
        metrics={"Total Return": tot},
        returns=pd.Series(asset["A"].to_numpy(dtype=float), index=idx),
        mode="fixed_selected_params",
        trade_history=th,
    )
    cfg = BootstrapConfig(
        enabled=True,
        n_samples=50,
        random_seed=11,
        random_wfo_architecture=RandomWfoArchitectureBootstrapConfig(enabled=False),
        block_shuffled_returns=BlockShuffledReturnsBootstrapConfig(
            enabled=False,
            block_size_days=20,
        ),
        block_shuffled_positions=BlockShuffledPositionsBootstrapConfig(
            enabled=True,
            block_size_days=100,
        ),
        random_strategy_parameters=RandomStrategyParametersBootstrapConfig(
            enabled=False,
            sample_size=100,
        ),
    )

    def metrics_from_returns(s: pd.Series) -> dict[str, float]:
        return {"Total Return": float(s.sum())}

    out = run_research_bootstrap(
        cfg=cfg,
        grid_results=[war],
        selected=sp,
        unseen_result=unseen,
        selection_metric="Total Return",
        metrics_from_returns=metrics_from_returns,
        asset_returns=asset,
        trade_history=th,
    )
    assert out is not None
    summary, rows = out
    bpos = summary["block_shuffled_positions"]
    assert bpos["enabled"] is True
    assert bpos["p_value"] == 1.0
    pos_rows = [r for r in rows if r.get("test") == "block_shuffled_positions"]
    assert len(pos_rows) == 1
