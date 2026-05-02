"""Parity tests for fast-path trial scenario overlay in BacktestEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_backtester.optimization.evaluator import (
    _canonical_with_merged_strategy_params,
    _coerce_trial_values,
)
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer


def _dummy_signal_raw() -> dict:
    return {
        "name": "trial_overlay_dummy",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {"open_long_prob": 0.1},
        "optimize": [
            {
                "parameter": "open_long_prob",
                "type": "float",
                "min_value": 0.05,
                "max_value": 0.15,
            }
        ],
        "train_window_months": 6,
        "test_window_months": 3,
    }


def test_coerce_trial_values_numpy_and_bool() -> None:
    out = _coerce_trial_values(
        {
            "a": np.float64(1.25),
            "b": np.int64(7),
            "c": np.bool_(True),
            "d": True,
            "e": False,
        }
    )
    assert out["a"] == pytest.approx(1.25) and isinstance(out["a"], float)
    assert out["b"] == 7 and isinstance(out["b"], int)
    assert out["c"] is True and isinstance(out["c"], bool)
    assert out["d"] is True
    assert out["e"] is False


def test_canonical_merge_matches_full_normalize() -> None:
    normalizer = ScenarioNormalizer()
    gc: dict = {}
    base = normalizer.normalize(scenario=_dummy_signal_raw(), global_config=gc)
    trial = {"open_long_prob": np.float64(0.11)}

    merged_dict = {**dict(base.strategy_params), **_coerce_trial_values(trial)}
    via_normalize = normalizer.normalize(
        scenario={**base.to_dict(), "strategy_params": merged_dict},
        global_config=gc,
    )
    via_replace = _canonical_with_merged_strategy_params(base, trial, gc)

    assert dict(via_normalize.strategy_params) == dict(via_replace.strategy_params)
    assert via_normalize.name == via_replace.name
    assert via_normalize.strategy == via_replace.strategy
