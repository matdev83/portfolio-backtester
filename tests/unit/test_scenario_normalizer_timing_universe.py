import pytest
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.scenario_normalizer import (
    ScenarioNormalizer,
    ScenarioNormalizationError,
)


@pytest.fixture
def mock_registry():
    with patch("src.portfolio_backtester.scenario_normalizer.get_strategy_registry") as mock_get:
        mock_reg = MagicMock()
        mock_reg.is_strategy_registered.return_value = True
        mock_reg.get_strategy_class.return_value = None
        mock_get.return_value = mock_reg
        yield mock_reg


def test_normalize_timing_no_conflict(mock_registry):
    normalizer = ScenarioNormalizer()
    global_config = {"rebalance_frequency": "M"}

    # Only top-level
    scenario = {"strategy": "S", "rebalance_frequency": "W"}
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.timing_config["rebalance_frequency"] == "W"

    # Only nested
    scenario = {"strategy": "S", "timing_config": {"rebalance_frequency": "D"}}
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.timing_config["rebalance_frequency"] == "D"

    # Both same
    scenario = {
        "strategy": "S",
        "rebalance_frequency": "W",
        "timing_config": {"rebalance_frequency": "W"},
    }
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.timing_config["rebalance_frequency"] == "W"


def test_normalize_timing_conflict(mock_registry):
    normalizer = ScenarioNormalizer()
    global_config = {}

    scenario = {
        "strategy": "S",
        "rebalance_frequency": "W",
        "timing_config": {"rebalance_frequency": "D"},
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config=global_config)
    assert "Conflict in timing configuration" in str(excinfo.value)
    assert "W" in str(excinfo.value)
    assert "D" in str(excinfo.value)


def test_normalize_universe_no_conflict(mock_registry):
    normalizer = ScenarioNormalizer()
    global_config = {}

    # Legacy 'universe' string
    scenario = {"strategy": "S", "universe": "sp500"}
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.universe_definition == {"type": "named", "name": "sp500"}

    # 'universe' list
    scenario = {"strategy": "S", "universe": ["AAPL", "MSFT"]}
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.universe_definition == {"type": "fixed", "tickers": ("AAPL", "MSFT")}

    # 'universe_config' mapping
    scenario = {"strategy": "S", "universe_config": {"type": "fixed", "tickers": ["SPY"]}}
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.universe_definition == {"type": "fixed", "tickers": ("SPY",)}


def test_normalize_universe_conflict(mock_registry):
    normalizer = ScenarioNormalizer()
    global_config = {}

    scenario = {
        "strategy": "S",
        "universe": "sp500",
        "universe_config": {"type": "fixed", "tickers": ["SPY"]},
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config=global_config)
    assert "Conflict in universe definition" in str(excinfo.value)
    assert "sp500" in str(excinfo.value)
    assert "SPY" in str(excinfo.value)


def test_normalize_universe_from_canonical_to_dict_roundtrip(mock_registry):
    """Optimization merges trial params via to_dict(); universe must survive re-normalization."""
    normalizer = ScenarioNormalizer()
    global_config = {}

    scenario = {
        "strategy": "S",
        "name": "roundtrip_u",
        "universe_config": {"type": "fixed", "tickers": ["QQQ"]},
    }
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert dict(canon.universe_definition) == {"type": "fixed", "tickers": ("QQQ",)}

    round_dict = canon.to_dict()
    round_dict["strategy_params"] = {"sl_atr_mult": 1.5}
    again = normalizer.normalize(scenario=round_dict, global_config=global_config)
    assert dict(again.universe_definition) == {"type": "fixed", "tickers": ("QQQ",)}
    assert dict(again.strategy_params)["sl_atr_mult"] == 1.5


def test_normalize_wfo_overrides(mock_registry):
    normalizer = ScenarioNormalizer()
    global_config = {"wfo_robustness_config": {"train_window_months": 12, "test_window_months": 3}}

    scenario = {"strategy": "S", "train_window_months": 24}
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.wfo_config["train_window_months"] == 24
    assert canon.wfo_config["test_window_months"] == 3


def test_normalize_extras_preservation(mock_registry):
    normalizer = ScenarioNormalizer()
    global_config = {}

    scenario = {"strategy": "S", "custom_key": "custom_value", "extras": {"another_key": 1}}
    canon = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert canon.extras["custom_key"] == "custom_value"
    assert canon.extras["another_key"] == 1
