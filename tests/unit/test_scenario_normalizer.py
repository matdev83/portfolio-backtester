import pytest
from src.portfolio_backtester.scenario_normalizer import (
    ScenarioNormalizer,
    ScenarioNormalizationError,
)


def test_scenario_normalizer_basic():
    """Test basic normalization with valid strategy and parameters."""
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY", "start_date": "2020-01-01"}
    scenario = {
        "name": "test",
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_months": 6,  # Valid parameter name
        "rebalance_frequency": "M",
        "universe": ["AAPL", "MSFT"],
    }

    canonical = normalizer.normalize(scenario=scenario, global_config=global_config)

    assert canonical.name == "test"
    assert canonical.strategy == "SimpleMomentumPortfolioStrategy"
    assert canonical.strategy_params["lookback_months"] == 6
    assert canonical.timing_config["rebalance_frequency"] == "M"
    assert canonical.universe_definition["type"] == "fixed"
    assert canonical.universe_definition["tickers"] == ("AAPL", "MSFT")


def test_scenario_normalizer_conflicts():
    """Test that conflicting universe definitions raise an error."""
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY"}
    scenario = {
        "name": "test",
        "strategy": "SimpleMomentumPortfolioStrategy",
        "universe": ["AAPL"],
        "universe_config": {"type": "fixed", "tickers": ["MSFT"]},
    }

    with pytest.raises(ScenarioNormalizationError, match="Conflict in universe definition"):
        normalizer.normalize(scenario=scenario, global_config=global_config)


def test_scenario_normalizer_legacy_flattening():
    """Test that valid legacy top-level params are flattened into strategy_params."""
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY"}
    scenario = {
        "name": "test",
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_months": 10,  # Valid parameter at top level
        "strategy_params": {"num_holdings": 20},  # Another valid param
    }

    canonical = normalizer.normalize(scenario=scenario, global_config=global_config)

    # Both should be in strategy_params
    assert canonical.strategy_params["lookback_months"] == 10
    assert canonical.strategy_params["num_holdings"] == 20


def test_scenario_normalizer_unknown_keys_in_extras():
    """Test that unknown keys are preserved in extras, not flattened."""
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY"}
    scenario = {
        "name": "test",
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_months": 6,  # Valid parameter
        "unknown_custom_key": "custom_value",  # Unknown key
        "strategy_params": {"num_holdings": 15},
    }

    canonical = normalizer.normalize(scenario=scenario, global_config=global_config)

    # Valid params should be in strategy_params
    assert canonical.strategy_params["lookback_months"] == 6
    assert canonical.strategy_params["num_holdings"] == 15

    # Unknown key should be in extras
    assert canonical.extras["unknown_custom_key"] == "custom_value"


def test_scenario_normalizer_missing_strategy():
    """Test that missing strategy raises an error when not in test mode."""
    import os
    from unittest.mock import patch

    normalizer = ScenarioNormalizer()
    # Temporarily remove PYTEST_CURRENT_TEST to test strict validation
    with patch.dict(os.environ, clear=True):
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]
        with pytest.raises(ScenarioNormalizationError, match="missing required 'strategy' key"):
            normalizer.normalize(scenario={"name": "test"}, global_config={})


def test_research_protocol_stays_in_extras_not_strategy_params() -> None:
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY"}
    scenario = {
        "name": "test",
        "strategy": "SimpleMomentumPortfolioStrategy",
        # `type` matches research protocol YAML (see parse_double_oos_wfo_protocol).
        "research_protocol": {"type": "double_oos_wfo"},
        "lookback_months": 6,
    }
    canonical = normalizer.normalize(scenario=scenario, global_config=global_config)
    assert dict(canonical.extras["research_protocol"]) == {"type": "double_oos_wfo"}
    assert "research_protocol" not in canonical.strategy_params
