import pytest
import logging
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

        # Mock a strategy class with tunable parameters
        mock_strategy_class = MagicMock()
        mock_strategy_class.tunable_parameters.return_value = {
            "lookback_months": {"default": 6},
            "num_holdings": {"default": 10},
        }
        mock_reg.get_strategy_class.return_value = mock_strategy_class
        mock_get.return_value = mock_reg
        yield mock_reg


def test_normalize_strategy_params_prefix_stripping(mock_registry, caplog):
    """Test that prefixed parameters are stripped and normalized."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "strategy_params": {
            "SimpleMomentumPortfolioStrategy.lookback_months": 20,
            "Momentum.lookback_months": 20,  # Different prefix but same param name
        },
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})

    assert canon.strategy_params["lookback_months"] == 20
    assert "Legacy normalization: stripped prefix" in caplog.text


def test_normalize_strategy_params_prefix_conflict(mock_registry):
    """Test that conflicting prefixed parameters raise an error."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "strategy_params": {"S1.lookback_months": 10, "S2.lookback_months": 20},
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "Conflict in 'strategy_params'" in str(excinfo.value)
    assert "10" in str(excinfo.value)
    assert "20" in str(excinfo.value)


def test_normalize_strategy_params_flattening(mock_registry, caplog):
    """Test that valid legacy top-level parameters are flattened."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_months": 12,  # Valid param at top level
        "strategy_params": {"num_holdings": 15},  # Another valid param
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})

    assert canon.strategy_params["lookback_months"] == 12
    assert canon.strategy_params["num_holdings"] == 15
    assert "Legacy normalization: flattened top-level key" in caplog.text


def test_normalize_strategy_params_flattening_conflict(mock_registry):
    """Test that conflicting top-level and strategy_params raise an error."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_months": 12,  # Top-level
        "strategy_params": {"lookback_months": 6},  # Different value in strategy_params
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "Ambiguous legacy parameter 'lookback_months'" in str(excinfo.value)


def test_normalize_strategy_params_unknown_not_flattened(mock_registry):
    """Test that unknown top-level parameters are NOT flattened, go to extras."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "unknown_param": "should_go_to_extras",  # Unknown param
        "strategy_params": {"lookback_months": 12},
    }
    canon = normalizer.normalize(scenario=scenario, global_config={})

    # Known param should be in strategy_params
    assert canon.strategy_params["lookback_months"] == 12
    # Unknown param should NOT be in strategy_params
    assert "unknown_param" not in canon.strategy_params
    # Unknown param should be in extras
    assert canon.extras["unknown_param"] == "should_go_to_extras"


def test_normalize_strategy_params_defaults(mock_registry):
    """Test that strategy defaults are applied for missing parameters."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "strategy_params": {"lookback_months": 12},  # Override one default
    }
    canon = normalizer.normalize(scenario=scenario, global_config={})

    assert canon.strategy_params["lookback_months"] == 12  # User-provided
    assert canon.strategy_params["num_holdings"] == 10  # Default value from mock
