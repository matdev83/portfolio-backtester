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
            "p1": {"default": 0},
        }
        mock_reg.get_strategy_class.return_value = mock_strategy_class
        mock_get.return_value = mock_reg
        yield mock_reg


def test_normalize_optimizer_selection(mock_registry, caplog):
    """Test that optuna is selected if multiple optimizers are provided."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "S",
        "optimizers": {"optuna": {"n_trials": 100}, "genetic": {"pop_size": 50}},
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})

    assert canon.optimizer_config["n_trials"] == 100
    assert "n_trials" in canon.optimizer_config
    assert "pop_size" not in canon.optimizer_config
    assert "Selected 'optuna'" in caplog.text


def test_normalize_optimizer_override_warning(mock_registry, caplog):
    """Test that top-level keys override optimizer-derived settings."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "S",
        "ga_num_generations": 200,  # Top level
        "optimizers": {"ga": {"ga_num_generations": 100}},
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})

    # Top-level should win, but a warning should be emitted
    assert canon.optimizer_config["ga_num_generations"] == 200
    assert "overridden by top-level setting" in caplog.text


def test_normalize_optimize_list(mock_registry):
    """Test that the optimize list is validated and normalized."""
    normalizer = ScenarioNormalizer()
    scenario = {"strategy": "S", "optimize": [{"parameter": "p1", "min": 1, "max": 10}]}
    canon = normalizer.normalize(scenario=scenario, global_config={})
    assert canon.optimize is not None
    assert len(canon.optimize) == 1
    assert canon.optimize[0]["parameter"] == "p1"


def test_normalize_optimizer_invalid_type(mock_registry):
    """Test that 'optimizers' must be a mapping."""
    normalizer = ScenarioNormalizer()
    scenario = {"strategy": "S", "optimizers": ["not", "a", "dict"]}
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "must be a mapping" in str(excinfo.value)


def test_normalize_optimize_list_validation(mock_registry):
    """Test that the optimize list is validated against strategy parameters."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "S",
        "optimize": [
            {"parameter": "lookback_months", "min": 1, "max": 12},
            {"parameter": "invalid_param", "min": 1, "max": 5},
        ],
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "Invalid optimization parameter 'invalid_param'" in str(excinfo.value)


def test_normalize_optimize_missing_parameter_key(mock_registry):
    """Test that optimize items must have a 'parameter' key."""
    normalizer = ScenarioNormalizer()
    scenario = {"strategy": "S", "optimize": [{"not_parameter": "lookback_months"}]}
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "missing 'parameter' key" in str(excinfo.value)
