import pytest
import logging
from src.portfolio_backtester.scenario_normalizer import ScenarioNormalizer, ScenarioNormalizationError

def test_normalize_optimizer_selection(caplog):
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "S",
        "optimizers": {
            "optuna": {"n_trials": 100},
            "genetic": {"pop_size": 50}
        }
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})
    
    assert canon.optimizer_config["n_trials"] == 100
    assert "n_trials" in canon.optimizer_config
    assert "pop_size" not in canon.optimizer_config
    assert "Selected 'optuna'" in caplog.text

def test_normalize_optimizer_override_warning(caplog):
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "S",
        "ga_num_generations": 200,  # Top level
        "optimizers": {
            "ga": {"ga_num_generations": 100}
        }
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})
    
    # Top-level should win, but a warning should be emitted
    assert canon.optimizer_config["ga_num_generations"] == 200
    assert "overridden by top-level setting" in caplog.text

def test_normalize_optimize_list():
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "S",
        "optimize": [
            {"parameter": "p1", "min": 1, "max": 10}
        ]
    }
    canon = normalizer.normalize(scenario=scenario, global_config={})
    assert len(canon.optimize) == 1
    assert canon.optimize[0]["parameter"] == "p1"

def test_normalize_optimizer_invalid_type():
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "S",
        "optimizers": ["not", "a", "dict"]
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "must be a mapping" in str(excinfo.value)
