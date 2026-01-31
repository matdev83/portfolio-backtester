import pytest
import logging
from src.portfolio_backtester.scenario_normalizer import ScenarioNormalizer, ScenarioNormalizationError

def test_normalize_strategy_params_prefix_stripping(caplog):
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "MyStrategy",
        "strategy_params": {
            "MyStrategy.lookback": 20,
            "momentum.lookback": 20  # Different prefix but same param name
        }
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})
    
    assert canon.strategy_params["lookback"] == 20
    assert "Normalized prefixed parameter" in caplog.text

def test_normalize_strategy_params_prefix_conflict():
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "MyStrategy",
        "strategy_params": {
            "S1.param": 10,
            "S2.param": 20
        }
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "Conflict in 'strategy_params'" in str(excinfo.value)
    assert "10" in str(excinfo.value)
    assert "20" in str(excinfo.value)

def test_normalize_strategy_params_flattening(caplog):
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "MyStrategy",
        "lookback": 63,  # Flat
        "strategy_params": {
            "hold_period": 21
        }
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})
    
    assert canon.strategy_params["lookback"] == 63
    assert canon.strategy_params["hold_period"] == 21
    assert "Flattened legacy top-level parameter" in caplog.text

def test_normalize_strategy_params_flattening_conflict():
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "MyStrategy",
        "lookback": 63,
        "strategy_params": {
            "lookback": 21
        }
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "Ambiguous legacy parameter 'lookback'" in str(excinfo.value)

def test_normalize_strategy_params_defaults():
    normalizer = ScenarioNormalizer()
    # SimpleMomentumPortfolioStrategy has defaults for num_holdings (10), lookback_months (6), etc.
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "strategy_params": {
            "lookback_months": 12
        }
    }
    canon = normalizer.normalize(scenario=scenario, global_config={})
    
    assert canon.strategy_params["lookback_months"] == 12
    assert canon.strategy_params["num_holdings"] == 10  # Default value
