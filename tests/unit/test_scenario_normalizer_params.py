import pytest
import logging
from src.portfolio_backtester.scenario_normalizer import ScenarioNormalizer, ScenarioNormalizationError

def test_normalize_strategy_params_prefix_stripping(caplog):
    """Test that prefixed parameters are stripped and normalized."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "strategy_params": {
            "SimpleMomentumPortfolioStrategy.lookback_months": 20,
            "Momentum.lookback_months": 20  # Different prefix but same param name
        }
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})
    
    assert canon.strategy_params["lookback_months"] == 20
    assert "Normalized prefixed parameter" in caplog.text

def test_normalize_strategy_params_prefix_conflict():
    """Test that conflicting prefixed parameters raise an error."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "strategy_params": {
            "S1.lookback_months": 10,
            "S2.lookback_months": 20
        }
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "Conflict in 'strategy_params'" in str(excinfo.value)
    assert "10" in str(excinfo.value)
    assert "20" in str(excinfo.value)

def test_normalize_strategy_params_flattening(caplog):
    """Test that valid legacy top-level parameters are flattened."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_months": 12,  # Valid param at top level
        "strategy_params": {
            "num_holdings": 15  # Another valid param
        }
    }
    with caplog.at_level(logging.WARNING):
        canon = normalizer.normalize(scenario=scenario, global_config={})
    
    assert canon.strategy_params["lookback_months"] == 12
    assert canon.strategy_params["num_holdings"] == 15
    assert "Flattened legacy top-level parameter" in caplog.text

def test_normalize_strategy_params_flattening_conflict():
    """Test that conflicting top-level and strategy_params raise an error."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_months": 12,  # Top-level
        "strategy_params": {
            "lookback_months": 6  # Different value in strategy_params
        }
    }
    with pytest.raises(ScenarioNormalizationError) as excinfo:
        normalizer.normalize(scenario=scenario, global_config={})
    assert "Ambiguous legacy parameter 'lookback_months'" in str(excinfo.value)

def test_normalize_strategy_params_unknown_not_flattened():
    """Test that unknown top-level parameters are NOT flattened, go to extras."""
    normalizer = ScenarioNormalizer()
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "unknown_param": "should_go_to_extras",  # Unknown param
        "strategy_params": {
            "lookback_months": 12
        }
    }
    canon = normalizer.normalize(scenario=scenario, global_config={})
    
    # Known param should be in strategy_params
    assert canon.strategy_params["lookback_months"] == 12
    # Unknown param should NOT be in strategy_params
    assert "unknown_param" not in canon.strategy_params
    # Unknown param should be in extras
    assert canon.extras["unknown_param"] == "should_go_to_extras"

def test_normalize_strategy_params_defaults():
    """Test that strategy defaults are applied for missing parameters."""
    normalizer = ScenarioNormalizer()
    # SimpleMomentumPortfolioStrategy has defaults for num_holdings (10), lookback_months (6), etc.
    scenario = {
        "strategy": "SimpleMomentumPortfolioStrategy",
        "strategy_params": {
            "lookback_months": 12  # Override one default
        }
    }
    canon = normalizer.normalize(scenario=scenario, global_config={})
    
    assert canon.strategy_params["lookback_months"] == 12  # User-provided
    assert canon.strategy_params["num_holdings"] == 10  # Default value
