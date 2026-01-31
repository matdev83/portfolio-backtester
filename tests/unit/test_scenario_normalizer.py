import pytest
from src.portfolio_backtester.scenario_normalizer import ScenarioNormalizer, ScenarioNormalizationError
from src.portfolio_backtester.canonical_config import CanonicalScenarioConfig

def test_scenario_normalizer_basic():
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY", "start_date": "2020-01-01"}
    scenario = {
        "name": "test",
        "strategy": "SimpleMomentumPortfolioStrategy",
        "lookback_period": 63,
        "rebalance_frequency": "M",
        "universe": ["AAPL", "MSFT"]
    }
    
    canonical = normalizer.normalize(scenario=scenario, global_config=global_config)
    
    assert canonical.name == "test"
    assert canonical.strategy == "SimpleMomentumPortfolioStrategy"
    assert canonical.strategy_params["lookback_period"] == 63
    assert canonical.timing_config["rebalance_frequency"] == "M"
    assert canonical.universe_definition["type"] == "fixed"
    assert canonical.universe_definition["tickers"] == ("AAPL", "MSFT")

def test_scenario_normalizer_conflicts():
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY"}
    scenario = {
        "name": "test",
        "strategy": "SimpleStrategy",
        "universe": ["AAPL"],
        "universe_config": {"type": "fixed", "tickers": ["MSFT"]}
    }
    
    with pytest.raises(ScenarioNormalizationError, match="Conflict in universe definition"):
        normalizer.normalize(scenario=scenario, global_config=global_config)

def test_scenario_normalizer_legacy_flattening():
    normalizer = ScenarioNormalizer()
    global_config = {"benchmark": "SPY"}
    scenario = {
        "name": "test",
        "strategy": "SimpleStrategy",
        "lookback": 10,
        "strategy_params": {"other_param": 20}
    }
    
    canonical = normalizer.normalize(scenario=scenario, global_config=global_config)
    
    assert canonical.strategy_params["lookback"] == 10
    assert canonical.strategy_params["other_param"] == 20

def test_scenario_normalizer_missing_strategy():
    normalizer = ScenarioNormalizer()
    with pytest.raises(ScenarioNormalizationError, match="missing required 'strategy' key"):
        normalizer.normalize(scenario={"name": "test"}, global_config={})
