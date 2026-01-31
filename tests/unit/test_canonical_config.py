import pytest
from frozendict import frozendict
from src.portfolio_backtester.canonical_config import CanonicalScenarioConfig, freeze_config

def test_freeze_config_recursive():
    data = {
        "a": 1,
        "b": [1, 2, {"c": 3}],
        "d": {"e": 4}
    }
    frozen = freeze_config(data)
    
    assert isinstance(frozen, frozendict)
    assert isinstance(frozen["b"], tuple)
    assert isinstance(frozen["b"][2], frozendict)
    assert isinstance(frozen["d"], frozendict)
    
    # Verify immutability
    with pytest.raises(TypeError):
        frozen["a"] = 2
    with pytest.raises(TypeError):
        frozen["d"]["e"] = 5

def test_canonical_scenario_config_immutability():
    config = CanonicalScenarioConfig(
        name="test_scenario",
        strategy="SimpleStrategy",
        strategy_params=frozendict({"param1": 10}),
        timing_config=frozendict({"rebalance_frequency": "M"})
    )
    
    assert config.name == "test_scenario"
    
    # Verify top-level immutability (dataclass frozen=True)
    with pytest.raises(AttributeError):
        config.name = "new_name"  # type: ignore
        
    # Verify nested immutability (frozendict)
    with pytest.raises(TypeError):
        config.strategy_params["param1"] = 20  # type: ignore

def test_canonical_scenario_config_from_dict():
    raw_data = {
        "name": "test_scenario",
        "strategy": "SimpleStrategy",
        "strategy_params": {"param1": 10},
        "timing_config": {"rebalance_frequency": "M"},
        "unknown_key": "val"
    }
    
    config = CanonicalScenarioConfig.from_dict(raw_data)
    
    assert config.name == "test_scenario"
    assert isinstance(config.strategy_params, frozendict)
    assert config.strategy_params["param1"] == 10
    
    # unknown_key should be in extras
    assert config.extras["unknown_key"] == "val"

def test_canonical_scenario_config_to_dict():
    config = CanonicalScenarioConfig(
        name="test_scenario",
        strategy="SimpleStrategy",
        strategy_params=frozendict({"param1": 10}),
        timing_config=frozendict({"rebalance_frequency": "M"})
    )
    
    as_dict = config.to_dict()
    assert isinstance(as_dict, dict)
    assert as_dict["name"] == "test_scenario"
    assert isinstance(as_dict["strategy_params"], dict)
    assert as_dict["strategy_params"]["param1"] == 10

def test_canonical_scenario_config_equality():
    raw_data = {
        "name": "test_scenario",
        "strategy": "SimpleStrategy",
        "strategy_params": {"param1": 10},
    }
    
    config1 = CanonicalScenarioConfig.from_dict(raw_data)
    config2 = CanonicalScenarioConfig.from_dict(raw_data)
    
    assert config1 == config2
    assert hash(config1) == hash(config2)
    
    raw_data2 = raw_data.copy()
    raw_data2["strategy_params"]["param1"] = 11
    config3 = CanonicalScenarioConfig.from_dict(raw_data2)
    
    assert config1 != config3
