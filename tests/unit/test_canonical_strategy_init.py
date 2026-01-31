import pytest
import pandas as pd
from src.portfolio_backtester.canonical_config import CanonicalScenarioConfig
from src.portfolio_backtester.strategies.builtins.signal.ema_roro_signal_strategy import EmaRoroSignalStrategy
from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming

def test_strategy_init_from_canonical_config_unification():
    """Test that strategy properly uses canonical config for providers and timing."""
    from src.portfolio_backtester.scenario_normalizer import ScenarioNormalizer
    normalizer = ScenarioNormalizer()

    # 1. Create canonical config with specific universe and timing
    raw_scenario = {
        "name": "test_scenario",
        "strategy": "EmaRoroSignalStrategy",
        "universe_config": {
            "type": "fixed",
            "tickers": ["AAPL", "MSFT"]
        },
        "strategy_params": {
            "fast_ema_days": 10,
            "slow_ema_days": 20
        },
        "timing_config": {
            "mode": "time_based",
            "rebalance_frequency": "W"
        }
    }
    config = normalizer.normalize(scenario=raw_scenario, global_config={})
    
    # 2. Instantiate strategy
    strategy = EmaRoroSignalStrategy(config)
    
    # 3. Verify canonical config is stored
    assert strategy.canonical_config == config
    
    # 4. Verify timing controller initialized from canonical timing_config
    timing_controller = strategy.get_timing_controller()
    assert isinstance(timing_controller, TimeBasedTiming)
    assert timing_controller.frequency == "W"
    
    # 5. Verify universe provider initialized from canonical universe_definition
    universe_provider = strategy.get_universe_provider()
    symbols = universe_provider.get_universe_symbols({})
    # THIS SHOULD FAIL UNTIL WE UNIFY PROVIDER INIT
    assert symbols == ["AAPL", "MSFT"]

def test_strategy_init_from_dict_legacy_compatibility():
    """Verify legacy dictionary-based initialization still works."""
    legacy_config = {
        "strategy_params": {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "universe_config": {
                "type": "fixed",
                "tickers": ["TSLA", "NVDA"]
            }
        },
        "timing_config": {
            "mode": "time_based",
            "rebalance_frequency": "M"
        }
    }
    
    strategy = EmaRoroSignalStrategy(legacy_config)
    assert strategy.canonical_config is None
    
    timing_controller = strategy.get_timing_controller()
    assert isinstance(timing_controller, TimeBasedTiming)
    assert timing_controller.frequency == "M"
    
    universe_provider = strategy.get_universe_provider()
    symbols = universe_provider.get_universe_symbols({})
    assert symbols == ["TSLA", "NVDA"]
