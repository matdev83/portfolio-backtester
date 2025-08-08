import pytest
from portfolio_backtester.strategies.signal.uvxy_rsi_signal_strategy import UvxyRsiSignalStrategy
from portfolio_backtester.strategies.signal.ema_roro_signal_strategy import EmaRoroSignalStrategy

@pytest.mark.parametrize("strategy_class, config, expected_tunable_params, expected_non_universe_reqs", [
    (UvxyRsiSignalStrategy, {"strategy_params": {"rsi_period": 2, "rsi_threshold": 30.0}}, {"rsi_period", "rsi_threshold"}, ["SPY"]),
    (EmaRoroSignalStrategy, {"fast_ema_days": 10, "slow_ema_days": 20}, {'fast_ema_days', 'slow_ema_days', 'leverage', 'risk_off_leverage_multiplier'}, [])
])
class TestStrategyInitialization:
    def test_strategy_initialization(self, strategy_class, config, expected_tunable_params, expected_non_universe_reqs):
        strategy = strategy_class(config)
        assert isinstance(strategy, strategy_class)

    def test_tunable_parameters(self, strategy_class, config, expected_tunable_params, expected_non_universe_reqs):
        strategy = strategy_class(config)
        # tunable_parameters() returns a dict, we check that the expected parameter names are keys
        tunable_params = strategy.tunable_parameters()
        assert isinstance(tunable_params, dict)
        assert set(tunable_params.keys()) == expected_tunable_params

    def test_non_universe_data_requirements(self, strategy_class, config, expected_tunable_params, expected_non_universe_reqs):
        strategy = strategy_class(config)
        assert strategy.get_non_universe_data_requirements() == expected_non_universe_reqs