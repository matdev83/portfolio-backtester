from portfolio_backtester.strategies._core.base.base.signal_strategy import (
    SignalStrategy,
)


class MomentumSignalStrategy(SignalStrategy):
    """Dummy momentum_strategy for testing."""

    def __init__(self, params):
        super().__init__(params)

    def get_universe(self, global_config):
        return [("AAPL", 1.0), ("GOOGL", 1.0)]
