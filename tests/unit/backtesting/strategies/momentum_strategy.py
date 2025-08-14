"""
Dummy momentum_strategy for testing purposes.
"""
from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """Dummy momentum_strategy for testing."""
    def __init__(self, params):
        super().__init__(params)

    def get_universe(self, global_config):
        return [("AAPL", 1.0), ("GOOGL", 1.0)]
