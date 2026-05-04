"""
Dummy Strategy for Testing YAML validation

This strategy is specifically designed for testing YAML validation functionality.
It defines specific tunable parameters that tests can validate against.
"""

from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from portfolio_backtester.strategies._core.base.base.signal_strategy import (
    SignalStrategy,
)
from portfolio_backtester.strategies._core.target_generation import StrategyContext


class DummyStrategyForTestingSignalStrategy(SignalStrategy):
    """Dummy strategy specifically for testing YAML validation."""

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.open_long_prob = self.strategy_params.get("open_long_prob", 0.1)
        self.close_long_prob = self.strategy_params.get("close_long_prob", 0.01)

        self.symbol = self.strategy_params.get("symbol", "SPY")

        # Seed the random number generator for reproducibility
        self.seed = self.strategy_params.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "open_long_prob": {
                "type": "float",
                "min": 0.01,  # Test expects values below this to trigger validation error
                "max": 1.0,
                "default": 0.1,
                "required": True,
            },
            "close_long_prob": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.01,
                "required": False,
            },
        }

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate signals based on random chance.
        """
        if current_date is None:
            current_date = all_historical_data.index[-1]

        # Create simple random signals
        weights = pd.Series(0.0, index=[self.symbol])
        if self.rng.random() < self.open_long_prob:
            weights[self.symbol] = 1.0

        return pd.DataFrame([weights], index=[current_date])

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        """Return a deterministic full-scan target matrix for YAML-validation tests."""
        checkpoint = self.rng.bit_generator.state
        self.rng = np.random.default_rng(self.seed)
        try:
            targets = pd.DataFrame(
                0.0,
                index=context.rebalance_dates,
                columns=context.universe_tickers,
                dtype=float,
            )
            for current_date in context.rebalance_dates:
                hist = context.asset_data.loc[context.asset_data.index <= current_date]
                bench = context.benchmark_data.loc[context.benchmark_data.index <= current_date]
                nu = context.non_universe_data.loc[context.non_universe_data.index <= current_date]
                row = self.generate_signals(
                    hist,
                    bench,
                    nu,
                    current_date=current_date,
                    start_date=context.wfo_start_date,
                    end_date=context.wfo_end_date,
                )
                if not row.empty:
                    targets.loc[current_date, :] = (
                        row.iloc[0].reindex(targets.columns).fillna(0.0).to_numpy(dtype=float)
                    )
            return targets
        finally:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = checkpoint

    def __str__(self):
        return f"DummyStrategyForTestingSignalStrategy({self.symbol})"
