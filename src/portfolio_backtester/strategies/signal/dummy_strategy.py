"""
Dummy Strategy for Testing

This strategy is designed for testing purposes. It uses random chance to generate signals.
- Universe: Single symbol (SPY)
- Position: Long only
- Logic:
    - If flat, open a long position with 10% probability.
    - If long, close the position with 1% probability.
"""

from typing import Optional, Set

import pandas as pd
import numpy as np

from ..base.signal_strategy import SignalStrategy

class DummyStrategyForTesting(SignalStrategy):
    """Dummy strategy for testing purposes."""

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.long_only = strategy_config.get('long_only', True)
        self.symbol = strategy_config.get('symbol', 'SPY')
        self.open_long_prob = strategy_config.get('open_long_prob', 0.1)
        self.close_long_prob = strategy_config.get('close_long_prob', 0.01)
        self.dummy_param_1 = strategy_config.get('dummy_param_1', 10)
        self.dummy_param_2 = strategy_config.get('dummy_param_2', 20)
        # Seed the random number generator for reproducibility
        self.seed = strategy_config.get('seed', 42)
        self.rng = np.random.default_rng(self.seed)

    @classmethod
    def tunable_parameters(cls) -> Set[str]:
        """Return set of tunable parameters for this strategy."""
        return {'open_long_prob', 'close_long_prob', 'dummy_param_1', 'dummy_param_2'}

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

        # Create a seeded random number generator for deterministic signals
        open_long_random = self.rng.random(size=len(all_historical_data.index))
        close_long_random = self.rng.random(size=len(all_historical_data.index))

        # Create a DataFrame with the random numbers
        random_df = pd.DataFrame(
            {
                'open_long': open_long_random,
                'close_long': close_long_random,
            },
            index=all_historical_data.index
        )

        # Generate signals based on the random numbers
        signals = pd.Series(0.0, index=all_historical_data.index)
        is_long = False
        for i in range(len(signals)):
            if is_long:
                if random_df['close_long'].iloc[i] < self.close_long_prob:
                    is_long = False
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = 1.0
            else:
                if random_df['open_long'].iloc[i] < self.open_long_prob:
                    is_long = True
                    signals.iloc[i] = 1.0
                else:
                    signals.iloc[i] = 0.0

        # Create the final DataFrame with the signal for the current date
        weights = pd.Series(0.0, index=[self.symbol])
        if current_date in signals.index:
            weights[self.symbol] = signals.loc[current_date]
        
        return pd.DataFrame([weights], index=[current_date])

    def __str__(self):
        return f"DummyStrategyForTesting({self.symbol})"
