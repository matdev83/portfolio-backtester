"""
Dummy Strategy for Testing

This strategy is designed for testing purposes. It uses random chance to generate signals.
- Universe: Single symbol (SPY)
- Position: Long only
- Logic:
    - If flat, open a long position with 10% probability.
    - If long, close the position with 1% probability.

To run the optimizer on this strategy, use the following command:

.venv\Scripts\python -m src.portfolio_backtester.backtester --mode optimize --scenario-filename config/scenarios/signal/dummy_strategy/dummy_strategy_test.yaml
"""

from typing import Optional, Set, Dict, Any

import pandas as pd
import numpy as np

from ..base.signal_strategy import SignalStrategy
from ...risk_management.stop_loss_handlers import AtrBasedStopLoss, NoStopLoss

class DummyStrategyForTesting(SignalStrategy):
    """Dummy strategy for testing purposes."""

    def _initialize_stop_loss_handler(self):
        # Default to no stop-loss unless explicitly requested; this avoids requiring full OHLC data
        stop_loss_type = self.strategy_params.get("dummy_strategy_for_testing.stop_loss_type", "NoStopLoss")
        if stop_loss_type == "AtrBasedStopLoss":
            return AtrBasedStopLoss(self.strategy_params, self.strategy_params)
        else:
            return NoStopLoss(self.strategy_params, self.strategy_params)

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.long_only = self.strategy_params.get('dummy_strategy_for_testing.long_only', True)
        self.symbol = self.strategy_params.get('dummy_strategy_for_testing.symbol', 'SPY')
        self.open_long_prob = self.strategy_params.get('dummy_strategy_for_testing.open_long_prob', 0.1)
        self.close_long_prob = self.strategy_params.get('dummy_strategy_for_testing.close_long_prob', 0.01)
        self.dummy_param_1 = self.strategy_params.get('dummy_strategy_for_testing.dummy_param_1', 10)
        self.dummy_param_2 = self.strategy_params.get('dummy_strategy_for_testing.dummy_param_2', 20)
        # Seed the random number generator for reproducibility
        self.seed = self.strategy_params.get('dummy_strategy_for_testing.seed', 42)
        self.rng = np.random.default_rng(self.seed)
        self.stop_loss_handler = self._initialize_stop_loss_handler()
        self.entry_prices = pd.Series(dtype=float)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'open_long_prob': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'required': True},
            'close_long_prob': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.01, 'required': False},
            'dummy_param_1': {'type': 'int', 'min': 1, 'max': 100, 'default': 10, 'required': False},
            'dummy_param_2': {'type': 'int', 'min': 1, 'max': 100, 'default': 20, 'required': False},
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

        # Apply stop-loss if a handler is configured
        if self.stop_loss_handler:
            # Calculate stop levels first
            stop_levels = self.stop_loss_handler.calculate_stop_levels(
                current_date,
                all_historical_data,
                weights,
                self.entry_prices,
            )

            # Extract current close prices and align their index with the weights Series
            current_close_prices = (
                all_historical_data
                .xs('Close', level='Field', axis=1)
                .loc[current_date]
                .reindex(weights.index)
            )

            # Apply stop-loss logic using the aligned Series
            weights = self.stop_loss_handler.apply_stop_loss(
                current_date,
                current_close_prices,
                weights,
                self.entry_prices,
                stop_levels,
            )

        return pd.DataFrame([weights], index=[current_date])

    def __str__(self):
        return f"DummyStrategyForTesting({self.symbol})"
