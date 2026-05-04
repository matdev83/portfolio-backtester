"""
Dummy Strategy for Testing

This strategy is designed for testing purposes. It uses random chance to generate signals.
- Universe: Single symbol (SPY)
- Position: Long only
- Logic:
    - If flat, open a long position with 10% probability.
    - If long, close the position with 1% probability.

To run the optimizer on this strategy, use the following command:

.venv/Scripts/python -m portfolio_backtester.backtester --mode optimize --scenario-filename config/scenarios/builtins/signal/dummy_signal_strategy/default.yaml
"""

from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from ...strategies._core.base.base.signal_strategy import SignalStrategy
from ...strategies._core.target_generation import StrategyContext
from ...risk_management.stop_loss_handlers import AtrBasedStopLoss, NoStopLoss


def _slice_historical_to_date(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    calendar_index: pd.Index,
) -> pd.DataFrame:
    """Slice ``df`` to rows on or before ``as_of``, optionally capped by ``calendar_index`` max."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    end = pd.Timestamp(as_of)
    out = df.loc[df.index <= end]
    if calendar_index is not None and len(calendar_index) > 0:
        cal_max = pd.Timestamp(calendar_index.max())
        out = out.loc[out.index <= cal_max]
    return out


class DummySignalStrategy(SignalStrategy):
    """Dummy strategy for testing purposes."""

    def _initialize_stop_loss_handler(self):
        # Default to no stop-loss unless explicitly requested; this avoids requiring full OHLC data
        stop_loss_type = self.strategy_params.get("stop_loss_type", "NoStopLoss")
        if stop_loss_type == "AtrBasedStopLoss":
            return AtrBasedStopLoss(self.strategy_params, self.strategy_params)
        else:
            return NoStopLoss(self.strategy_params, self.strategy_params)

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.trade_longs = self.strategy_params.get("trade_longs", True)
        self.trade_shorts = self.strategy_params.get("trade_shorts", False)
        self.symbol = self.strategy_params.get("symbol", "SPY")
        self.open_long_prob = self.strategy_params.get("open_long_prob", 0.1)
        self.close_long_prob = self.strategy_params.get("close_long_prob", 0.01)

        # Seed the random number generator for reproducibility
        self.seed = self.strategy_params.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self.stop_loss_handler = self._initialize_stop_loss_handler()
        self.entry_prices = pd.Series(dtype=float)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "open_long_prob": {
                "type": "float",
                "min": 0.0,
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
        Generates trading signals for the dummy strategy.
        """
        if current_date is None:
            current_date = all_historical_data.index[-1]

        # Generate random numbers for the entire series at once
        open_long_random = self.rng.random(size=len(all_historical_data.index))
        close_long_random = self.rng.random(size=len(all_historical_data.index))

        # Create boolean masks for entry and exit signals
        open_long_mask = open_long_random < self.open_long_prob
        close_long_mask = close_long_random < self.close_long_prob

        # Create a forward-filled series to represent being in a long position
        signals = pd.Series(0.0, index=all_historical_data.index)
        signals[open_long_mask] = 1.0
        signals[close_long_mask] = 0.0
        signals = signals.ffill()

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
                all_historical_data.xs("Close", level="Field", axis=1)
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

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        """Return a deterministic full-scan target matrix for testing scenarios."""
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
                nu = _slice_historical_to_date(
                    context.non_universe_data,
                    current_date,
                    context.asset_data.index,
                )
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
        return f"DummySignalStrategy({self.symbol})"

    # Fallback helpers expected by integration tests
    def _apply_signal_strategy_stop_loss(
        self,
        weights: pd.Series,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame,
        current_prices: pd.Series,
    ) -> pd.Series:
        if not self.stop_loss_handler or self.stop_loss_handler.__class__.__name__ == "NoStopLoss":
            return weights
        try:
            stop_levels = self.stop_loss_handler.calculate_stop_levels(
                current_date=current_date,
                asset_ohlc_history=all_historical_data,
                current_weights=weights,
                entry_prices=self.entry_prices,
            )
            applied = self.stop_loss_handler.apply_stop_loss(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=weights,
                entry_prices=self.entry_prices,
                stop_levels=stop_levels,
            )
            # Ensure return type is a Series
            return pd.Series(applied) if not isinstance(applied, pd.Series) else applied
        except Exception:
            return weights

    def _apply_signal_strategy_take_profit(
        self,
        weights: pd.Series,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame,
        current_prices: pd.Series,
    ) -> pd.Series:
        # This dummy strategy does not implement take-profit; return weights unchanged
        return weights
