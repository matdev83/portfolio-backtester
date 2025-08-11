import pandas as pd
from typing import Any, Dict, Optional

from ...strategies._core.base.base.base_strategy import BaseStrategy


class StopLossTesterStrategy(BaseStrategy):
    """Diagnostic strategy for testing stop-loss functionality."""

    def __init__(self, strategy_params: Dict[str, Any]):
        super().__init__(strategy_params)
        # Initialize entry prices tracking
        self.entry_prices: Optional[pd.Series] = None

        # Get parameters with simplified names
        self.stop_loss_type = self.strategy_params.get("stop_loss_type", "atr")
        self.atr_length = self.strategy_params.get("atr_length", 14)
        self.atr_multiple = self.strategy_params.get("atr_multiple", 2.0)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary defining the tunable parameters for this strategy."""
        return {
            "stop_loss_type": {
                "type": "str",
                "default": "atr",
                "options": ["atr", "trailing"],
            },
            "atr_length": {
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 30,
                "step": 1,
            },
            "atr_multiple": {
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 5.0,
                "step": 0.5,
            },
        }

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Generate simple long signals for all assets in universe."""

        # Get asset names from multi-index columns
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            assets = all_historical_data.columns.get_level_values(0).unique()
        else:
            assets = all_historical_data.columns

        # Simple strategy: go long on all assets
        weights = pd.Series(1.0 / len(assets), index=assets)

        return pd.DataFrame([weights], index=[current_date])
