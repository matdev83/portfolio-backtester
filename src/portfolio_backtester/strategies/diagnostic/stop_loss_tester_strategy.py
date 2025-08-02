import pandas as pd
from typing import Any, Dict, Optional, Set

from ..base.base_strategy import BaseStrategy

class StopLossTesterStrategy(BaseStrategy):
    """Diagnostic strategy for testing stop-loss functionality."""
    
    def __init__(self, strategy_params: Dict[str, Any]):
        super().__init__(strategy_params)
        # Initialize entry prices tracking
        self.entry_prices = {}
        
    @classmethod
    def tunable_parameters(cls) -> Set[str]:
        return {
            "stop_loss_type",
            "atr_length", 
            "atr_multiple"
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
