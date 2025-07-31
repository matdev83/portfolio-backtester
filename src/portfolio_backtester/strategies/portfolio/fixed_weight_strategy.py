
from typing import Dict, Any
import pandas as pd
from ..base.base_strategy import BaseStrategy

class FixedWeightStrategy(BaseStrategy):
    """
    A strategy that maintains a fixed allocation of weights for a given set of assets.
    The weights can be provided as a dictionary or as individual parameters (e.g., 'spy_weight').

    This strategy handles weights in the following way:
    1.  If the sum of weights is less than 1.0, the remainder is held as cash (0% return).
    2.  If the sum of weights is greater than 1.0, the weights are proportionally scaled down
        so that their sum equals 1.0 (i.e., no leverage is used).
    """

    def __init__(self, strategy_params: Dict[str, Any]):
        super().__init__(strategy_params)
        # First, check for a 'weights' dictionary
        self.weights = self.strategy_params.get("weights")

        # If not found, construct weights from individual parameters
        if not self.weights:
            self.weights = {
                key.replace("_weight", "").upper(): value
                for key, value in self.strategy_params.items()
                if key.endswith("_weight")
            }
        
        if not self.weights or not isinstance(self.weights, dict):
            raise ValueError("FixedWeightStrategy requires either a 'weights' dictionary or individual asset weight parameters (e.g., 'spy_weight') in strategy_params.")

    def generate_signals(self, all_historical_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Returns the fixed weights for the given date, handling cash and leverage as described
        in the class docstring.
        """
        current_date = kwargs.get("current_date")
        
        weight_sum = sum(self.weights.values())
        
        if weight_sum > 1.0:
            # Scale down weights if they sum to more than 1.0
            final_weights = {asset: weight / weight_sum for asset, weight in self.weights.items()}
        else:
            # Otherwise, use the weights as is, with the remainder in cash
            final_weights = self.weights

        signals = pd.Series(final_weights, name=current_date)
        return pd.DataFrame([signals])

    @classmethod
    def tunable_parameters(cls) -> set:
        """
        This strategy's tunable parameters are dynamically determined by the universe.
        The optimization configuration should define them (e.g., 'spy_weight').
        """
        return set()
