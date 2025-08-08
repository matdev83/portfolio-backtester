from typing import Dict, Any, Optional
import pandas as pd
from ..base.portfolio_strategy import PortfolioStrategy as BaseStrategy

# Import strategy base interface for composition instead of inheritance
from ...interfaces.strategy_base_interface import IStrategyBase, StrategyBaseFactory


class FixedWeightPortfolioStrategy(BaseStrategy):
    """
    A strategy that maintains a fixed allocation of weights for a given set of assets.
    The weights can be provided as a dictionary or as individual parameters (e.g., 'spy_weight').

    This strategy handles weights in the following way:
    1.  If the sum of weights is less than 1.0, the remainder is held as cash (0% return).
    2.  If the sum of weights is greater than 1.0, the weights are proportionally scaled down
        so that their sum equals 1.0 (i.e., no leverage is used).
    """

    def __init__(self, strategy_params: Dict[str, Any]):
        # Use composition instead of inheritance - create strategy base via factory
        self._strategy_base: IStrategyBase = StrategyBaseFactory.create_strategy_base(
            strategy_params
        )

        # Still call super() for BaseStrategy compatibility, but minimize dependency
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
            raise ValueError(
                "FixedWeightPortfolioStrategy requires either a 'weights' dictionary or individual asset weight parameters (e.g., 'spy_weight') in strategy_params."
            )

    def generate_signals(self, all_historical_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Returns the fixed weights for the given date, handling cash and leverage as described
        in the class docstring.
        """
        current_date = kwargs.get("current_date")

        if self.weights is not None:
            weight_sum = sum(self.weights.values())

            if weight_sum > 1.0:
                # Scale down weights if they sum to more than 1.0
                final_weights = {
                    asset: weight / weight_sum for asset, weight in self.weights.items()
                }
            else:
                # Otherwise, use the weights as is, with the remainder in cash
                final_weights = self.weights
        else:
            final_weights = {}

        signals = pd.Series(final_weights, name=current_date)
        result_df = pd.DataFrame([signals])

        # Enforce trade direction constraints - this will raise an exception if violated
        result_df = self._enforce_trade_direction_constraints(result_df)

        return result_df

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns tunable parameters for the fixed weight strategy.

        This strategy supports dynamic weight parameters for any asset.
        The framework will automatically recognize parameters ending with '_weight'
        as tunable asset weight parameters.

        Always tunable parameters:
        - rebalance_frequency: How often to rebalance the portfolio

        Dynamically tunable parameters:
        - Any parameter ending with '_weight' (e.g., 'spy_weight', 'gld_weight')
        """
        # Return a comprehensive set that includes common weight parameters
        # The optimization system will filter based on what's actually defined
        # in the scenario configuration
        return {
            "rebalance_frequency": {
                "type": "categorical",
                "values": ["D", "W", "M", "Q", "Y"],
                "description": "How often to rebalance the portfolio",
            },
            # Common asset weight parameters - these will be dynamically recognized
            "spy_weight": {
                "type": "float",
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Weight allocation for SPY",
            },
            "gld_weight": {
                "type": "float",
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Weight allocation for GLD",
            },
            # Add more common assets as needed
            "qqq_weight": {
                "type": "float",
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Weight allocation for QQQ",
            },
            "tlt_weight": {
                "type": "float",
                "min_value": 0.0,
                "max_value": 1.0,
                "description": "Weight allocation for TLT",
            },
        }

    # Delegate base strategy methods to interface instead of using super()
    def get_timing_controller(self):
        """Get the timing controller via interface delegation."""
        return self._strategy_base.get_timing_controller()

    def supports_daily_signals(self) -> bool:
        """Check if strategy supports daily signals via interface delegation."""
        return bool(self._strategy_base.supports_daily_signals())

    def get_roro_signal(self):
        """Get RoRo signal - returns None by default for this portfolio strategy."""
        return None

    def get_stop_loss_handler(self):
        """Get stop loss handler via interface delegation."""
        return self._strategy_base.get_stop_loss_handler()

    def get_universe(self, global_config: Dict[str, Any]):
        """Get universe via interface delegation."""
        return self._strategy_base.get_universe(global_config)

    def get_universe_method_with_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ):
        """Get universe with date context via interface delegation."""
        return self._strategy_base.get_universe_method_with_date(global_config, current_date)

    def get_non_universe_data_requirements(self):
        """Get non-universe data requirements via interface delegation."""
        return self._strategy_base.get_non_universe_data_requirements()

    def get_synthetic_data_requirements(self) -> bool:
        """Get synthetic data requirements via interface delegation."""
        return bool(self._strategy_base.get_synthetic_data_requirements())

    def get_minimum_required_periods(self) -> int:
        """Get minimum required periods via interface delegation."""
        return int(self._strategy_base.get_minimum_required_periods())

    def validate_data_sufficiency(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ):
        """Validate data sufficiency via interface delegation."""
        return self._strategy_base.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None,
    ):
        """Filter universe by data availability via interface delegation."""
        return self._strategy_base.filter_universe_by_data_availability(
            all_historical_data, current_date, min_periods_override
        )
