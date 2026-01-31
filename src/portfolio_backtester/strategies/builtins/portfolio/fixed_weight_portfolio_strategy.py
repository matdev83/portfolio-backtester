from __future__ import annotations

from typing import Dict, Any, Optional, Union, Mapping, TYPE_CHECKING

import pandas as pd

from ..._core.base.base.portfolio_strategy import PortfolioStrategy

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig


class FixedWeightPortfolioStrategy(PortfolioStrategy):
    """Simple fixed-weight portfolio strategy.
...
    This minimal implementation serves as a framework-provided baseline
    that users may extend in `strategies/user/**`.
    """

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)


    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        # No tunables for the baseline fixed-weight strategy
        return {}

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = pd.Timestamp(all_historical_data.index[-1])

        if isinstance(all_historical_data.columns, pd.MultiIndex):
            tickers = all_historical_data.columns.get_level_values(0).unique().tolist()
        else:
            tickers = list(all_historical_data.columns)

        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)
        if len(tickers) == 0:
            return result

        equal_weight = 1.0 / len(tickers)
        result.loc[current_date, :] = equal_weight
        return result


__all__ = ["FixedWeightPortfolioStrategy"]
