from __future__ import annotations

from typing import Dict, Any, List, Optional, Union, Mapping, TYPE_CHECKING

import pandas as pd

from ..._core.base.base.portfolio_strategy import PortfolioStrategy
from ..._core.target_generation import StrategyContext, default_benchmark_ticker

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig


class FixedWeightPortfolioStrategy(PortfolioStrategy):
    """Simple fixed-weight portfolio strategy.

    Full-period authoring API: :py:meth:`generate_target_weights`.
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

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        cols = list(context.universe_tickers)
        idx = pd.DatetimeIndex(context.rebalance_dates)
        fill_value = float("nan") if context.use_sparse_nan_for_inactive_rows else 0.0
        result = pd.DataFrame(fill_value, index=idx, columns=cols, dtype=float)
        if len(idx) == 0 or len(cols) == 0:
            return result
        equal_weight = 1.0 / float(len(cols))
        result.iloc[:, :] = equal_weight
        return result

    def generate_signal_matrix(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame],
        rebalance_dates: pd.DatetimeIndex,
        universe_tickers: List[str],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        use_sparse_nan_for_inactive_rows: bool = False,
    ) -> Optional[pd.DataFrame]:
        ctx = StrategyContext.from_standard_inputs(
            asset_data=all_historical_data,
            benchmark_data=benchmark_historical_data,
            non_universe_data=non_universe_historical_data,
            rebalance_dates=rebalance_dates,
            universe_tickers=universe_tickers,
            benchmark_ticker=default_benchmark_ticker(benchmark_historical_data, universe_tickers),
            wfo_start_date=start_date,
            wfo_end_date=end_date,
            use_sparse_nan_for_inactive_rows=use_sparse_nan_for_inactive_rows,
        )
        return self.generate_target_weights(ctx)

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
