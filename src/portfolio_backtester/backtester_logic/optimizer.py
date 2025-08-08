from typing import Any, Dict, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..core import Backtester


def run_optimization(
    self: "Backtester",
    scenario_config: Dict[str, Any],
    monthly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    rets_full: pd.DataFrame,
) -> Any:
    """Wrapper for the optimization logic."""
    from .optimization import run_optimization as run_opt

    return run_opt(self, scenario_config, monthly_data, daily_data, rets_full)
