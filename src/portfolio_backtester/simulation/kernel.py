from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, cast

import numpy as np
import pandas as pd

from ..backtester_logic.portfolio_simulation_input import PortfolioSimulationInput
from ..numba_kernels import canonical_portfolio_simulation_kernel


@dataclass(frozen=True)
class SimulationResult:
    portfolio_values: pd.Series
    daily_returns: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    per_asset_cost_fraction: np.ndarray
    total_cost_fraction: np.ndarray


def _resolve_ref_portfolio_value(global_config: Mapping[str, Any] | None) -> float:
    if isinstance(global_config, dict):
        return float(global_config.get("portfolio_value", 100_000.0))
    return 100_000.0


def _resolve_allocation_mode_int(scenario_config: Mapping[str, Any] | None) -> int:
    mode = "reinvestment"
    if isinstance(scenario_config, dict):
        mode = str(scenario_config.get("allocation_mode", "reinvestment"))
    if mode in ("reinvestment", "compound"):
        return 0
    return 1


def simulate_portfolio(
    sim_input: PortfolioSimulationInput,
    *,
    global_config: Optional[Mapping[str, Any]] = None,
    scenario_config: Optional[Mapping[str, Any]] = None,
) -> SimulationResult:
    ref_pv = _resolve_ref_portfolio_value(global_config)
    initial_pv = ref_pv
    alloc = _resolve_allocation_mode_int(scenario_config)

    transaction_costs_bps: float | None = None
    if isinstance(scenario_config, dict):
        cc = scenario_config.get("costs_config")
        if isinstance(cc, dict):
            raw = cc.get("transaction_costs_bps")
            if raw is not None:
                transaction_costs_bps = float(raw)

    use_simple_bps = transaction_costs_bps is not None
    if use_simple_bps:
        bps_val = float(cast(float, transaction_costs_bps))
    else:
        bps_val = 0.0

    gc = global_config if isinstance(global_config, dict) else {}
    commission_per_share = float(gc.get("commission_per_share", 0.005))
    commission_min_per_order = float(gc.get("commission_min_per_order", 1.0))
    commission_max_percent = float(gc.get("commission_max_percent_of_trade", 0.005))
    slippage_bps = float(gc.get("slippage_bps", 2.5))

    pv, cash, pos, pa_frac, tot_frac, dret = canonical_portfolio_simulation_kernel(
        initial_pv,
        alloc,
        sim_input.execution_timing,
        sim_input.weights_target.astype(np.float64),
        sim_input.execution_prices.astype(np.float64),
        sim_input.execution_price_mask.astype(np.bool_),
        sim_input.close_prices.astype(np.float64),
        sim_input.close_price_mask.astype(np.bool_),
        sim_input.rebalance_mask.astype(np.bool_),
        use_simple_bps,
        bps_val,
        commission_per_share,
        commission_min_per_order,
        commission_max_percent,
        slippage_bps,
        ref_pv,
        1e-9,
    )

    idx = sim_input.dates
    cols = list(sim_input.tickers)
    return SimulationResult(
        portfolio_values=pd.Series(pv, index=idx, dtype=float),
        daily_returns=pd.Series(dret, index=idx, dtype=float),
        cash=pd.Series(cash, index=idx, dtype=float),
        positions=pd.DataFrame(pos, index=idx, columns=cols),
        per_asset_cost_fraction=pa_frac,
        total_cost_fraction=tot_frac,
    )
