from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Optional, cast

import numpy as np
import pandas as pd

from ..backtester_logic.portfolio_simulation_input import PortfolioSimulationInput
from ..numba_kernels import canonical_portfolio_simulation_kernel

EXECUTION_LEDGER_COLUMNS: Final[tuple[str, ...]] = (
    "decision_date_idx",
    "decision_date",
    "execution_date_idx",
    "execution_date",
    "ticker",
    "quantity",
    "execution_price",
    "execution_value",
    "cash_before",
    "cash_after",
    "position_before",
    "position_after",
    "cost",
)


def _empty_execution_ledger_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EXECUTION_LEDGER_COLUMNS))


def _execution_ledger_to_df(
    ledger: np.ndarray,
    ledger_count: int,
    dates: pd.DatetimeIndex | pd.Index,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    if ledger_count <= 0:
        return _empty_execution_ledger_df()
    chunk = ledger[: int(ledger_count)]
    dec_idx = chunk[:, 0].astype(np.int64, copy=False)
    exe_idx = chunk[:, 1].astype(np.int64, copy=False)
    ai = chunk[:, 2].astype(np.int64, copy=False)
    out_decision_dates = dates.take(dec_idx).values
    out_execution_dates = dates.take(exe_idx).values
    out_tickers = np.array([tickers[int(j)] for j in ai], dtype=object)
    return pd.DataFrame(
        {
            "decision_date_idx": dec_idx,
            "decision_date": out_decision_dates,
            "execution_date_idx": exe_idx,
            "execution_date": out_execution_dates,
            "ticker": out_tickers,
            "quantity": chunk[:, 3],
            "execution_price": chunk[:, 4],
            "execution_value": chunk[:, 5],
            "cash_before": chunk[:, 6],
            "cash_after": chunk[:, 7],
            "position_before": chunk[:, 8],
            "position_after": chunk[:, 9],
            "cost": chunk[:, 10],
        },
        columns=list(EXECUTION_LEDGER_COLUMNS),
    )


@dataclass(frozen=True)
class SimulationResult:
    """Outputs from ``simulate_portfolio`` / ``canonical_portfolio_simulation_kernel``.

    Cost fraction arrays measure transaction costs **per calendar row** as a fraction of the
    simulator reference portfolio value (``global_config["portfolio_value"]``), not versus
    intraday NAV unless callers align those inputs.
    """

    portfolio_values: pd.Series
    daily_returns: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    per_asset_transaction_cost_frac_of_reference_pv: np.ndarray
    total_daily_transaction_cost_frac_of_reference_pv: np.ndarray
    execution_ledger: pd.DataFrame = field(default_factory=_empty_execution_ledger_df)


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

    pv, cash, pos, pa_frac, tot_frac, dret, led, led_n = canonical_portfolio_simulation_kernel(
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
    execution_ledger = _execution_ledger_to_df(led, int(led_n), idx, sim_input.tickers)
    return SimulationResult(
        portfolio_values=pd.Series(pv, index=idx, dtype=float),
        daily_returns=pd.Series(dret, index=idx, dtype=float),
        cash=pd.Series(cash, index=idx, dtype=float),
        positions=pd.DataFrame(pos, index=idx, columns=cols),
        per_asset_transaction_cost_frac_of_reference_pv=pa_frac,
        total_daily_transaction_cost_frac_of_reference_pv=tot_frac,
        execution_ledger=execution_ledger,
    )
