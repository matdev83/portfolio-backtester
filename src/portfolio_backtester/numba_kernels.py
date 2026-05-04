from typing import Tuple

import numpy as np

# Direct import of Numba - no fallback needed since Numba is a hard dependency
from numba import njit


@njit(cache=True, fastmath=False)
def _weights_diff_abs(weights: np.ndarray) -> np.ndarray:
    """
    Compute absolute day-over-day change in weights per asset.
    weights: [T, N]
    returns: [T, N] with first row = abs(weights[0])
    """
    T, N = weights.shape
    out: np.ndarray = np.empty((T, N), dtype=weights.dtype)
    # first day: treat as full rebalance to current weights
    for j in range(N):
        out[0, j] = abs(weights[0, j])
    for i in range(1, T):
        for j in range(N):
            out[i, j] = abs(weights[i, j] - weights[i - 1, j])
    return out


@njit(cache=True, fastmath=False)
def _masked_weighted_sum(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Vectorized masked sum across columns per row.
    values: [T, N], weights: [T, N], mask: [T, N] (True = valid)
    returns: [T]
    """
    T, N = values.shape
    out: np.ndarray = np.zeros(T, dtype=values.dtype)
    for i in range(T):
        acc = 0.0
        for j in range(N):
            if mask[i, j]:
                acc += values[i, j] * weights[i, j]
        out[i] = acc
    return out


@njit(cache=True, fastmath=False)
def _cumprod_1p(x: np.ndarray) -> np.ndarray:
    """
    Cumulative product of (1 + x) in a numerically stable loop.
    x: [T]
    """
    T = x.shape[0]
    out: np.ndarray = np.empty(T, dtype=x.dtype)
    c = 1.0
    for i in range(T):
        c = c * (1.0 + x[i])
        out[i] = c
    return out


@njit(cache=True, fastmath=False)
def position_and_pnl_kernel(
    weights_for_returns: np.ndarray,  # [T, N]
    rets: np.ndarray,  # [T, N]
    mask: np.ndarray,  # [T, N], True where rets valid
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute daily gross portfolio return, equity curve, and per-day turnover."""
    daily_gross = _masked_weighted_sum(rets, weights_for_returns, mask)
    equity_curve = _cumprod_1p(daily_gross)

    weights_diff = _weights_diff_abs(weights_for_returns)
    T, N = weights_for_returns.shape
    turnover = np.zeros(T, dtype=weights_for_returns.dtype)
    for i in range(T):
        s = 0.0
        for j in range(N):
            s += weights_diff[i, j]
        turnover[i] = s

    return daily_gross, equity_curve, turnover


@njit(cache=True, fastmath=False)
def drifting_weights_returns_kernel(
    target_weights_for_returns: np.ndarray,  # [T, N] target weights (typically shifted by 1 day)
    rets: np.ndarray,  # [T, N]
    mask: np.ndarray,  # [T, N]
    eps: float = 1e-9,
) -> np.ndarray:
    """DEPRECATED for new production code: drifting-weight return path; kept for tests/integration.

    Compute daily gross returns assuming buy-and-hold between rebalance dates.

    The key idea:
    - On days when the target weights change, we rebalance to those target weights.
    - On other days, we hold shares constant, so weights drift with returns.

    This avoids the unrealistic "daily rebalance" effect that can inflate results.

    Args:
        target_weights_for_returns: Target weights applied to returns on each day.
            In the surrounding code, this should typically be `weights_daily.shift(1)`.
        rets: Daily asset returns.
        mask: Valid-return mask.
        eps: Threshold for detecting a rebalance (weight changes).

    Returns:
        Daily gross portfolio returns, shape [T]. First element is 0.0.
    """
    T, N = target_weights_for_returns.shape
    out = np.zeros(T, dtype=rets.dtype)

    # Current drifting weights (interpreted as fractions of equity, can be <0 or >1)
    w = np.empty(N, dtype=rets.dtype)
    for j in range(N):
        w[j] = target_weights_for_returns[0, j]

    for i in range(1, T):
        # Detect rebalance: any weight changed materially
        rebalance = False
        for j in range(N):
            if abs(target_weights_for_returns[i, j] - target_weights_for_returns[i - 1, j]) > eps:
                rebalance = True
                break
        if rebalance:
            for j in range(N):
                w[j] = target_weights_for_returns[i, j]

        # Compute daily gross return using current weights and today's returns
        gross = 0.0
        for j in range(N):
            if mask[i, j]:
                gross += w[j] * rets[i, j]
        out[i] = gross

        # Drift weights forward (shares held constant) via weight evolution:
        # w_{t+1} = w_t * (1 + r_t) / (1 + gross)
        denom = 1.0 + gross
        if denom <= eps:
            # Pathological: portfolio value collapsed; keep weights unchanged to avoid blow-ups
            continue
        for j in range(N):
            r = rets[i, j] if mask[i, j] else 0.0
            w[j] = w[j] * (1.0 + r) / denom

    return out


# NumPy fallback function removed - using only optimized Numba implementation


@njit(cache=True, fastmath=False)
def detailed_commission_slippage_kernel(
    weights_current: np.ndarray,  # [T, N] weights at day t (for turnover calc)
    close_prices: np.ndarray,  # [T, N] close prices
    portfolio_value: float,
    commission_per_share: float,
    commission_min_per_order: float,
    commission_max_percent: float,
    slippage_bps: float,
    price_mask: np.ndarray,  # [T, N] True where price valid (>0) and finite
) -> Tuple[np.ndarray, np.ndarray]:
    """DEPRECATED for new production code: legacy IBKR-style turnover cost helper; tests only.

    Compute per-day total transaction cost fraction of portfolio value using an IBKR-style model.

    Returns:
      per_day_cost_fraction: [T]
      per_day_per_asset_cost_fraction: [T, N]
    """
    T, N = weights_current.shape
    out: np.ndarray = np.zeros(T, dtype=weights_current.dtype)
    out_detailed: np.ndarray = np.zeros((T, N), dtype=weights_current.dtype)

    for i in range(T):
        day_cost = 0.0
        for j in range(N):
            dw = abs(weights_current[i, j] - (weights_current[i - 1, j] if i > 0 else 0.0))
            if dw <= 0.0:
                continue
            if not price_mask[i, j]:
                continue
            price = close_prices[i, j]
            if price <= 0.0 or not np.isfinite(price):
                continue

            trade_value = dw * portfolio_value
            shares = trade_value / price

            # Commission per trade with min and max caps
            commission_trade = 0.0
            if shares > 0.0:
                commission_trade = shares * commission_per_share
                if commission_trade < commission_min_per_order:
                    commission_trade = commission_min_per_order
                max_commission = trade_value * commission_max_percent
                if commission_trade > max_commission:
                    commission_trade = max_commission

            # Slippage
            slippage_amount = trade_value * (slippage_bps / 10000.0)

            asset_cost = commission_trade + slippage_amount
            day_cost += asset_cost
            out_detailed[i, j] = asset_cost / portfolio_value if portfolio_value > 0.0 else 0.0

        out[i] = day_cost / portfolio_value if portfolio_value > 0.0 else 0.0

    return out, out_detailed


@njit(cache=True, fastmath=True)  # Enable fastmath for better performance
def trade_tracking_kernel(
    initial_portfolio_value: float,
    allocation_mode: int,  # 0 for reinvestment, 1 for fixed
    weights: np.ndarray,  # [T, N]
    prices: np.ndarray,  # [T, N]
    price_mask: np.ndarray,  # [T, N] boolean mask for valid prices
    commissions: np.ndarray,  # [T, N] detailed commission costs as fraction of portfolio
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DEPRECATED for new production paths except ``numba_trade_tracker``: legacy holdings stepping kernel.

    Vectorized trade tracking with realistic rebalancing semantics.

    Critical: weights are interpreted as **target weights on rebalancing dates**.
    Between rebalances, the portfolio holds shares constant and weights drift with prices.

    Args:
        initial_portfolio_value: Starting value.
        allocation_mode: 0 for 'reinvestment'/compounding, 1 for 'fixed' capital base.
        weights: Daily target weights (typically ffilled across non-rebalance days).
        prices: Daily close prices.
        price_mask: Mask for valid prices.
        commissions: Per-asset, per-day commission/slippage **fraction**.

    Returns:
        (portfolio_values, cash_values, positions)
    """
    T, N = weights.shape
    portfolio_values = np.zeros(T, dtype=np.float64)
    cash_values = np.zeros(T, dtype=np.float64)
    positions = np.zeros((T, N), dtype=np.float64)

    last_valid_prices = np.zeros(N, dtype=np.float64)

    # --- Day 0: open initial positions from cash ---
    capital_base = initial_portfolio_value
    current_prices0 = prices[0]
    current_mask0 = price_mask[0]

    target_weights0 = weights[0]
    target_dollar_values0 = target_weights0 * capital_base
    safe_prices0 = np.where(current_prices0 > 0, current_prices0, 1.0)
    target_positions0 = (target_dollar_values0 / safe_prices0) * current_mask0

    positions[0] = target_positions0
    for j in range(N):
        if current_mask0[j]:
            last_valid_prices[j] = current_prices0[j]

    holdings_value0 = np.sum(positions[0] * last_valid_prices)
    total_commission_cost0 = np.sum(commissions[0] * capital_base * current_mask0)
    cash_values[0] = capital_base - holdings_value0 - total_commission_cost0
    portfolio_values[0] = cash_values[0] + holdings_value0

    eps = 1e-9

    for i in range(1, T):
        # --- Valuation Stage (Start of Day) ---
        last_positions = positions[i - 1]
        current_prices = prices[i]
        current_mask = price_mask[i]

        valuation_prices = np.empty(N, dtype=np.float64)
        for j in range(N):
            if current_mask[j]:
                last_valid_prices[j] = current_prices[j]
            valuation_prices[j] = last_valid_prices[j]

        holdings_value = np.sum(last_positions * valuation_prices)
        portfolio_values[i] = cash_values[i - 1] + holdings_value

        # Default: no trades => positions and cash unchanged
        positions[i] = last_positions
        cash_values[i] = cash_values[i - 1]

        # Determine capital base for new trades based on allocation mode
        if allocation_mode == 0:  # reinvestment
            capital_base = portfolio_values[i]

        # Rebalance only if weights changed meaningfully day-over-day
        rebalance = False
        for j in range(N):
            if abs(weights[i, j] - weights[i - 1, j]) > eps:
                rebalance = True
                break
        if not rebalance:
            continue

        # --- Rebalancing Stage (End of Day) ---
        target_weights = weights[i]
        target_dollar_values = target_weights * capital_base
        target_positions = np.empty(N, dtype=np.float64)
        for j in range(N):
            if current_mask[j]:
                target_positions[j] = target_dollar_values[j] / current_prices[j]
            else:
                # Missing/invalid price: keep existing shares and value by last valid close.
                target_positions[j] = last_positions[j]

        positions[i] = target_positions

        new_holdings_value = np.sum(positions[i] * valuation_prices)
        total_commission_cost = np.sum(commissions[i] * capital_base * current_mask)

        cash_values[i] = portfolio_values[i] - new_holdings_value - total_commission_cost
        portfolio_values[i] = cash_values[i] + new_holdings_value

    return portfolio_values, cash_values, positions


@njit(cache=True, fastmath=False)
def _per_trade_cost_dollars_kernel(
    trade_value: float,
    abs_share_delta: float,
    use_simple_bps: bool,
    transaction_costs_bps: float,
    commission_per_share: float,
    commission_min_per_order: float,
    commission_max_percent: float,
    slippage_bps: float,
) -> float:
    if trade_value <= 0.0 or (not np.isfinite(trade_value)):
        return 0.0
    if use_simple_bps:
        return trade_value * (transaction_costs_bps / 10000.0)
    if abs_share_delta <= 0.0:
        return 0.0
    commission_trade = abs_share_delta * commission_per_share
    if commission_trade < commission_min_per_order:
        commission_trade = commission_min_per_order
    max_commission = trade_value * commission_max_percent
    if commission_trade > max_commission:
        commission_trade = max_commission
    slippage_amount = trade_value * (slippage_bps / 10000.0)
    return commission_trade + slippage_amount


@njit(cache=True, fastmath=False)
def canonical_portfolio_simulation_kernel(
    initial_portfolio_value: float,
    allocation_mode: int,
    execution_timing: int,
    weights: np.ndarray,
    execution_prices: np.ndarray,
    execution_price_mask: np.ndarray,
    close_prices: np.ndarray,
    close_price_mask: np.ndarray,
    rebalance_mask: np.ndarray,
    use_simple_bps: bool,
    transaction_costs_bps: float,
    commission_per_share: float,
    commission_min_per_order: float,
    commission_max_percent: float,
    slippage_bps: float,
    ref_portfolio_value: float,
    eps: float,
):
    """Share/cash path with execution vs close valuation and rebalance_mask-driven trades."""
    T, N = weights.shape
    portfolio_values = np.zeros(T, dtype=np.float64)
    cash_values = np.zeros(T, dtype=np.float64)
    positions = np.zeros((T, N), dtype=np.float64)
    per_asset_cost_frac = np.zeros((T, N), dtype=np.float64)
    total_cost_frac = np.zeros(T, dtype=np.float64)
    daily_returns = np.zeros(T, dtype=np.float64)

    ref_denom = ref_portfolio_value if ref_portfolio_value > eps else 1.0
    last_valid_close = np.zeros(N, dtype=np.float64)

    for j in range(N):
        if close_price_mask[0, j]:
            last_valid_close[j] = close_prices[0, j]

    do_rebalance_0 = rebalance_mask[0]
    if not do_rebalance_0:
        for j in range(N):
            if abs(weights[0, j]) > eps:
                do_rebalance_0 = True
                break

    last_positions0 = np.zeros(N, dtype=np.float64)
    positions0 = np.zeros(N, dtype=np.float64)
    day0_cost_dollars = 0.0

    if do_rebalance_0:
        capital_base0 = initial_portfolio_value
        tdv0 = weights[0] * capital_base0
        for j in range(N):
            if execution_price_mask[0, j] and execution_prices[0, j] > 0.0:
                positions0[j] = tdv0[j] / execution_prices[0, j]
            else:
                positions0[j] = last_positions0[j]
        for j in range(N):
            if not execution_price_mask[0, j]:
                continue
            exc_price = execution_prices[0, j]
            if exc_price <= 0.0:
                continue
            dsh0 = positions0[j] - last_positions0[j]
            td = abs(dsh0) * exc_price
            c0 = _per_trade_cost_dollars_kernel(
                td,
                abs(dsh0),
                use_simple_bps,
                transaction_costs_bps,
                commission_per_share,
                commission_min_per_order,
                commission_max_percent,
                slippage_bps,
            )
            day0_cost_dollars += c0
            per_asset_cost_frac[0, j] = c0 / ref_denom
        total_cost_frac[0] = day0_cost_dollars / ref_denom

    holdings_value0 = np.sum(positions0 * last_valid_close)
    cash_values[0] = initial_portfolio_value - holdings_value0 - day0_cost_dollars
    portfolio_values[0] = cash_values[0] + holdings_value0
    positions[0] = positions0

    for i in range(1, T):
        prev_last_close = np.empty(N, dtype=np.float64)
        for j in range(N):
            prev_last_close[j] = last_valid_close[j]

        close_row = close_prices[i]
        close_row_mask = close_price_mask[i]
        for j in range(N):
            if close_row_mask[j]:
                last_valid_close[j] = close_row[j]

        last_positions = positions[i - 1]
        holdings_mark = np.sum(last_positions * last_valid_close)
        portfolio_values[i] = cash_values[i - 1] + holdings_mark

        positions[i] = last_positions
        cash_values[i] = cash_values[i - 1]

        if not rebalance_mask[i]:
            continue

        exec_row = execution_prices[i]
        exec_row_mask = execution_price_mask[i]

        if allocation_mode == 0:
            if execution_timing == 1:
                capital_base_use = cash_values[i - 1]
                for j in range(N):
                    if exec_row_mask[j] and exec_row[j] > 0.0:
                        capital_base_use += last_positions[j] * exec_row[j]
                    else:
                        capital_base_use += last_positions[j] * prev_last_close[j]
            else:
                capital_base_use = portfolio_values[i]
        else:
            capital_base_use = initial_portfolio_value

        target_dollar_vals = weights[i] * capital_base_use
        target_positions = np.empty(N, dtype=np.float64)
        for j in range(N):
            if exec_row_mask[j] and exec_row[j] > 0.0:
                target_positions[j] = target_dollar_vals[j] / exec_row[j]
            else:
                target_positions[j] = last_positions[j]

        day_cost_dollars = 0.0
        for j in range(N):
            if not exec_row_mask[j]:
                continue
            exc_price = exec_row[j]
            if exc_price <= 0.0:
                continue
            dsh = target_positions[j] - last_positions[j]
            td = abs(dsh) * exc_price
            cday = _per_trade_cost_dollars_kernel(
                td,
                abs(dsh),
                use_simple_bps,
                transaction_costs_bps,
                commission_per_share,
                commission_min_per_order,
                commission_max_percent,
                slippage_bps,
            )
            day_cost_dollars += cday
            per_asset_cost_frac[i, j] = cday / ref_denom

        total_cost_frac[i] = day_cost_dollars / ref_denom
        positions[i] = target_positions

        new_holdings = np.sum(positions[i] * last_valid_close)
        cash_values[i] = portfolio_values[i] - new_holdings - day_cost_dollars
        portfolio_values[i] = cash_values[i] + new_holdings

    if initial_portfolio_value > eps:
        daily_returns[0] = portfolio_values[0] / initial_portfolio_value - 1.0

    for i in range(1, T):
        prev_pv = portfolio_values[i - 1]
        if prev_pv > eps:
            daily_returns[i] = portfolio_values[i] / prev_pv - 1.0

    return (
        portfolio_values,
        cash_values,
        positions,
        per_asset_cost_frac,
        total_cost_frac,
        daily_returns,
    )


@njit(cache=True, fastmath=True)
def trade_lifecycle_kernel(
    positions: np.ndarray,
    prices: np.ndarray,
    dates: np.ndarray,
    commissions: np.ndarray,
    initial_capital: float,
    out_trades: np.ndarray,
    out_open_pos: np.ndarray,
) -> int:
    """
    Numba-jitted kernel to identify and process the lifecycle of trades.
    Args:
        positions: [T, N] positions
        prices: [T, N] prices
        dates: [T] dates
        commissions: [T, N] commissions
        initial_capital: float
        out_trades: Pre-allocated array for completed trades.
        out_open_pos: Pre-allocated array for open positions tracking.
    Returns:
        int: Number of completed trades found.
    """
    n_days, n_assets = positions.shape
    trade_count = 0

    # Reset open positions buffer
    # It is assumed out_open_pos is initialized to zeros or doesn't matter,
    # but strictly we should zero it out if it's reused.
    # Here we assume caller provides clean or valid buffer.
    # For safety in this kernel loop we rely on "is_open" flag which is false by default if zeros.

    # We can iterate and clear if needed, but assuming zeros from caller is standard for Numba kernels.

    for i in range(1, n_days):  # Start from the second day
        for j in range(n_assets):
            current_pos = positions[i, j]
            prev_pos = positions[i - 1, j]
            price = prices[i, j]
            date = dates[i]
            commission = commissions[i, j]

            # If there is a change in position, a trade has occurred
            if abs(current_pos - prev_pos) > 1e-6:
                trade_quantity = current_pos - prev_pos

                # Check if we are reducing or flipping an existing position
                is_reducing_or_flipping = False
                if out_open_pos[j]["is_open"]:
                    if np.sign(trade_quantity) != np.sign(out_open_pos[j]["quantity"]):
                        is_reducing_or_flipping = True

                if is_reducing_or_flipping:
                    # We are closing the existing position (fully or partially) and potentially reopening

                    # Calculate commission split based on virtual volumes
                    total_virtual_vol = abs(prev_pos) + abs(current_pos)
                    comm_close = 0.0
                    comm_open = 0.0
                    if total_virtual_vol > 1e-9:
                        comm_close = commission * (abs(prev_pos) / total_virtual_vol)
                        comm_open = commission * (abs(current_pos) / total_virtual_vol)
                    else:
                        comm_close = commission

                    out_trades[trade_count]["ticker_idx"] = j
                    out_trades[trade_count]["entry_date"] = out_open_pos[j]["entry_date"]
                    out_trades[trade_count]["exit_date"] = date
                    out_trades[trade_count]["entry_price"] = out_open_pos[j]["entry_price"]
                    out_trades[trade_count]["exit_price"] = price
                    out_trades[trade_count]["quantity"] = out_open_pos[j]["quantity"]
                    out_trades[trade_count]["pnl"] = (
                        price - out_open_pos[j]["entry_price"]
                    ) * out_open_pos[j]["quantity"]
                    out_trades[trade_count]["commission"] = (
                        out_open_pos[j]["total_commission"] + comm_close
                    )
                    trade_count += 1
                    out_open_pos[j]["is_open"] = False

                    # Potentially Open New Position (if current_pos is not zero)
                    if abs(current_pos) > 1e-6:
                        out_open_pos[j]["is_open"] = True
                        out_open_pos[j]["entry_date"] = date
                        out_open_pos[j]["entry_price"] = price
                        out_open_pos[j]["quantity"] = current_pos
                        out_open_pos[j]["total_commission"] = comm_open

                else:
                    # We are extending the position or opening from zero
                    if not out_open_pos[j]["is_open"]:
                        # Opening from zero
                        if abs(current_pos) > 1e-6:
                            out_open_pos[j]["is_open"] = True
                            out_open_pos[j]["entry_date"] = date
                            out_open_pos[j]["entry_price"] = price
                            out_open_pos[j]["quantity"] = current_pos
                            out_open_pos[j]["total_commission"] = commission
                    else:
                        # Adding to existing position
                        out_open_pos[j]["quantity"] += trade_quantity
                        out_open_pos[j]["total_commission"] += commission

    return trade_count


# NumPy fallback function removed - using only optimized Numba implementation
