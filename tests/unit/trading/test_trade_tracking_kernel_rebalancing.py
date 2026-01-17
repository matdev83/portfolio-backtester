import numpy as np
import pytest

from portfolio_backtester.numba_kernels import trade_tracking_kernel


@pytest.mark.unit
@pytest.mark.fast
def test_trade_tracking_kernel_does_not_rebalance_daily_when_weights_constant() -> None:
    """Ensure the kernel holds shares constant between rebalance dates.

    This test is designed to catch the prior behavior where the kernel effectively
    rebalanced every day (by recomputing target shares daily), which can create
    unrealistic "volatility pumping" returns.

    Scenario:
    - Two assets that mean-revert over 2 days.
    - 50/50 target weights are constant for all days.
    - Buy-and-hold should end flat; daily rebalancing would produce a gain.
    """

    # Prices: A up then down, B down then up
    prices = np.array(
        [
            [100.0, 100.0],
            [110.0, 90.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    price_mask = np.ones_like(prices, dtype=np.bool_)

    # Constant target weights (no rebalance after day 0)
    weights = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )

    # No transaction costs
    commissions = np.zeros_like(prices, dtype=np.float64)

    pv, cash, pos = trade_tracking_kernel(
        initial_portfolio_value=1000.0,
        allocation_mode=0,  # reinvestment
        weights=weights,
        prices=prices,
        price_mask=price_mask,
        commissions=commissions,
    )

    # Buy-and-hold ends flat in this symmetric path
    assert pv.shape == (3,)
    assert pos.shape == (3, 2)

    assert pv[0] == pytest.approx(1000.0)
    assert pv[1] == pytest.approx(1000.0)
    assert pv[2] == pytest.approx(1000.0)

    # Shares should remain constant after day 0
    assert pos[1, 0] == pytest.approx(pos[0, 0])
    assert pos[1, 1] == pytest.approx(pos[0, 1])
    assert pos[2, 0] == pytest.approx(pos[0, 0])
    assert pos[2, 1] == pytest.approx(pos[0, 1])

    # Cash should remain constant after day 0 (no trades)
    assert cash[1] == pytest.approx(cash[0])
    assert cash[2] == pytest.approx(cash[0])
