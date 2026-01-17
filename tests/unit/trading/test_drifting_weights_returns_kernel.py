import numpy as np
import pytest

from portfolio_backtester.numba_kernels import drifting_weights_returns_kernel


@pytest.mark.unit
@pytest.mark.fast
def test_drifting_weights_returns_kernel_buy_and_hold_no_volatility_pumping() -> None:
    """Constant target weights should not create volatility pumping.

    Two assets mean-revert over 2 days:
    - With buy-and-hold (no rebalance after day 0), the portfolio ends flat.
    - A buggy daily-rebalance implementation would show a gain.
    """

    # Day 0 is the initial portfolio establishment (return forced to 0 in kernel)
    # Asset returns chosen to be symmetric and mean-reverting:
    # Day 1: A +10%, B -10%
    # Day 2: A -9.0909%, B +11.1111% => prices return to baseline (100 -> 110 -> 100, 100 -> 90 -> 100)
    rets = np.array(
        [
            [0.0, 0.0],
            [0.10, -0.10],
            [-0.09090909, 0.11111111],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(rets, dtype=np.bool_)

    # Constant target weights (shifted-by-1 semantics are handled by caller; here we keep constant)
    w = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    port_rets = drifting_weights_returns_kernel(w, rets, mask)
    assert port_rets.shape == (3,)
    assert port_rets[0] == pytest.approx(0.0)

    equity = float(np.prod(1.0 + port_rets))
    assert equity == pytest.approx(1.0, abs=1e-6)
