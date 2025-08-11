import pandas as pd

from portfolio_backtester.backtester_logic.portfolio_logic import (
    _calculate_position_weights,
)


def test_calculate_position_weights_basic():
    """_calculate_position_weights should convert position quantities into weights."""
    current_positions = {"AAA": 10, "BBB": 5}  # quantities
    prices = pd.Series({"AAA": 100, "BBB": 200})
    base_portfolio_value = 10 * 100 + 5 * 200  # 1000 + 1000 = 2000

    weights = _calculate_position_weights(current_positions, prices, base_portfolio_value)

    # Expected weights
    expected_weights = pd.Series(
        {
            "AAA": (10 * 100) / base_portfolio_value,
            "BBB": (5 * 200) / base_portfolio_value,
        },
        index=["AAA", "BBB"],
    )

    pd.testing.assert_series_equal(weights, expected_weights)

    # Assets with zero quantity or missing price should produce zero weight
    current_positions_extra = {"AAA": 10, "BBB": 5, "CCC": 0}
    prices_extra = pd.Series({"AAA": 100, "BBB": 200, "CCC": 50})
    weights_extra = _calculate_position_weights(
        current_positions_extra, prices_extra, base_portfolio_value
    )

    # Weight for CCC should be zero
    assert weights_extra["CCC"] == 0.0
