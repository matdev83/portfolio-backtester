import pytest

from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy


class MockStrategyForTest(BaseStrategy):
    pass


"""
Test script to verify the universe check functionality.
"""


def test_empty_universe_raises_error():
    """Test that an empty universe raises a ValueError."""
    strategy = MockStrategyForTest({})

    with pytest.raises(ValueError, match="No universe configuration found"):
        strategy.get_universe({})


def test_non_empty_universe_works():
    """Test that a non-empty universe works correctly."""
    strategy = MockStrategyForTest({})

    global_config = {"universe": ["AAPL", "GOOGL", "MSFT"]}
    universe = strategy.get_universe(global_config)
    expected = [("AAPL", 1.0), ("GOOGL", 1.0), ("MSFT", 1.0)]
    assert universe == expected
