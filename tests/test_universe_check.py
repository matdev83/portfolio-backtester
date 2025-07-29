
"""
Test script to verify the universe check functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Direct import to avoid dependency issues
from src.portfolio_backtester.strategies.base_strategy import BaseStrategy


class MockStrategyForTest(BaseStrategy):
    pass


import pytest

def test_empty_universe_raises_error():
    """Test that an empty universe raises a ValueError."""
    strategy = MockStrategyForTest({})
    
    with pytest.raises(ValueError, match="Global config universe is empty"):
        strategy.get_universe({})

def test_non_empty_universe_works():
    """Test that a non-empty universe works correctly."""
    strategy = MockStrategyForTest({})
    
    global_config = {"universe": ["AAPL", "GOOGL", "MSFT"]}
    universe = strategy.get_universe(global_config)
    expected = [("AAPL", 1.0), ("GOOGL", 1.0), ("MSFT", 1.0)]
    assert universe == expected