"""Tests for BaseMetaStrategy class."""

import pytest
import pandas as pd

from portfolio_backtester.strategies._core.base.meta_strategy import BaseMetaStrategy


class DummyMetaStrategy(BaseMetaStrategy):
    """Helper implementation of BaseMetaStrategy for testing (avoid pytest collection issues).

    NOTE: Pytest cannot collect classes with a user-defined __init__ method as
    test classes. We keep this helper here but rename to avoid conflicting with
    pytest collection behavior. Tests will instantiate this class directly.
    """

    # Provide a lightweight no-arg __init__ so pytest can treat this as a helper
    # class without treating it as a test class. We implement __new__ to avoid
    # class-level __init__ detection by pytest's collector.
    def __new__(cls, config=None):
        obj = super(DummyMetaStrategy, cls).__new__(cls)
        # Initialize via BaseMetaStrategy.__init__ normally
        BaseMetaStrategy.__init__(obj, config if config is not None else {"initial_capital": 1000000, "allocations": []})
        return obj

    def allocate_capital(self):
        return {allocation.strategy_id: allocation.weight for allocation in self.allocations}

# Backwards-compatible factory function expected by some tests.
# Exposing a function prevents pytest from attempting to collect a class
# named `TestMetaStrategy` (which would conflict with pytest's collection rules).
def TestMetaStrategy(config=None):
    return DummyMetaStrategy(config)


class TestBaseMetaStrategy:
    """Test cases for BaseMetaStrategy."""

    def test_initialization_with_valid_config(self):
        """Test meta strategy initialization with valid configuration."""
        config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {"param1": "value1"},
                    "weight": 0.6,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {"param2": "value2"},
                    "weight": 0.4,
                },
            ],
        }

        meta_strategy = DummyMetaStrategy(config)

        assert meta_strategy.initial_capital == 1000000
        assert meta_strategy.available_capital == 1000000
        assert meta_strategy.cumulative_pnl == 0.0
        assert len(meta_strategy.allocations) == 2
        assert meta_strategy.allocations[0].strategy_id == "strategy1"
        assert meta_strategy.allocations[0].weight == 0.6
        assert meta_strategy.allocations[1].strategy_id == "strategy2"
        assert meta_strategy.allocations[1].weight == 0.4

    def test_allocation_validation_weights_sum_to_one(self):
        """Test that allocation weights must sum to 1.0."""
        config = {
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.5,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.3,  # Sum = 0.8, should fail
                },
            ]
        }

        with pytest.raises(ValueError, match="Allocation weights must sum to 1.0"):
            TestMetaStrategy(config)

    def test_allocation_validation_negative_weights(self):
        """Test that negative allocation weights are rejected."""
        config = {
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": -0.2,  # Negative weight should fail
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 1.2,
                },
            ]
        }

        with pytest.raises(ValueError, match="Allocation weight cannot be negative"):
            TestMetaStrategy(config)

    def test_allocation_validation_minimum_allocation(self):
        """Test minimum allocation constraint."""
        config = {
            "min_allocation": 0.1,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.05,  # Below minimum
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.95,
                },
            ],
        }

        with pytest.raises(ValueError, match="below minimum"):
            TestMetaStrategy(config)

    def test_calculate_sub_strategy_capital(self):
        """Test capital allocation calculation."""
        config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.6,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.4,
                },
            ],
        }

        meta_strategy = DummyMetaStrategy(config)
        capital_allocations = meta_strategy.calculate_sub_strategy_capital()

        assert capital_allocations["strategy1"] == 600000  # 60% of 1M
        assert capital_allocations["strategy2"] == 400000  # 40% of 1M

    def test_update_available_capital(self):
        """Test capital update with sub-strategy returns."""
        config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.6,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.4,
                },
            ],
        }

        meta_strategy = DummyMetaStrategy(config)

        # Simulate returns: strategy1 +5%, strategy2 -2%
        returns = {"strategy1": 0.05, "strategy2": -0.02}  # 5% return  # -2% return

        meta_strategy.update_available_capital(returns)

        # Expected P&L: (600000 * 0.05) + (400000 * -0.02) = 30000 - 8000 = 22000
        expected_pnl = 22000
        expected_capital = 1000000 + expected_pnl

        assert meta_strategy.cumulative_pnl == expected_pnl
        assert meta_strategy.available_capital == expected_capital

    def test_aggregate_signals_basic(self):
        """Test basic signal aggregation."""
        config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.6,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 0.4,
                },
            ],
        }

        meta_strategy = DummyMetaStrategy(config)
        current_date = pd.Timestamp("2023-01-01")

        # Create mock signals
        signals1 = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.5]}, index=[current_date])

        signals2 = pd.DataFrame({"AAPL": [0.3], "GOOGL": [0.7]}, index=[current_date])

        sub_strategy_signals = {"strategy1": signals1, "strategy2": signals2}

        aggregated = meta_strategy.aggregate_signals(sub_strategy_signals, current_date)

        # Expected: AAPL = 0.6*0.5 + 0.4*0.3 = 0.3 + 0.12 = 0.42
        #          MSFT = 0.6*0.5 + 0.4*0 = 0.3
        #          GOOGL = 0.6*0 + 0.4*0.7 = 0.28

        assert aggregated.loc[current_date, "AAPL"] == pytest.approx(0.42)
        assert aggregated.loc[current_date, "MSFT"] == pytest.approx(0.3)
        assert aggregated.loc[current_date, "GOOGL"] == pytest.approx(0.28)

    def test_aggregate_signals_empty_input(self):
        """Test signal aggregation with empty input."""
        config = {
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "TestStrategy",
                    "strategy_params": {},
                    "weight": 1.0,
                }
            ]
        }

        meta_strategy = DummyMetaStrategy(config)
        current_date = pd.Timestamp("2023-01-01")

        # Empty signals
        aggregated = meta_strategy.aggregate_signals({}, current_date)
        assert aggregated.empty

    def test_tunable_parameters(self):
        """Test tunable parameters."""
        # Backwards-compatible reference: some tests expect TestMetaStrategy name
        # to exist. Provide an alias to our DummyMetaStrategy defined above.
        TestMetaStrategy = DummyMetaStrategy
        params = TestMetaStrategy.tunable_parameters()
        expected_params = {"initial_capital", "min_allocation", "rebalance_threshold"}
        assert expected_params.issubset(params)
