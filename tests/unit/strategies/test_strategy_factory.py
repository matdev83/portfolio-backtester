"""Tests for StrategyFactory class."""

import pytest

from portfolio_backtester.strategies._core.strategy_factory import StrategyFactory
from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, strategy_params):
        super().__init__(strategy_params)

    def generate_signals(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def clear_strategy_factory_registry():
    """Fixture to clear the StrategyFactory registry before and after each test."""
    StrategyFactory.clear_registry()
    yield
    StrategyFactory.clear_registry()


class TestStrategyFactory:
    """Test cases for StrategyFactory."""

    def test_dynamic_discovery_and_registration(self):
        """Test that the factory dynamically discovers and registers strategies."""
        # The new strategy registry should be populated automatically
        registry = StrategyFactory.get_registered_strategies()
        assert len(registry) > 0
        # Check for a known strategy that should be discovered
        assert "SimpleMomentumPortfolioStrategy" in registry

    def test_create_strategy_success(self):
        """Test successful creation of a dynamically discovered strategy."""
        params = {"param1": "value1"}
        # Use a known strategy that is dynamically discovered
        strategy = StrategyFactory.create_strategy("SimpleMomentumPortfolioStrategy", params)
        from portfolio_backtester.strategies.portfolio.simple_momentum_portfolio_strategy import (
            SimpleMomentumPortfolioStrategy,
        )

        assert isinstance(strategy, SimpleMomentumPortfolioStrategy)

    def test_create_strategy_unknown_class(self):
        """Test creation with unknown strategy class."""
        with pytest.raises(ValueError, match="Unknown strategy class"):
            StrategyFactory.create_strategy("UnknownStrategy", {})

    def test_manual_registration_prohibited(self):
        """Test that manual registration is prohibited."""
        with pytest.raises(RuntimeError, match="MANUAL STRATEGY REGISTRATION IS PROHIBITED"):
            StrategyFactory.register_strategy("TestStrategy", MockStrategy)

    def test_circular_dependency_detection(self):
        """Test circular dependency detection for meta strategies."""
        # Since manual registration is prohibited, we can't test circular dependency
        # detection in the same way. This test is now focused on the prohibition.
        # The circular dependency detection is tested implicitly through auto-discovery
        # when actual meta strategies are created.

        # We can add a TestMetaStrategy to the circular detection set directly
        # to simulate the scenario without going through registration
        StrategyFactory._circular_detection.add("TestMetaStrategy")

        with pytest.raises(ValueError, match="Circular dependency detected"):
            StrategyFactory.create_strategy("TestMetaStrategy", {})

    def test_clear_registry(self):
        """Test registry clearing."""
        # First, populate the registry
        strategies_before = StrategyFactory.get_registered_strategies()
        assert len(strategies_before) > 0

        # Now, clear it
        StrategyFactory.clear_registry()

        # After clearing, the registry should start fresh and get populated again
        strategies_after = StrategyFactory.get_registered_strategies()
        # The registry will be repopulated automatically, so we check that it works
        assert len(strategies_after) > 0
