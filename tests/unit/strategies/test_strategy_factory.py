"""Tests for StrategyFactory class."""

import pytest
from unittest.mock import Mock, patch

from src.portfolio_backtester.strategies.strategy_factory import StrategyFactory
from src.portfolio_backtester.strategies.base.base_strategy import BaseStrategy


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
        # The _populate_registry_if_needed method should be called automatically
        # by get_registered_strategies or create_strategy.
        registry = StrategyFactory.get_registered_strategies()
        assert len(registry) > 0
        # Check for a known strategy that should be discovered
        assert "MomentumStrategy" in registry

    def test_create_strategy_success(self):
        """Test successful creation of a dynamically discovered strategy."""
        params = {"param1": "value1"}
        # Use a known strategy that is dynamically discovered
        strategy = StrategyFactory.create_strategy("MomentumStrategy", params)
        from src.portfolio_backtester.strategies.portfolio.momentum_strategy import MomentumStrategy
        assert isinstance(strategy, MomentumStrategy)

    def test_create_strategy_unknown_class(self):
        """Test creation with unknown strategy class."""
        with pytest.raises(ValueError, match="Unknown strategy class"):
            StrategyFactory.create_strategy("UnknownStrategy", {})

    def test_circular_dependency_detection(self):
        """Test circular dependency detection for meta strategies."""
        # Manually register a mock meta strategy to simulate the scenario
        StrategyFactory.register_strategy("TestMetaStrategy", MockStrategy)
        StrategyFactory._circular_detection.add("TestMetaStrategy")
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            StrategyFactory.create_strategy("TestMetaStrategy", {})

    def test_clear_registry(self):
        """Test registry clearing."""
        # First, populate the registry
        StrategyFactory.get_registered_strategies()
        assert len(StrategyFactory._registry) > 0

        # Now, clear it
        StrategyFactory.clear_registry()
        assert len(StrategyFactory._registry) == 0
        assert not StrategyFactory._registry_populated
