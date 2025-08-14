"""
Unit tests for the strategy_factory module.
"""
from unittest.mock import Mock, patch
from portfolio_backtester.interfaces.strategy_factory import (
    DefaultStrategyFactory,
    StrategyFactoryRegistry,
    IStrategyFactory,
)


class TestDefaultStrategyFactory:
    """Test suite for DefaultStrategyFactory."""

    @patch("portfolio_backtester.interfaces.strategy_specification_handler.create_polymorphic_strategy_factory")
    def test_create_strategy(self, mock_create_factory):
        """Test that create_strategy delegates to the polymorphic factory."""
        mock_factory = Mock()
        mock_create_factory.return_value = mock_factory
        factory = DefaultStrategyFactory()
        factory.create_strategy("spec", {})
        mock_factory.create_strategy.assert_called_once_with("spec", {})


class TestStrategyFactoryRegistry:
    """Test suite for StrategyFactoryRegistry."""

    def test_register_and_get_factory(self):
        """Test that a factory can be registered and retrieved."""
        mock_factory = Mock(spec=IStrategyFactory)
        StrategyFactoryRegistry.register_factory("test", mock_factory)
        assert StrategyFactoryRegistry.get_factory("test") == mock_factory

    def test_get_default_factory(self):
        """Test that the default factory is returned when requested."""
        default_factory = StrategyFactoryRegistry.get_factory("default")
        assert isinstance(default_factory, DefaultStrategyFactory)

    def test_get_nonexistent_factory(self):
        """Test that the default factory is returned for a nonexistent factory."""
        nonexistent_factory = StrategyFactoryRegistry.get_factory("nonexistent")
        assert isinstance(nonexistent_factory, DefaultStrategyFactory)
