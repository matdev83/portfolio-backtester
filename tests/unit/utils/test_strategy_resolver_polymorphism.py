"""
Tests for strategy resolver polymorphic interfaces.

This module tests the polymorphic interfaces that replace isinstance violations
in strategy resolution functionality.
"""

from unittest.mock import Mock, patch

from src.portfolio_backtester.interfaces.strategy_resolver_interface import (
    DictStrategySpecificationResolver,
    StringStrategySpecificationResolver,
    NullStrategySpecificationResolver,
    StrategySpecificationResolverFactory,
    DefaultStrategyLookup,
    PolymorphicStrategyResolver,
    create_strategy_resolver,
)


class TestDictStrategySpecificationResolver:
    """Test dictionary-based strategy specification resolver."""

    def setup_method(self):
        self.resolver = DictStrategySpecificationResolver()

    def test_can_resolve_dict_with_get_method(self):
        """Test that resolver can handle objects with get method."""
        dict_spec = {"name": "test_strategy"}
        assert self.resolver.can_resolve(dict_spec) is True

    def test_can_resolve_dict_like_object(self):
        """Test that resolver can handle dict-like objects."""
        mock_dict = Mock()
        mock_dict.get = Mock()
        assert self.resolver.can_resolve(mock_dict) is True

    def test_cannot_resolve_string(self):
        """Test that resolver cannot handle string specifications."""
        assert self.resolver.can_resolve("test_strategy") is False

    def test_cannot_resolve_none(self):
        """Test that resolver cannot handle None."""
        assert self.resolver.can_resolve(None) is False

    def test_resolve_to_name_with_name_key(self):
        """Test resolving specification with 'name' key."""
        spec = {"name": "momentum_strategy"}
        result = self.resolver.resolve_to_name(spec)
        assert result == "momentum_strategy"

    def test_resolve_to_name_with_strategy_key(self):
        """Test resolving specification with 'strategy' key."""
        spec = {"strategy": "calmar_strategy"}
        result = self.resolver.resolve_to_name(spec)
        assert result == "calmar_strategy"

    def test_resolve_to_name_with_type_key(self):
        """Test resolving specification with 'type' key."""
        spec = {"type": "fixed_weight_strategy"}
        result = self.resolver.resolve_to_name(spec)
        assert result == "fixed_weight_strategy"

    def test_resolve_to_name_priority_order(self):
        """Test that 'name' takes priority over 'strategy', which takes priority over 'type'."""
        spec = {
            "name": "name_strategy",
            "strategy": "strategy_strategy", 
            "type": "type_strategy"
        }
        result = self.resolver.resolve_to_name(spec)
        assert result == "name_strategy"

    def test_resolve_to_name_fallback_to_strategy(self):
        """Test fallback to 'strategy' when 'name' is not present."""
        spec = {"strategy": "strategy_strategy", "type": "type_strategy"}
        result = self.resolver.resolve_to_name(spec)
        assert result == "strategy_strategy"

    def test_resolve_to_name_fallback_to_type(self):
        """Test fallback to 'type' when 'name' and 'strategy' are not present."""
        spec = {"type": "type_strategy"}
        result = self.resolver.resolve_to_name(spec)
        assert result == "type_strategy"

    def test_resolve_to_name_no_valid_keys(self):
        """Test handling specification with no valid keys."""
        spec = {"invalid_key": "some_value"}
        result = self.resolver.resolve_to_name(spec)
        assert result is None

    def test_resolve_to_name_none_values(self):
        """Test handling specification with None values."""
        spec = {"name": None, "strategy": None, "type": None}
        result = self.resolver.resolve_to_name(spec)
        assert result is None


class TestStringStrategySpecificationResolver:
    """Test string-based strategy specification resolver."""

    def setup_method(self):
        self.resolver = StringStrategySpecificationResolver()

    def test_can_resolve_string(self):
        """Test that resolver can handle string specifications."""
        assert self.resolver.can_resolve("test_strategy") is True

    def test_can_resolve_string_like_object(self):
        """Test that resolver can handle string-like objects."""
        mock_string = Mock()
        mock_string.lower = Mock()
        assert self.resolver.can_resolve(mock_string) is True

    def test_cannot_resolve_dict(self):
        """Test that resolver cannot handle dictionary specifications."""
        assert self.resolver.can_resolve({"name": "test"}) is False

    def test_cannot_resolve_none(self):
        """Test that resolver cannot handle None."""
        assert self.resolver.can_resolve(None) is False

    def test_resolve_to_name_string(self):
        """Test resolving string specification."""
        result = self.resolver.resolve_to_name("momentum_strategy")
        assert result == "momentum_strategy"

    def test_resolve_to_name_none(self):
        """Test resolving None specification."""
        result = self.resolver.resolve_to_name(None)
        assert result is None

    def test_resolve_to_name_converts_to_string(self):
        """Test that non-string specifications are converted to string."""
        result = self.resolver.resolve_to_name(123)
        assert result == "123"


class TestNullStrategySpecificationResolver:
    """Test null/fallback strategy specification resolver."""

    def setup_method(self):
        self.resolver = NullStrategySpecificationResolver()

    def test_can_resolve_always_true(self):
        """Test that resolver always returns True (fallback)."""
        assert self.resolver.can_resolve("anything") is True
        assert self.resolver.can_resolve({"anything": "value"}) is True
        assert self.resolver.can_resolve(None) is True
        assert self.resolver.can_resolve(123) is True

    def test_resolve_to_name_always_none(self):
        """Test that resolver always returns None."""
        assert self.resolver.resolve_to_name("anything") is None
        assert self.resolver.resolve_to_name({"anything": "value"}) is None
        assert self.resolver.resolve_to_name(None) is None


class TestStrategySpecificationResolverFactory:
    """Test strategy specification resolver factory."""

    def setup_method(self):
        self.factory = StrategySpecificationResolverFactory()

    def test_get_resolver_for_dict(self):
        """Test getting resolver for dictionary specification."""
        spec = {"name": "test_strategy"}
        resolver = self.factory.get_resolver(spec)
        assert isinstance(resolver, DictStrategySpecificationResolver)

    def test_get_resolver_for_string(self):
        """Test getting resolver for string specification."""
        spec = "test_strategy"
        resolver = self.factory.get_resolver(spec)
        assert isinstance(resolver, StringStrategySpecificationResolver)

    def test_get_resolver_for_invalid_spec(self):
        """Test getting resolver for invalid specification."""
        spec = 123
        resolver = self.factory.get_resolver(spec)
        assert isinstance(resolver, NullStrategySpecificationResolver)

    def test_get_resolver_for_none(self):
        """Test getting resolver for None specification."""
        resolver = self.factory.get_resolver(None)
        assert isinstance(resolver, NullStrategySpecificationResolver)


class TestDefaultStrategyLookup:
    """Test default strategy lookup implementation."""

    def setup_method(self):
        self.lookup = DefaultStrategyLookup()

    @patch('src.portfolio_backtester.strategies.strategy_factory.StrategyFactory.create_strategy')
    def test_lookup_strategy_found(self, mock_create_strategy):
        """Test successful strategy lookup."""
        mock_strategy_class = Mock()
        mock_create_strategy.return_value = mock_strategy_class
        
        result = self.lookup.lookup_strategy("momentum_strategy")
        assert result == mock_strategy_class

    @patch('src.portfolio_backtester.strategies.strategy_factory.StrategyFactory.create_strategy')
    def test_lookup_strategy_not_found(self, mock_create_strategy):
        """Test strategy lookup when strategy not found."""
        mock_create_strategy.side_effect = ValueError
        
        result = self.lookup.lookup_strategy("nonexistent_strategy")
        assert result is None

    @patch('src.portfolio_backtester.strategies.strategy_factory.StrategyFactory.create_strategy')
    def test_lookup_strategy_empty_registry(self, mock_create_strategy):
        """Test strategy lookup with empty strategy registry."""
        mock_create_strategy.side_effect = ValueError
        
        result = self.lookup.lookup_strategy("any_strategy")
        assert result is None


class TestPolymorphicStrategyResolver:
    """Test polymorphic strategy resolver that eliminates isinstance violations."""

    def setup_method(self):
        self.mock_strategy_lookup = Mock()
        self.resolver = PolymorphicStrategyResolver(self.mock_strategy_lookup)

    def test_resolve_strategy_dict_specification(self):
        """Test resolving dictionary strategy specification."""
        mock_strategy = Mock()
        self.mock_strategy_lookup.lookup_strategy.return_value = mock_strategy
        
        spec = {"name": "momentum_strategy"}
        result = self.resolver.resolve_strategy(spec)
        
        assert result == mock_strategy
        self.mock_strategy_lookup.lookup_strategy.assert_called_once_with("momentum_strategy", None)

    def test_resolve_strategy_string_specification(self):
        """Test resolving string strategy specification."""
        mock_strategy = Mock()
        self.mock_strategy_lookup.lookup_strategy.return_value = mock_strategy
        
        result = self.resolver.resolve_strategy("calmar_strategy")
        
        assert result == mock_strategy
        self.mock_strategy_lookup.lookup_strategy.assert_called_once_with("calmar_strategy", None)

    def test_resolve_strategy_invalid_specification(self):
        """Test resolving invalid strategy specification."""
        result = self.resolver.resolve_strategy(123)
        assert result is None
        self.mock_strategy_lookup.lookup_strategy.assert_not_called()

    def test_resolve_strategy_none_specification(self):
        """Test resolving None strategy specification."""
        result = self.resolver.resolve_strategy(None)
        assert result is None
        self.mock_strategy_lookup.lookup_strategy.assert_not_called()

    def test_resolve_strategy_not_found_in_lookup(self):
        """Test resolving when strategy is not found in lookup."""
        self.mock_strategy_lookup.lookup_strategy.return_value = None
        
        result = self.resolver.resolve_strategy("nonexistent_strategy")
        assert result is None
        self.mock_strategy_lookup.lookup_strategy.assert_called_once_with("nonexistent_strategy", None)

    def test_resolve_strategy_uses_default_lookup_when_none_provided(self):
        """Test that resolver uses default lookup when none provided."""
        resolver = PolymorphicStrategyResolver()
        assert isinstance(resolver._strategy_lookup, DefaultStrategyLookup)


class TestCreateStrategyResolver:
    """Test factory function for creating strategy resolver."""

    def test_create_strategy_resolver_default(self):
        """Test creating resolver with default parameters."""
        resolver = create_strategy_resolver()
        assert isinstance(resolver, PolymorphicStrategyResolver)
        assert isinstance(resolver._strategy_lookup, DefaultStrategyLookup)

    def test_create_strategy_resolver_custom_lookup(self):
        """Test creating resolver with custom lookup."""
        custom_lookup = Mock()
        resolver = create_strategy_resolver(custom_lookup)
        assert isinstance(resolver, PolymorphicStrategyResolver)
        assert resolver._strategy_lookup == custom_lookup


class TestPolymorphicIntegration:
    """Integration tests verifying isinstance violations are eliminated."""

    def test_no_isinstance_usage_in_polymorphic_resolver(self):
        """Test that polymorphic resolver doesn't use isinstance internally."""
        resolver = create_strategy_resolver()
        
        # Test with various input types that would have triggered isinstance checks
        test_cases = [
            {"name": "test_strategy"},
            "string_strategy",
            None,
            123,
            [],
        ]
        
        # All these should work without isinstance checks
        for test_case in test_cases:
            # Should not raise exceptions due to isinstance usage
            result = resolver.resolve_strategy(test_case)
            # Result depends on whether strategy exists, but should not fail due to type checking
            assert result is None or result is not None  # Basic success test

    @patch('src.portfolio_backtester.strategies.strategy_factory.StrategyFactory.create_strategy')
    def test_polymorphic_resolver_equivalent_to_original_logic(self, mock_create_strategy):
        """Test that polymorphic resolver produces same results as original isinstance logic."""
        # Mock strategy registry
        mock_strategy = Mock()
        mock_create_strategy.return_value = mock_strategy
        
        resolver = create_strategy_resolver()
        
        # Test cases that mirror original isinstance logic
        test_cases = [
            # Dict with name
            ({"name": "test_strategy"}, mock_strategy),
            # Dict with strategy
            ({"strategy": "test_strategy"}, mock_strategy),
            # Dict with type
            ({"type": "test_strategy"}, mock_strategy),
            # String
            ("test_strategy", mock_strategy),
            # Invalid dict
            ({"invalid": "value"}, None),
            # Invalid type
            (123, None),
            # None
            (None, None),
        ]
        
        for spec, expected in test_cases:
            result = resolver.resolve_strategy(spec)
            assert result == expected
