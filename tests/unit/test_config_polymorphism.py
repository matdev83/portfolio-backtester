"""
Tests for configuration module polymorphism interfaces.

This module tests the polymorphic interfaces that replaced isinstance checks
in config_initializer.py, ensuring SOLID principle compliance and backward compatibility.
"""

from unittest.mock import Mock, patch
from typing import Set, Any, Dict

from portfolio_backtester.interfaces.parameter_extractor import (
    ParameterExtractor,
    StringParameterExtractor,
    DictParameterExtractor,
    ParameterExtractorFactory,
)
from portfolio_backtester.interfaces.optimization_populator import (
    LegacyOptimizationPopulator,
    ModernOptimizationPopulator,
    CompositeOptimizationPopulator,
    OptimizationPopulatorFactory,
)
from portfolio_backtester.config_initializer import (
    _get_strategy_tunable_params,
    populate_default_optimizations,
)


class TestParameterExtractor:
    """Test cases for parameter extraction interfaces."""

    def test_string_parameter_extractor_can_handle(self):
        """Test StringParameterExtractor can handle string inputs."""
        extractor = StringParameterExtractor()
        assert extractor.can_handle("momentum_strategy")
        assert not extractor.can_handle({"name": "momentum_strategy"})
        assert not extractor.can_handle(123)

    def test_dict_parameter_extractor_can_handle(self):
        """Test DictParameterExtractor can handle dictionary inputs."""
        extractor = DictParameterExtractor()
        assert extractor.can_handle({"name": "momentum_strategy"})
        assert not extractor.can_handle("momentum_strategy")
        assert not extractor.can_handle(123)

    @patch("portfolio_backtester.utils._resolve_strategy")
    def test_string_parameter_extractor_extract_parameters(self, mock_resolve):
        """Test StringParameterExtractor parameter extraction."""
        mock_strategy = Mock()
        mock_strategy.tunable_parameters.return_value = ["param1", "param2"]
        mock_resolve.return_value = mock_strategy

        extractor = StringParameterExtractor()
        result = extractor.extract_parameters("momentum_strategy")

        assert result == {"param1", "param2"}
        mock_resolve.assert_called_once_with("momentum_strategy")

    @patch("portfolio_backtester.utils._resolve_strategy")
    def test_dict_parameter_extractor_extract_parameters(self, mock_resolve):
        """Test DictParameterExtractor parameter extraction from dict."""
        mock_strategy = Mock()
        mock_strategy.tunable_parameters.return_value = ["param1", "param2"]
        mock_resolve.return_value = mock_strategy

        extractor = DictParameterExtractor()
        strategy_spec = {"name": "momentum_strategy"}
        result = extractor.extract_parameters(strategy_spec)

        assert result == {"param1", "param2"}
        mock_resolve.assert_called_once_with("momentum_strategy")

    def test_dict_parameter_extractor_alternative_keys(self):
        """Test DictParameterExtractor handles alternative dict keys."""
        with patch("portfolio_backtester.utils._resolve_strategy") as mock_resolve:
            mock_strategy = Mock()
            mock_strategy.tunable_parameters.return_value = ["param1"]
            mock_resolve.return_value = mock_strategy

            extractor = DictParameterExtractor()

            # Test 'strategy' key
            result = extractor.extract_parameters({"strategy": "test_strategy"})
            assert result == {"param1"}

            # Test 'type' key
            result = extractor.extract_parameters({"type": "test_strategy"})
            assert result == {"param1"}

    def test_parameter_extractor_factory(self):
        """Test ParameterExtractorFactory selects correct extractor."""
        factory = ParameterExtractorFactory()

        # Test string input
        extractor = factory.get_extractor("momentum_strategy")
        assert isinstance(extractor, StringParameterExtractor)

        # Test dict input
        extractor = factory.get_extractor({"name": "momentum_strategy"})
        assert isinstance(extractor, DictParameterExtractor)

        # Test unknown input falls back to string
        extractor = factory.get_extractor(123)
        assert isinstance(extractor, StringParameterExtractor)

    @patch("portfolio_backtester.utils._resolve_strategy")
    def test_parameter_extractor_factory_extract_parameters(self, mock_resolve):
        """Test ParameterExtractorFactory parameter extraction."""
        mock_strategy = Mock()
        mock_strategy.tunable_parameters.return_value = ["param1", "param2"]
        mock_resolve.return_value = mock_strategy

        factory = ParameterExtractorFactory()

        # Test string extraction
        result = factory.extract_parameters("momentum_strategy")
        assert result == {"param1", "param2"}

        # Test dict extraction
        result = factory.extract_parameters({"name": "momentum_strategy"})
        assert result == {"param1", "param2"}


class TestOptimizationPopulator:
    """Test cases for optimization population interfaces."""

    def test_legacy_optimization_populator_can_handle(self):
        """Test LegacyOptimizationPopulator can handle legacy format."""
        populator = LegacyOptimizationPopulator()

        config_with_optimize = {"optimize": [{"parameter": "param1"}]}
        assert populator.can_handle(config_with_optimize)

        config_without_optimize = {"strategy": "test"}
        assert not populator.can_handle(config_without_optimize)

        assert not populator.can_handle("not_dict")

    def test_legacy_optimization_populator_extract_parameters(self):
        """Test LegacyOptimizationPopulator parameter extraction."""
        populator = LegacyOptimizationPopulator()

        config = {
            "optimize": [
                {"parameter": "param1"},
                {"parameter": "param2"},
            ]
        }

        result = populator.extract_optimized_parameters(config)
        assert result == {"param1", "param2"}

        # Test empty config
        result = populator.extract_optimized_parameters({})
        assert result == set()

    def test_modern_optimization_populator_can_handle(self):
        """Test ModernOptimizationPopulator can handle modern format."""
        populator = ModernOptimizationPopulator()

        modern_config = {"strategy": {"params": {"param1": {"optimization": {"range": [1, 10]}}}}}
        assert populator.can_handle(modern_config)

        legacy_config = {"optimize": [{"parameter": "param1"}]}
        assert not populator.can_handle(legacy_config)

    def test_modern_optimization_populator_extract_parameters(self):
        """Test ModernOptimizationPopulator parameter extraction."""
        populator = ModernOptimizationPopulator()

        config = {
            "strategy": {
                "params": {
                    "param1": {"optimization": {"range": [1, 10]}},
                    "param2": {"optimization": {"choices": ["a", "b"]}},
                    "param3": {"value": 5},  # Not optimized
                }
            }
        }

        result = populator.extract_optimized_parameters(config)
        assert result == {"param1", "param2"}

    def test_modern_optimization_populator_populate_strategy_params(self):
        """Test ModernOptimizationPopulator strategy params population."""
        populator = ModernOptimizationPopulator()

        config = {
            "strategy": {
                "params": {
                    "param1": {"optimization": {"range": [1, 10]}},
                    "param2": {"optimization": {"choices": ["a", "b"]}},
                }
            }
        }

        scenario_config: Dict[str, Any] = {"strategy_params": {}}
        populator.populate_strategy_params(config, scenario_config)

        assert scenario_config["strategy_params"]["param1"] == 1  # First range value
        assert scenario_config["strategy_params"]["param2"] == "a"  # First choice

    def test_composite_optimization_populator(self):
        """Test CompositeOptimizationPopulator combines multiple populators."""
        populator = CompositeOptimizationPopulator()

        # Config with both legacy and modern formats
        config = {
            "optimize": [{"parameter": "legacy_param"}],
            "strategy": {"params": {"modern_param": {"optimization": {"range": [1, 5]}}}},
        }

        result = populator.extract_optimized_parameters(config)
        assert "legacy_param" in result
        assert "modern_param" in result

    def test_optimization_populator_factory(self):
        """Test OptimizationPopulatorFactory returns composite populator."""
        factory = OptimizationPopulatorFactory()

        populator = factory.get_populator({"test": "config"})
        assert isinstance(populator, CompositeOptimizationPopulator)


class TestConfigInitializerBackwardCompatibility:
    """Test backward compatibility of refactored config_initializer functions."""

    @patch("portfolio_backtester.utils._resolve_strategy")
    def test_get_strategy_tunable_params_string_input(self, mock_resolve):
        """Test _get_strategy_tunable_params with string input (original behavior)."""
        mock_strategy = Mock()
        mock_strategy.tunable_parameters.return_value = ["param1", "param2"]
        mock_resolve.return_value = mock_strategy

        result = _get_strategy_tunable_params("momentum_strategy")
        assert result == {"param1", "param2"}

    @patch("portfolio_backtester.utils._resolve_strategy")
    def test_get_strategy_tunable_params_dict_input(self, mock_resolve):
        """Test _get_strategy_tunable_params with dict input (original behavior)."""
        mock_strategy = Mock()
        mock_strategy.tunable_parameters.return_value = ["param1", "param2"]
        mock_resolve.return_value = mock_strategy

        strategy_spec = {"name": "momentum_strategy"}
        result = _get_strategy_tunable_params(strategy_spec)
        assert result == {"param1", "param2"}

    def test_populate_default_optimizations_backward_compatibility(self):
        """Test populate_default_optimizations maintains original behavior."""
        # Test scenario with legacy format
        scenarios: list[Dict[str, Any]] = [
            {
                "name": "test_scenario",
                "strategy": "momentum_strategy",
                "optimize": [{"parameter": "existing_param"}],
                "strategy_params": {},
            }
        ]

        optimizer_defaults: Dict[str, Dict[str, Any]] = {
            "existing_param": {"type": "int", "low": 1, "high": 10}
        }

        with patch(
            "portfolio_backtester.config_initializer._get_strategy_tunable_params"
        ) as mock_get_params:
            mock_get_params.return_value = {"existing_param", "new_param"}

            populate_default_optimizations(scenarios, optimizer_defaults)

            # Should preserve existing optimize list behavior
            assert len(scenarios[0]["optimize"]) == 1
            assert scenarios[0]["optimize"][0]["parameter"] == "existing_param"

    def test_populate_default_optimizations_modern_format(self):
        """Test populate_default_optimizations handles modern format."""
        scenarios: list[Dict[str, Any]] = [
            {
                "name": "test_scenario",
                "strategy": {"params": {"param1": {"optimization": {"range": [1, 10]}}}},
                "optimize": [],
                "strategy_params": {},
            }
        ]

        optimizer_defaults: Dict[str, Dict[str, Any]] = {}

        populate_default_optimizations(scenarios, optimizer_defaults)

        # Should populate strategy_params from modern format
        assert "param1" in scenarios[0]["strategy_params"]
        assert scenarios[0]["strategy_params"]["param1"] == 1


class TestSOLIDCompliance:
    """Test SOLID principle compliance of the new interfaces."""

    def test_single_responsibility_principle(self):
        """Test that each class has a single responsibility."""
        # ParameterExtractor classes should only handle parameter extraction
        string_extractor = StringParameterExtractor()

        # Each extractor has focused responsibility
        assert hasattr(string_extractor, "can_handle")
        assert hasattr(string_extractor, "extract_parameters")
        assert len([method for method in dir(string_extractor) if not method.startswith("_")]) == 2

    def test_open_closed_principle(self):
        """Test that the system is open for extension but closed for modification."""

        # Can create new parameter extractors without modifying existing code
        class CustomParameterExtractor(ParameterExtractor):
            def can_handle(self, strategy_spec: Any) -> bool:
                return isinstance(strategy_spec, list)

            def extract_parameters(self, strategy_spec: Any) -> Set[str]:
                return {"custom_param"}

        # Factory can be extended with new extractors
        factory = ParameterExtractorFactory()
        factory._extractors.append(CustomParameterExtractor())

        result = factory.extract_parameters(["test"])
        assert result == {"custom_param"}

    def test_liskov_substitution_principle(self):
        """Test that derived classes can substitute base classes."""
        # All extractors should be substitutable
        extractors = [StringParameterExtractor(), DictParameterExtractor()]

        for extractor in extractors:
            # All should implement the interface correctly
            assert hasattr(extractor, "can_handle")
            assert hasattr(extractor, "extract_parameters")
            assert callable(extractor.can_handle)
            assert callable(extractor.extract_parameters)

    def test_interface_segregation_principle(self):
        """Test that interfaces are focused and not forced to implement unused methods."""
        # Each interface has minimal, focused methods
        extractor = StringParameterExtractor()
        populator = LegacyOptimizationPopulator()

        # Interfaces don't force implementation of unrelated methods
        assert not hasattr(extractor, "populate_strategy_params")
        assert not hasattr(populator, "extract_parameters")

    def test_dependency_inversion_principle(self):
        """Test that high-level modules don't depend on low-level modules."""
        # Factory depends on abstractions, not concrete implementations
        factory = ParameterExtractorFactory()

        # Factory works with any ParameterExtractor implementation
        for extractor in factory._extractors:
            assert isinstance(extractor, ParameterExtractor)
