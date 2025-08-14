"""
Unit tests for the optimization_converter module.
"""
import pytest
from portfolio_backtester.interfaces.optimization_converter import (
    DefaultOptimizationSpecConverter,
    GridSearchSpecConverter,
    OptimizationSpecConverterFactory,
)


class TestDefaultOptimizationSpecConverter:
    """Test suite for DefaultOptimizationSpecConverter."""

    def setup_method(self):
        """Set up the test environment."""
        self.converter = DefaultOptimizationSpecConverter()

    def test_convert_parameter_space(self):
        """Test the conversion of the parameter space."""
        optimization_config = {
            "optimize": [
                {"name": "param1", "min": 0, "max": 1},
                {"name": "param2", "min": 1, "max": 2},
            ]
        }
        parameter_space = self.converter.convert_parameter_space(optimization_config)
        assert parameter_space == {"param1": (0.0, 1.0), "param2": (1.0, 2.0)}

    def test_validate_parameter_bounds(self):
        """Test the validation of parameter bounds."""
        assert self.converter.validate_parameter_bounds("p", {"min": 0, "max": 1})
        assert not self.converter.validate_parameter_bounds("p", {"min": 1, "max": 0})
        assert not self.converter.validate_parameter_bounds("p", {"min": 0})
        assert not self.converter.validate_parameter_bounds("p", {"max": 1})

    def test_normalize_parameter_value(self):
        """Test the normalization of parameter values."""
        assert self.converter.normalize_parameter_value("p", 1.0, {"type": "int"}) == 1
        assert self.converter.normalize_parameter_value("p", 1, {"type": "float"}) == 1.0
        assert self.converter.normalize_parameter_value("p", 1, {"type": "bool"}) is True
        assert self.converter.normalize_parameter_value("p", 0, {"type": "bool"}) is False


class TestGridSearchSpecConverter:
    """Test suite for GridSearchSpecConverter."""

    def setup_method(self):
        """Set up the test environment."""
        self.converter = GridSearchSpecConverter()

    def test_convert_parameter_space(self):
        """Test the conversion of the parameter space for grid search."""
        optimization_config = {
            "optimize": [
                {"name": "param1", "min": 0, "max": 1, "steps": 2},
                {"name": "param2", "min": 1, "max": 2, "steps": 3},
            ]
        }
        parameter_space = self.converter.convert_parameter_space(optimization_config)
        assert parameter_space == {"param1": [0.0, 1.0], "param2": [1.0, 1.5, 2.0]}


class TestOptimizationSpecConverterFactory:
    """Test suite for OptimizationSpecConverterFactory."""

    def test_create_converter_default(self):
        """Test that the factory creates a DefaultOptimizationSpecConverter by default."""
        converter = OptimizationSpecConverterFactory.create_converter()
        assert isinstance(converter, DefaultOptimizationSpecConverter)

    def test_create_converter_grid(self):
        """Test that the factory creates a GridSearchSpecConverter for the 'grid' method."""
        converter = OptimizationSpecConverterFactory.create_converter("grid")
        assert isinstance(converter, GridSearchSpecConverter)
