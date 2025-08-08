"""
OptimizationSpecConverter interface and implementations for polymorphic parameter space conversion.

Replaces isinstance checks in optimization parameter handling with proper polymorphic behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class IOptimizationSpecConverter(ABC):
    """Abstract interface for converting optimization specifications."""

    @abstractmethod
    def convert_parameter_space(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert optimization configuration to parameter space specification.

        Args:
            optimization_config: Optimization configuration dictionary

        Returns:
            Converted parameter space specification

        Raises:
            ValueError: If optimization configuration is invalid
        """
        pass

    @abstractmethod
    def validate_parameter_bounds(self, param_name: str, param_spec: Dict[str, Any]) -> bool:
        """
        Validate parameter bounds specification.

        Args:
            param_name: Parameter name
            param_spec: Parameter specification with bounds

        Returns:
            True if bounds are valid, False otherwise
        """
        pass

    @abstractmethod
    def normalize_parameter_value(
        self, param_name: str, value: Any, param_spec: Dict[str, Any]
    ) -> Any:
        """
        Normalize parameter value according to its specification.

        Args:
            param_name: Parameter name
            value: Parameter value to normalize
            param_spec: Parameter specification

        Returns:
            Normalized parameter value

        Raises:
            ValueError: If value cannot be normalized
        """
        pass


class DefaultOptimizationSpecConverter(IOptimizationSpecConverter):
    """Default implementation for optimization specification conversion."""

    def convert_parameter_space(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert optimization config to parameter space using existing logic.

        Args:
            optimization_config: Optimization configuration

        Returns:
            Parameter space specification
        """
        try:
            parameter_space = {}
            optimize_list = optimization_config.get("optimize", [])

            if not isinstance(optimize_list, list):
                raise ValueError(f"Expected list for 'optimize', got {type(optimize_list)}")

            for param_spec in optimize_list:
                if not isinstance(param_spec, dict):
                    logger.warning(f"Skipping non-dict parameter spec: {param_spec}")
                    continue

                param_name = param_spec.get("name")
                if not param_name:
                    logger.warning(f"Parameter spec missing 'name': {param_spec}")
                    continue

                # Convert bounds
                bounds = self._extract_bounds(param_spec)
                if bounds:
                    parameter_space[param_name] = bounds
                else:
                    logger.warning(f"Could not extract bounds for parameter {param_name}")

            return parameter_space

        except Exception as e:
            logger.error(f"Failed to convert parameter space: {e}")
            raise ValueError(f"Parameter space conversion failed: {e}")

    def validate_parameter_bounds(self, param_name: str, param_spec: Dict[str, Any]) -> bool:
        """
        Validate parameter bounds are properly specified.

        Args:
            param_name: Parameter name
            param_spec: Parameter specification

        Returns:
            True if bounds are valid
        """
        try:
            min_val = param_spec.get("min")
            max_val = param_spec.get("max")

            if min_val is None or max_val is None:
                return False

            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                return False

            return min_val < max_val

        except Exception:
            return False

    def normalize_parameter_value(
        self, param_name: str, value: Any, param_spec: Dict[str, Any]
    ) -> Any:
        """
        Normalize parameter value according to its type specification.

        Args:
            param_name: Parameter name
            value: Value to normalize
            param_spec: Parameter specification

        Returns:
            Normalized value
        """
        try:
            param_type = param_spec.get("type", "float")

            if param_type == "int":
                if isinstance(value, (int, float)):
                    return int(value)
                else:
                    raise ValueError(f"Cannot convert {value} to int")

            elif param_type == "float":
                if isinstance(value, (int, float)):
                    return float(value)
                else:
                    raise ValueError(f"Cannot convert {value} to float")

            elif param_type == "bool":
                if isinstance(value, bool):
                    return value
                elif isinstance(value, (int, float)):
                    return bool(value)
                else:
                    raise ValueError(f"Cannot convert {value} to bool")

            else:
                # Return as-is for unknown types
                return value

        except Exception as e:
            logger.error(f"Failed to normalize parameter {param_name}: {e}")
            raise ValueError(f"Parameter normalization failed: {e}")

    def _extract_bounds(self, param_spec: Dict[str, Any]) -> Union[Tuple[float, float], None]:
        """
        Extract parameter bounds from specification.

        Args:
            param_spec: Parameter specification

        Returns:
            Tuple of (min, max) bounds or None if invalid
        """
        min_val = param_spec.get("min")
        max_val = param_spec.get("max")

        if min_val is None or max_val is None:
            return None

        try:
            return (float(min_val), float(max_val))
        except (ValueError, TypeError):
            return None


class GridSearchSpecConverter(IOptimizationSpecConverter):
    """Converter for grid search optimization specifications."""

    def convert_parameter_space(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to grid search parameter space."""
        try:
            parameter_space = {}
            optimize_list = optimization_config.get("optimize", [])

            for param_spec in optimize_list:
                if not isinstance(param_spec, dict):
                    continue

                param_name = param_spec.get("name")
                if not param_name:
                    continue

                # For grid search, create discrete values
                values = self._create_grid_values(param_spec)
                if values:
                    parameter_space[param_name] = values

            return parameter_space

        except Exception as e:
            logger.error(f"Failed to convert grid search parameter space: {e}")
            raise ValueError(f"Grid search conversion failed: {e}")

    def validate_parameter_bounds(self, param_name: str, param_spec: Dict[str, Any]) -> bool:
        """Validate grid search bounds."""
        return DefaultOptimizationSpecConverter().validate_parameter_bounds(param_name, param_spec)

    def normalize_parameter_value(
        self, param_name: str, value: Any, param_spec: Dict[str, Any]
    ) -> Any:
        """Normalize grid search parameter value."""
        return DefaultOptimizationSpecConverter().normalize_parameter_value(
            param_name, value, param_spec
        )

    def _create_grid_values(self, param_spec: Dict[str, Any]) -> Union[List[float], None]:
        """Create discrete grid values from parameter specification."""
        try:
            min_val = param_spec.get("min")
            max_val = param_spec.get("max")
            steps = param_spec.get("steps", 10)

            if min_val is None or max_val is None:
                return None

            import numpy as np

            values = np.linspace(float(min_val), float(max_val), int(steps)).tolist()
            return [float(v) for v in values]

        except Exception:
            return None


class OptimizationSpecConverterFactory:
    """Factory for creating appropriate optimization spec converters."""

    @staticmethod
    def create_converter(optimization_method: str = "default") -> IOptimizationSpecConverter:
        """
        Create appropriate converter based on optimization method.

        Args:
            optimization_method: Optimization method name

        Returns:
            Appropriate converter implementation
        """
        if optimization_method.lower() == "grid":
            return GridSearchSpecConverter()
        else:
            return DefaultOptimizationSpecConverter()
