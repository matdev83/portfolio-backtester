"""
Optimization population interfaces for configuration module polymorphism.

This module provides interfaces to replace isinstance checks in config_initializer.py
with polymorphic strategies for handling different optimization configuration formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Set


class OptimizationPopulator(ABC):
    """Interface for populating optimization parameters from configuration data."""

    @abstractmethod
    def can_handle(self, config_data: Any) -> bool:
        """Check if this populator can handle the given configuration data."""
        pass

    @abstractmethod
    def extract_optimized_parameters(self, config_data: Any) -> Set[str]:
        """Extract parameters that are being optimized from the configuration."""
        pass

    @abstractmethod
    def populate_strategy_params(self, config_data: Any, scenario_config: Dict[str, Any]) -> None:
        """Populate strategy_params with default values from optimization config."""
        pass


class LegacyOptimizationPopulator(OptimizationPopulator):
    """Handles legacy 'optimize' section format."""

    def can_handle(self, config_data: Any) -> bool:
        """Check if config has legacy optimize section."""
        return isinstance(config_data, dict) and "optimize" in config_data

    def extract_optimized_parameters(self, config_data: Any) -> Set[str]:
        """Extract parameters from legacy optimize section."""
        if not isinstance(config_data, dict):
            return set()

        optimize_list = config_data.get("optimize", [])
        return {opt_spec["parameter"] for opt_spec in optimize_list if isinstance(opt_spec, dict)}

    def populate_strategy_params(self, config_data: Any, scenario_config: Dict[str, Any]) -> None:
        """No additional population needed for legacy format."""
        pass


class ModernOptimizationPopulator(OptimizationPopulator):
    """Handles modern 'strategy.params.param_name.optimization' format."""

    def can_handle(self, config_data: Any) -> bool:
        """Check if config has modern strategy.params.optimization format."""
        if not isinstance(config_data, dict):
            return False

        strategy_config = config_data.get("strategy", {})
        if not isinstance(strategy_config, dict):
            return False

        params_config = strategy_config.get("params", {})
        if not isinstance(params_config, dict):
            return False

        # Check if any param has optimization config
        for param_config in params_config.values():
            if isinstance(param_config, dict) and "optimization" in param_config:
                return True

        return False

    def extract_optimized_parameters(self, config_data: Any) -> Set[str]:
        """Extract parameters from modern optimization format."""
        if not isinstance(config_data, dict):
            return set()

        strategy_config = config_data.get("strategy", {})
        if not isinstance(strategy_config, dict):
            return set()

        params_config = strategy_config.get("params", {})
        if not isinstance(params_config, dict):
            return set()

        optimized_params = set()
        for param_name, param_config in params_config.items():
            if isinstance(param_config, dict) and "optimization" in param_config:
                optimized_params.add(param_name)

        return optimized_params

    def populate_strategy_params(self, config_data: Any, scenario_config: Dict[str, Any]) -> None:
        """Populate strategy_params with defaults from modern optimization config."""
        if not isinstance(config_data, dict):
            return

        strategy_config = config_data.get("strategy", {})
        if not isinstance(strategy_config, dict):
            return

        params_config = strategy_config.get("params", {})
        if not isinstance(params_config, dict):
            return

        if "strategy_params" not in scenario_config:
            scenario_config["strategy_params"] = {}

        for param_name, param_config in params_config.items():
            if isinstance(param_config, dict) and "optimization" in param_config:
                # Add default value to strategy_params if not present
                if param_name not in scenario_config["strategy_params"]:
                    opt_config = param_config["optimization"]

                    if "range" in opt_config:
                        # For range-based parameters, use the first value as default
                        range_values = opt_config["range"]
                        if len(range_values) == 2:
                            default_value = range_values[0]  # Use minimum as default
                            scenario_config["strategy_params"][param_name] = default_value
                    elif "choices" in opt_config:
                        # For categorical parameters, use the first choice as default
                        choices = opt_config["choices"]
                        if choices:
                            scenario_config["strategy_params"][param_name] = choices[0]


class CompositeOptimizationPopulator(OptimizationPopulator):
    """Combines legacy and modern optimization populators."""

    def __init__(self):
        self._populators = [
            LegacyOptimizationPopulator(),
            ModernOptimizationPopulator(),
        ]

    def can_handle(self, config_data: Any) -> bool:
        """Check if any populator can handle the config data."""
        return any(populator.can_handle(config_data) for populator in self._populators)

    def extract_optimized_parameters(self, config_data: Any) -> Set[str]:
        """Extract parameters using all applicable populators."""
        all_params = set()
        for populator in self._populators:
            if populator.can_handle(config_data):
                all_params.update(populator.extract_optimized_parameters(config_data))
        return all_params

    def populate_strategy_params(self, config_data: Any, scenario_config: Dict[str, Any]) -> None:
        """Populate strategy params using all applicable populators."""
        for populator in self._populators:
            if populator.can_handle(config_data):
                populator.populate_strategy_params(config_data, scenario_config)


class OptimizationPopulatorFactory:
    """Factory for creating appropriate optimization populators."""

    def __init__(self):
        self._default_populator = CompositeOptimizationPopulator()

    def get_populator(self, config_data: Any) -> OptimizationPopulator:
        """Get the appropriate optimization populator for the given config data."""
        return self._default_populator

    def extract_optimized_parameters(self, config_data: Any) -> Set[str]:
        """Extract optimized parameters using the appropriate populator."""
        populator = self.get_populator(config_data)
        return populator.extract_optimized_parameters(config_data)

    def populate_strategy_params(self, config_data: Any, scenario_config: Dict[str, Any]) -> None:
        """Populate strategy params using the appropriate populator."""
        populator = self.get_populator(config_data)
        populator.populate_strategy_params(config_data, scenario_config)
