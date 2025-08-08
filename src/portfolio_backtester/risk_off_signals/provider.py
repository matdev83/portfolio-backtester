"""
Risk-off Signal Provider Interface

Defines provider interfaces for managing risk-off signal generators in strategies.
This module follows the Provider pattern used throughout the framework, implementing
SOLID principles and Dependency Inversion Principle (DIP).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import IRiskOffSignalGenerator


class IRiskOffSignalProvider(ABC):
    """
    Interface for providing risk-off signal generators to strategies.

    This interface follows the same pattern as other provider interfaces in the framework,
    supporting both configuration-based and fixed signal generator instantiation.

    Design Principles:
    - Single Responsibility: Only responsible for providing signal generators
    - Dependency Inversion: Strategies depend on this abstraction, not concrete generators
    - Open/Closed: Open for extension (new providers), closed for modification
    """

    @abstractmethod
    def get_risk_off_signal_generator(self) -> "IRiskOffSignalGenerator":
        """
        Get the risk-off signal generator instance.

        Returns:
            IRiskOffSignalGenerator instance

        Raises:
            ValueError: If signal generator cannot be created
        """
        pass

    @abstractmethod
    def get_risk_off_config(self) -> Dict[str, Any]:
        """
        Get the risk-off signal configuration parameters.

        Returns:
            Dictionary of risk-off signal configuration parameters
        """
        pass

    @abstractmethod
    def supports_risk_off_signals(self) -> bool:
        """
        Check if this provider supports active risk-off signal generation.

        Returns:
            True if risk-off signals are active, False if using NoRiskOffSignalGenerator
        """
        pass


class ConfigBasedRiskOffSignalProvider(IRiskOffSignalProvider):
    """
    Risk-off signal provider that uses strategy configuration.

    This provider creates signal generators based on strategy configuration,
    following the same pattern as other config-based providers in the framework.

    Configuration Structure:
    {
        "risk_off_signal_config": {
            "type": "NoRiskOffSignalGenerator",  # or "DummyRiskOffSignalGenerator", etc.
            # ... type-specific parameters
        }
    }
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        """
        Initialize provider with strategy configuration.

        Args:
            strategy_config: Strategy configuration dictionary
        """
        self.strategy_config = strategy_config
        self.risk_off_config = strategy_config.get("risk_off_signal_config", {})

    def get_risk_off_signal_generator(self) -> "IRiskOffSignalGenerator":
        """Get risk-off signal generator instance using configuration."""
        from .implementations import (
            NoRiskOffSignalGenerator,
            DummyRiskOffSignalGenerator,
        )

        generator_type_name = self.risk_off_config.get("type", "NoRiskOffSignalGenerator")

        # Registry of available generator types
        generator_registry = {
            "NoRiskOffSignalGenerator": NoRiskOffSignalGenerator,
            "DummyRiskOffSignalGenerator": DummyRiskOffSignalGenerator,
            # Future implementations can be added here:
            # "VixBasedRiskOffSignalGenerator": VixBasedRiskOffSignalGenerator,
            # "TechnicalRiskOffSignalGenerator": TechnicalRiskOffSignalGenerator,
        }

        generator_class = generator_registry.get(generator_type_name)
        if generator_class is None:
            available_types = list(generator_registry.keys())
            raise ValueError(
                f"Unknown risk-off signal generator type: {generator_type_name}. "
                f"Available types: {available_types}"
            )

        # Validate configuration before creating generator
        temp_generator = generator_class({})
        is_valid, error_msg = temp_generator.validate_configuration(self.risk_off_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for {generator_type_name}: {error_msg}")

        return generator_class(self.risk_off_config)

    def get_risk_off_config(self) -> Dict[str, Any]:
        """Get risk-off signal configuration parameters."""
        return dict(self.risk_off_config)

    def supports_risk_off_signals(self) -> bool:
        """Check if risk-off signals are active."""
        generator_type = self.risk_off_config.get("type", "NoRiskOffSignalGenerator")
        return bool(generator_type != "NoRiskOffSignalGenerator")


class FixedRiskOffSignalProvider(IRiskOffSignalProvider):
    """
    Simple risk-off signal provider for a fixed generator type.

    Useful for strategies that always use a specific risk-off signal generator.
    Follows the same pattern as other fixed providers in the framework.
    """

    def __init__(self, generator_type: str, generator_params: Optional[Dict[str, Any]] = None):
        """
        Initialize with fixed risk-off signal generator configuration.

        Args:
            generator_type: Type of risk-off signal generator to use
            generator_params: Optional parameters for the generator
        """
        self.generator_type = generator_type
        self.generator_params = generator_params or {}

    def get_risk_off_signal_generator(self) -> "IRiskOffSignalGenerator":
        """Get the configured risk-off signal generator instance."""
        from .implementations import (
            NoRiskOffSignalGenerator,
            DummyRiskOffSignalGenerator,
        )

        # Registry of available generator types
        generator_registry = {
            "NoRiskOffSignalGenerator": NoRiskOffSignalGenerator,
            "DummyRiskOffSignalGenerator": DummyRiskOffSignalGenerator,
        }

        generator_class = generator_registry.get(self.generator_type)
        if generator_class is None:
            available_types = list(generator_registry.keys())
            raise ValueError(
                f"Unknown risk-off signal generator type: {self.generator_type}. "
                f"Available types: {available_types}"
            )

        return generator_class(self.generator_params)

    def get_risk_off_config(self) -> Dict[str, Any]:
        """Get risk-off signal configuration."""
        config = {"type": self.generator_type}
        config.update(self.generator_params)
        return config

    def supports_risk_off_signals(self) -> bool:
        """Check if the fixed generator supports active risk-off signals."""
        return self.generator_type != "NoRiskOffSignalGenerator"


class RiskOffSignalProviderFactory:
    """
    Factory for creating risk-off signal providers.

    Follows the factory pattern used throughout the framework,
    simplifying provider creation and promoting loose coupling.
    """

    @staticmethod
    def create_provider(
        strategy_config: Dict[str, Any], provider_type: str = "config"
    ) -> IRiskOffSignalProvider:
        """
        Create a risk-off signal provider instance.

        Args:
            strategy_config: Strategy configuration dictionary
            provider_type: Type of provider to create ("config", "fixed")

        Returns:
            IRiskOffSignalProvider instance

        Raises:
            ValueError: If provider_type is unknown
        """
        if provider_type == "config":
            return ConfigBasedRiskOffSignalProvider(strategy_config)
        elif provider_type == "fixed":
            risk_off_config = strategy_config.get("risk_off_signal_config", {})
            generator_type = risk_off_config.get("type", "NoRiskOffSignalGenerator")
            return FixedRiskOffSignalProvider(generator_type, risk_off_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def create_config_provider(strategy_config: Dict[str, Any]) -> ConfigBasedRiskOffSignalProvider:
        """Create a configuration-based risk-off signal provider."""
        return ConfigBasedRiskOffSignalProvider(strategy_config)

    @staticmethod
    def create_fixed_provider(
        generator_type: str, generator_params: Optional[Dict[str, Any]] = None
    ) -> FixedRiskOffSignalProvider:
        """Create a fixed risk-off signal provider."""
        return FixedRiskOffSignalProvider(generator_type, generator_params)

    @staticmethod
    def get_default_provider(strategy_config: Dict[str, Any]) -> IRiskOffSignalProvider:
        """
        Get a default risk-off signal provider for backward compatibility.

        Args:
            strategy_config: Strategy configuration dictionary

        Returns:
            IRiskOffSignalProvider instance with NoRiskOffSignalGenerator as default
        """
        # Ensure default risk-off signal config if none specified
        if "risk_off_signal_config" not in strategy_config:
            strategy_config = strategy_config.copy()
            strategy_config["risk_off_signal_config"] = {"type": "NoRiskOffSignalGenerator"}

        return ConfigBasedRiskOffSignalProvider(strategy_config)
