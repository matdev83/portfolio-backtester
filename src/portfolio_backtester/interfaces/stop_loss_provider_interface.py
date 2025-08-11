"""
Stop Loss Provider Interface

Defines interfaces for managing stop loss handlers in strategies.
This interface enhances the existing stop loss system by providing
clear abstractions for stop loss configuration and instantiation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..risk_management.stop_loss_handlers import BaseStopLoss


class IStopLossProvider(ABC):
    """
    Interface for providing stop loss handlers to strategies.

    This interface defines how strategies obtain and configure their stop loss handlers,
    supporting both simple and advanced stop loss implementations.
    """

    @abstractmethod
    def get_stop_loss_handler(self) -> "BaseStopLoss":
        """
        Get the stop loss handler instance.

        Returns:
            BaseStopLoss instance

        Raises:
            ValueError: If stop loss handler cannot be created
        """
        pass

    @abstractmethod
    def get_stop_loss_config(self) -> Dict[str, Any]:
        """
        Get the stop loss configuration parameters.

        Returns:
            Dictionary of stop loss configuration parameters
        """
        pass

    @abstractmethod
    def supports_stop_loss(self) -> bool:
        """
        Check if this provider supports active stop loss functionality.

        Returns:
            True if stop loss is active, False if using NoStopLoss
        """
        pass

    def validate_stop_loss_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate stop loss configuration parameters.

        Args:
            config: Stop loss configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return (True, "")

    def get_required_parameters(self) -> Dict[str, type]:
        """
        Get required parameters for this stop loss type.

        Returns:
            Dictionary mapping parameter names to their expected types
        """
        return {}


class ConfigBasedStopLossProvider(IStopLossProvider):
    """
    Stop loss provider that uses strategy configuration.

    This provider integrates with the existing stop loss system
    to provide flexible stop loss instantiation based on configuration.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        """
        Initialize provider with strategy configuration.

        Args:
            strategy_config: Strategy configuration dictionary
        """
        self.strategy_config = strategy_config
        self.stop_loss_config = strategy_config.get("stop_loss_config", {})

    def get_stop_loss_handler(self) -> "BaseStopLoss":
        """Get stop loss handler instance using configuration."""
        from ..risk_management.stop_loss_handlers import BaseStopLoss, NoStopLoss, AtrBasedStopLoss

        sl_type_name = self.stop_loss_config.get("type", "NoStopLoss")

        handler_class: type[BaseStopLoss] = NoStopLoss
        if sl_type_name == "AtrBasedStopLoss":
            handler_class = AtrBasedStopLoss
        elif sl_type_name in ["PercentageStopLoss", "TrailingStopLoss", "TimeBasedStopLoss"]:
            raise NotImplementedError(f"Stop loss type '{sl_type_name}' not yet implemented")
        elif sl_type_name != "NoStopLoss":
            # Try to get from registry if we implement one later
            raise ValueError(f"Unknown stop loss type: {sl_type_name}")

        return handler_class(self.strategy_config, self.stop_loss_config)

    def get_stop_loss_config(self) -> Dict[str, Any]:
        """Get stop loss configuration parameters."""
        return dict(self.stop_loss_config)

    def supports_stop_loss(self) -> bool:
        """Check if stop loss is active."""
        sl_type_name = self.stop_loss_config.get("type", "NoStopLoss")
        return bool(sl_type_name != "NoStopLoss")

    def validate_stop_loss_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate stop loss configuration."""
        sl_type = config.get("type", "NoStopLoss")

        valid_types = [
            "NoStopLoss",
            "AtrBasedStopLoss",
            # Note: Additional types will be added when implemented:
            # "PercentageStopLoss", "TrailingStopLoss", "TimeBasedStopLoss"
        ]

        if sl_type not in valid_types:
            return (False, f"Invalid stop loss type '{sl_type}'. Valid types: {valid_types}")

        # Type-specific validation
        if sl_type == "AtrBasedStopLoss":
            atr_length = config.get("atr_length", 14)
            atr_multiple = config.get("atr_multiple", 2.5)

            if not isinstance(atr_length, int) or atr_length < 1:
                return (False, "AtrBasedStopLoss requires positive integer 'atr_length'")

            if not isinstance(atr_multiple, (int, float)) or atr_multiple <= 0:
                return (False, "AtrBasedStopLoss requires positive 'atr_multiple'")

        elif sl_type == "PercentageStopLoss":
            stop_percentage = config.get("stop_percentage")
            if stop_percentage is None:
                return (False, "PercentageStopLoss requires 'stop_percentage' parameter")

            if not isinstance(stop_percentage, (int, float)) or not (0 < stop_percentage < 1):
                return (False, "PercentageStopLoss 'stop_percentage' must be between 0 and 1")

        elif sl_type == "TrailingStopLoss":
            trail_percentage = config.get("trail_percentage")
            if trail_percentage is None:
                return (False, "TrailingStopLoss requires 'trail_percentage' parameter")

            if not isinstance(trail_percentage, (int, float)) or not (0 < trail_percentage < 1):
                return (False, "TrailingStopLoss 'trail_percentage' must be between 0 and 1")

        elif sl_type == "TimeBasedStopLoss":
            max_holding_days = config.get("max_holding_days")
            if max_holding_days is None:
                return (False, "TimeBasedStopLoss requires 'max_holding_days' parameter")

            if not isinstance(max_holding_days, int) or max_holding_days < 1:
                return (False, "TimeBasedStopLoss 'max_holding_days' must be positive integer")

        return (True, "")

    def get_required_parameters(self) -> Dict[str, type]:
        """Get required parameters for the configured stop loss type."""
        sl_type = self.stop_loss_config.get("type", "NoStopLoss")

        if sl_type == "AtrBasedStopLoss":
            return {
                "atr_length": int,
                "atr_multiple": float,
            }
        elif sl_type == "PercentageStopLoss":
            return {
                "stop_percentage": float,
            }
        elif sl_type == "TrailingStopLoss":
            return {
                "trail_percentage": float,
            }
        elif sl_type == "TimeBasedStopLoss":
            return {
                "max_holding_days": int,
            }

        return {}


class FixedStopLossProvider(IStopLossProvider):
    """
    Simple stop loss provider for a fixed stop loss type.

    Useful for strategies that always use a specific stop loss handler.
    """

    def __init__(self, stop_loss_type: str, stop_loss_params: Optional[Dict[str, Any]] = None):
        """
        Initialize with fixed stop loss configuration.

        Args:
            stop_loss_type: Type of stop loss handler to use
            stop_loss_params: Optional parameters for the stop loss handler
        """
        self.stop_loss_type = stop_loss_type
        self.stop_loss_params = stop_loss_params or {}

    def get_stop_loss_handler(self) -> "BaseStopLoss":
        """Get the configured stop loss handler instance."""
        from ..risk_management.stop_loss_handlers import BaseStopLoss, NoStopLoss, AtrBasedStopLoss

        handler_class: type[BaseStopLoss] = NoStopLoss
        if self.stop_loss_type == "AtrBasedStopLoss":
            handler_class = AtrBasedStopLoss
        elif self.stop_loss_type in ["PercentageStopLoss", "TrailingStopLoss", "TimeBasedStopLoss"]:
            raise NotImplementedError(f"Stop loss type '{self.stop_loss_type}' not yet implemented")
        elif self.stop_loss_type != "NoStopLoss":
            raise ValueError(f"Unknown stop loss type: {self.stop_loss_type}")

        # Create minimal strategy config for the handler
        strategy_config = {"stop_loss_config": self.get_stop_loss_config()}
        stop_loss_config = {"type": self.stop_loss_type, **self.stop_loss_params}

        return handler_class(strategy_config, stop_loss_config)

    def get_stop_loss_config(self) -> Dict[str, Any]:
        """Get stop loss configuration."""
        config = {"type": self.stop_loss_type}
        config.update(self.stop_loss_params)
        return config

    def supports_stop_loss(self) -> bool:
        """Check if the fixed stop loss is active."""
        return self.stop_loss_type != "NoStopLoss"


class StopLossProviderFactory:
    """
    Factory for creating stop loss providers.

    Simplifies stop loss provider creation and promotes loose coupling.
    """

    @staticmethod
    def create_provider(
        strategy_config: Dict[str, Any], provider_type: str = "config"
    ) -> IStopLossProvider:
        """
        Create a stop loss provider instance.

        Args:
            strategy_config: Strategy configuration dictionary
            provider_type: Type of provider to create ("config", "fixed")

        Returns:
            IStopLossProvider instance

        Raises:
            ValueError: If provider_type is unknown
        """
        if provider_type == "config":
            return ConfigBasedStopLossProvider(strategy_config)
        elif provider_type == "fixed":
            stop_loss_config = strategy_config.get("stop_loss_config", {})
            stop_loss_type = stop_loss_config.get("type", "NoStopLoss")
            return FixedStopLossProvider(stop_loss_type, stop_loss_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def create_config_provider(strategy_config: Dict[str, Any]) -> ConfigBasedStopLossProvider:
        """Create a configuration-based stop loss provider."""
        return ConfigBasedStopLossProvider(strategy_config)

    @staticmethod
    def create_fixed_provider(
        stop_loss_type: str, stop_loss_params: Optional[Dict[str, Any]] = None
    ) -> FixedStopLossProvider:
        """Create a fixed stop loss provider."""
        return FixedStopLossProvider(stop_loss_type, stop_loss_params)

    @staticmethod
    def get_default_provider(strategy_config: Dict[str, Any]) -> IStopLossProvider:
        """
        Get a default stop loss provider for backward compatibility.

        Args:
            strategy_config: Strategy configuration dictionary

        Returns:
            IStopLossProvider instance with NoStopLoss as default
        """
        # Ensure default stop loss config if none specified
        if "stop_loss_config" not in strategy_config:
            strategy_config = strategy_config.copy()
            strategy_config["stop_loss_config"] = {"type": "NoStopLoss"}

        return ConfigBasedStopLossProvider(strategy_config)


# Additional stop loss implementations will be added in future updates:
# - PercentageStopLoss: Fixed percentage from entry price
# - TrailingStopLoss: Trailing stop following high-water mark
# - TimeBasedStopLoss: Exit after maximum holding period
