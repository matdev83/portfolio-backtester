"""
Take Profit Provider Interface

Defines interfaces for managing take profit handlers in strategies.
This interface provides clear abstractions for take profit configuration
and instantiation, working alongside the stop loss system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, Mapping

if TYPE_CHECKING:
    from ..risk_management.take_profit_handlers import BaseTakeProfit
    from ..canonical_config import CanonicalScenarioConfig


class ITakeProfitProvider(ABC):
    """
    Interface for providing take profit handlers to strategies.

    This interface defines how strategies obtain and configure their take profit handlers,
    supporting both simple and advanced take profit implementations.
    """

    @abstractmethod
    def get_take_profit_handler(self) -> "BaseTakeProfit":
        """
        Get the take profit handler instance.

        Returns:
            BaseTakeProfit instance

        Raises:
            ValueError: If take profit handler cannot be created
        """
        pass

    @abstractmethod
    def get_take_profit_config(self) -> Dict[str, Any]:
        """
        Get the take profit configuration parameters.

        Returns:
            Dictionary of take profit configuration parameters
        """
        pass

    @abstractmethod
    def supports_take_profit(self) -> bool:
        """
        Check if this provider supports active take profit functionality.

        Returns:
            True if take profit is active, False if using NoTakeProfit
        """
        pass

    def validate_take_profit_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate take profit configuration parameters.

        Args:
            config: Take profit configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return (True, "")

    def get_required_parameters(self) -> Dict[str, type]:
        """
        Get required parameters for this take profit type.

        Returns:
            Dictionary mapping parameter names to their expected types
        """
        return {}


class ConfigBasedTakeProfitProvider(ITakeProfitProvider):
    """
    Take profit provider that uses strategy configuration.

    This provider provides flexible take profit instantiation based on configuration,
    working alongside the existing stop loss system.
    """

    def __init__(self, strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig]):
        """
        Initialize provider with strategy configuration or canonical config.

        Args:
            strategy_config: Strategy configuration dictionary or canonical config
        """
        from ..canonical_config import CanonicalScenarioConfig

        self.strategy_config = strategy_config
        self.take_profit_config: Dict[str, Any] = {}

        if isinstance(strategy_config, CanonicalScenarioConfig):
            tp_cfg = strategy_config.strategy_params.get("take_profit_config")
            if tp_cfg is not None:
                self.take_profit_config = dict(tp_cfg)
        else:
            tp_raw = strategy_config.get("take_profit_config")
            # Legacy support: look inside strategy_params
            if tp_raw is None and isinstance(strategy_config, Mapping):
                sp = strategy_config.get("strategy_params", {})
                if isinstance(sp, Mapping):
                    tp_raw = sp.get("take_profit_config")

            if tp_raw is not None:
                self.take_profit_config = dict(tp_raw)

    def get_take_profit_handler(self) -> "BaseTakeProfit":
        """Get take profit handler instance using configuration."""
        from ..risk_management.take_profit_handlers import (
            NoTakeProfit,
            AtrBasedTakeProfit,
        )
        from ..canonical_config import CanonicalScenarioConfig

        tp_type_name = self.take_profit_config.get("type", "NoTakeProfit")

        handler_class: type[BaseTakeProfit] = NoTakeProfit
        if tp_type_name == "AtrBasedTakeProfit":
            handler_class = AtrBasedTakeProfit
        elif tp_type_name in [
            "PercentageTakeProfit",
            "TrailingTakeProfit",
            "TimeBasedTakeProfit",
        ]:
            raise NotImplementedError(f"Take profit type '{tp_type_name}' not yet implemented")
        elif tp_type_name != "NoTakeProfit":
            # Try to get from registry if we implement one later
            raise ValueError(f"Unknown take profit type: {tp_type_name}")

        # Ensure we pass a dict to the handler
        if isinstance(self.strategy_config, CanonicalScenarioConfig):
            strategy_config_dict = self.strategy_config.to_dict()
        else:
            strategy_config_dict = dict(self.strategy_config)

        return handler_class(strategy_config_dict, self.take_profit_config)

    def get_take_profit_config(self) -> Dict[str, Any]:
        """Get take profit configuration parameters."""
        return dict(self.take_profit_config)

    def supports_take_profit(self) -> bool:
        """Check if take profit is active."""
        tp_type_name = self.take_profit_config.get("type", "NoTakeProfit")
        return bool(tp_type_name != "NoTakeProfit")

    def validate_take_profit_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate take profit configuration."""
        tp_type = config.get("type", "NoTakeProfit")

        valid_types = [
            "NoTakeProfit",
            "AtrBasedTakeProfit",
            # Note: Additional types will be added when implemented:
            # "PercentageTakeProfit", "TrailingTakeProfit", "TimeBasedTakeProfit"
        ]

        if tp_type not in valid_types:
            return (
                False,
                f"Invalid take profit type '{tp_type}'. Valid types: {valid_types}",
            )

        # Type-specific validation
        if tp_type == "AtrBasedTakeProfit":
            atr_length = config.get("atr_length", 14)
            atr_multiple = config.get("atr_multiple", 2.0)

            if not isinstance(atr_length, int) or atr_length < 1:
                return (
                    False,
                    "AtrBasedTakeProfit requires positive integer 'atr_length'",
                )

            if not isinstance(atr_multiple, (int, float)) or atr_multiple <= 0:
                return (False, "AtrBasedTakeProfit requires positive 'atr_multiple'")

        elif tp_type == "PercentageTakeProfit":
            profit_percentage = config.get("profit_percentage")
            if profit_percentage is None:
                return (
                    False,
                    "PercentageTakeProfit requires 'profit_percentage' parameter",
                )

            if not isinstance(profit_percentage, (int, float)) or profit_percentage <= 0:
                return (
                    False,
                    "PercentageTakeProfit 'profit_percentage' must be positive",
                )

        elif tp_type == "TrailingTakeProfit":
            trail_percentage = config.get("trail_percentage")
            if trail_percentage is None:
                return (
                    False,
                    "TrailingTakeProfit requires 'trail_percentage' parameter",
                )

            if not isinstance(trail_percentage, (int, float)) or not (0 < trail_percentage < 1):
                return (
                    False,
                    "TrailingTakeProfit 'trail_percentage' must be between 0 and 1",
                )

        elif tp_type == "TimeBasedTakeProfit":
            max_holding_days = config.get("max_holding_days")
            if max_holding_days is None:
                return (
                    False,
                    "TimeBasedTakeProfit requires 'max_holding_days' parameter",
                )

            if not isinstance(max_holding_days, int) or max_holding_days < 1:
                return (
                    False,
                    "TimeBasedTakeProfit 'max_holding_days' must be positive integer",
                )

        return (True, "")

    def get_required_parameters(self) -> Dict[str, type]:
        """Get required parameters for the configured take profit type."""
        tp_type = self.take_profit_config.get("type", "NoTakeProfit")

        if tp_type == "AtrBasedTakeProfit":
            return {
                "atr_length": int,
                "atr_multiple": float,
            }
        elif tp_type == "PercentageTakeProfit":
            return {
                "profit_percentage": float,
            }
        elif tp_type == "TrailingTakeProfit":
            return {
                "trail_percentage": float,
            }
        elif tp_type == "TimeBasedTakeProfit":
            return {
                "max_holding_days": int,
            }

        return {}


class FixedTakeProfitProvider(ITakeProfitProvider):
    """
    Simple take profit provider for a fixed take profit type.

    Useful for strategies that always use a specific take profit handler.
    """

    def __init__(self, take_profit_type: str, take_profit_params: Optional[Dict[str, Any]] = None):
        """
        Initialize with fixed take profit configuration.

        Args:
            take_profit_type: Type of take profit handler to use
            take_profit_params: Optional parameters for the take profit handler
        """
        self.take_profit_type = take_profit_type
        self.take_profit_params = take_profit_params or {}

    def get_take_profit_handler(self) -> "BaseTakeProfit":
        """Get the configured take profit handler instance."""
        from ..risk_management.take_profit_handlers import (
            NoTakeProfit,
            AtrBasedTakeProfit,
        )

        handler_class: type[BaseTakeProfit] = NoTakeProfit
        if self.take_profit_type == "AtrBasedTakeProfit":
            handler_class = AtrBasedTakeProfit
        elif self.take_profit_type in [
            "PercentageTakeProfit",
            "TrailingTakeProfit",
            "TimeBasedTakeProfit",
        ]:
            raise NotImplementedError(
                f"Take profit type '{self.take_profit_type}' not yet implemented"
            )
        elif self.take_profit_type != "NoTakeProfit":
            raise ValueError(f"Unknown take profit type: {self.take_profit_type}")

        # Create minimal strategy config for the handler
        strategy_config = {"take_profit_config": self.get_take_profit_config()}
        take_profit_config = {"type": self.take_profit_type, **self.take_profit_params}

        return handler_class(strategy_config, take_profit_config)

    def get_take_profit_config(self) -> Dict[str, Any]:
        """Get take profit configuration."""
        config = {"type": self.take_profit_type}
        config.update(self.take_profit_params)
        return config

    def supports_take_profit(self) -> bool:
        """Check if the fixed take profit is active."""
        return self.take_profit_type != "NoTakeProfit"


class TakeProfitProviderFactory:
    """
    Factory for creating take profit providers.

    Simplifies take profit provider creation and promotes loose coupling.
    """

    @staticmethod
    def create_provider(
        strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig],
        provider_type: str = "config",
    ) -> ITakeProfitProvider:
        """
        Create a take profit provider instance.

        Args:
            strategy_config: Strategy configuration dictionary or canonical config
            provider_type: Type of provider to create ("config", "fixed")

        Returns:
            ITakeProfitProvider instance

        Raises:
            ValueError: If provider_type is unknown
        """
        from ..canonical_config import CanonicalScenarioConfig

        if provider_type == "config":
            return ConfigBasedTakeProfitProvider(strategy_config)
        elif provider_type == "fixed":
            if isinstance(strategy_config, CanonicalScenarioConfig):
                take_profit_config = dict(
                    strategy_config.strategy_params.get("take_profit_config", {})
                )
            else:
                take_profit_config = strategy_config.get("take_profit_config", {})

            take_profit_type = take_profit_config.get("type", "NoTakeProfit")
            return FixedTakeProfitProvider(take_profit_type, take_profit_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def create_config_provider(
        strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig],
    ) -> ConfigBasedTakeProfitProvider:
        """Create a configuration-based take profit provider."""
        return ConfigBasedTakeProfitProvider(strategy_config)

    @staticmethod
    def create_fixed_provider(
        take_profit_type: str, take_profit_params: Optional[Dict[str, Any]] = None
    ) -> FixedTakeProfitProvider:
        """Create a fixed take profit provider."""
        return FixedTakeProfitProvider(take_profit_type, take_profit_params)

    @staticmethod
    def get_default_provider(
        strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig],
    ) -> ITakeProfitProvider:
        """
        Get a default take profit provider for backward compatibility.

        Args:
            strategy_config: Strategy configuration dictionary or canonical config

        Returns:
            ITakeProfitProvider instance with NoTakeProfit as default
        """
        from ..canonical_config import CanonicalScenarioConfig

        # Ensure default take profit config if none specified
        if isinstance(strategy_config, CanonicalScenarioConfig):
            return ConfigBasedTakeProfitProvider(strategy_config)

        if "take_profit_config" not in strategy_config:
            new_config = dict(strategy_config)
            new_config["take_profit_config"] = {"type": "NoTakeProfit"}
            return ConfigBasedTakeProfitProvider(new_config)

        return ConfigBasedTakeProfitProvider(strategy_config)


# Additional take profit implementations will be added in future updates:
# - PercentageTakeProfit: Fixed percentage profit from entry price
# - TrailingTakeProfit: Trailing take profit following low-water mark
# - TimeBasedTakeProfit: Take profit after minimum holding period
