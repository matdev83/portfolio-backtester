"""
Position Sizer Provider Interface

Defines interfaces for managing position sizing in strategies.
This interface enhances the existing position sizer system by providing
clear abstractions for position sizer configuration and instantiation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, Mapping

if TYPE_CHECKING:
    from ..portfolio.base_sizer import BasePositionSizer
    from ..canonical_config import CanonicalScenarioConfig


class IPositionSizerProvider(ABC):
    """
    Interface for providing position sizers to strategies.

    This interface defines how strategies obtain and configure their position sizers,
    supporting both static and dynamic position sizing approaches.
    """

    @abstractmethod
    def get_position_sizer(self) -> "BasePositionSizer":
        """
        Get the position sizer instance.

        Returns:
            BasePositionSizer instance

        Raises:
            ValueError: If position sizer cannot be created
        """
        pass

    @abstractmethod
    def get_position_sizer_config(self) -> Dict[str, Any]:
        """
        Get the position sizer configuration parameters.

        Returns:
            Dictionary of position sizer configuration parameters
        """
        pass

    @abstractmethod
    def supports_dynamic_sizing(self) -> bool:
        """
        Check if this provider supports dynamic position sizing.

        Dynamic sizing means the sizer can adjust its behavior based on
        market conditions, volatility, or other dynamic factors.

        Returns:
            True if dynamic sizing is supported, False otherwise
        """
        pass

    def validate_position_sizer_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate position sizer configuration parameters.

        Args:
            config: Position sizer configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return (True, "")

    def get_parameter_mapping(self) -> Dict[str, str]:
        """
        Get parameter mapping for legacy parameter names.

        Returns:
            Dictionary mapping old parameter names to new ones
        """
        return {}


class ConfigBasedPositionSizerProvider(IPositionSizerProvider):
    """
    Position sizer provider that uses strategy configuration.

    This provider integrates with the existing position_sizer system
    to provide flexible position sizer instantiation based on configuration.
    """

    def __init__(self, strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig]):
        """
        Initialize provider with strategy configuration or canonical config.

        Args:
            strategy_config: Strategy configuration dictionary or canonical config
        """
        from ..canonical_config import CanonicalScenarioConfig

        self.strategy_config = strategy_config
        if isinstance(strategy_config, CanonicalScenarioConfig):
            self.strategy_params = dict(strategy_config.strategy_params)
        else:
            self.strategy_params = strategy_config.get("strategy_params", {})
            # Legacy support: look inside strategy_params if not at top level
            if (
                isinstance(self.strategy_params, Mapping)
                and "position_sizer" in self.strategy_params
            ):
                # We'll handle this in get_position_sizer_config and get_position_sizer
                pass

    def get_position_sizer(self) -> "BasePositionSizer":
        """Get position sizer instance using configuration."""
        from ..portfolio.position_sizer import get_position_sizer_from_config
        from ..canonical_config import CanonicalScenarioConfig

        if isinstance(self.strategy_config, CanonicalScenarioConfig):
            # Convert to dict for compatibility with existing sizer factory
            config_dict = {"position_sizer": self.strategy_config.position_sizer or "equal_weight"}
            return get_position_sizer_from_config(config_dict)

        # Mapping to Dict conversion for compatibility
        config_dict = dict(self.strategy_config)
        # Legacy support: look inside strategy_params
        if config_dict.get("position_sizer") is None and "strategy_params" in config_dict:
            sp = config_dict["strategy_params"]
            if isinstance(sp, Mapping) and "position_sizer" in sp:
                config_dict["position_sizer"] = sp["position_sizer"]

        return get_position_sizer_from_config(config_dict)

    def get_position_sizer_config(self) -> Dict[str, Any]:
        """Get position sizer configuration parameters."""
        from ..portfolio.position_sizer import SIZER_PARAM_MAPPING
        from ..canonical_config import CanonicalScenarioConfig

        # Get base config
        if isinstance(self.strategy_config, CanonicalScenarioConfig):
            config = {"position_sizer": self.strategy_config.position_sizer or "equal_weight"}
        else:
            sizer_name = self.strategy_config.get("position_sizer")
            # Legacy support: look inside strategy_params
            if sizer_name is None and "strategy_params" in self.strategy_config:
                sp = self.strategy_config["strategy_params"]
                if isinstance(sp, Mapping):
                    sizer_name = sp.get("position_sizer")

            config = {"position_sizer": sizer_name or "equal_weight"}

        # Extract sizer-specific parameters from strategy_params
        filtered_params = {}
        for key, value in self.strategy_params.items():
            if key in SIZER_PARAM_MAPPING:
                new_key = SIZER_PARAM_MAPPING[key]
                filtered_params[new_key] = value

        config.update(filtered_params)
        return config

    def supports_dynamic_sizing(self) -> bool:
        """Check if the configured sizer supports dynamic sizing."""
        sizer_name = self.strategy_config.get("position_sizer", "equal_weight")

        # Dynamic sizers are those that use market data for sizing decisions
        dynamic_sizers = {
            "rolling_sharpe",
            "rolling_sortino",
            "rolling_beta",
            "rolling_benchmark_corr",
            "rolling_downside_volatility",
        }

        return sizer_name in dynamic_sizers

    def validate_position_sizer_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate position sizer configuration."""
        from ..portfolio.position_sizer import SIZER_REGISTRY

        sizer_name = config.get("position_sizer", "equal_weight")

        # Check if sizer exists
        if sizer_name not in SIZER_REGISTRY and sizer_name != "direct":
            valid_sizers = list(SIZER_REGISTRY.keys()) + ["direct"]
            return (
                False,
                f"Invalid position sizer '{sizer_name}'. Valid options: {valid_sizers}",
            )

        # Validate sizer-specific parameters
        if sizer_name == "rolling_downside_volatility":
            window = config.get("window")
            if window is not None and (not isinstance(window, int) or window < 1):
                return (
                    False,
                    "rolling_downside_volatility sizer requires positive integer 'window'",
                )

        elif sizer_name in [
            "rolling_sharpe",
            "rolling_sortino",
            "rolling_beta",
            "rolling_benchmark_corr",
        ]:
            window = config.get("window")
            if window is not None and (not isinstance(window, int) or window < 1):
                return (False, f"{sizer_name} sizer requires positive integer 'window'")

        return (True, "")

    def get_parameter_mapping(self) -> Dict[str, str]:
        """Get parameter mapping for legacy parameter names."""
        from ..portfolio.position_sizer import SIZER_PARAM_MAPPING

        return SIZER_PARAM_MAPPING.copy()


class FixedPositionSizerProvider(IPositionSizerProvider):
    """
    Simple position sizer provider for a fixed sizer type.

    Useful for strategies that always use a specific position sizer.
    """

    def __init__(self, sizer_name: str, sizer_params: Optional[Dict[str, Any]] = None):
        """
        Initialize with fixed sizer configuration.

        Args:
            sizer_name: Name of the position sizer to use
            sizer_params: Optional parameters for the sizer
        """
        self.sizer_name = sizer_name
        self.sizer_params = sizer_params or {}

    def get_position_sizer(self) -> "BasePositionSizer":
        """Get the configured position sizer instance."""
        from ..portfolio.position_sizer import get_position_sizer

        return get_position_sizer(self.sizer_name)

    def get_position_sizer_config(self) -> Dict[str, Any]:
        """Get position sizer configuration."""
        config = {"position_sizer": self.sizer_name}
        config.update(self.sizer_params)
        return config

    def supports_dynamic_sizing(self) -> bool:
        """Check if the fixed sizer supports dynamic sizing."""
        dynamic_sizers = {
            "rolling_sharpe",
            "rolling_sortino",
            "rolling_beta",
            "rolling_benchmark_corr",
            "rolling_downside_volatility",
        }

        return self.sizer_name in dynamic_sizers


class PositionSizerProviderFactory:
    """
    Factory for creating position sizer providers.

    Simplifies position sizer provider creation and promotes loose coupling.
    """

    @staticmethod
    def create_provider(
        strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig],
        provider_type: str = "config",
    ) -> IPositionSizerProvider:
        """
        Create a position sizer provider instance.

        Args:
            strategy_config: Strategy configuration dictionary or canonical config
            provider_type: Type of provider to create ("config", "fixed")

        Returns:
            IPositionSizerProvider instance

        Raises:
            ValueError: If provider_type is unknown
        """
        from ..canonical_config import CanonicalScenarioConfig

        if provider_type == "config":
            return ConfigBasedPositionSizerProvider(strategy_config)
        elif provider_type == "fixed":
            if isinstance(strategy_config, CanonicalScenarioConfig):
                sizer_name = strategy_config.position_sizer or "equal_weight"
                sizer_params = dict(strategy_config.strategy_params)
            else:
                sizer_name = strategy_config.get("position_sizer", "equal_weight")
                sizer_params = strategy_config.get("strategy_params", {})
            return FixedPositionSizerProvider(sizer_name, sizer_params)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def create_config_provider(
        strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig],
    ) -> ConfigBasedPositionSizerProvider:
        """Create a configuration-based position sizer provider."""
        return ConfigBasedPositionSizerProvider(strategy_config)

    @staticmethod
    def create_fixed_provider(
        sizer_name: str, sizer_params: Optional[Dict[str, Any]] = None
    ) -> FixedPositionSizerProvider:
        """Create a fixed position sizer provider."""
        return FixedPositionSizerProvider(sizer_name, sizer_params)

    @staticmethod
    def get_default_provider(
        strategy_config: Union[Mapping[str, Any], CanonicalScenarioConfig],
    ) -> IPositionSizerProvider:
        """
        Get a default position sizer provider for backward compatibility.

        Args:
            strategy_config: Strategy configuration dictionary or canonical config

        Returns:
            IPositionSizerProvider instance with equal weight as default
        """
        from ..canonical_config import CanonicalScenarioConfig

        # Ensure default position sizer if none specified
        if isinstance(strategy_config, CanonicalScenarioConfig):
            # Canonical config always has position_sizer (can be None)
            return ConfigBasedPositionSizerProvider(strategy_config)

        if "position_sizer" not in strategy_config:
            # We need to modify it, so we cast to dict if it's not
            new_config = dict(strategy_config)
            new_config["position_sizer"] = "equal_weight"
            return ConfigBasedPositionSizerProvider(new_config)

        return ConfigBasedPositionSizerProvider(strategy_config)
