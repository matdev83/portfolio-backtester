"""
Universe Provider Interface

Defines interfaces for managing strategy universes in a modular and testable way.
This interface enhances the existing universe management system by providing
clear abstractions for different universe resolution strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import warnings

# Suppress deprecation warning for internal provider usage
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"DEPRECATED: resolve_universe_config is deprecated",
)


class IUniverseProvider(ABC):
    """
    Interface for providing universe data to strategies.

    This interface defines how strategies obtain their universe of tradeable symbols,
    supporting both static and dynamic universe resolution.
    """

    @abstractmethod
    def get_universe_symbols(self, global_config: Dict[str, Any]) -> List[str]:
        """
        Get the universe symbols for static universe resolution.

        Args:
            global_config: Global configuration dictionary

        Returns:
            List of ticker symbols

        Raises:
            ValueError: If universe cannot be resolved or is empty
        """
        pass

    @abstractmethod
    def get_dynamic_universe_symbols(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[str]:
        """
        Get the universe symbols with date context for dynamic resolution.

        Args:
            global_config: Global configuration dictionary
            current_date: Current date for universe resolution

        Returns:
            List of ticker symbols

        Raises:
            ValueError: If universe cannot be resolved or is empty
        """
        pass

    @abstractmethod
    def supports_dynamic_universe(self) -> bool:
        """
        Check if this provider supports dynamic universe resolution.

        Returns:
            True if dynamic universe is supported, False otherwise
        """
        pass

    def get_non_universe_data_requirements(self) -> List[str]:
        """
        Get additional tickers required for calculations but not for trading.

        Returns:
            List of ticker symbols required for strategy calculations
        """
        return []

    def validate_universe_config(self, universe_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate universe configuration parameters.

        Args:
            universe_config: Universe configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return (True, "")


class IUniverseWeightProvider(ABC):
    """
    Interface for providing universe with weights.

    Extends basic universe provision to include weights for each symbol.
    This maintains backward compatibility with existing BaseStrategy.get_universe
    which returns List[Tuple[str, float]].
    """

    @abstractmethod
    def get_universe_with_weights(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get the universe of assets with weights.

        Args:
            global_config: Global configuration dictionary

        Returns:
            List of (ticker, weight) tuples
        """
        pass

    @abstractmethod
    def get_universe_with_weights_and_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[Tuple[str, float]]:
        """
        Get the universe of assets with weights using date context.

        Args:
            global_config: Global configuration dictionary
            current_date: Current date for universe resolution

        Returns:
            List of (ticker, weight) tuples
        """
        pass


class ConfigBasedUniverseProvider(IUniverseProvider, IUniverseWeightProvider):
    """
    Universe provider that uses strategy configuration and universe resolver.

    This provider integrates with the existing universe_resolver system
    to provide flexible universe resolution based on strategy configuration.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        """
        Initialize provider with strategy configuration.

        Args:
            strategy_config: Strategy configuration dictionary
        """
        self.strategy_config = strategy_config
        self.universe_config = strategy_config.get("universe_config")

    def get_universe_symbols(self, global_config: Dict[str, Any]) -> List[str]:
        """Get universe symbols using configuration-based resolution."""
        from ..universe_resolver import resolve_universe_config

        if self.universe_config:
            # Let errors propagate so BaseStrategy can log and handle fallback
            return list(resolve_universe_config(self.universe_config))

        # Use global config universe
        default_universe = global_config.get("universe", [])
        if not isinstance(default_universe, list) or not default_universe:
            raise ValueError("No universe configuration found")
        return list(default_universe)  # Ensure it's List[str]

    def get_dynamic_universe_symbols(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[str]:
        """Get universe symbols with date context."""
        from ..universe_resolver import resolve_universe_config

        if self.universe_config:
            # Let errors propagate so BaseStrategy can log and handle fallback
            return list(resolve_universe_config(self.universe_config, current_date))

        # Use global config universe
        default_universe = global_config.get("universe", [])
        if not isinstance(default_universe, list) or not default_universe:
            raise ValueError("No universe configuration found")
        return list(default_universe)  # Ensure it's List[str]

    def supports_dynamic_universe(self) -> bool:
        """Check if universe config supports dynamic resolution."""
        if not self.universe_config:
            return False

        # Dynamic support is available for method-based universes
        universe_type = self.universe_config.get("type")
        return bool(universe_type == "method")

    def get_universe_with_weights(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get universe with equal weights (1.0 for each symbol)."""
        symbols = self.get_universe_symbols(global_config)
        return [(symbol, 1.0) for symbol in symbols]

    def get_universe_with_weights_and_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[Tuple[str, float]]:
        """Get universe with equal weights using date context."""
        symbols = self.get_dynamic_universe_symbols(global_config, current_date)
        return [(symbol, 1.0) for symbol in symbols]

    def validate_universe_config(self, universe_config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate universe configuration."""
        if not universe_config:
            return (False, "Universe configuration is empty")

        universe_type = universe_config.get("type")
        if not universe_type:
            return (False, "Universe configuration must specify 'type'")

        valid_types = ["method", "fixed", "named", "single_symbol"]
        if universe_type not in valid_types:
            return (False, f"Invalid universe type '{universe_type}'. Valid types: {valid_types}")

        # Type-specific validation
        if universe_type == "fixed":
            tickers = universe_config.get("tickers", [])
            if not isinstance(tickers, list) or not tickers:
                return (False, "Fixed universe type requires non-empty 'tickers' list")

        elif universe_type == "named":
            universe_name = universe_config.get("universe_name")
            universe_names = universe_config.get("universe_names")
            if not universe_name and not universe_names:
                return (
                    False,
                    "Named universe type requires 'universe_name' or 'universe_names'",
                )

        elif universe_type == "single_symbol":
            ticker = universe_config.get("params", {}).get("ticker") or universe_config.get(
                "ticker"
            )
            if not ticker:
                return (False, "Single symbol universe type requires 'ticker' parameter")

        return (True, "")


class FixedListUniverseProvider(IUniverseProvider, IUniverseWeightProvider):
    """
    Simple universe provider for fixed list of symbols.

    Useful for strategies that trade a specific set of symbols.
    """

    def __init__(self, symbols: List[str], weights: Optional[List[float]] = None):
        """
        Initialize with fixed list of symbols.

        Args:
            symbols: List of ticker symbols
            weights: Optional list of weights (defaults to equal weights)
        """
        if not symbols:
            raise ValueError("Symbol list cannot be empty")

        self.symbols = symbols
        if weights:
            if len(weights) != len(symbols):
                raise ValueError("Weights list must match symbols list length")
            self.weights = weights
        else:
            self.weights = [1.0] * len(symbols)

    def get_universe_symbols(self, global_config: Dict[str, Any]) -> List[str]:
        """Return the fixed list of symbols."""
        return self.symbols.copy()

    def get_dynamic_universe_symbols(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[str]:
        """Return the fixed list of symbols (ignores date)."""
        return self.symbols.copy()

    def supports_dynamic_universe(self) -> bool:
        """Fixed universe does not support dynamic resolution."""
        return False

    def get_universe_with_weights(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get universe with configured weights."""
        return list(zip(self.symbols, self.weights))

    def get_universe_with_weights_and_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[Tuple[str, float]]:
        """Get universe with configured weights (ignores date)."""
        return list(zip(self.symbols, self.weights))


class UniverseProviderFactory:
    """
    Factory for creating universe providers.

    Simplifies universe provider creation and promotes loose coupling.
    """

    @staticmethod
    def create_provider(
        strategy_config: Dict[str, Any], provider_type: str = "config"
    ) -> IUniverseProvider:
        """
        Create a universe provider instance.

        Args:
            strategy_config: Strategy configuration dictionary
            provider_type: Type of provider to create ("config", "fixed")

        Returns:
            IUniverseProvider instance

        Raises:
            ValueError: If provider_type is unknown
        """
        if provider_type == "config":
            return ConfigBasedUniverseProvider(strategy_config)
        elif provider_type == "fixed":
            # Extract symbols from strategy config
            symbols = strategy_config.get("universe", [])
            if not symbols:
                raise ValueError("Fixed provider requires 'universe' in strategy config")
            return FixedListUniverseProvider(symbols)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def create_config_provider(strategy_config: Dict[str, Any]) -> ConfigBasedUniverseProvider:
        """Create a configuration-based universe provider."""
        return ConfigBasedUniverseProvider(strategy_config)

    @staticmethod
    def create_fixed_provider(
        symbols: List[str], weights: Optional[List[float]] = None
    ) -> FixedListUniverseProvider:
        """Create a fixed list universe provider."""
        return FixedListUniverseProvider(symbols, weights)
