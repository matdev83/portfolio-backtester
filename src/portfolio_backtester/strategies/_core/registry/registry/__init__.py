"""SOLID-compliant strategy registry package."""

from .strategy_registry import (
    StrategyRegistry,
    get_strategy_registry,
    clear_strategy_registry,
)

from .solid_strategy_registry import (
    StrategyRegistryFactory,
    AutoDiscoveryStrategyRegistry,
    FileSystemStrategyDiscoveryEngine,
    ConcreteStrategyValidator,
)

from portfolio_backtester.interfaces.strategy_registry_interface import (
    IStrategyRegistry,
    IStrategyDiscoveryEngine,
    IStrategyValidator,
    StrategyDiscoveryError,
)

__all__ = [
    # Backward compatibility (delegates to SOLID implementation)
    "StrategyRegistry",
    "get_strategy_registry",
    "clear_strategy_registry",
    # SOLID-compliant interfaces
    "IStrategyRegistry",
    "IStrategyDiscoveryEngine",
    "IStrategyValidator",
    "StrategyDiscoveryError",
    # SOLID-compliant implementations
    "StrategyRegistryFactory",
    "AutoDiscoveryStrategyRegistry",
    "FileSystemStrategyDiscoveryEngine",
    "ConcreteStrategyValidator",
]
