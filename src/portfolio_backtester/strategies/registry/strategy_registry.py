"""
SOLID-compliant strategy registry with unified automatic discovery.

🚨 FOR CODING AGENTS: MANUAL STRATEGY REGISTRATION IS STRICTLY PROHIBITED! 🚨

=== AUTOMATIC DISCOVERY ONLY ===
This registry follows SOLID principles with clear separation of concerns:
- Registry operations (IStrategyRegistry)
- Strategy discovery (IStrategyDiscoveryEngine)
- Strategy validation (IStrategyValidator)
- Factory pattern for dependency injection

TO CREATE NEW STRATEGIES (for coding agents):
1. Create properly named strategy class:
   - Signal: YourNameSignalStrategy(SignalStrategy)
   - Portfolio: YourNamePortfolioStrategy(PortfolioStrategy)
   - Meta: YourNameMetaStrategy(BaseMetaStrategy)

2. Save in correct directory with proper filename:
   - Signal: src/portfolio_backtester/strategies/signal/your_name_signal_strategy.py
   - Portfolio: src/portfolio_backtester/strategies/portfolio/your_name_portfolio_strategy.py
   - Meta: src/portfolio_backtester/strategies/meta/your_name_meta_strategy.py

3. Implement all abstract methods (make class concrete)

4. System will AUTOMATICALLY discover and register it - NO manual steps needed!

🚫 NEVER call registry.register_strategy() or similar methods!
🚫 NEVER manually add to registry._strategies dict!
🚫 NEVER import and hardcode strategy class names!

This module provides SOLID-compliant strategy discovery following framework patterns:
- Single Responsibility Principle (SRP): Separate classes for registry, discovery, validation
- Open/Closed Principle (OCP): Extensible through dependency injection
- Dependency Inversion Principle (DIP): Depends on abstractions, not concretions
- Interface Segregation Principle (ISP): Separate interfaces for different concerns
"""

from typing import Dict, Type, Optional
import logging

from .solid_strategy_registry import (
    StrategyRegistryFactory,
    IStrategyRegistry
)

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    SOLID-compliant strategy registry with backward compatibility.

    🚨 FOR CODING AGENTS: NO MANUAL REGISTRATION ALLOWED! 🚨

    This registry now follows SOLID principles:
    - Single Responsibility: Delegates to specialized components
    - Open/Closed: Extensible through dependency injection
    - Dependency Inversion: Uses abstractions, not concretions

    Any attempt to manually register strategies will raise RuntimeError.
    Use automatic discovery by creating properly named strategy files in correct directories.
    """

    def __init__(self) -> None:
        """Initialize SOLID-compliant strategy registry."""
        # Use factory to create SOLID-compliant implementation
        self._solid_registry: IStrategyRegistry = StrategyRegistryFactory.create_default_registry()
        self._initialized = True

    def get_strategy_class(self, name: str) -> Optional[Type]:
        """Get strategy class by name."""
        return self._solid_registry.get_strategy_class(name)

    def get_all_strategies(self) -> Dict[str, Type]:
        """Get all discovered strategies."""
        return self._solid_registry.get_all_strategies()

    def is_strategy_registered(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return self._solid_registry.is_strategy_registered(name)

    def register_strategy(self, name: str, strategy_class: Type) -> None:
        """
        🚨 MANUAL STRATEGY REGISTRATION IS STRICTLY PROHIBITED! 🚨

        This method is blocked to prevent manual strategy registration.
        Strategies are discovered AUTOMATICALLY - manual registration is not allowed.

        FOR CODING AGENTS: DO NOT CALL THIS METHOD!
        - Create properly named strategy classes
        - Place in correct directories with correct filenames
        - The system will auto-discover them

        Args:
            name: Strategy name identifier (ignored)
            strategy_class: Strategy class to register (ignored)

        Raises:
            RuntimeError: Always - manual registration is prohibited
        """
        # Delegate to SOLID implementation's register_strategy method
        if hasattr(self._solid_registry, 'register_strategy'):
            self._solid_registry.register_strategy(name, strategy_class)
        else:
            # Fallback if interface doesn't have register_strategy
            raise RuntimeError(
                "🚨 MANUAL STRATEGY REGISTRATION IS STRICTLY PROHIBITED! 🚨\n\n"
                "Use AUTOMATIC DISCOVERY instead:\n"
                "1. Create properly named class (e.g. YourNameSignalStrategy)\n"
                "2. Place in correct directory with proper filename\n"
                "3. Ensure class is concrete (implements all abstract methods)\n"
                "4. System will automatically discover and register it!\n\n"
                "DO NOT hardcode class names or call register_strategy()!"
            )

    def add_strategy(self, name: str, strategy_class: Type) -> None:
        """🚨 PROHIBITED! Use automatic discovery instead."""
        self.register_strategy(name, strategy_class)  # Will raise RuntimeError

    def register(self, name: str, strategy_class: Type) -> None:
        """🚨 PROHIBITED! Use automatic discovery instead."""
        self.register_strategy(name, strategy_class)  # Will raise RuntimeError

    def add(self, name: str, strategy_class: Type) -> None:
        """🚨 PROHIBITED! Use automatic discovery instead."""
        self.register_strategy(name, strategy_class)  # Will raise RuntimeError

    def __setitem__(self, name: str, strategy_class: Type) -> None:
        """🚨 PROHIBITED! Direct assignment blocked."""
        # Direct assignment is not supported by SOLID interface
        raise RuntimeError(
            "🚨 DIRECT STRATEGY ASSIGNMENT IS PROHIBITED! 🚨\n"
            "Use automatic discovery instead - create properly named strategy files!"
        )

    def __setattr__(self, name: str, value) -> None:
        """Protect internal attributes from external manipulation."""
        # Allow setting during initialization
        if not hasattr(self, "_initialized"):
            super().__setattr__(name, value)
            return

        # Block direct manipulation of strategy storage after initialization
        if name.startswith("_strategies") or name == "_solid_registry":
            raise RuntimeError(
                "🚨 DIRECT MANIPULATION OF STRATEGY STORAGE IS PROHIBITED! 🚨\n"
                "Strategies are managed automatically - use auto-discovery instead!"
            )

        # Allow other attributes to be set normally
        super().__setattr__(name, value)


# Singleton instance
_registry_instance: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """
    Get the singleton strategy registry instance.

    🚨 FOR CODING AGENTS: This is the ONLY way to access the registry!
    - DO NOT create StrategyRegistry() directly
    - DO NOT manipulate the returned registry manually
    - Use only read-only methods: get_strategy_class(), get_all_strategies(), is_strategy_registered()
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = StrategyRegistry()
        logger.debug("Created strategy registry singleton")
    return _registry_instance


def clear_strategy_registry() -> None:
    """
    Clear the global strategy registry (for testing only).

    🚨 FOR CODING AGENTS: DO NOT CALL THIS IN PRODUCTION CODE!
    This is only for unit tests - normal code should never clear the registry.
    """
    global _registry_instance
    _registry_instance = None
    logger.debug("Cleared strategy registry singleton")
