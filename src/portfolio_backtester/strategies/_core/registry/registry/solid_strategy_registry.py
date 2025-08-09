"""
SOLID-compliant strategy registry implementation.

This module implements strategy registry following SOLID principles,
with clear separation of concerns and dependency inversion.
"""

import logging
from typing import Dict, List, Optional, Type, Any

from portfolio_backtester.interfaces.strategy_registry_interface import (
    IStrategyRegistry,
    IStrategyDiscoveryEngine,
    IStrategyValidator,
    StrategyDiscoveryError,
)

logger = logging.getLogger(__name__)


class AutoDiscoveryStrategyRegistry(IStrategyRegistry):
    """
    SOLID-compliant strategy registry with automatic discovery.

    🚨 FOR CODING AGENTS: NO MANUAL REGISTRATION ALLOWED! 🚨

    This registry follows SRP by focusing ONLY on registry operations.
    Discovery and validation are handled by injected dependencies.

    Follows DIP by depending on abstractions (IStrategyDiscoveryEngine, IStrategyValidator)
    rather than concrete implementations.
    """

    def __init__(self, discovery_engine: IStrategyDiscoveryEngine, validator: IStrategyValidator):
        """
        Initialize registry with dependency injection.

        Args:
            discovery_engine: Strategy discovery implementation
            validator: Strategy validation implementation
        """
        self._discovery_engine = discovery_engine
        self._validator = validator
        self._strategies: Dict[str, Type[Any]] = {}
        self._discovered = False

        # Alias support removed: use canonical class names only

    def get_strategy_class(self, name: str) -> Optional[Type[Any]]:
        """Get strategy class by canonical name only (no aliases)."""
        self._ensure_discovered()

        # Check for direct name first
        strategy_class = self._strategies.get(name)
        if strategy_class:
            return strategy_class

        return None

    def get_all_strategies(self) -> Dict[str, Type[Any]]:
        """Get all discovered strategies."""
        self._ensure_discovered()
        return self._strategies.copy()

    def is_strategy_registered(self, name: str) -> bool:
        """Check if a strategy is registered by canonical name only (no aliases)."""
        self._ensure_discovered()

        # Check direct name first
        if name in self._strategies:
            return True

        return False

    def get_strategy_count(self) -> int:
        """
        Get total number of registered strategies.

        Returns:
            Number of registered strategies
        """
        self._ensure_discovered()
        return len(self._strategies)

    def register_strategy(self, name: str, strategy_class: Type[Any]) -> None:
        """
        🚨 MANUAL STRATEGY REGISTRATION IS STRICTLY PROHIBITED! 🚨

        This method is blocked to prevent manual strategy registration.
        Strategies are discovered AUTOMATICALLY - manual registration is not allowed.

        FOR CODING AGENTS: DO NOT CALL THIS METHOD!
        - Create properly named strategy classes
        - Place in correct directories with correct filenames
        - The system will auto-discover them

        Raises:
            RuntimeError: Always - manual registration is prohibited
        """
        raise RuntimeError(
            "🚨 MANUAL STRATEGY REGISTRATION IS STRICTLY PROHIBITED! 🚨\n\n"
            "Use AUTOMATIC DISCOVERY instead:\n"
            "1. Create properly named class (e.g. YourNameSignalStrategy)\n"
            "2. Place in correct directory with proper filename\n"
            "3. Ensure class is concrete (implements all abstract methods)\n"
            "4. System will automatically discover and register it!\n\n"
            "DO NOT hardcode class names or call register_strategy()!"
        )

    def __setitem__(self, name: str, strategy_class: Type[Any]) -> None:
        """🚨 PROHIBITED! Direct assignment blocked."""
        raise RuntimeError(
            "🚨 DIRECT STRATEGY ASSIGNMENT IS PROHIBITED! 🚨\n"
            "Use automatic discovery instead - create properly named strategy files!"
        )

    def _ensure_discovered(self) -> None:
        """Ensure strategies have been discovered."""
        if not self._discovered:
            self._discover_strategies()
            self._discovered = True

    def _discover_strategies(self) -> None:
        """
        Discover and validate strategies using injected dependencies.

        This method delegates to the discovery engine and validator,
        following the Single Responsibility Principle.
        """
        logger.debug("Starting SOLID-compliant strategy discovery")

        try:
            # Use injected discovery engine (DIP compliance)
            discovered_strategies = self._discovery_engine.discover_strategies()

            # Filter using injected validator (DIP compliance)
            valid_strategies = {}
            for name, strategy_class in discovered_strategies.items():
                if self._validator.is_valid_strategy(strategy_class):
                    valid_strategies[name] = strategy_class
                    logger.debug(f"Validated strategy: {name}")
                else:
                    errors = self._validator.get_validation_errors(strategy_class)
                    logger.debug(f"Invalid strategy {name}: {errors}")

            self._strategies = valid_strategies
            logger.info(
                f"SOLID strategy discovery completed: {len(self._strategies)} strategies found"
            )

        except Exception as e:
            logger.error(f"Strategy discovery failed: {e}")
            raise StrategyDiscoveryError(f"Failed to discover strategies: {e}")


class FileSystemStrategyDiscoveryEngine(IStrategyDiscoveryEngine):
    """
    Concrete strategy discovery engine using filesystem scanning.

    Follows SRP by focusing only on discovery operations.
    Follows OCP by accepting configurable discovery paths.
    """

    def __init__(self, discovery_paths: Optional[List[str]] = None):
        """
        Initialize discovery engine with configurable paths.

        Args:
            discovery_paths: List of directory names to scan (defaults to standard paths)
        """
        # Default discovery covers new layout first (builtins/user),
        # plus legacy paths and testing for backward compatibility during migration.
        self._discovery_paths = discovery_paths or [
            # New layout
            "builtins/portfolio",
            "builtins/signal",
            "builtins/meta",
            "user/portfolio",
            "user/signal",
            "user/meta",
            # Legacy layout (to be removed once migration completes)
            "portfolio",
            "signal",
            "meta",
            # Special-case for tests
            "testing",
        ]

    def discover_strategies(self) -> Dict[str, Type[Any]]:
        """Discover strategies by scanning filesystem."""
        import importlib
        import pkgutil
        import os

        discovered = {}

        # Get strategies directory path
        from portfolio_backtester import strategies as strategies_pkg

        strategies_dir = os.path.dirname(strategies_pkg.__file__)

        for subdir in self._discovery_paths:
            if subdir == "testing":
                # Special case for testing directory - it's under portfolio_backtester.testing.strategies
                testing_package = "portfolio_backtester.testing.strategies"
                testing_path = os.path.join(
                    os.path.dirname(strategies_dir), "testing", "strategies"
                )
                if os.path.isdir(testing_path):
                    for module_info in pkgutil.walk_packages(
                        [testing_path], prefix=f"{testing_package}."
                    ):
                        try:
                            importlib.import_module(module_info.name)
                            logger.debug(f"Imported testing module: {module_info.name}")
                        except Exception as e:
                            logger.debug(f"Skipped testing module {module_info.name}: {e}")
            else:
                # Regular strategy directories under portfolio_backtester.strategies
                package_name = "portfolio_backtester.strategies"
                subdir_path = os.path.join(strategies_dir, subdir)
                if os.path.isdir(subdir_path):
                    # Build module prefix with dots for nested paths
                    dotted_subdir = subdir.replace(os.sep, ".").replace("/", ".")
                    module_prefix = f"{package_name}.{dotted_subdir}."
                    for module_info in pkgutil.walk_packages([subdir_path], prefix=module_prefix):
                        try:
                            importlib.import_module(module_info.name)
                            logger.debug(f"Imported strategy module: {module_info.name}")
                        except Exception as e:
                            logger.debug(f"Skipped module {module_info.name}: {e}")

        # Get all subclasses from imported modules
        from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy

        all_subclasses = self._get_all_subclasses(BaseStrategy)

        for cls in all_subclasses:
            discovered[cls.__name__] = cls

        return discovered

    def get_discovery_paths(self) -> List[str]:
        """Get the discovery paths."""
        return self._discovery_paths.copy()

    def _get_all_subclasses(self, base_class: Type[Any]) -> set[Type[Any]]:
        """Recursively get all subclasses."""
        all_subclasses = set()
        for subclass in base_class.__subclasses__():
            all_subclasses.add(subclass)
            all_subclasses.update(self._get_all_subclasses(subclass))
        return all_subclasses


class ConcreteStrategyValidator(IStrategyValidator):
    """
    Concrete strategy validator implementation.

    Follows SRP by focusing only on validation operations.
    """

    def __init__(self):
        """Initialize validator with base strategy types."""
        # Import base strategy types
        from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy
        from portfolio_backtester.strategies._core.base.signal_strategy import SignalStrategy
        from portfolio_backtester.strategies._core.base.portfolio_strategy import PortfolioStrategy
        from portfolio_backtester.strategies._core.base.meta_strategy import BaseMetaStrategy

        self._base_types = {BaseStrategy, SignalStrategy, PortfolioStrategy, BaseMetaStrategy}

    def is_valid_strategy(self, strategy_class: Type[Any]) -> bool:
        """Check if a class is a valid concrete strategy."""
        import inspect
        from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy

        return (
            inspect.isclass(strategy_class)
            and issubclass(strategy_class, BaseStrategy)
            and strategy_class not in self._base_types
            and not inspect.isabstract(strategy_class)
            and not bool(getattr(strategy_class, "__abstractmethods__", frozenset()))
        )

    def get_validation_errors(self, strategy_class: Type[Any]) -> List[str]:
        """Get detailed validation errors."""
        import inspect
        from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy

        errors = []

        if not inspect.isclass(strategy_class):
            errors.append("Not a class")

        if not issubclass(strategy_class, BaseStrategy):
            errors.append("Does not inherit from BaseStrategy")

        if strategy_class in self._base_types:
            errors.append("Is a base class, not a concrete strategy")

        if inspect.isabstract(strategy_class):
            errors.append("Is abstract class")

        if bool(getattr(strategy_class, "__abstractmethods__", frozenset())):
            abstract_methods: frozenset[str] = getattr(
                strategy_class, "__abstractmethods__", frozenset()
            )
            errors.append(f"Has unimplemented abstract methods: {list(abstract_methods)}")

        return errors

    def get_base_strategy_types(self) -> List[Type[Any]]:
        """Get the base strategy types that should be excluded."""
        return list(self._base_types)


class StrategyRegistryFactory:
    """
    Factory for creating SOLID-compliant strategy registries.

    Follows the same pattern as other framework factories.
    Provides dependency injection for discovery and validation.
    """

    @staticmethod
    def create_registry(
        discovery_paths: Optional[List[str]] = None,
        discovery_engine: Optional[IStrategyDiscoveryEngine] = None,
        validator: Optional[IStrategyValidator] = None,
    ) -> IStrategyRegistry:
        """
        Create a strategy registry with dependency injection.

        Args:
            discovery_paths: Custom discovery paths (optional)
            discovery_engine: Custom discovery engine (optional)
            validator: Custom validator (optional)

        Returns:
            IStrategyRegistry instance
        """
        if discovery_engine is None:
            discovery_engine = FileSystemStrategyDiscoveryEngine(discovery_paths)

        if validator is None:
            validator = ConcreteStrategyValidator()

        return AutoDiscoveryStrategyRegistry(discovery_engine, validator)

    @staticmethod
    def create_default_registry() -> IStrategyRegistry:
        """Create registry with default dependencies."""
        return StrategyRegistryFactory.create_registry()
