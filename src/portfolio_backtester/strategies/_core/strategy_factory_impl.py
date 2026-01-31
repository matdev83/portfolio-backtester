"""
Strategy factory for dynamic instantiation of strategies with AUTOMATIC DISCOVERY ONLY.

=== FOR CODING AGENTS ===
🚨 NEVER CALL StrategyFactory.register_strategy() MANUALLY! 🚨

Strategies are AUTOMATICALLY DISCOVERED based on proper naming and location.
Manual registration is PROHIBITED and will raise an error.

TO CREATE A NEW STRATEGY:
1. Choose strategy type: SignalStrategy, PortfolioStrategy, or BaseMetaStrategy
2. Create class with proper naming:
   - Signal strategies: class YourNameSignalStrategy(SignalStrategy)
   - Portfolio strategies: class YourNamePortfolioStrategy(PortfolioStrategy)
   - Meta strategies: class YourNameMetaStrategy(BaseMetaStrategy)
3. Save in correct directory with proper filename:
   - Signal: src/portfolio_backtester/strategies/signal/your_name_signal_strategy.py
   - Portfolio: src/portfolio_backtester/strategies/portfolio/your_name_portfolio_strategy.py
   - Meta: src/portfolio_backtester/strategies/meta/your_name_meta_strategy.py
4. Ensure class is concrete (not abstract) - implement all required methods

The system will AUTOMATICALLY find and register your strategy. No manual steps needed!
"""

from typing import Dict, Any, Type, Set, Optional, cast, Mapping, Union
import logging

from .base.base.base_strategy import BaseStrategy
from .registry import get_strategy_registry

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Factory for creating strategy instances using SOLID-compliant registry.

    This factory delegates to the centralized strategy registry system,
    eliminating complex discovery mechanisms and following DIP principles.
    """

    _circular_detection: Set[str] = set()

    @classmethod
    def _get_registry(cls):
        """Get the strategy registry instance."""
        return get_strategy_registry()

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        🚨 MANUAL REGISTRATION IS PROHIBITED! 🚨

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
        raise RuntimeError(
            "🚨 MANUAL STRATEGY REGISTRATION IS PROHIBITED! 🚨\n\n"
            "Use AUTOMATIC DISCOVERY instead:\n"
            "1. Create properly named class (e.g. YourNameSignalStrategy)\n"
            "2. Place in correct directory with proper filename\n"
            "3. Ensure class is concrete (implements all abstract methods)\n"
            "4. System will automatically discover and register it!\n\n"
            "DO NOT hardcode class names or call register_strategy()!"
        )

    @classmethod
    def create_strategy(
        cls,
        strategy_class: str,
        strategy_params: Union[Mapping[str, Any], "CanonicalScenarioConfig"],
        global_config: Optional[Dict[str, Any]] = None,
    ) -> BaseStrategy:
        """
        Create a strategy instance from class name and parameters.

        Args:
            strategy_class: Name of the strategy class
            strategy_params: Parameters for strategy initialization or full canonical config
            global_config: Global configuration (for meta strategies)

        Returns:
            Instantiated strategy

        Raises:
            ValueError: If strategy class is unknown or circular dependency detected
        """
        from .base.base.base_strategy import BaseStrategy
        from ...canonical_config import CanonicalScenarioConfig

        registry = cls._get_registry()

        # Check for circular dependencies in meta strategies
        if strategy_class in cls._circular_detection:
            raise ValueError(f"Circular dependency detected for strategy: {strategy_class}")

        strategy_cls = registry.get_strategy_class(strategy_class)
        if strategy_cls is None:
            available_strategies = list(registry.get_all_strategies().keys())
            raise ValueError(
                f"Unknown strategy class: {strategy_class}. Available: {available_strategies}"
            )

        # Add to circular detection set for meta strategies
        if strategy_class.endswith("MetaStrategy"):
            cls._circular_detection.add(strategy_class)

        # Process strategy_params
        if isinstance(strategy_params, CanonicalScenarioConfig):
            # If passed canonical config, we use it directly for instantiation
            # but for processed_params (if we still need it for some reason) we use strategy_params
            init_arg: Union[Mapping[str, Any], CanonicalScenarioConfig] = strategy_params
            # Extract params for prefix processing if we were to pass dict
            params_to_process = strategy_params.strategy_params
        else:
            init_arg = strategy_params
            params_to_process = strategy_params

        processed_params = {}
        if params_to_process:
            # Handle both prefixed and non-prefixed parameters
            for key, value in params_to_process.items():
                if "." in key:
                    # Remove the prefix (everything before the first dot)
                    param_name = key.split(".", 1)[1]
                    processed_params[param_name] = value
                else:
                    # Keep non-prefixed params as is
                    processed_params[key] = value

        # Note: If init_arg was CanonicalScenarioConfig, we might want to pass 
        # processed_params instead if the strategy doesn't handle the canonical object yet.
        # But BaseStrategy does handle it. 
        # However, some strategies might override __init__ and expect a dict.
        # So we should probably pass the processed dict if it's not a canonical object,
        # or find a way to merge them.
        
        if not isinstance(init_arg, CanonicalScenarioConfig):
            init_arg = processed_params

        try:
            # Check if this is a meta strategy that accepts global_config
            if strategy_class.endswith("MetaStrategy") and global_config is not None:
                # Meta strategies accept global_config as second parameter
                instance = strategy_cls(init_arg, global_config=global_config)
            else:
                instance = strategy_cls(init_arg)

            return cast(BaseStrategy, instance)
        finally:
            # Remove from circular detection set
            cls._circular_detection.discard(strategy_class)

    @classmethod
    def get_registered_strategies(cls) -> Dict[str, Type[BaseStrategy]]:
        """Get all registered strategies."""
        registry = cls._get_registry()
        return cast(Dict[str, Type[BaseStrategy]], registry.get_all_strategies())

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the strategy registry (testing utility)."""
        from .registry import clear_strategy_registry

        clear_strategy_registry()
        cls._circular_detection.clear()
