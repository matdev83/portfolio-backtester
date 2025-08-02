"""
Strategy Configuration Validator

This module validates that all discovered strategies have corresponding configuration files
in the proper directory structure.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Set

from .strategies import enumerate_strategies_with_params

logger = logging.getLogger(__name__)

# Configuration directory paths
SCENARIOS_DIR = Path(__file__).parent.parent.parent / "config" / "scenarios"


def get_strategy_config_directory(strategy_name: str) -> Path:
    """
    Get the expected configuration directory for a strategy.
    
    Args:
        strategy_name: The strategy name (snake_case)
        
    Returns:
        Path to the expected configuration directory
    """
    # Determine strategy type based on the strategy class
    discovered_strategies = enumerate_strategies_with_params()
    strategy_class = discovered_strategies.get(strategy_name)
    
    if not strategy_class:
        return None
    
    # Import base classes to check inheritance
    from .strategies.base.portfolio_strategy import PortfolioStrategy
    from .strategies.base.signal_strategy import SignalStrategy
    from .strategies.base.meta_strategy import BaseMetaStrategy
    
    # Determine the strategy type directory
    if issubclass(strategy_class, BaseMetaStrategy):
        strategy_type = "meta"
    elif issubclass(strategy_class, PortfolioStrategy):
        strategy_type = "portfolio"
    elif issubclass(strategy_class, SignalStrategy):
        strategy_type = "signal"
    elif "diagnostic" in strategy_class.__module__.lower():
        strategy_type = "diagnostic"
    else:
        # Default to portfolio for unknown types
        strategy_type = "portfolio"
    
    # Handle special naming cases
    if strategy_name.endswith("_strategy"):
        directory_name = strategy_name
    else:
        directory_name = f"{strategy_name}_strategy"
    
    return SCENARIOS_DIR / strategy_type / directory_name


def find_existing_config_files(strategy_name: str) -> List[Path]:
    """
    Find all existing configuration files for a strategy.
    
    Args:
        strategy_name: The strategy name (snake_case)
        
    Returns:
        List of paths to existing configuration files
    """
    # Try multiple possible directory names
    possible_dirs = []
    
    # Get the expected directory
    config_dir = get_strategy_config_directory(strategy_name)
    if config_dir:
        possible_dirs.append(config_dir)
    
    # Also try without the "_strategy" suffix for existing directories
    discovered_strategies = enumerate_strategies_with_params()
    strategy_class = discovered_strategies.get(strategy_name)
    
    if strategy_class:
        from .strategies.base.portfolio_strategy import PortfolioStrategy
        from .strategies.base.signal_strategy import SignalStrategy
        from .strategies.base.meta_strategy import BaseMetaStrategy
        
        # Determine the strategy type directory
        if issubclass(strategy_class, BaseMetaStrategy):
            strategy_type = "meta"
        elif issubclass(strategy_class, PortfolioStrategy):
            strategy_type = "portfolio"
        elif issubclass(strategy_class, SignalStrategy):
            strategy_type = "signal"
        elif "diagnostic" in strategy_class.__module__.lower():
            strategy_type = "diagnostic"
        else:
            strategy_type = "portfolio"
        
        # Try directory without "_strategy" suffix
        alt_dir = SCENARIOS_DIR / strategy_type / strategy_name
        if alt_dir not in possible_dirs:
            possible_dirs.append(alt_dir)
    
    config_files = []
    for config_dir in possible_dirs:
        if config_dir and config_dir.exists():
            for file_path in config_dir.glob("*.yaml"):
                config_files.append(file_path)
    
    return config_files


def is_concrete_strategy_requiring_config(strategy_name: str, strategy_class) -> bool:
    """
    Check if a strategy is concrete and requires a configuration file.
    
    Args:
        strategy_name: The strategy name
        strategy_class: The strategy class
        
    Returns:
        True if the strategy requires a config file
    """
    import inspect
    
    # Skip aliases and test strategies
    if strategy_name in ['dummy', 'dummy_strategy', 'dummy_strategy_for_testing']:
        return False
    
    # Skip abstract base classes
    if inspect.isabstract(strategy_class):
        return False
    
    # Skip base strategy classes that are not meant to be used directly
    base_class_names = ['BaseStrategy', 'PortfolioStrategy', 'SignalStrategy', 'BaseMetaStrategy']
    if strategy_class.__name__ in base_class_names:
        return False
    
    # Skip if it's just an alias pointing to the same class
    # (This handles cases where multiple names point to the same strategy)
    return True


def validate_strategy_configs() -> tuple[bool, List[str]]:
    """
    Validate that all discovered strategies have at least one configuration file.
    
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    logger.info("Validating strategy configuration files...")
    
    # Get all discovered strategies
    discovered_strategies = enumerate_strategies_with_params()
    
    if not discovered_strategies:
        return False, ["No strategies were discovered. This indicates a problem with strategy discovery."]
    
    errors = []
    missing_configs = []
    
    for strategy_name, strategy_class in discovered_strategies.items():
        # Skip strategies that don't require config files
        if not is_concrete_strategy_requiring_config(strategy_name, strategy_class):
            continue
            
        config_files = find_existing_config_files(strategy_name)
        
        if not config_files:
            expected_dir = get_strategy_config_directory(strategy_name)
            missing_configs.append({
                'strategy_name': strategy_name,
                'strategy_class': strategy_class.__name__,
                'expected_dir': expected_dir
            })
    
    if missing_configs:
        errors.append("❌ Missing configuration files for the following strategies:")
        errors.append("")
        
        for missing in missing_configs:
            errors.append(f"  Strategy: {missing['strategy_name']} ({missing['strategy_class']})")
            errors.append(f"  Expected directory: {missing['expected_dir']}")
            errors.append(f"  Required: At least one .yaml file in the directory")
            errors.append("")
        
        errors.append("To fix this issue:")
        errors.append("1. Create the missing directories")
        errors.append("2. Add at least one .yaml configuration file (e.g., 'default.yaml') in each directory")
        errors.append("3. The configuration file should contain valid strategy parameters")
        errors.append("")
        errors.append("Example configuration file structure:")
        errors.append("  strategy_class: YourStrategyClassName")
        errors.append("  parameters:")
        errors.append("    param1: value1")
        errors.append("    param2: value2")
        
        return False, errors
    
    logger.info(f"✅ All {len(discovered_strategies)} discovered strategies have configuration files")
    return True, []


def create_missing_config_directories() -> List[str]:
    """
    Create missing configuration directories for strategies that don't have them.
    
    Returns:
        List of created directory paths
    """
    discovered_strategies = enumerate_strategies_with_params()
    created_dirs = []
    
    for strategy_name, strategy_class in discovered_strategies.items():
        # Skip aliases and test strategies
        if strategy_name in ['dummy', 'dummy_strategy', 'dummy_strategy_for_testing']:
            continue
            
        config_dir = get_strategy_config_directory(strategy_name)
        if config_dir and not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(config_dir))
    
    return created_dirs


if __name__ == "__main__":
    # CLI for testing the validator
    import sys
    
    is_valid, errors = validate_strategy_configs()
    
    if is_valid:
        print("✅ All strategies have configuration files!")
        sys.exit(0)
    else:
        print("\n".join(errors))
        sys.exit(1)