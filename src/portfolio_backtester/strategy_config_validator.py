import os
from pathlib import Path

def get_strategy_files(strategies_dir):
    """Get all strategy files from the strategies directory."""
    return [p for p in Path(strategies_dir).rglob('*.py') if p.is_file() and p.name != '__init__.py']

def get_default_config_files(strategies_dir):
    """Get all default.yaml configuration files from the strategies directory."""
    return [p for p in Path(strategies_dir).rglob('default.yaml') if p.is_file()]

def get_strategy_name_from_path(strategy_path):
    """Extracts the strategy name from the file path."""
    return strategy_path.stem

def get_strategy_dir_from_path(strategy_path):
    """Extracts the strategy directory from the file path."""
    return strategy_path.parent.name

def validate_strategy_configs(strategies_dir, scenarios_dir):
    """
    Validates that each strategy has a corresponding default.yaml configuration file.
    """
    strategy_files = get_strategy_files(strategies_dir)
    
    # Get configuration files from scenarios directory
    scenarios_path = Path(scenarios_dir)
    config_files = list(scenarios_path.rglob('default.yaml'))
    strategy_names_with_config = {f.parent.name for f in config_files}

    validation_errors = []
    
    # Only validate actual strategy files, not base classes or utilities
    strategy_categories = ['diagnostic', 'meta', 'portfolio', 'signal']
    
    for strategy_file in strategy_files:
        # Skip base classes and utility files
        if strategy_file.parent.name in ['base'] or strategy_file.name in [
            'candidate_weights.py', 'leverage_and_smoothing.py', 'strategy_factory.py'
        ]:
            continue
            
        # Only check strategies in known categories
        if strategy_file.parent.name not in strategy_categories:
            continue
            
        strategy_name = get_strategy_name_from_path(strategy_file)

        if strategy_name not in strategy_names_with_config:
            validation_errors.append(
                f"Strategy '{strategy_name}' does not have a corresponding "
                f"default.yaml configuration file in scenarios directory."
            )

    return len(validation_errors) == 0, validation_errors
