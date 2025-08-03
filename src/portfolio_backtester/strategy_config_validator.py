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
    all_strategy_files = get_strategy_files(strategies_dir)
    
    # Get configuration files from scenarios directory
    scenarios_path = Path(scenarios_dir)
    config_files = list(scenarios_path.rglob('default.yaml'))
    strategy_names_with_config = {f.parent.name for f in config_files}

    validation_errors = []
    
    # Only validate actual strategy files, not base classes or utilities
    strategy_categories = ['diagnostic', 'meta', 'portfolio', 'signal']

    # Group strategy files by their parent directory (strategy directory)
    strategy_dirs_to_files = {}
    for py_file in all_strategy_files:
        # Skip base classes and utility files at the root of strategies_dir
        if py_file.parent.name in ['base'] or py_file.name in [
            'candidate_weights.py', 'leverage_and_smoothing.py', 'strategy_factory.py'
        ]:
            continue

        strategy_dir = py_file.parent
        
        # Check if the strategy's parent directory (the category) is known
        if strategy_dir.parent.name not in strategy_categories:
            continue
            
        if strategy_dir not in strategy_dirs_to_files:
            strategy_dirs_to_files[strategy_dir] = []
        strategy_dirs_to_files[strategy_dir].append(py_file)

    # Now process each unique strategy directory
    for strategy_dir, files_in_dir in strategy_dirs_to_files.items():
        # The strategy name is the name of its directory
        strategy_name = strategy_dir.name

        if strategy_name not in strategy_names_with_config:
            validation_errors.append(
                f"Strategy '{strategy_name}' does not have a corresponding "
                f"default.yaml configuration file in scenarios directory."
            )
            # If there's no config, we might not want to check .py file naming for this strategy,
            # or we might, depending on desired behavior. For now, let's skip .py checks if no config.
            continue

        # New: Check Python file naming consistency within the strategy's directory
        python_files_in_dir = [f for f in strategy_dir.glob('*.py') if f.is_file() and f.name != '__init__.py']
        
        for py_file in python_files_in_dir:
            if strategy_name not in py_file.name:
                validation_errors.append(
                    f"Python file '{py_file.name}' in strategy '{strategy_name}' directory "
                    f"does not contain the strategy name ('{strategy_name}') in its filename. "
                    f"Consider renaming or moving this file for better organization."
                )

    return len(validation_errors) == 0, validation_errors
