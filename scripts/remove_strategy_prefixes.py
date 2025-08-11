#!/usr/bin/env python3
"""
Script to remove strategy prefixes from all YAML configuration files.

This script converts configuration files from the old prefixed format:
  strategy_params:
    momentum.lookback_months: 6
    momentum.num_holdings: 10

To the new simplified format:
  strategy_params:
    lookback_months: 6
    num_holdings: 10

Usage: python scripts/remove_strategy_prefixes.py
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def remove_prefixes_from_params(params: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Remove strategy prefixes from parameter keys."""
    if not isinstance(params, dict):
        return params

    cleaned_params = {}
    prefix = f"{strategy_name}."

    for key, value in params.items():
        if key.startswith(prefix):
            # Remove the prefix
            clean_key = key[len(prefix) :]
            cleaned_params[clean_key] = value
        else:
            # Keep non-prefixed keys as-is
            cleaned_params[key] = value

    return cleaned_params


def process_yaml_file(file_path: Path) -> bool:
    """Process a single YAML file to remove strategy prefixes. Returns True if changed."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return False

        strategy_name = data.get("strategy")
        if not strategy_name or not isinstance(strategy_name, str):
            return False

        strategy_params = data.get("strategy_params")
        if not isinstance(strategy_params, dict):
            return False

        # Remove prefixes from strategy_params
        original_params = dict(strategy_params)
        cleaned_params = remove_prefixes_from_params(strategy_params, strategy_name)

        # Check if any changes were made
        if cleaned_params == original_params:
            return False

        # Update the data
        data["strategy_params"] = cleaned_params

        # Write back to file
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process all YAML files in the config directory."""
    config_dir = Path("config/scenarios")

    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}")
        return

    # Find all YAML files
    yaml_files = list(config_dir.rglob("*.yaml"))

    if not yaml_files:
        print("No YAML files found in config directory")
        return

    print(f"Found {len(yaml_files)} YAML files to process...")

    changed_files = []

    for yaml_file in yaml_files:
        try:
            # Use a safe relative path display
            try:
                display_path = yaml_file.relative_to(Path.cwd())
            except ValueError:
                display_path = yaml_file

            if process_yaml_file(yaml_file):
                changed_files.append(yaml_file)
                print(f"✅ Updated: {display_path}")
            else:
                print(f"⏭️  Skipped: {display_path} (no prefixes found)")
        except Exception as e:
            print(f"❌ Error: {display_path}: {e}")

    print(f"\n🎉 Summary: Updated {len(changed_files)} files")

    if changed_files:
        print("\nUpdated files:")
        for file_path in changed_files:
            try:
                display_path = file_path.relative_to(Path.cwd())
            except ValueError:
                display_path = file_path
            print(f"  - {display_path}")

    print("\n✨ All YAML configuration files have been updated to use simplified parameter syntax!")


if __name__ == "__main__":
    main()
