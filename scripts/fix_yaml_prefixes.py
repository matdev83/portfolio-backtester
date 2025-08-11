"""Utility script to ensure all strategy_params keys are namespaced with the top-level strategy.

Usage (from project root):
    python scripts/fix_yaml_prefixes.py

The script modifies YAML files in-place under config/scenarios/.
It adds `<strategy>.` prefix to any key inside `strategy_params` that does
not already start with that prefix or any prefix (i.e., it contains no dot).

For meta-strategy scenarios, only the meta strategy's own `strategy_params`
block is updated – nested `allocations[*].strategy_params` are left untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCEN_DIR = ROOT / "config" / "scenarios"


def process_strategy_params(strategy_name: str, params: dict[str, Any]) -> bool:
    """Add prefix where missing. Returns True if any change made."""
    changed = False
    for key in list(params.keys()):
        if key.startswith(f"{strategy_name}."):
            continue  # already correct
        if "." in key:
            # assume already namespaced for something else – leave untouched
            continue
        new_key = f"{strategy_name}.{key}"
        params[new_key] = params.pop(key)
        changed = True
    return changed


def fix_file(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"YAML parse error in {path}: {e}")
            return False

    if not isinstance(data, dict):
        return False

    strategy_name = data.get("strategy")
    if not strategy_name or not isinstance(strategy_name, str):
        return False

    changed = False

    # Top-level strategy_params
    if isinstance(data.get("strategy_params"), dict):
        if process_strategy_params(strategy_name, data["strategy_params"]):
            changed = True

    # If meta-strategy, don't touch nested allocations' parameters
    # (They already provide their own namespacing context.)

    if changed:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    return changed


def main() -> None:
    changed_files = []
    for yaml_path in SCEN_DIR.rglob("*.yaml"):
        if fix_file(yaml_path):
            changed_files.append(yaml_path.relative_to(ROOT))

    if changed_files:
        print("Updated prefixes in:")
        for p in changed_files:
            print(f"  {p}")
    else:
        print("All YAML files already use prefixed parameters.")


if __name__ == "__main__":
    main()
