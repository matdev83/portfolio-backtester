"""Advanced strategy configuration cross-validator.

This module extends the existing YAML strategy config validator with cross-check functionality:
it properly checks for the existence and correctness of config files for existing strategies
and looks for folders in config/scenarios/ related to strategies that don't exist (maybe
got deleted or renamed), and for strategies that in their content refer to strategy names
that are not valid (maybe got renamed or deleted). This improves the config validation
process and allows for detection and removal of stale/obsolete config files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from .strategy_config_validator import validate_strategy_configs
from .interfaces import StrategyResolverFactory
from .yaml_validator import YamlValidator

logger = logging.getLogger(__name__)


class StrategyConfigCrossValidator:
    """Advanced cross-validator for strategy configurations and references."""

    def __init__(self, src_strategies_dir: str, config_scenarios_dir: str):
        """Initialize the cross-validator.

        Args:
            src_strategies_dir: Path to src/portfolio_backtester/strategies
            config_scenarios_dir: Path to config/scenarios
        """
        self.src_strategies_dir = Path(src_strategies_dir)
        self.config_scenarios_dir = Path(config_scenarios_dir)
        self.strategy_resolver = StrategyResolverFactory.create()

    def validate_cross_references(self) -> Tuple[bool, List[str]]:
        """Perform comprehensive cross-reference validation.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []

        # 1. Run existing basic validator first
        basic_valid, basic_errors = validate_strategy_configs(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )
        if not basic_valid:
            all_errors.extend(basic_errors)

        # 2. Check for stale config folders (configs without corresponding strategies)
        stale_errors = self._check_stale_config_folders()
        all_errors.extend(stale_errors)

        # 3. Check for invalid strategy references in config files
        reference_errors = self._check_invalid_strategy_references()
        all_errors.extend(reference_errors)

        # 4. Check for strategy name references in strategy code itself
        # (e.g., meta-strategies referencing non-existent sub-strategies)
        code_reference_errors = self._check_strategy_code_references()
        all_errors.extend(code_reference_errors)

        return len(all_errors) == 0, all_errors

    def _get_valid_strategy_names(self) -> Set[str]:
        """Get set of valid strategy names by directly importing and scanning strategy classes."""
        valid_names = set()

        # Import all strategy modules to register classes
        self._import_all_strategy_modules()

        # Get all BaseStrategy subclasses
        from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy

        all_strategies = self._collect_all_subclasses(BaseStrategy)

        for strategy_class in all_strategies:
            if self._is_concrete_strategy_class(strategy_class):
                # Add the class name
                class_name = strategy_class.__name__
                valid_names.add(class_name)

                # Add snake_case version of class name (common convention)
                snake_case = self._class_name_to_snake_case(class_name)
                valid_names.add(snake_case)

                # Add name without suffix (e.g., "momentum" from "MomentumPortfolioStrategy")
                base_name = self._extract_base_strategy_name(class_name)
                if base_name:
                    valid_names.add(base_name)

        return valid_names

    def _import_all_strategy_modules(self) -> None:
        """Import all strategy modules to ensure classes are available."""
        import importlib

        # Import from strategy subdirectories
        package_name = "portfolio_backtester.strategies"
        strategies_path = self.src_strategies_dir

        for subdir in ["portfolio", "signal", "meta", "diagnostic"]:
            subdir_path = strategies_path / subdir
            if subdir_path.is_dir():
                for py_file in subdir_path.glob("*.py"):
                    if py_file.name.startswith("__"):
                        continue

                    module_name = f"{package_name}.{subdir}.{py_file.stem}"
                    try:
                        importlib.import_module(module_name)
                    except Exception as e:
                        logger.debug(f"Failed to import {module_name}: {e}")
                        continue

    def _collect_all_subclasses(self, base_class) -> set:
        """Recursively collect all subclasses of a base class."""
        all_subclasses = set()
        for subclass in base_class.__subclasses__():
            all_subclasses.add(subclass)
            all_subclasses.update(self._collect_all_subclasses(subclass))
        return all_subclasses

    def _is_concrete_strategy_class(self, cls) -> bool:
        """Check if a class is a concrete strategy (not abstract or base class)."""
        import inspect
        from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy
        from portfolio_backtester.strategies._core.base.signal_strategy import SignalStrategy
        from portfolio_backtester.strategies._core.base.portfolio_strategy import PortfolioStrategy
        from portfolio_backtester.strategies._core.base.meta_strategy import BaseMetaStrategy

        # Known base classes that should not be considered concrete
        base_classes = {BaseStrategy, SignalStrategy, PortfolioStrategy, BaseMetaStrategy}

        # Check if it has abstract methods
        has_abstract_methods = bool(getattr(cls, "__abstractmethods__", frozenset()))

        return (
            inspect.isclass(cls)
            and issubclass(cls, BaseStrategy)
            and cls not in base_classes
            and not inspect.isabstract(cls)
            and not has_abstract_methods
        )

    def _class_name_to_snake_case(self, class_name: str) -> str:
        """Convert class name from PascalCase to snake_case."""
        import re

        # Insert underscore before uppercase letters (except the first one)
        snake_case = re.sub(r"(?<!^)([A-Z])", r"_\1", class_name)
        return snake_case.lower()

    def _extract_base_strategy_name(self, class_name: str) -> Optional[str]:
        """Extract base strategy name from class name.

        E.g., "MomentumPortfolioStrategy" -> "momentum"
        """
        # Remove common suffixes and convert to snake_case
        suffixes = ["PortfolioStrategy", "SignalStrategy", "MetaStrategy", "Strategy"]

        for suffix in suffixes:
            if class_name.endswith(suffix):
                base_name = class_name[: -len(suffix)]
                return self._class_name_to_snake_case(base_name)

        return None

    def _check_stale_config_folders(self) -> List[str]:
        """Check for config folders that don't have corresponding strategies.

        Supports both legacy layout and new layout:
          - legacy: config/scenarios/<category>/<strategy>
          - new:    config/scenarios/{builtins|user}/<category>/<strategy>
        """
        errors: List[str] = []
        valid_strategy_names = self._get_valid_strategy_names()

        strategy_categories = ["diagnostic", "meta", "portfolio", "signal"]

        # Roots to scan: builtins/, user/, and legacy top-level
        roots = [
            ("builtins", True),
            ("user", True),
            ("", False),  # legacy (no root component)
        ]

        for root, has_root in roots:
            for category in strategy_categories:
                category_path = (
                    (self.config_scenarios_dir / root / category)
                    if has_root
                    else (self.config_scenarios_dir / category)
                )

                if not category_path.exists():
                    continue

                for strategy_folder in category_path.iterdir():
                    if not strategy_folder.is_dir():
                        continue

                    strategy_name = strategy_folder.name

                    # Check if this strategy name exists in any valid form
                    if not any(
                        name == strategy_name or strategy_name in name or name in strategy_name
                        for name in valid_strategy_names
                    ):
                        # Double-check by trying to resolve via strategy resolver
                        resolved_strategy = self.strategy_resolver.resolve_strategy(strategy_name)

                        if resolved_strategy is None:
                            # Build path string for message
                            prefix = f"{root}/" if has_root else ""
                            rel_path = f"{prefix}{category}/{strategy_name}"
                            errors.append(
                                f"Stale config folder detected: '{rel_path}' - "
                                f"no corresponding strategy implementation found. "
                                f"This config folder should be removed or the strategy should be restored."
                            )

        return errors

    def _check_invalid_strategy_references(self) -> List[str]:
        """Check for invalid strategy references in YAML config files."""
        errors: List[str] = []
        valid_strategy_names = self._get_valid_strategy_names()

        # Find all YAML files in scenarios
        yaml_files = list(self.config_scenarios_dir.rglob("*.yaml")) + list(
            self.config_scenarios_dir.rglob("*.yml")
        )

        for yaml_file in yaml_files:
            config_errors = self._validate_yaml_strategy_references(yaml_file, valid_strategy_names)
            errors.extend(config_errors)

        return errors

    def _validate_yaml_strategy_references(
        self, yaml_file: Path, valid_strategy_names: Set[str]
    ) -> List[str]:
        """Validate strategy references in a single YAML file."""
        errors: List[str] = []

        # Load and parse YAML file
        validator = YamlValidator()
        is_valid, data, yaml_errors = validator.validate_file(yaml_file)

        if not is_valid or data is None:
            # Skip files with YAML syntax errors - they'll be caught by other validators
            return errors

        if not isinstance(data, dict):
            return errors

        # Check 'strategy' field
        strategy_name = data.get("strategy")
        if strategy_name and isinstance(strategy_name, str):
            if not self._is_valid_strategy_reference(strategy_name, valid_strategy_names):
                errors.append(
                    f"Invalid strategy reference in {yaml_file.relative_to(self.config_scenarios_dir)}: "
                    f"strategy '{strategy_name}' does not exist or cannot be resolved"
                )

        # Check 'strategy_class' field if present
        strategy_class = data.get("strategy_class")
        if strategy_class and isinstance(strategy_class, str):
            if not self._is_valid_strategy_class_reference(strategy_class, valid_strategy_names):
                errors.append(
                    f"Invalid strategy_class reference in {yaml_file.relative_to(self.config_scenarios_dir)}: "
                    f"strategy_class '{strategy_class}' does not exist"
                )

        # Check meta-strategy allocations if present
        if self._is_meta_strategy_config(data):
            allocation_errors = self._check_meta_strategy_allocations(
                yaml_file, data, valid_strategy_names
            )
            errors.extend(allocation_errors)

        return errors

    def _is_valid_strategy_reference(
        self, strategy_name: str, valid_strategy_names: Set[str]
    ) -> bool:
        """Check if a strategy reference is valid."""
        # Direct check
        if strategy_name in valid_strategy_names:
            return True

        # Try to resolve via strategy resolver
        resolved_strategy = self.strategy_resolver.resolve_strategy(strategy_name)
        return resolved_strategy is not None

    def _is_valid_strategy_class_reference(
        self, strategy_class: str, valid_strategy_names: Set[str]
    ) -> bool:
        """Check if a strategy_class reference is valid."""
        # First, consult the SOLID registry – unit tests may mock it
        try:
            from portfolio_backtester.strategies._core.registry import get_strategy_registry

            registry = get_strategy_registry()
            registered = registry.get_all_strategies()
            if strategy_class in registered.keys():
                return True
        except Exception:
            # Non-fatal: fall back to introspection below
            pass

        # Import all strategy modules to register classes
        self._import_all_strategy_modules()

        # Get all BaseStrategy subclasses
        from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy

        all_strategies = self._collect_all_subclasses(BaseStrategy)

        # Check if the strategy class name matches any concrete strategy
        for strategy_cls in all_strategies:
            if strategy_cls.__name__ == strategy_class and self._is_concrete_strategy_class(
                strategy_cls
            ):
                return True

        return False

    def _is_meta_strategy_config(self, config_data: Dict[str, Any]) -> bool:
        """Check if a config is for a meta-strategy."""
        strategy_name = config_data.get("strategy", "")

        # Check if strategy name suggests meta-strategy
        if "meta" in strategy_name.lower():
            return True

        # Check if strategy_params contains allocations
        strategy_params = config_data.get("strategy_params", {})
        if "allocations" in strategy_params:
            return True

        # Try to resolve and check type
        resolved_strategy = self.strategy_resolver.resolve_strategy(strategy_name)
        if resolved_strategy:
            return self.strategy_resolver.is_meta_strategy(resolved_strategy)

        return False

    def _check_meta_strategy_allocations(
        self, yaml_file: Path, config_data: Dict[str, Any], valid_strategy_names: Set[str]
    ) -> List[str]:
        """Check meta-strategy allocations for invalid strategy references."""
        errors: List[str] = []

        strategy_params = config_data.get("strategy_params", {})
        allocations = strategy_params.get("allocations")

        if not allocations or not isinstance(allocations, list):
            return errors

        for i, allocation in enumerate(allocations):
            if not isinstance(allocation, dict):
                continue

            strategy_id = allocation.get("strategy_id")
            if strategy_id and isinstance(strategy_id, str):
                if not self._is_valid_strategy_reference(strategy_id, valid_strategy_names):
                    errors.append(
                        f"Invalid strategy reference in meta-strategy allocation "
                        f"{yaml_file.relative_to(self.config_scenarios_dir)} "
                        f"(allocation #{i+1}): strategy_id '{strategy_id}' does not exist"
                    )

        return errors

    def _check_strategy_code_references(self) -> List[str]:
        """Check for invalid strategy references within strategy code itself."""
        errors = []

        # This is more complex and would require parsing Python code
        # For now, focus on common patterns in meta-strategies and configuration

        # Find Python files in strategy directories
        strategy_files = list(self.src_strategies_dir.rglob("*.py"))

        for py_file in strategy_files:
            # Skip __init__.py and other utility files
            if py_file.name in ["__init__.py", "__pycache__"] or "__pycache__" in str(py_file):
                continue

            file_errors = self._check_python_file_strategy_references(py_file)
            errors.extend(file_errors)

        return errors

    def _check_python_file_strategy_references(self, py_file: Path) -> List[str]:
        """Check strategy references in a Python file."""
        errors: List[str] = []

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            logger.warning(f"Could not read Python file {py_file}: {e}")
            return errors

        # Look for common patterns of strategy references
        # This is a simple regex-based approach - could be enhanced with AST parsing

        import re

        # Look for string literals that might be strategy names
        # Common patterns: _resolve_strategy("strategy_name"), "strategy": "name", etc.
        patterns = [
            r'_resolve_strategy\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'"strategy"\s*:\s*["\']([^"\']+)["\']',
            r"'strategy'\s*:\s*[\"']([^\"']+)[\"']",
            r'strategy_name\s*=\s*["\']([^"\']+)["\']',
        ]

        valid_strategy_names = self._get_valid_strategy_names()

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                strategy_name = match.group(1)

                # Skip obviously non-strategy strings
                if (
                    len(strategy_name) < 3
                    or strategy_name.startswith("_")
                    or strategy_name in ["test", "debug"]
                ):
                    continue

                if not self._is_valid_strategy_reference(strategy_name, valid_strategy_names):
                    line_num = content[: match.start()].count("\n") + 1
                    errors.append(
                        f"Potential invalid strategy reference in {py_file.relative_to(self.src_strategies_dir)} "
                        f"(line {line_num}): '{strategy_name}' may not exist"
                    )

        return errors


def validate_strategy_config_cross_references(
    src_strategies_dir: str, config_scenarios_dir: str
) -> Tuple[bool, List[str]]:
    """Convenience function to run comprehensive cross-reference validation.

    Args:
        src_strategies_dir: Path to src/portfolio_backtester/strategies directory
        config_scenarios_dir: Path to config/scenarios directory

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    validator = StrategyConfigCrossValidator(src_strategies_dir, config_scenarios_dir)
    return validator.validate_cross_references()


if __name__ == "__main__":
    # Example usage for testing
    import sys

    if len(sys.argv) >= 3:
        src_dir = sys.argv[1]
        config_dir = sys.argv[2]

        is_valid, errors = validate_strategy_config_cross_references(src_dir, config_dir)

        if is_valid:
            print("✅ All strategy configuration cross-references are valid!")
        else:
            print(f"❌ Found {len(errors)} cross-reference validation errors:")
            for i, error in enumerate(errors, 1):
                print(f"{i:3d}. {error}")

        sys.exit(0 if is_valid else 1)
    else:
        print(
            "Usage: python strategy_config_cross_validator.py <src_strategies_dir> <config_scenarios_dir>"
        )
        sys.exit(1)
