"""
Provider Interface Enforcement Module

This module provides mechanisms to enforce the use of provider interfaces
and prevent direct usage of legacy functions in new code.

**REQUIRED USAGE PATTERNS:**
1. Use strategy.get_universe_provider() instead of resolve_universe_config()
2. Use strategy.get_position_sizer_provider() instead of get_position_sizer_from_config()
3. Use strategy.get_stop_loss_provider() instead of direct stop loss instantiation
4. Always pass strategy parameter to size_positions() function

**ENFORCEMENT RULES:**
- New code MUST NOT import legacy functions directly
- All strategy workflow functions MUST use provider interfaces
- Tests MUST use strategy.get_* methods instead of direct function calls
"""

import warnings
from typing import Any, Dict
from functools import wraps


class LegacyUsageError(Exception):
    """Raised when legacy functions are used instead of provider interfaces."""

    pass


class ProviderEnforcementWarning(UserWarning):
    """Warning for deprecated legacy usage."""

    pass


def deprecated_legacy_function(replacement_guidance: str):
    """
    Decorator to mark legacy functions as deprecated and guide users to provider system.

    Args:
        replacement_guidance: Instructions on how to use the provider system instead
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = (
                f"DEPRECATED: {func.__name__} is deprecated. "
                f"Use the provider system instead: {replacement_guidance}"
            )
            warnings.warn(warning_msg, ProviderEnforcementWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def enforce_strategy_parameter(func):
    """
    Decorator to enforce that workflow functions receive strategy parameter.

    This ensures functions like size_positions() are called with strategy object
    so they can use provider interfaces.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if strategy is provided as positional or keyword argument
        strategy_provided = (
            len(args) >= 7 and args[6] is not None
        ) or (  # strategy as 7th positional arg
            "strategy" in kwargs and kwargs["strategy"] is not None
        )

        if not strategy_provided:
            raise LegacyUsageError(
                f"{func.__name__} must be called with strategy parameter to use provider interfaces. "
                f"Example: {func.__name__}(..., strategy=strategy_instance)"
            )

        return func(*args, **kwargs)

    return wrapper


def validate_provider_usage(strategy_instance: Any) -> Dict[str, bool]:
    """
    Validate that a strategy instance properly uses provider interfaces.

    Args:
        strategy_instance: Strategy instance to validate

    Returns:
        Dict mapping provider types to their initialization status

    Raises:
        LegacyUsageError: If providers are not properly initialized
    """
    validation_results = {}

    try:
        # Test universe provider
        universe_provider = strategy_instance.get_universe_provider()
        validation_results["universe_provider"] = universe_provider is not None

        # Test position sizer provider
        position_sizer_provider = strategy_instance.get_position_sizer_provider()
        validation_results["position_sizer_provider"] = position_sizer_provider is not None

        # Test stop loss provider
        stop_loss_provider = strategy_instance.get_stop_loss_provider()
        validation_results["stop_loss_provider"] = stop_loss_provider is not None

    except (AttributeError, RuntimeError) as e:
        raise LegacyUsageError(
            f"Strategy instance does not properly support provider interfaces: {e}"
        ) from e

    # Check if all providers are initialized
    if not all(validation_results.values()):
        missing_providers = [k for k, v in validation_results.items() if not v]
        raise LegacyUsageError(f"Strategy missing required providers: {missing_providers}")

    return validation_results


# Registry of allowed provider patterns
ALLOWED_PROVIDER_IMPORTS = {
    "from ...interfaces.universe_provider_interface import UniverseProviderFactory",
    "from ...interfaces.position_sizer_provider_interface import PositionSizerProviderFactory",
    "from ...interfaces.stop_loss_provider_interface import StopLossProviderFactory",
}

# Registry of forbidden legacy patterns
FORBIDDEN_LEGACY_PATTERNS = {
    "from ..universe_resolver import resolve_universe_config",
    "from ..portfolio.position_sizer import get_position_sizer_from_config",
    "resolve_universe_config(",
    "get_position_sizer_from_config(",
}


def check_code_for_legacy_usage(code_content: str, file_path: str = "unknown") -> None:
    """
    Check code content for prohibited legacy usage patterns.

    Args:
        code_content: String content of code to check
        file_path: Path to file being checked (for error reporting)

    Raises:
        LegacyUsageError: If legacy patterns are found
    """
    violations = []

    for line_num, line in enumerate(code_content.splitlines(), 1):
        line_stripped = line.strip()

        # Skip comments and empty lines
        if not line_stripped or line_stripped.startswith("#"):
            continue

        # Check for forbidden patterns
        for pattern in FORBIDDEN_LEGACY_PATTERNS:
            if pattern in line_stripped:
                violations.append(f"Line {line_num}: {line_stripped}")

    if violations:
        raise LegacyUsageError(
            f"Legacy usage patterns found in {file_path}:\n"
            + "\n".join(violations)
            + "\n\nUse provider interfaces instead. See interfaces/enforcement.py for guidance."
        )


# Export enforcement utilities
__all__ = [
    "LegacyUsageError",
    "ProviderEnforcementWarning",
    "deprecated_legacy_function",
    "enforce_strategy_parameter",
    "validate_provider_usage",
    "check_code_for_legacy_usage",
    "ALLOWED_PROVIDER_IMPORTS",
    "FORBIDDEN_LEGACY_PATTERNS",
]
