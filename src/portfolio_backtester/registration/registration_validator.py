"""Registration validator for validating component registration data.

This module implements comprehensive validation logic for component registration
following the Single Responsibility Principle. It focuses solely on validation
of registration parameters, data integrity, and business rules.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RegistrationValidator:
    """Validates registration data and enforces registration rules.

    This class is responsible for:
    - Validating registration data structure and types
    - Enforcing naming conventions and constraints
    - Validating component metadata
    - Checking business rules for registration
    - Providing detailed validation error messages

    Follows SRP by focusing solely on validation logic.
    """

    def __init__(
        self,
        *,
        custom_validators: Optional[List[Callable[[Dict[str, Any]], List[str]]]] = None,
        naming_pattern: Optional[str] = None,
        reserved_names: Optional[Set[str]] = None,
        max_alias_count: int = 10,
    ):
        """Initialize the registration validator.

        Args:
            custom_validators: Optional list of custom validation functions
            naming_pattern: Optional regex pattern for valid names
            reserved_names: Optional set of reserved names that cannot be used
            max_alias_count: Maximum number of aliases allowed per component
        """
        self._custom_validators = custom_validators or []
        self._naming_pattern = re.compile(naming_pattern) if naming_pattern else None
        self._reserved_names = reserved_names or set()
        self._max_alias_count = max_alias_count

        logger.debug("RegistrationValidator initialized")

    def validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate registration data.

        Args:
            data: The registration data to validate

        Returns:
            List of validation error messages
        """
        errors: List[str] = []

        # Basic structure validation
        errors.extend(self._validate_basic_structure(data))

        # Name validation
        if "name" in data:
            errors.extend(self._validate_name(data["name"]))

        # Component validation
        if "component" in data:
            errors.extend(self._validate_component(data["component"]))

        # Aliases validation
        if "aliases" in data:
            errors.extend(self._validate_aliases(data["aliases"], data.get("name")))

        # Metadata validation
        if "metadata" in data:
            errors.extend(self._validate_metadata(data["metadata"]))

        # Custom validation
        for validator in self._custom_validators:
            try:
                custom_errors = validator(data)
                if custom_errors:
                    errors.extend(custom_errors)
            except Exception as e:
                errors.append(f"Custom validation failed: {e}")
                logger.warning(f"Custom validator failed: {e}")

        return errors

    def _validate_basic_structure(self, data: Any) -> List[str]:
        """Validate basic data structure requirements.

        Args:
            data: Registration data

        Returns:
            List of validation errors
        """
        errors: List[str] = []

        if not isinstance(data, dict):
            errors.append("Registration data must be a dictionary")
            return errors

        # Required fields
        if "name" not in data:
            errors.append("Missing required 'name' field")

        if "component" not in data:
            errors.append("Missing required 'component' field")

        return errors

    def _validate_name(self, name: Any) -> List[str]:
        """Validate component name.

        Args:
            name: Component name to validate

        Returns:
            List of validation errors
        """
        errors: List[str] = []

        if not isinstance(name, str):
            errors.append(f"Name must be a string, got {type(name).__name__}")
            return errors

        if not name.strip():
            errors.append("Name cannot be empty or whitespace only")
            return errors

        # Check reserved names
        if name.lower() in {n.lower() for n in self._reserved_names}:
            errors.append(f"Name '{name}' is reserved and cannot be used")

        # Check naming pattern
        if self._naming_pattern and not self._naming_pattern.match(name):
            errors.append(f"Name '{name}' does not match required pattern")

        # Length validation
        if len(name) > 100:
            errors.append("Name cannot exceed 100 characters")

        # Basic character validation
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", name):
            errors.append("Name can only contain letters, numbers, underscores, hyphens, and dots")

        return errors

    def _validate_component(self, component: Any) -> List[str]:
        """Validate the component itself.

        Args:
            component: Component to validate

        Returns:
            List of validation errors
        """
        errors: List[str] = []

        if component is None:
            errors.append("Component cannot be None")

        # Additional component-specific validation can be added here
        # For example, checking if component has required methods/attributes

        return errors

    def _validate_aliases(self, aliases: Any, component_name: Optional[str] = None) -> List[str]:
        """Validate component aliases.

        Args:
            aliases: Aliases to validate
            component_name: Name of the component (for reference)

        Returns:
            List of validation errors
        """
        errors: List[str] = []

        if not isinstance(aliases, list):
            errors.append(f"Aliases must be a list, got {type(aliases).__name__}")
            return errors

        if len(aliases) > self._max_alias_count:
            errors.append(f"Too many aliases: {len(aliases)} (max: {self._max_alias_count})")

        seen_aliases: Set[str] = set()
        for i, alias in enumerate(aliases):
            if not isinstance(alias, str):
                errors.append(f"Alias at index {i} must be a string, got {type(alias).__name__}")
                continue

            if not alias.strip():
                errors.append(f"Alias at index {i} cannot be empty or whitespace only")
                continue

            # Check for duplicate aliases
            alias_lower = alias.lower()
            if alias_lower in seen_aliases:
                errors.append(f"Duplicate alias: '{alias}'")
            seen_aliases.add(alias_lower)

            # Alias cannot be the same as component name
            if component_name and alias.lower() == component_name.lower():
                errors.append(f"Alias '{alias}' cannot be the same as component name")

            # Apply same validation rules as names
            alias_errors = self._validate_name(alias)
            for error in alias_errors:
                errors.append(f"Alias '{alias}': {error}")

        return errors

    def _validate_metadata(self, metadata: Any) -> List[str]:
        """Validate component metadata.

        Args:
            metadata: Metadata to validate

        Returns:
            List of validation errors
        """
        errors: List[str] = []

        if not isinstance(metadata, dict):
            errors.append(f"Metadata must be a dictionary, got {type(metadata).__name__}")
            return errors

        # Validate metadata keys and values
        for key, value in metadata.items():
            if not isinstance(key, str):
                errors.append(f"Metadata key must be string, got {type(key).__name__}: {key}")

            # Validate specific metadata fields if needed
            if key == "version" and value is not None:
                if not isinstance(value, str):
                    errors.append(f"Metadata 'version' must be string, got {type(value).__name__}")

            if key == "description" and value is not None:
                if not isinstance(value, str):
                    errors.append(
                        f"Metadata 'description' must be string, got {type(value).__name__}"
                    )
                elif len(value) > 1000:
                    errors.append("Metadata 'description' cannot exceed 1000 characters")

        return errors

    def add_custom_validator(self, validator: Callable[[Dict[str, Any]], List[str]]) -> None:
        """Add a custom validator function.

        Args:
            validator: Function that takes registration data and returns list of errors
        """
        self._custom_validators.append(validator)
        logger.debug("Added custom validator")

    def set_naming_pattern(self, pattern: str) -> None:
        """Set a regex pattern for valid names.

        Args:
            pattern: Regex pattern string
        """
        try:
            self._naming_pattern = re.compile(pattern)
            logger.debug(f"Set naming pattern: {pattern}")
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def add_reserved_name(self, name: str) -> None:
        """Add a reserved name that cannot be used.

        Args:
            name: Name to reserve
        """
        self._reserved_names.add(name)
        logger.debug(f"Added reserved name: {name}")

    def remove_reserved_name(self, name: str) -> bool:
        """Remove a reserved name.

        Args:
            name: Name to unreserve

        Returns:
            True if name was removed, False if not found
        """
        if name in self._reserved_names:
            self._reserved_names.remove(name)
            logger.debug(f"Removed reserved name: {name}")
            return True
        return False
