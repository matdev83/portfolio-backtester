"""Registration manager for handling component registrations.

This module implements the registration management functionality following
the Single Responsibility Principle. It manages the lifecycle of component
registrations including registration, deregistration, and retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .registration_validator import RegistrationValidator

logger = logging.getLogger(__name__)


class RegistrationManager:
    """Manages the registration and lifecycle of components.

    This class is responsible for:
    - Registering components with unique names
    - Deregistering components
    - Retrieving registered components
    - Managing component metadata
    - Ensuring registration integrity

    Follows SRP by focusing solely on registration management operations.
    """

    def __init__(self, validator: Optional[RegistrationValidator] = None):
        """Initialize the registration manager.

        Args:
            validator: Optional validator for registration data validation
        """
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._validator = validator or RegistrationValidator()
        self._aliases: Dict[str, str] = {}

        logger.debug("RegistrationManager initialized")

    def register(
        self,
        name: str,
        component: Any,
        *,
        aliases: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """Register a component with a unique name.

        Args:
            name: Unique name for the component
            component: Component to register
            aliases: Optional list of alternative names
            metadata: Optional metadata for the component
            force: If True, overwrite existing registration

        Raises:
            ValueError: If name is already registered and force=False
            ValueError: If validation fails
        """
        if not name or not isinstance(name, str):
            raise ValueError("Component name must be a non-empty string")

        # Validate registration data
        validation_data = {
            "name": name,
            "component": component,
            "aliases": aliases or [],
            "metadata": metadata or {},
        }

        validation_errors = self._validator.validate(validation_data)
        if validation_errors:
            error_msg = "; ".join(validation_errors)
            raise ValueError(f"Registration validation failed: {error_msg}")

        # Check for existing registration
        if name in self._registry and not force:
            raise ValueError(
                f"Component '{name}' is already registered. Use force=True to overwrite."
            )

        # Register the component
        self._registry[name] = component
        self._metadata[name] = metadata or {}

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases and not force:
                    raise ValueError(
                        f"Alias '{alias}' is already in use. Use force=True to overwrite."
                    )
                self._aliases[alias] = name
                logger.debug(f"Registered alias: {alias} -> {name}")

        logger.info(f"Successfully registered component: {name}")

    def deregister(self, name: str) -> bool:
        """Deregister a component by name.

        Args:
            name: Name of the component to deregister

        Returns:
            True if component was deregistered, False if not found
        """
        if name not in self._registry:
            logger.warning(f"Component '{name}' is not registered")
            return False

        # Remove the component and its metadata
        del self._registry[name]
        self._metadata.pop(name, None)

        # Remove any aliases pointing to this component
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
            logger.debug(f"Removed alias: {alias}")

        logger.info(f"Successfully deregistered component: {name}")
        return True

    def get_component(self, name: str) -> Optional[Any]:
        """Retrieve a component by name or alias.

        Args:
            name: Name or alias of the component

        Returns:
            The registered component or None if not found
        """
        # Check direct registry first
        if name in self._registry:
            return self._registry[name]

        # Check aliases
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._registry.get(actual_name)

        return None

    def is_registered(self, name: str) -> bool:
        """Check if a component is registered.

        Args:
            name: Name or alias to check

        Returns:
            True if the component is registered
        """
        return name in self._registry or name in self._aliases

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered component.

        Args:
            name: Name or alias of the component

        Returns:
            Component metadata or None if not found
        """
        # Resolve alias to actual name if needed
        actual_name = self._aliases.get(name, name)
        return self._metadata.get(actual_name)

    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a registered component.

        Args:
            name: Name or alias of the component
            metadata: New metadata to set

        Returns:
            True if metadata was updated, False if component not found
        """
        # Resolve alias to actual name if needed
        actual_name = self._aliases.get(name, name)

        if actual_name not in self._registry:
            return False

        self._metadata[actual_name] = metadata
        logger.debug(f"Updated metadata for component: {actual_name}")
        return True

    def get_registry_stats(self) -> Dict[str, int]:
        """Get statistics about the registry.

        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_components": len(self._registry),
            "total_aliases": len(self._aliases),
            "components_with_metadata": len([k for k, v in self._metadata.items() if v]),
        }

    def clear_registry(self) -> None:
        """Clear all registrations (for testing/cleanup)."""
        self._registry.clear()
        self._metadata.clear()
        self._aliases.clear()
        logger.info("Registry cleared")
