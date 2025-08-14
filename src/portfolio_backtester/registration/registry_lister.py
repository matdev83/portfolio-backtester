"""Registry lister for querying and listing registered components.

This module implements comprehensive listing and querying functionality for
registered components following the Single Responsibility Principle. It focuses
solely on read-only operations for discovering and inspecting components.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RegistryLister:
    """Lists and queries registered components.

    This class is responsible for:
    - Listing all registered components
    - Filtering components by various criteria
    - Searching components by name patterns
    - Providing component information and metadata
    - Generating registry reports and statistics

    Follows SRP by focusing solely on read-only registry queries and listings.
    """

    def __init__(self, registry_manager):
        """Initialize the registry lister.

        Args:
            registry_manager: The RegistrationManager instance to query
        """
        self._manager = registry_manager
        logger.debug("RegistryLister initialized")

    def list_components(self, *, include_aliases: bool = False) -> List[str]:
        """List all registered component names.

        Args:
            include_aliases: If True, include aliases in the list

        Returns:
            List of registered component names (and optionally aliases)
        """
        names = list(self._manager._registry.keys())

        if include_aliases:
            aliases = list(self._manager._aliases.keys())
            names.extend(aliases)

        return sorted(names)

    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a component.

        Args:
            name: Name or alias of the component

        Returns:
            Dictionary with component information or None if not found
        """
        # Resolve alias to actual name if needed
        actual_name = self._manager._aliases.get(name, name)

        if actual_name not in self._manager._registry:
            return None

        component = self._manager._registry[actual_name]
        metadata = self._manager._metadata.get(actual_name, {})

        # Find all aliases for this component
        aliases = [
            alias for alias, target in self._manager._aliases.items() if target == actual_name
        ]

        return {
            "name": actual_name,
            "component": component,
            "component_type": type(component).__name__,
            "component_module": getattr(type(component), "__module__", "unknown"),
            "aliases": aliases,
            "metadata": metadata,
            "is_alias": name != actual_name,
            "resolved_name": actual_name if name != actual_name else None,
        }

    def search_components(self, pattern: str, *, case_sensitive: bool = False) -> List[str]:
        """Search for components by name pattern.

        Args:
            pattern: Search pattern (supports wildcards * and ?)
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of matching component names
        """
        import fnmatch

        all_names = self.list_components(include_aliases=True)

        if case_sensitive:
            matches = fnmatch.filter(all_names, pattern)
        else:
            pattern_lower = pattern.lower()
            matches = fnmatch.filter([name.lower() for name in all_names], pattern_lower)
            # Map back to original case
            matches = [name for name in all_names if name.lower() in matches]

        return sorted(matches)

    def filter_components(
        self, criterion: Union[Callable[[Any], bool], Dict[str, Any]]
    ) -> List[str]:
        """Filter components by criterion.

        Args:
            criterion: Either a callable that evaluates components, or a dict
                      specifying metadata criteria to match

        Returns:
            List of names for components meeting the criterion
        """
        matching = []

        for name in self._manager._registry.keys():
            component = self._manager._registry[name]

            if callable(criterion):
                # Use the callable criterion
                try:
                    if criterion(component):
                        matching.append(name)
                except Exception as e:
                    logger.warning(f"Filter criterion failed for component '{name}': {e}")

            elif isinstance(criterion, dict):
                # Use metadata-based filtering
                metadata = self._manager._metadata.get(name, {})
                if self._matches_metadata_criteria(metadata, criterion):
                    matching.append(name)

        return sorted(matching)

    def filter_by_type(self, component_type: type) -> List[str]:
        """Filter components by their type.

        Args:
            component_type: Type to filter by

        Returns:
            List of component names of the specified type
        """
        return self.filter_components(lambda comp: isinstance(comp, component_type))

    def filter_by_metadata(self, **metadata_criteria: Any) -> List[str]:
        """Filter components by metadata criteria.

        Args:
            **metadata_criteria: Key-value pairs to match in metadata

        Returns:
            List of component names matching the metadata criteria
        """
        return self.filter_components(metadata_criteria)

    def get_components_with_aliases(self) -> Dict[str, List[str]]:
        """Get a mapping of component names to their aliases.

        Returns:
            Dictionary mapping component names to lists of their aliases
        """
        result = {}

        for name in self._manager._registry.keys():
            aliases = [alias for alias, target in self._manager._aliases.items() if target == name]
            result[name] = aliases

        return result

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the registry.

        Returns:
            Dictionary with registry summary information
        """
        stats = self._manager.get_registry_stats()

        # Get type distribution
        type_counts: Dict[str, int] = {}
        for component in self._manager._registry.values():
            type_name = type(component).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Get components with most aliases
        alias_counts = {}
        for name in self._manager._registry.keys():
            aliases = [alias for alias, target in self._manager._aliases.items() if target == name]
            if aliases:
                alias_counts[name] = len(aliases)

        return {
            "statistics": stats,
            "type_distribution": type_counts,
            "components_with_aliases": len(alias_counts),
            "most_aliased": dict(
                sorted(alias_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "all_names": self.list_components(),
            "all_aliases": list(self._manager._aliases.keys()),
        }

    def list_by_type(self) -> Dict[str, List[str]]:
        """List components grouped by their type.

        Returns:
            Dictionary mapping type names to lists of component names
        """
        result: Dict[str, List[str]] = {}

        for name, component in self._manager._registry.items():
            type_name = type(component).__name__
            if type_name not in result:
                result[type_name] = []
            result[type_name].append(name)

        # Sort component names within each type
        for type_name in result:
            result[type_name].sort()

        return result

    def find_similar_names(self, name: str, max_results: int = 5) -> List[Tuple[str, float]]:
        """Find components with names similar to the given name.

        Args:
            name: Name to find similar matches for
            max_results: Maximum number of results to return

        Returns:
            List of tuples (component_name, similarity_score) sorted by similarity
        """
        from difflib import SequenceMatcher

        all_names = self.list_components(include_aliases=True)
        similarities = []

        for other_name in all_names:
            if other_name == name:
                continue

            similarity = SequenceMatcher(None, name.lower(), other_name.lower()).ratio()
            similarities.append((other_name, similarity))

        # Sort by similarity score (descending) and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def validate_registry_integrity(self) -> List[str]:
        """Validate the integrity of the registry.

        Returns:
            List of integrity issues found
        """
        issues = []

        # Check for orphaned aliases
        for alias, target in self._manager._aliases.items():
            if target not in self._manager._registry:
                issues.append(
                    f"Orphaned alias '{alias}' points to non-existent component '{target}'"
                )

        # Check for orphaned metadata
        for name in self._manager._metadata.keys():
            if name not in self._manager._registry:
                issues.append(f"Orphaned metadata for non-existent component '{name}'")

        # Check for None components
        for name, component in self._manager._registry.items():
            if component is None:
                issues.append(f"Component '{name}' is None")

        return issues

    def _matches_metadata_criteria(
        self, metadata: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the given criteria.

        Args:
            metadata: Component metadata
            criteria: Criteria to match

        Returns:
            True if metadata matches all criteria
        """
        for key, expected_value in criteria.items():
            if key not in metadata:
                return False

            actual_value = metadata[key]

            # Support different matching modes
            if isinstance(expected_value, dict) and "$regex" in expected_value:
                # Regex matching
                import re

                pattern = expected_value["$regex"]
                if not re.search(pattern, str(actual_value)):
                    return False
            elif isinstance(expected_value, dict) and "$in" in expected_value:
                # Value in list matching
                if actual_value not in expected_value["$in"]:
                    return False
            else:
                # Exact matching
                if actual_value != expected_value:
                    return False

        return True
