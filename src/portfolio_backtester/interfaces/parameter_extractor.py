"""
Parameter extraction interfaces for configuration module polymorphism.

This module provides interfaces to replace isinstance checks in config_initializer.py
with polymorphic strategies, following the Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from typing import Set, Any


class ParameterExtractor(ABC):
    """Interface for extracting tunable parameters from strategy specifications."""

    @abstractmethod
    def can_handle(self, strategy_spec: Any) -> bool:
        """Check if this extractor can handle the given strategy specification."""
        pass

    @abstractmethod
    def extract_parameters(self, strategy_spec: Any) -> Set[str]:
        """Extract tunable parameters from the strategy specification."""
        pass


class StringParameterExtractor(ParameterExtractor):
    """Extracts parameters from string strategy specifications."""

    def can_handle(self, strategy_spec: Any) -> bool:
        """Check if the strategy spec is a string."""
        return isinstance(strategy_spec, str)

    def extract_parameters(self, strategy_spec: Any) -> Set[str]:
        """Extract parameters from string strategy specification."""
        if not strategy_spec:
            return set()

        # Resolve using polymorphic resolver so tests can mock _resolve_strategy
        try:
            from ..utils import _resolve_strategy
        except Exception:
            return set()

        strat = _resolve_strategy(str(strategy_spec))
        if strat is None:
            return set()

        try:
            tunables = getattr(strat, "tunable_parameters", None)
            if callable(tunables):
                result = tunables()
                return set(result) if not isinstance(result, dict) else set(result.keys())
        except Exception:
            return set()
        return set()


class DictParameterExtractor(ParameterExtractor):
    """Extracts parameters from dictionary strategy specifications."""

    def can_handle(self, strategy_spec: Any) -> bool:
        """Check if the strategy spec is a dictionary."""
        return isinstance(strategy_spec, dict)

    def extract_parameters(self, strategy_spec: Any) -> Set[str]:
        """Extract parameters from dictionary strategy specification."""
        if not isinstance(strategy_spec, dict):
            return set()

        strategy_name = (
            strategy_spec.get("name") or strategy_spec.get("strategy") or strategy_spec.get("type")
        )

        if not strategy_name:
            return set()

        # Resolve using polymorphic resolver so tests can mock _resolve_strategy
        try:
            from ..utils import _resolve_strategy
        except Exception:
            return set()

        strat = _resolve_strategy(str(strategy_name))
        if strat is None:
            return set()

        try:
            tunables = getattr(strat, "tunable_parameters", None)
            if callable(tunables):
                result = tunables()
                return set(result) if not isinstance(result, dict) else set(result.keys())
        except Exception:
            return set()
        return set()


class ParameterExtractorFactory:
    """Factory for creating appropriate parameter extractors."""

    def __init__(self):
        self._extractors = [
            StringParameterExtractor(),
            DictParameterExtractor(),
        ]

    def get_extractor(self, strategy_spec: Any) -> ParameterExtractor:
        """Get the appropriate parameter extractor for the given strategy spec."""
        for extractor in self._extractors:
            if extractor.can_handle(strategy_spec):
                return extractor

        # Fallback to string extractor for unknown types
        return StringParameterExtractor()

    def extract_parameters(self, strategy_spec: Any) -> Set[str]:
        """Extract parameters using the appropriate extractor."""
        extractor = self.get_extractor(strategy_spec)
        return extractor.extract_parameters(strategy_spec)
