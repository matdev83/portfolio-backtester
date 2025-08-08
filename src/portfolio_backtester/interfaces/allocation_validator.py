"""
Allocation validator interface for meta strategy validation.

This module provides interfaces for validating meta strategy allocations
without using isinstance checks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from portfolio_backtester.yaml_validator import YamlError, YamlErrorType


class IAllocationValidator(ABC):
    """Interface for validating meta strategy allocations."""
    
    @abstractmethod
    def validate_meta_strategy_logic(
        self, scenario_data: Dict[str, Any], strategy_class: Any, file_path_str: Optional[str]
    ) -> List[Any]:
        """Validate meta strategy specific configuration."""
        pass
    
    @abstractmethod
    def validate_allocations(self, allocations: Any, file_path_str: Optional[str]) -> List[Any]:
        """Validate meta strategy allocations structure."""
        pass


class DefaultAllocationValidator(IAllocationValidator):
    """Default implementation of allocation validator."""
    
    def validate_meta_strategy_logic(
        self, scenario_data: Dict[str, Any], strategy_class: Any, file_path_str: Optional[str]
    ) -> List[Any]:
        """Validate meta strategy specific configuration."""
        # YamlError and YamlErrorType imported at module level
        errors = []
        
        # Check for universe_config in meta strategies
        if "universe_config" in scenario_data:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        "Meta strategies cannot define 'universe_config'. "
                        "Meta strategies inherit their universe from their sub-strategies."
                    ),
                    file_path=file_path_str,
                )
            )
        
        # Check for universe in meta strategies
        if "universe" in scenario_data:
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        "Meta strategies cannot define 'universe'. "
                        "Meta strategies inherit their universe from their sub-strategies."
                    ),
                    file_path=file_path_str,
                )
            )
            
        # Check for allocations in strategy_params
        strategy_params = scenario_data.get("strategy_params", {})
        strategy_name = scenario_data.get("strategy", "")
        
        if isinstance(strategy_params, dict):
            has_allocations = False
            for key in strategy_params:
                if key.endswith(".allocations") or key == "allocations":
                    has_allocations = True
                    break
                    
            if not has_allocations:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Meta strategy '{strategy_name}' is missing required 'allocations' parameter. "
                            "Meta strategies must define allocations for their sub-strategies."
                        ),
                        file_path=file_path_str,
                    )
                )
        
        return errors
    
    def validate_allocations(self, allocations: Any, file_path_str: Optional[str]) -> List[Any]:
        """Validate meta strategy allocations structure."""
        # YamlError and YamlErrorType imported at module level
        errors = []
        
        if not isinstance(allocations, dict) and not isinstance(allocations, list):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        "Meta strategy allocations must be a dictionary or list. "
                        f"Got {type(allocations).__name__} instead."
                    ),
                    file_path=file_path_str,
                )
            )
            return errors
            
        if isinstance(allocations, dict):
            # Check that all values are numeric
            for strategy, weight in allocations.items():
                if not isinstance(weight, (int, float)):
                    errors.append(
                        YamlError(
                            error_type=YamlErrorType.VALIDATION_ERROR,
                            message=(
                                f"Allocation weight for '{strategy}' must be a number. "
                                f"Got {type(weight).__name__} instead."
                            ),
                            file_path=file_path_str,
                        )
                    )
        
        return errors


class AllocationValidatorFactory:
    """Factory for creating allocation validators."""
    
    @staticmethod
    def create() -> IAllocationValidator:
        """Create a new allocation validator instance."""
        return DefaultAllocationValidator()