"""
Factory pattern for creating parameter generators.

This module implements the factory pattern for creating parameter generators,
providing a unified interface for instantiating different optimization backends
with proper error handling and extensibility.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


from .parameter_generator import ParameterGeneratorError as BaseParameterGeneratorError


class ParameterGeneratorError(BaseParameterGeneratorError):
    """Base exception for parameter generator creation errors."""
    pass


class UnknownOptimizerError(ParameterGeneratorError):
    """Raised when an unknown optimizer type is requested."""
    pass


class OptimizerImportError(ParameterGeneratorError):
    """Raised when an optimizer's dependencies cannot be imported."""
    pass


def create_parameter_generator(
    optimizer_type: str, 
    random_state: Optional[int] = None,
    **kwargs: Any
) -> 'ParameterGenerator':
    """Factory function to create parameter generators.
    
    This function creates parameter generator instances based on the optimizer type
    string. It handles proper initialization with random state for reproducible
    results and provides clear error messages for unknown types.
    
    Args:
        optimizer_type: String identifier for the optimizer type
                       Supported values: "optuna", "genetic", "mock"
        random_state: Random seed for reproducible results (optional)
        **kwargs: Additional keyword arguments passed to the generator constructor
        
    Returns:
        ParameterGenerator: Instance of the requested parameter generator
        
    Raises:
        UnknownOptimizerError: If the optimizer type is not supported
        OptimizerImportError: If the optimizer's dependencies cannot be imported
        ParameterGeneratorError: For other parameter generator creation errors
        
    Examples:
        >>> # Create Optuna parameter generator
        >>> generator = create_parameter_generator("optuna", random_state=42)
        
        >>> # Create genetic algorithm parameter generator
        >>> generator = create_parameter_generator("genetic", random_state=123)
        
        >>> # Use optuna for testing
        >>> generator = create_parameter_generator("optuna", random_state=456)
    """
    if not isinstance(optimizer_type, str):
        raise ParameterGeneratorError(
            f"optimizer_type must be a string, got {type(optimizer_type)}"
        )
    
    optimizer_type = optimizer_type.lower().strip()
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Creating parameter generator: type={optimizer_type}, "
            f"random_state={random_state}, kwargs={kwargs}"
        )
    
    try:
        if optimizer_type == "optuna":
            return _create_optuna_generator(random_state, **kwargs)
        elif optimizer_type == "genetic":
            return _create_genetic_generator(random_state, **kwargs)
        # Mock generator removed - use optuna or genetic for testing
        else:
            available_types = ["optuna", "genetic"]
            raise UnknownOptimizerError(
                f"Unknown optimizer type: '{optimizer_type}'. "
                f"Available types: {available_types}"
            )
    
    except (ImportError, ModuleNotFoundError) as e:
        raise OptimizerImportError(
            f"Failed to import dependencies for optimizer '{optimizer_type}': {e}"
        ) from e
    except Exception as e:
        if isinstance(e, (UnknownOptimizerError, OptimizerImportError)):
            raise
        raise ParameterGeneratorError(
            f"Failed to create parameter generator '{optimizer_type}': {e}"
        ) from e


def _create_optuna_generator(random_state: Optional[int], **kwargs) -> 'ParameterGenerator':
    """Create an Optuna parameter generator.
    
    Args:
        random_state: Random seed for reproducible results
        **kwargs: Additional arguments for OptunaParameterGenerator
        
    Returns:
        OptunaParameterGenerator instance
    """
    try:
        from .generators.optuna_generator import OptunaParameterGenerator
        return OptunaParameterGenerator(random_state=random_state, **kwargs)
    except ImportError as e:
        raise OptimizerImportError(
            "Optuna parameter generator requires 'optuna' package. "
            "Install with: pip install optuna"
        ) from e


def _create_genetic_generator(random_state: Optional[int], **kwargs) -> 'ParameterGenerator':
    """Create a genetic algorithm parameter generator.
    
    Args:
        random_state: Random seed for reproducible results
        **kwargs: Additional arguments for GeneticParameterGenerator
        
    Returns:
        GeneticParameterGenerator instance
    """
    try:
        from .generators.genetic_generator import GeneticParameterGenerator
        return GeneticParameterGenerator(random_state=random_state, **kwargs)
    except ImportError as e:
        raise OptimizerImportError(
            "Genetic parameter generator requires 'pygad' package. "
            "Install with: pip install pygad"
        ) from e


# Mock generator removed - use Optuna or Genetic generators for testing


def get_available_optimizers() -> Dict[str, Dict[str, Any]]:
    """Get information about available optimizer types.
    
    Returns:
        Dictionary mapping optimizer names to their information including
        description, dependencies, and availability status.
    """
    optimizers = {
        "optuna": {
            "description": "Tree-structured Parzen Estimator (TPE) optimization with pruning",
            "dependencies": ["optuna"],
            "supports_multi_objective": True,
            "supports_pruning": True,
            "available": False
        },
        "genetic": {
            "description": "Genetic algorithm optimization using PyGAD",
            "dependencies": ["pygad"],
            "supports_multi_objective": False,
            "supports_pruning": False,
            "available": False
        },
        "mockREMOVED": {
            "description": "Mock parameter generator for testing purposes",
            "dependencies": [],
            "supports_multi_objective": True,
            "supports_pruning": False,
            "available": True
        }
    }
    
    # Check availability of each optimizer
    for optimizer_name, info in optimizers.items():
        if optimizer_name == "mockREMOVED":
            continue  # Mock is always available
            
        try:
            create_parameter_generator(optimizer_name)
            info["available"] = True
        except (OptimizerImportError, UnknownOptimizerError):
            info["available"] = False
        except Exception:
            # Other errors might indicate the optimizer is available but misconfigured
            info["available"] = True
    
    return optimizers


def validate_optimizer_type(optimizer_type: str) -> bool:
    """Validate that an optimizer type is supported and available.
    
    Args:
        optimizer_type: String identifier for the optimizer type
        
    Returns:
        True if the optimizer is supported and available, False otherwise
    """
    try:
        create_parameter_generator(optimizer_type)
        return True
    except (UnknownOptimizerError, OptimizerImportError):
        return False
    except Exception:
        # Other errors might indicate the optimizer is available but misconfigured
        return True