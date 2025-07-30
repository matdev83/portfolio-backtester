"""
Parameter generators for different optimization backends.

This package contains implementations of the ParameterGenerator interface
for various optimization backends including Optuna, genetic algorithms,
and mock generators for testing.
"""

# Import all generators to make them available at package level
__all__ = []

# Conditional imports for optional dependencies
try:
    from .optuna_generator import OptunaParameterGenerator
    __all__.append('OptunaParameterGenerator')
except ImportError:
    pass

try:
    from .genetic_generator import GeneticParameterGenerator
    __all__.append('GeneticParameterGenerator')
except ImportError:
    pass