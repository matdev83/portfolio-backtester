"""
API Stability Protection Module

This module provides decorators and utilities to protect critical method signatures
from breaking changes in a multi-developer codebase.
"""

from .protection import validate_signature, api_stable, deprecated, deprecated_signature
from .exceptions import SignatureViolationError, ParameterViolationError, ReturnTypeViolationError

__all__ = [
    'validate_signature',
    'api_stable',
    'deprecated',
    'deprecated_signature',
    'SignatureViolationError',
    'ParameterViolationError',
    'ReturnTypeViolationError'
]