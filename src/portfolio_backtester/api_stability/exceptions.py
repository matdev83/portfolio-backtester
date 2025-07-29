"""
Custom exceptions for API stability protection.
"""


class SignatureViolationError(Exception):
    """Raised when a method signature is violated."""
    pass


class ParameterViolationError(SignatureViolationError):
    """Raised when method parameters are invalid."""
    pass


class ReturnTypeViolationError(SignatureViolationError):
    """Raised when return type doesn't match expected type."""
    pass