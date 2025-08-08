"""
Flag storage and retrieval system.

This module handles the low-level storage and retrieval of feature flags,
including environment variables and thread-local overrides.
"""

import os
import threading
from typing import Optional


class FlagStore:
    """
    Handles flag storage and retrieval from multiple sources.

    This class manages the storage hierarchy:
    1. Thread-local overrides (highest priority)
    2. Environment variables
    3. Default values (lowest priority)
    """

    # Thread-local storage for context manager overrides
    _local = threading.local()

    @classmethod
    def get_env_flag(cls, env_var: str, default: str = "false") -> bool:
        """Get boolean value from environment variable."""
        return os.environ.get(env_var, default).lower() in ("true", "1", "yes", "on")

    @classmethod
    def get_override(cls, flag_name: str) -> Optional[bool]:
        """Get override value from thread-local storage."""
        if not hasattr(cls._local, "overrides"):
            return None
        # Dict is of bool values; .get may return None if not present.
        return cls._local.overrides.get(flag_name)  # type: ignore[no-any-return]

    @classmethod
    def set_override(cls, flag_name: str, value: bool) -> None:
        """Set override value in thread-local storage."""
        if not hasattr(cls._local, "overrides"):
            cls._local.overrides = {}
        cls._local.overrides[flag_name] = value

    @classmethod
    def clear_override(cls, flag_name: str) -> None:
        """Clear override value from thread-local storage."""
        if hasattr(cls._local, "overrides") and flag_name in cls._local.overrides:
            del cls._local.overrides[flag_name]

    @classmethod
    def get_flag(cls, flag_name: str, default: bool = False) -> bool:
        """
        Get the current value of a feature flag.

        This method checks for a thread-local override first, then checks
        the corresponding environment variable, and finally falls back to the
        provided default.

        Args:
            flag_name (str): The snake_case name of the flag.
            default (bool): The default value if no override or env var is set.

        Returns:
            bool: The current value of the flag.
        """
        override = cls.get_override(flag_name)
        if override is not None:
            return override

        env_var_name = flag_name.upper()
        return cls.get_env_flag(env_var_name, str(default).lower())
