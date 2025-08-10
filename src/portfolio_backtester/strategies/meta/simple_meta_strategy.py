"""Backward-compatibility shim for SimpleMetaStrategy.

Historically, tests and user code imported:
    from portfolio_backtester.strategies.meta.simple_meta_strategy import SimpleMetaStrategy

The implementation was moved under `strategies.builtins.meta`. This module preserves the
old import path by re-exporting the class from its new location.
"""

from __future__ import annotations

from ..builtins.meta.simple_meta_strategy import SimpleMetaStrategy as _SimpleMetaStrategy

__all__ = ["SimpleMetaStrategy"]

# Public alias to preserve the old import path
SimpleMetaStrategy = _SimpleMetaStrategy
