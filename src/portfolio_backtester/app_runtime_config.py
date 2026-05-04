"""Explicit application runtime configuration for CLI and programmatic entry points.

Prefer constructing :class:`AppRuntimeConfig` after :func:`portfolio_backtester.config_loader.load_config`
and passing ``runtime.global_config`` / ``runtime.scenarios`` through call chains instead of reading
``config_loader.GLOBAL_CONFIG`` from ad hoc modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AppRuntimeConfig:
    """Snapshot of global YAML-derived config and scenario list."""

    global_config: Dict[str, Any]
    scenarios: List[Dict[str, Any]]
