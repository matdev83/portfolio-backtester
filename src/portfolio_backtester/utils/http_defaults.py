"""Shared defaults for outbound HTTP and related retry timing.

Policy: HTTP timeout seconds and simple sleep-based retry delays live here so
callers do not scatter magic numbers. Retry loops should exist in at most one
layer (no tenacity-on-requests stacking without an explicit design).
"""

from __future__ import annotations

# SEC / OpenFIGI and similar small JSON fetches
DEFAULT_HTTP_TIMEOUT_SEC: float = 30.0

# Optuna study load race in parallel workers (see parallel_optimization_runner)
OPTUNA_STUDY_LOAD_RETRY_SLEEP_SEC: float = 1.0
OPTUNA_STUDY_LOAD_MAX_ATTEMPTS: int = 3
