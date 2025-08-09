from pathlib import Path

ZERO_RET_EPS = 1e-8  # |returns| below this is considered "zero"

# Default Optuna storage configuration
# Use an absolute path anchored at the project root to avoid CWD-dependent behavior.

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPTUNA_DB_PATH = (_PROJECT_ROOT / "data" / "optuna_studies.db").resolve()

# SQLAlchemy/Optuna accept forward slashes in Windows paths; use as_posix() for portability
DEFAULT_OPTUNA_STORAGE_URL = f"sqlite:///{_OPTUNA_DB_PATH.as_posix()}"
