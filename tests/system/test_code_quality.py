import subprocess
import pytest
import sys
import os
from pathlib import Path

# Determine project root
# This file is in tests/system/test_code_quality.py
# So root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Determine executable paths based on platform and environment
if sys.platform == "win32":
    VENV_SCRIPTS = PROJECT_ROOT / ".venv" / "Scripts"
    RUFF_EXE = VENV_SCRIPTS / "ruff.exe"
    MYPY_EXE = VENV_SCRIPTS / "mypy.exe"
else:
    VENV_BIN = PROJECT_ROOT / ".venv" / "bin"
    RUFF_EXE = VENV_BIN / "ruff"
    MYPY_EXE = VENV_BIN / "mypy"

def test_ruff_linting():
    """
    Run ruff linting on src directory with --fix.
    Fails if unfixable errors are found.
    """
    if not RUFF_EXE.exists():
        # Fallback to trying to run from PATH if not in venv
        # But for this environment we expect it in .venv
        pytest.skip(f"ruff executable not found at {RUFF_EXE}")

    print(f"Running ruff from: {RUFF_EXE}")
    
    # Run ruff check src --fix
    cmd = [str(RUFF_EXE), "check", "src", "--fix"]
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    
    # Check output for unfixable errors
    if result.returncode != 0:
        pytest.fail(f"Ruff found unfixable errors:\n{result.stdout}\n{result.stderr}")

def test_mypy_type_checking():
    """
    Run mypy type checking on src directory.
    Fails if type errors are found.
    """
    if not MYPY_EXE.exists():
        pytest.skip(f"mypy executable not found at {MYPY_EXE}")
        
    print(f"Running mypy from: {MYPY_EXE}")
    
    # Run mypy src
    cmd = [str(MYPY_EXE), "src"]
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    
    if result.returncode != 0:
        pytest.fail(f"Mypy found type errors:\n{result.stdout}\n{result.stderr}")
