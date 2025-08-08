#!/usr/bin/env python3

import subprocess
import sys
import re


def run_mypy_on_all():
    """Run mypy on all files with a timeout."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml", "src"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=".",
            timeout=60,  # 60 second timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print("MyPy execution timed out")
        sys.exit(1)
    except Exception as e:
        print(f"Error running MyPy: {e}")
        sys.exit(1)


def run_mypy_on_path(path):
    """Run mypy on a specific file path."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml", path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=".",
            timeout=30,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"MyPy execution timed out on {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running MyPy on {path}: {e}")
        sys.exit(1)


def find_first_error_file(output):
    """Find the first file with errors, excluding spy_holdings.py."""
    lines = output.splitlines()
    for line in lines:
        match = re.match(r"^(src[\\/][^:]+\.py):\d+:", line)
        if match:
            filename = match.group(1)
            if "spy_holdings.py" not in filename:
                return filename
    return None


def main():
    # Step 1: Run mypy on all files to find first error file
    print("Step 1: Finding first file with MyPy errors...")
    mypy_output = run_mypy_on_all()

    first_file = find_first_error_file(mypy_output)
    if not first_file:
        print("No MyPy errors found in files other than spy_holdings.py")
        return

    print(f"Found first file with errors: {first_file}")

    # Step 2: Run mypy specifically on that file path to get full error output
    print(f"Step 2: Running MyPy on {first_file} to get detailed errors...")
    file_output = run_mypy_on_path(first_file)

    if file_output.strip():
        print(f"\nFull MyPy output for {first_file}:")
        print(file_output)
    else:
        print(f"No specific errors found for {first_file} (unexpected)")


if __name__ == "__main__":
    main()
