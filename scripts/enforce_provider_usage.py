#!/usr/bin/env python3
"""
Provider Interface Usage Enforcement Script

This script checks for violations of the provider interface usage patterns
and prevents legacy code patterns from being introduced.

Usage:
    python scripts/enforce_provider_usage.py [path_to_check]

This can be integrated into:
- Pre-commit hooks
- CI/CD pipelines
- IDE linting
- Development workflows

Exit codes:
    0: No violations found
    1: Violations found
    2: Script error
"""

import sys
import argparse
import ast
from pathlib import Path
from typing import List, Tuple


def check_file_for_violations(file_path: Path) -> List[Tuple[int, str]]:
    """
    Check a single file for provider interface violations.

    Returns:
        List of (line_number, violation_description) tuples
    """
    violations: List[Tuple[int, str]] = []

    if not file_path.exists() or not file_path.is_file():
        return violations

    # Basic file type check (filtering is done in main())
    if file_path.suffix != ".py":
        return violations

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return violations

    lines = content.splitlines()

    # Patterns that indicate legacy usage (violations)
    legacy_patterns = {
        "from ..universe_resolver import resolve_universe_config": "Use UniverseProviderFactory instead of direct resolve_universe_config import",
        "from ..portfolio.position_sizer import get_position_sizer_from_config": "Use PositionSizerProviderFactory instead of direct get_position_sizer_from_config import",
        "resolve_universe_config(": "Use strategy.get_universe_provider().get_universe_symbols() instead",
        "get_position_sizer_from_config(": "Use strategy.get_position_sizer_provider().get_position_sizer() instead",
    }

    # Exception patterns (allowed usage)
    allowed_contexts = {
        "interfaces/universe_provider_interface.py",
        "interfaces/position_sizer_provider_interface.py",
        "interfaces/stop_loss_provider_interface.py",
        "universe_resolver.py",
        "position_sizer.py",
        "enforcement.py",
    }

    # Check if this file is in allowed contexts
    file_str = str(file_path).replace("\\", "/")
    is_allowed_context = any(context in file_str for context in allowed_contexts)

    if is_allowed_context:
        return violations  # Skip enforcement for provider implementation files

    # Check each line for violations
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # Skip comments and empty lines
        if not line_stripped or line_stripped.startswith("#"):
            continue

        # Check for legacy patterns
        for pattern, message in legacy_patterns.items():
            if pattern in line_stripped:
                violations.append((line_num, f"{message}\n    Found: {line_stripped}"))

    return violations


class SizePositionsVisitor(ast.NodeVisitor):
    """
    AST visitor to check size_positions() function calls.
    """

    def __init__(self, lines: List[str]):
        self.violations: List[Tuple[int, str]] = []
        self.lines = lines

    def visit_Call(self, node: ast.Call) -> None:
        # Check if this is a call to size_positions
        if isinstance(node.func, ast.Name) and node.func.id == "size_positions":

            self._check_size_positions_call(node)

        self.generic_visit(node)

    def _check_size_positions_call(self, node: ast.Call) -> None:
        """
        Check if a size_positions call has the required strategy parameter.
        """
        # Expected positional arguments: signals, scenario_config, price_data_monthly_closes,
        # price_data_daily_ohlc, universe_tickers, benchmark_ticker, strategy
        expected_positional_args = 7

        # Check positional arguments
        positional_count = len(node.args)

        # Check keyword arguments for 'strategy'
        has_strategy_kwarg = any(kw.arg == "strategy" for kw in node.keywords if kw.arg is not None)

        # If we have the right number of positional args OR strategy as keyword, it's valid
        if positional_count >= expected_positional_args or has_strategy_kwarg:
            return  # This call is valid

        # Get the line for error reporting
        line_num = getattr(node, "lineno", 0)
        if line_num > 0 and line_num <= len(self.lines):
            line_content = self.lines[line_num - 1].strip()
        else:
            line_content = "Unknown line"

        self.violations.append(
            (
                line_num,
                f"size_positions() missing strategy parameter. "
                f"Found {positional_count} positional args, expected {expected_positional_args} "
                f"or 'strategy' keyword argument.\n    Found: {line_content}",
            )
        )


def check_size_positions_calls(file_path: Path) -> List[Tuple[int, str]]:
    """
    Check that size_positions() is called with strategy parameter using AST parsing.

    Returns:
        List of (line_number, violation_description) tuples
    """
    violations: List[Tuple[int, str]] = []

    if not file_path.exists() or not file_path.is_file() or file_path.suffix != ".py":
        return violations

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return violations

    # Parse the file with AST
    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError:
        # Skip files with syntax errors
        return violations

    lines = content.splitlines()
    visitor = SizePositionsVisitor(lines)
    visitor.visit(tree)

    return visitor.violations


def get_function_call_context(lines: List[str], line_num: int, context_lines: int = 3) -> str:
    """
    Get context around a function call for better error reporting.
    """
    start = max(0, line_num - context_lines - 1)
    end = min(len(lines), line_num + context_lines)

    context = []
    for i in range(start, end):
        marker = ">>> " if i == line_num - 1 else "    "
        context.append(f"{marker}{i + 1:3}: {lines[i]}")

    return "\n".join(context)


def validate_file_accessibility(file_path: Path) -> bool:
    """
    Check if file is accessible and should be processed.
    """
    # Skip certain directories completely
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
    if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
        return False

    # Skip non-Python files
    if file_path.suffix != ".py":
        return False

    # Skip test files (but allow them to be checked with --include-tests)
    if "test" in str(file_path).lower():
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Enforce provider interface usage patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/enforce_provider_usage.py                    # Check src/ directory
  python scripts/enforce_provider_usage.py src/              # Check specific directory  
  python scripts/enforce_provider_usage.py file.py           # Check single file
  python scripts/enforce_provider_usage.py --include-tests   # Include test files
  python scripts/enforce_provider_usage.py --verbose         # Show detailed output
        """,
    )
    parser.add_argument("path", nargs="?", default="src", help="Path to check (default: src)")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument(
        "--include-tests", action="store_true", help="Include test files in checking"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output and context"
    )
    parser.add_argument(
        "--ignore-enforcement-hints",
        action="store_true",
        help="Ignore enforcement.py hint text (documentation-only violations)",
    )

    args = parser.parse_args()

    check_path = Path(args.path)
    if not check_path.exists():
        print(f"Error: Path {check_path} does not exist")
        return 2

    total_violations = 0
    total_files_checked = 0

    # Get all Python files to check
    if check_path.is_file():
        files_to_check = [check_path]
    else:
        all_files = list(check_path.rglob("*.py"))
        files_to_check = []

        for file_path in all_files:
            # Apply filtering based on arguments
            # Skip certain directories completely
            skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Skip non-Python files
            if file_path.suffix != ".py":
                continue

            # Skip test files unless --include-tests is specified
            # Only skip files that are actually test files (in test directories or named test_*)
            if not args.include_tests:
                file_str_lower = str(file_path).lower()
                is_test_file = (
                    "/test/" in file_str_lower
                    or "\\test\\" in file_str_lower
                    or "/tests/" in file_str_lower
                    or "\\tests\\" in file_str_lower
                    or file_path.name.startswith("test_")
                    or file_path.name.startswith("conftest")
                )
                if is_test_file:
                    continue

            files_to_check.append(file_path)

    if args.verbose:
        print(f"🔍 Checking {len(files_to_check)} files for provider interface violations...")
        print(f"📁 Search path: {check_path.absolute()}")
        if args.include_tests:
            print("🧪 Including test files")
        print("=" * 70)
    else:
        print(f"🔍 Checking {len(files_to_check)} files for provider interface violations...")
        print("=" * 70)

    for file_path in files_to_check:
        total_files_checked += 1

        # Skip files that should be ignored
        file_str = str(file_path).replace("\\", "/")

        # Special handling for enforcement.py documentation
        is_enforcement_file = "enforcement.py" in file_str

        # Check for legacy pattern violations
        violations = check_file_for_violations(file_path)

        # Check for size_positions calls (skip enforcement.py for documentation)
        if not is_enforcement_file or not args.ignore_enforcement_hints:
            size_violations = check_size_positions_calls(file_path)

            # Filter out documentation-only violations in enforcement.py
            if is_enforcement_file and args.ignore_enforcement_hints:
                size_violations = [
                    (line_num, msg)
                    for line_num, msg in size_violations
                    if not any(
                        doc_hint in msg.lower()
                        for doc_hint in [
                            "always pass strategy parameter to size_positions",
                            "ensure functions like size_positions",
                        ]
                    )
                ]

            violations.extend(size_violations)

        if violations:
            total_violations += len(violations)
            print(f"\n❌ {file_path}")
            for line_num, message in violations:
                if args.verbose:
                    # Show context for verbose mode
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        context = get_function_call_context(lines, line_num, 2)
                        print(f"   Line {line_num}: {message}")
                        print(f"   Context:\n{context}")
                        print()
                    except Exception:
                        print(f"   Line {line_num}: {message}")
                else:
                    print(f"   Line {line_num}: {message}")

    print("\n" + "=" * 70)

    if total_violations == 0:
        print("✅ No provider interface violations found!")
        return 0
    else:
        print(f"❌ Found {total_violations} violations")
        print("\n📖 ENFORCEMENT GUIDE:")
        print("1. Use strategy.get_universe_provider() instead of resolve_universe_config()")
        print(
            "2. Use strategy.get_position_sizer_provider() instead of get_position_sizer_from_config()"
        )
        print("3. Use strategy.get_stop_loss_provider() for stop loss handling")
        print("4. Always pass strategy parameter to size_positions()")
        print("\nSee src/portfolio_backtester/interfaces/enforcement.py for more details.")

        return 1


if __name__ == "__main__":
    sys.exit(main())
