"""
Test coverage validation script.
Ensures no regression in test coverage during refactoring.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Repository root (this file lives under tests/unit/core/).
_REPO_ROOT = Path(__file__).resolve().parents[3]


class CoverageValidator:
    """Validates test coverage and ensures no regression."""

    def __init__(self) -> None:
        self.min_coverage_threshold = 80.0
        self.critical_modules_threshold = 95.0
        self.critical_modules = [
            "src/portfolio_backtester/backtester.py",
            "src/portfolio_backtester/strategies/",
            "src/portfolio_backtester/timing/",
        ]

    def run_coverage_analysis(self) -> bool:
        """Run coverage analysis and return results."""
        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["pytest", "--cov=src", "--cov-report=json", "--cov-report=term-missing", "-q"],
                capture_output=True,
                text=True,
                cwd=_REPO_ROOT,
            )

            if result.returncode != 0:
                logger.error("Tests failed: %s", result.stderr)
                return False

            # Load coverage data
            coverage_file = _REPO_ROOT / "coverage.json"
            if not coverage_file.exists():
                logger.error("Coverage file not found")
                return False

            with open(coverage_file, encoding="utf-8") as f:
                coverage_data = json.load(f)

            return self.validate_coverage(coverage_data)

        except Exception as e:
            logger.exception("Coverage analysis failed: %s", e)
            return False

    def validate_coverage(self, coverage_data: dict) -> bool:
        """Validate coverage meets requirements."""
        total_coverage = coverage_data["totals"]["percent_covered"]

        logger.info("\nTest Coverage Analysis")
        logger.info("%s", "=" * 50)
        logger.info("Overall Coverage: %.1f%%", total_coverage)

        # Check overall coverage
        if total_coverage < self.min_coverage_threshold:
            logger.error(
                "Overall coverage %.1f%% below threshold %.1f%%",
                total_coverage,
                self.min_coverage_threshold,
            )
            return False
        logger.info(
            "Overall coverage meets threshold (%.1f%%)",
            self.min_coverage_threshold,
        )

        # Check critical modules
        critical_issues = []
        for module_pattern in self.critical_modules:
            module_coverage = self.get_module_coverage(coverage_data, module_pattern)
            if module_coverage < self.critical_modules_threshold:
                critical_issues.append(f"{module_pattern}: {module_coverage:.1f}%")

        if critical_issues:
            logger.error(
                "Critical modules below %.1f%% threshold:",
                self.critical_modules_threshold,
            )
            for issue in critical_issues:
                logger.error("   - %s", issue)
            return False
        logger.info(
            "All critical modules meet threshold (%.1f%%)",
            self.critical_modules_threshold,
        )

        # Report top uncovered files
        self.report_uncovered_files(coverage_data)

        return True

    def get_module_coverage(self, coverage_data: dict, module_pattern: str) -> float:
        """Get coverage for modules matching pattern."""
        matching_files = []
        for file_path in coverage_data["files"]:
            if module_pattern in file_path:
                matching_files.append(coverage_data["files"][file_path])

        if not matching_files:
            return 100.0  # No files found, assume covered

        total_statements = sum(f["summary"]["num_statements"] for f in matching_files)
        covered_statements = sum(f["summary"]["covered_lines"] for f in matching_files)

        if total_statements == 0:
            return 100.0

        return float((covered_statements / total_statements) * 100)

    def report_uncovered_files(self, coverage_data: dict) -> None:
        """Report files with lowest coverage."""
        file_coverages = []
        for file_path, file_data in coverage_data["files"].items():
            if "src/portfolio_backtester" in file_path:
                coverage = file_data["summary"]["percent_covered"]
                file_coverages.append((file_path, coverage))

        # Sort by coverage (lowest first)
        file_coverages.sort(key=lambda x: x[1])

        logger.info("\nFiles with Lowest Coverage:")
        logger.info("%s", "=" * 50)
        for file_path, coverage in file_coverages[:10]:  # Top 10 lowest
            if coverage < 90:  # Only show files below 90%
                logger.info("   %5.1f%% - %s", coverage, file_path)

    def validate_test_count(self) -> bool:
        """Validate that we have sufficient test count."""
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q"],
                capture_output=True,
                text=True,
                cwd=_REPO_ROOT,
            )

            if result.returncode != 0:
                logger.error("Test collection failed: %s", result.stderr)
                return False

            # Count collected tests
            lines = result.stdout.split("\n")
            test_count = 0
            for line in lines:
                if "collected" in line and "items" in line:
                    # Extract number from "collected X items"
                    words = line.split()
                    for i, word in enumerate(words):
                        if word == "collected" and i + 1 < len(words):
                            try:
                                test_count = int(words[i + 1])
                                break
                            except ValueError:
                                continue

            logger.info("\nTest Count Analysis")
            logger.info("%s", "=" * 50)
            logger.info("Total Tests Collected: %s", test_count)

            # Validate minimum test count
            min_tests = 200  # Expect at least 200 tests
            if test_count < min_tests:
                logger.error("Test count %s below minimum %s", test_count, min_tests)
                return False
            logger.info("Test count meets minimum requirement (%s)", min_tests)

            return True

        except Exception as e:
            logger.exception("Test count validation failed: %s", e)
            return False

    def validate_test_organization(self) -> bool:
        """Validate test organization structure."""
        logger.info("\nTest Organization Validation")
        logger.info("%s", "=" * 50)

        required_dirs = ["tests/unit", "tests/integration", "tests/fixtures", "tests/base"]

        missing_dirs = []
        for dir_path in required_dirs:
            full_path = _REPO_ROOT / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)

        if missing_dirs:
            logger.error("Missing required directories:")
            for dir_path in missing_dirs:
                logger.error("   - %s", dir_path)
            return False
        logger.info("All required directories exist")

        # Count files in each category
        unit_tests = len(list((_REPO_ROOT / "tests/unit").rglob("test_*.py")))
        integration_tests = len(list((_REPO_ROOT / "tests/integration").rglob("test_*.py")))

        logger.info("   Unit tests: %s", unit_tests)
        logger.info("   Integration tests: %s", integration_tests)

        return True


def main() -> int:
    """Main validation function."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Starting Test Suite Validation")
    logger.info("%s", "=" * 60)

    validator = CoverageValidator()

    # Run all validations
    validations = [
        ("Test Organization", validator.validate_test_organization),
        ("Test Count", validator.validate_test_count),
        ("Test Coverage", validator.run_coverage_analysis),
    ]

    all_passed = True
    for name, validation_func in validations:
        try:
            if not validation_func():
                all_passed = False
        except Exception as e:
            logger.exception("%s validation failed with error: %s", name, e)
            all_passed = False

    logger.info("\n%s", "=" * 60)
    if all_passed:
        logger.info("SUCCESS: All validations passed! Test suite refactoring successful.")
        return 0
    logger.error("FAILED: Some validations failed. Please review and fix issues.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
