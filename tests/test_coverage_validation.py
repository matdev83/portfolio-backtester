"""
Test coverage validation script.
Ensures no regression in test coverage during refactoring.
"""

import subprocess
import sys
import json
import os
from pathlib import Path


class CoverageValidator:
    """Validates test coverage and ensures no regression."""
    
    def __init__(self):
        self.min_coverage_threshold = 80.0
        self.critical_modules_threshold = 95.0
        self.critical_modules = [
            'src/portfolio_backtester/backtester.py',
            'src/portfolio_backtester/strategies/',
            'src/portfolio_backtester/timing/',
        ]
    
    def run_coverage_analysis(self):
        """Run coverage analysis and return results."""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                '--cov=src',
                '--cov-report=json',
                '--cov-report=term-missing',
                '-q'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode != 0:
                print(f"❌ Tests failed: {result.stderr}")
                return False
            
            # Load coverage data
            coverage_file = Path(__file__).parent.parent / 'coverage.json'
            if not coverage_file.exists():
                print("❌ Coverage file not found")
                return False
            
            with open(coverage_file) as f:
                coverage_data = json.load(f)
            
            return self.validate_coverage(coverage_data)
            
        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return False
    
    def validate_coverage(self, coverage_data):
        """Validate coverage meets requirements."""
        total_coverage = coverage_data['totals']['percent_covered']
        
        print(f"\nTest Coverage Analysis")
        print(f"{'='*50}")
        print(f"Overall Coverage: {total_coverage:.1f}%")
        
        # Check overall coverage
        if total_coverage < self.min_coverage_threshold:
            print(f"❌ Overall coverage {total_coverage:.1f}% below threshold {self.min_coverage_threshold}%")
            return False
        else:
            print(f"✅ Overall coverage meets threshold ({self.min_coverage_threshold}%)")
        
        # Check critical modules
        critical_issues = []
        for module_pattern in self.critical_modules:
            module_coverage = self.get_module_coverage(coverage_data, module_pattern)
            if module_coverage < self.critical_modules_threshold:
                critical_issues.append(f"{module_pattern}: {module_coverage:.1f}%")
        
        if critical_issues:
            print(f"❌ Critical modules below {self.critical_modules_threshold}% threshold:")
            for issue in critical_issues:
                print(f"   - {issue}")
            return False
        else:
            print(f"✅ All critical modules meet threshold ({self.critical_modules_threshold}%)")
        
        # Report top uncovered files
        self.report_uncovered_files(coverage_data)
        
        return True
    
    def get_module_coverage(self, coverage_data, module_pattern):
        """Get coverage for modules matching pattern."""
        matching_files = []
        for file_path in coverage_data['files']:
            if module_pattern in file_path:
                matching_files.append(coverage_data['files'][file_path])
        
        if not matching_files:
            return 100.0  # No files found, assume covered
        
        total_statements = sum(f['summary']['num_statements'] for f in matching_files)
        covered_statements = sum(f['summary']['covered_lines'] for f in matching_files)
        
        if total_statements == 0:
            return 100.0
        
        return (covered_statements / total_statements) * 100
    
    def report_uncovered_files(self, coverage_data):
        """Report files with lowest coverage."""
        file_coverages = []
        for file_path, file_data in coverage_data['files'].items():
            if 'src/portfolio_backtester' in file_path:
                coverage = file_data['summary']['percent_covered']
                file_coverages.append((file_path, coverage))
        
        # Sort by coverage (lowest first)
        file_coverages.sort(key=lambda x: x[1])
        
        print(f"\nFiles with Lowest Coverage:")
        print(f"{'='*50}")
        for file_path, coverage in file_coverages[:10]:  # Top 10 lowest
            if coverage < 90:  # Only show files below 90%
                print(f"   {coverage:5.1f}% - {file_path}")
    
    def validate_test_count(self):
        """Validate that we have sufficient test count."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                '--collect-only', '-q'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode != 0:
                print(f"❌ Test collection failed: {result.stderr}")
                return False
            
            # Count collected tests
            lines = result.stdout.split('\n')
            test_count = 0
            for line in lines:
                if 'collected' in line and 'items' in line:
                    # Extract number from "collected X items"
                    words = line.split()
                    for i, word in enumerate(words):
                        if word == 'collected' and i + 1 < len(words):
                            try:
                                test_count = int(words[i + 1])
                                break
                            except ValueError:
                                continue
            
            print(f"\nTest Count Analysis")
            print(f"{'='*50}")
            print(f"Total Tests Collected: {test_count}")
            
            # Validate minimum test count
            min_tests = 200  # Expect at least 200 tests
            if test_count < min_tests:
                print(f"❌ Test count {test_count} below minimum {min_tests}")
                return False
            else:
                print(f"✅ Test count meets minimum requirement ({min_tests})")
            
            return True
            
        except Exception as e:
            print(f"❌ Test count validation failed: {e}")
            return False
    
    def validate_test_organization(self):
        """Validate test organization structure."""
        print(f"\nTest Organization Validation")
        print(f"{'='*50}")
        
        required_dirs = [
            'tests/unit',
            'tests/integration',
            'tests/fixtures',
            'tests/base'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = Path(__file__).parent.parent / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"❌ Missing required directories:")
            for dir_path in missing_dirs:
                print(f"   - {dir_path}")
            return False
        else:
            print("✅ All required directories exist")
        
        # Count files in each category
        unit_tests = len(list(Path('tests/unit').rglob('test_*.py')))
        integration_tests = len(list(Path('tests/integration').rglob('test_*.py')))
        
        print(f"   Unit tests: {unit_tests}")
        print(f"   Integration tests: {integration_tests}")
        
        return True


def main():
    """Main validation function."""
    print("Starting Test Suite Validation")
    print("="*60)
    
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
            print(f"❌ {name} validation failed with error: {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("SUCCESS: All validations passed! Test suite refactoring successful.")
        return 0
    else:
        print("FAILED: Some validations failed. Please review and fix issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())