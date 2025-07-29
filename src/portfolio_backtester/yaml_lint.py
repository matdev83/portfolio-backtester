
"""
YAML Linter CLI Tool

A command-line tool for validating and linting YAML configuration files
with detailed error reporting and suggestions.
"""

import sys
import argparse
from pathlib import Path
from typing import List

from .yaml_validator import YamlValidator


def lint_files(file_paths: List[str], verbose: bool = False) -> bool:
    """
    Lint multiple YAML files and report results.
    
    Args:
        file_paths: List of file paths to validate
        verbose: Whether to show verbose output
        
    Returns:
        True if all files are valid, False otherwise
    """
    validator = YamlValidator()
    all_valid = True
    
    for file_path in file_paths:
        if verbose:
            print(f"Validating {file_path}...")
        
        is_valid, data, errors = validator.validate_file(file_path)
        
        if is_valid:
            print(f"[OK] {file_path}")
            if verbose and data:
                print(f"   Contains {len(data)} top-level keys")
        else:
            print(f"[ERROR] {file_path}")
            print(validator.format_errors(errors))
            all_valid = False
    
    return all_valid


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='YAML Linter - Validate YAML files with detailed error reporting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config/parameters.yaml
  %(prog)s config/*.yaml
  %(prog)s --verbose config/parameters.yaml config/scenarios.yaml
  %(prog)s --config-check  # Check default configuration files
        """
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        help='YAML files to validate'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--config-check', '-c',
        action='store_true',
        help='Check default configuration files (parameters.yaml and scenarios.yaml)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show errors, suppress success messages'
    )
    
    args = parser.parse_args()
    
    # Determine which files to check
    files_to_check = []
    if args.config_check:
        # Check default configuration files
        config_dir = Path(__file__).parent.parent.parent / "config"
        files_to_check = [
            str(config_dir / "parameters.yaml"),
            str(config_dir / "scenarios.yaml")
        ]
    elif args.files:
        files_to_check = args.files
    else:
        parser.print_help()
        sys.exit(1)
    
    # Validate files exist
    missing_files = []
    for file_path in files_to_check:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("[ERROR] The following files do not exist:")
        for file_path in missing_files:
            print(f"   {file_path}")
        sys.exit(1)
    
    # Lint the files
    if not args.quiet:
        print(f"Linting {len(files_to_check)} YAML file(s)...")
        print("-" * 50)
    
    all_valid = lint_files(files_to_check, verbose=args.verbose and not args.quiet)
    
    if not args.quiet:
        print("-" * 50)
        if all_valid:
            print(f"[OK] All {len(files_to_check)} file(s) are valid!")
        else:
            print("[ERROR] Some files have errors. Please fix them and try again.")
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()