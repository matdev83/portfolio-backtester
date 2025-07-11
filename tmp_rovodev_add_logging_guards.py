#!/usr/bin/env python3
"""
Script to add comprehensive logging level guards to prevent unnecessary 
parameter serialization and string formatting when logging is disabled.
"""

import os
import re
from pathlib import Path

def add_logging_guards_to_file(file_path):
    """Add logging level guards to expensive logging calls in a single file."""
    print(f"Processing {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Patterns for logging calls that need guards (expensive string formatting)
    guard_patterns = [
        # DEBUG level guards
        (r'(\s+)(logger\.debug\(f"[^"]*\{[^}]+\}[^"]*"\))',
         r'\1if logger.isEnabledFor(logging.DEBUG):\n\1    \2'),
        (r'(\s+)(self\.logger\.debug\(f"[^"]*\{[^}]+\}[^"]*"\))',
         r'\1if self.logger.isEnabledFor(logging.DEBUG):\n\1    \2'),
        
        # INFO level guards for expensive operations
        (r'(\s+)(logger\.info\(f"[^"]*\{[^}]+\}[^"]*"\))',
         r'\1if logger.isEnabledFor(logging.INFO):\n\1    \2'),
        (r'(\s+)(self\.logger\.info\(f"[^"]*\{[^}]+\}[^"]*"\))',
         r'\1if self.logger.isEnabledFor(logging.INFO):\n\1    \2'),
        
        # WARNING level guards for expensive operations
        (r'(\s+)(logger\.warning\(f"[^"]*\{[^}]+\}[^"]*"\))',
         r'\1if logger.isEnabledFor(logging.WARNING):\n\1    \2'),
        (r'(\s+)(self\.logger\.warning\(f"[^"]*\{[^}]+\}[^"]*"\))',
         r'\1if self.logger.isEnabledFor(logging.WARNING):\n\1    \2'),
    ]
    
    # Apply the patterns
    for pattern, replacement in guard_patterns:
        # Find all matches first to avoid infinite loops
        matches = list(re.finditer(pattern, content))
        
        # Process matches in reverse order to maintain positions
        for match in reversed(matches):
            # Check if this logging call is already guarded
            start_pos = match.start()
            
            # Look backwards to see if there's already a guard
            lines_before = content[:start_pos].split('\n')
            if len(lines_before) >= 2:
                prev_line = lines_before[-2].strip()
                if 'isEnabledFor' in prev_line:
                    continue  # Already guarded, skip
            
            # Apply the replacement
            new_content = content[:match.start()] + re.sub(pattern, replacement, match.group()) + content[match.end():]
            if new_content != content:
                changes_made.append(f"Added guard for: {match.group(2)[:50]}...")
                content = new_content
    
    # Specific patterns for backtester.py verbose logging
    if 'backtester.py' in file_path:
        specific_patterns = [
            # Signals head/tail logging - very verbose
            (r'(\s+)(logger\.info\(f"Signals head:\\n\{signals\.head\(\)\}"\))',
             r'\1if logger.isEnabledFor(logging.DEBUG):\n\1    logger.debug(f"Signals head:\\n{signals.head()}")'),
            (r'(\s+)(logger\.info\(f"Signals tail:\\n\{signals\.tail\(\)\}"\))',
             r'\1if logger.isEnabledFor(logging.DEBUG):\n\1    logger.debug(f"Signals tail:\\n{signals.tail()}")'),
            (r'(\s+)(logger\.info\(f"Sized signals head:\\n\{sized_signals\.head\(\)\}"\))',
             r'\1if logger.isEnabledFor(logging.DEBUG):\n\1    logger.debug(f"Sized signals head:\\n{sized_signals.head()}")'),
            (r'(\s+)(logger\.info\(f"Sized signals tail:\\n\{signals\.tail\(\)\}"\))',
             r'\1if logger.isEnabledFor(logging.DEBUG):\n\1    logger.debug(f"Sized signals tail:\\n{signals.tail()}")'),
            
            # Portfolio returns logging - very verbose
            (r'(\s+)(logger\.info\(f"Portfolio net returns calculated for \{scenario_config\[\'name\'\]\}\. First few net returns: \{portfolio_rets_net\.head\(\)\.to_dict\(\)\}"\))',
             r'\1if logger.isEnabledFor(logging.DEBUG):\n\1    logger.debug(f"Portfolio net returns calculated for {scenario_config[\'name\']}. First few net returns: {portfolio_rets_net.head().to_dict()}")'),
            (r'(\s+)(logger\.info\(f"Net returns index: \{portfolio_rets_net\.index\.min\(\)\} to \{portfolio_rets_net\.index\.max\(\)\}"\))',
             r'\1if logger.isEnabledFor(logging.DEBUG):\n\1    logger.debug(f"Net returns index: {portfolio_rets_net.index.min()} to {portfolio_rets_net.index.max()}")'),
        ]
        
        for pattern, replacement in specific_patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes_made.append(f"Changed verbose INFO to DEBUG: {pattern[:50]}...")
                content = new_content
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Made {len(changes_made)} changes")
        for change in changes_made:
            print(f"    - {change}")
        return True
    else:
        print(f"  - No changes needed")
        return False

def main():
    """Main function to add logging guards across the codebase."""
    
    # Files to process
    files_to_fix = [
        "src/portfolio_backtester/backtester.py",
        "src/portfolio_backtester/backtester_logic/optimization.py",
        "src/portfolio_backtester/backtester_logic/reporting.py",
        "src/portfolio_backtester/parallel_wfo.py", 
        "src/portfolio_backtester/parallel_monte_carlo.py",
        "src/portfolio_backtester/optimization/genetic_optimizer.py",
        "src/portfolio_backtester/reporting/optimizer_report_generator.py",
        "src/portfolio_backtester/data_cache.py",
        "src/portfolio_backtester/monte_carlo/asset_replacement.py",
        "src/portfolio_backtester/monte_carlo/synthetic_data_generator.py",
        "src/portfolio_backtester/data_sources/yfinance_data_source.py",
        "src/portfolio_backtester/universe_data/spy_holdings.py",
        "src/portfolio_backtester/portfolio/position_sizer.py",
    ]
    
    total_files_changed = 0
    
    print("Adding comprehensive logging level guards...")
    print("=" * 60)
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if add_logging_guards_to_file(file_path):
                total_files_changed += 1
        else:
            print(f"Warning: File not found: {file_path}")
    
    print("=" * 60)
    print(f"Completed! Modified {total_files_changed} files.")
    print("\nSummary of changes:")
    print("- Added logging level guards to prevent unnecessary string formatting")
    print("- Guards check if logging level is enabled before expensive operations")
    print("- Converted verbose INFO messages to DEBUG level where appropriate")
    print("- Performance improvement: no parameter serialization when logging disabled")

if __name__ == "__main__":
    main()