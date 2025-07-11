#!/usr/bin/env python3
"""
Script to fix common pylint issues in backtester.py
"""

import re

def fix_pylint_issues():
    """Fix common pylint issues."""
    
    # Read the file
    with open('src/portfolio_backtester/backtester.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix line length issues by breaking long lines
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        
        # Fix long lines (over 100 characters)
        if len(line) > 100:
            # Handle specific patterns
            if 'self.asset_replacement_manager = AssetReplacementManager(' in line:
                fixed_lines.append('        # Initialize Monte Carlo components if enabled')
                fixed_lines.append('        if self.monte_carlo_config.get(\'enable_synthetic_data\', False):')
                fixed_lines.append('            self.asset_replacement_manager = AssetReplacementManager(')
                fixed_lines.append('                self.monte_carlo_config)')
                continue
            elif 'self.synthetic_data_generator = SyntheticDataGenerator(' in line:
                fixed_lines.append('            self.synthetic_data_generator = SyntheticDataGenerator(')
                fixed_lines.append('                self.monte_carlo_config)')
                continue
            elif 'self.synthetic_data_validator = SyntheticDataValidator(' in line:
                fixed_lines.append('            self.synthetic_data_validator = SyntheticDataValidator(')
                fixed_lines.append('                self.monte_carlo_config.get(\'validation_config\', {}))')
                continue
            elif 'logger.info(' in line and len(line) > 100:
                # Break long logger statements
                indent = len(line) - len(line.lstrip())
                prefix = ' ' * indent
                if 'f"' in line:
                    # Find the f-string and break it
                    parts = line.split('f"', 1)
                    if len(parts) == 2:
                        before = parts[0]
                        after = parts[1]
                        if len(before + 'f"') < 80:
                            fixed_lines.append(before + 'f"')
                            fixed_lines.append(prefix + '    ' + after)
                            continue
            elif line.strip().startswith('assert') and len(line) > 100:
                # Break long assert statements
                indent = len(line) - len(line.lstrip())
                prefix = ' ' * indent
                if ', (' in line:
                    parts = line.split(', (', 1)
                    if len(parts) == 2:
                        fixed_lines.append(parts[0] + ', (')
                        fixed_lines.append(prefix + '    ' + parts[1])
                        continue
        
        fixed_lines.append(line)
    
    # Write back the fixed content
    with open('src/portfolio_backtester/backtester.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print("Fixed pylint issues in backtester.py")

if __name__ == "__main__":
    fix_pylint_issues()