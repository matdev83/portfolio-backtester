#!/usr/bin/env python3

"""
Script to run bandit security analysis with project-specific configuration.
"""

import subprocess
import sys
import os

def run_bandit():
    """Run bandit with our project configuration."""
    try:
        # Get the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Run bandit with our configuration
        cmd = [
            sys.executable, "-m", "bandit",
            "-r", "src",
            "-c", "bandit.yaml"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running bandit: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(run_bandit())