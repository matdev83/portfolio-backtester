#!/usr/bin/env python
"""
Shortcut script to run portfolio optimization on a given strategy YAML config file.
Usage:
    ./optimize.py <strategy_config.yaml> [<optional_optimizer_args>...]

This will invoke the optimizer with default settings, forwarding any extra arguments.
"""
import sys
import subprocess
import os

PYTHON_EXEC = os.path.join('.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join('.venv', 'bin', 'python')
BACKTESTER_MODULE = 'src.portfolio_backtester.backtester'


def print_usage():
    print("Usage: ./optimize.py <strategy_config.yaml> [<optional_optimizer_args>...]")
    print("Example: ./optimize.py config/scenarios/portfolio/calmar_momentum_strategy/default.yaml --optuna-trials 1000 --n-jobs -1")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    scenario_filename = sys.argv[1]
    extra_args = sys.argv[2:]

    cmd = [PYTHON_EXEC, '-m', BACKTESTER_MODULE, '--mode', 'optimize', '--scenario-filename', scenario_filename]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[optimize.py] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[optimize.py] Optimization failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()