#!/usr/bin/env python3
"""
Test script to identify import issues in the separation of concerns implementation.
"""

import sys
import traceback

def test_import(module_name, description=""):
    """Test importing a module and report results."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} - {description}")
        return True
    except Exception as e:
        print(f"✗ {module_name} - {description}")
        print(f"  Error: {e}")
        return False

def main():
    print("=== Testing Imports for Separation of Concerns ===\n")
    
    # Test feature flags
    print("--- Feature Flags ---")
    test_import("src.portfolio_backtester.feature_flags", "Feature flag system")
    
    # Test backtesting modules
    print("\n--- Backtesting Modules ---")
    test_import("src.portfolio_backtester.backtesting.results", "Backtesting results")
    test_import("src.portfolio_backtester.backtesting.strategy_backtester", "Strategy backtester")
    
    # Test optimization modules
    print("\n--- Optimization Modules ---")
    test_import("src.portfolio_backtester.optimization.results", "Optimization results")
    test_import("src.portfolio_backtester.optimization.parameter_generator", "Parameter generator base")
    test_import("src.portfolio_backtester.optimization.evaluator", "Backtest evaluator")
    test_import("src.portfolio_backtester.optimization.factory", "Optimization factory")
    test_import("src.portfolio_backtester.optimization.orchestrator", "Optimization orchestrator")
    
    # Test parameter generators
    print("\n--- Parameter Generators ---")
    test_import("src.portfolio_backtester.optimization.generators.optuna_generator", "Optuna generator")
    test_import("src.portfolio_backtester.optimization.generators.genetic_generator", "Genetic generator")
    
    # Test dependencies that backtester needs
    print("\n--- Backtester Dependencies ---")
    test_import("src.portfolio_backtester.strategies.base_strategy", "Base strategy")
    test_import("src.portfolio_backtester.strategies", "Strategies module")
    test_import("src.portfolio_backtester.backtester_logic.strategy_logic", "Strategy logic")
    test_import("src.portfolio_backtester.backtester_logic.portfolio_logic", "Portfolio logic")
    test_import("src.portfolio_backtester.backtester_logic.data_manager", "Data manager")
    test_import("src.portfolio_backtester.reporting.performance_metrics", "Performance metrics")
    test_import("src.portfolio_backtester.data_cache", "Data cache")
    
    # Test trading module
    print("\n--- Trading Module ---")
    test_import("src.portfolio_backtester.trading.trade_tracker", "Trade tracker")
    
    print("\n=== Import Test Complete ===")

if __name__ == "__main__":
    main()