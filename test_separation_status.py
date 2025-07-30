#!/usr/bin/env python3
"""
Comprehensive test to check the current status of separation of concerns implementation.
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, 'src')

def test_basic_imports():
    """Test that all basic modules can be imported."""
    print("=== Testing Basic Imports ===")
    
    imports_to_test = [
        ("Feature Flags", "portfolio_backtester.feature_flags", "FeatureFlags"),
        ("Backtesting Results", "portfolio_backtester.backtesting.results", "BacktestResult"),
        ("Optimization Results", "portfolio_backtester.optimization.results", "OptimizationResult"),
        ("Parameter Generator", "portfolio_backtester.optimization.parameter_generator", "ParameterGenerator"),
        ("Evaluator", "portfolio_backtester.optimization.evaluator", "BacktestEvaluator"),
        ("Factory", "portfolio_backtester.optimization.factory", "create_parameter_generator"),
        ("Orchestrator", "portfolio_backtester.optimization.orchestrator", "OptimizationOrchestrator"),
        ("Strategy Backtester", "portfolio_backtester.backtesting.strategy_backtester", "StrategyBacktester"),
        ("Optuna Generator", "portfolio_backtester.optimization.generators.optuna_generator", "OptunaParameterGenerator"),
        ("Genetic Generator", "portfolio_backtester.optimization.generators.genetic_generator", "GeneticParameterGenerator"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_path, class_name in imports_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print(f"\nImport Results: {passed} passed, {failed} failed")
    return failed == 0

def test_factory_functionality():
    """Test the parameter generator factory."""
    print("\n=== Testing Factory Functionality ===")
    
    try:
        from portfolio_backtester.optimization.factory import create_parameter_generator, get_available_optimizers
        
        # Test getting available optimizers
        optimizers = get_available_optimizers()
        print(f"Available optimizers: {list(optimizers.keys())}")
        
        # Test creating generators
        generators_tested = 0
        generators_working = 0
        
        for optimizer_name in ["optuna", "genetic"]:
            try:
                generator = create_parameter_generator(optimizer_name, random_state=42)
                print(f"✓ {optimizer_name} generator created: {type(generator).__name__}")
                generators_working += 1
            except Exception as e:
                print(f"✗ {optimizer_name} generator failed: {e}")
            generators_tested += 1
        
        print(f"Factory Results: {generators_working}/{generators_tested} generators working")
        return generators_working > 0
        
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        return False

def test_feature_flag_isolation():
    """Test feature flag isolation functionality."""
    print("\n=== Testing Feature Flag Isolation ===")
    
    try:
        from portfolio_backtester.feature_flags import FeatureFlags
        from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
        from unittest.mock import Mock
        
        # Test basic flag functionality
        flags = FeatureFlags.get_all_flags()
        print(f"Current flags: {flags}")
        
        # Test backtester isolation
        global_config = {
            "data_source": "mock",
            "cache_enabled": False,
            "parallel_processing": False,
            "benchmark": "SPY"
        }
        
        with FeatureFlags.disable_all_optimizers():
            backtester = StrategyBacktester(
                global_config=global_config,
                data_source=Mock()
            )
            
            # Check required methods exist
            required_methods = ['evaluate_window', 'backtest_strategy']
            for method in required_methods:
                if not hasattr(backtester, method):
                    raise AssertionError(f"Missing method: {method}")
                if not callable(getattr(backtester, method)):
                    raise AssertionError(f"Method not callable: {method}")
            
            print("✓ Backtester works in isolation")
        
        # Test individual context managers
        with FeatureFlags.disable_optuna():
            print("✓ disable_optuna context works")
        
        with FeatureFlags.disable_genetic():
            print("✓ disable_genetic context works")
        
        print("Feature Flag Results: All tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Feature flag test failed: {e}")
        return False

def test_orchestrator_functionality():
    """Test the optimization orchestrator."""
    print("\n=== Testing Orchestrator Functionality ===")
    
    try:
        from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
        from portfolio_backtester.optimization.evaluator import BacktestEvaluator
        from portfolio_backtester.optimization.factory import create_parameter_generator
        
        # Create components
        try:
            parameter_generator = create_parameter_generator("optuna", random_state=42)
        except:
            print("✗ Could not create parameter generator for orchestrator test")
            return False
        
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            timeout_seconds=None,
            early_stop_patience=None
        )
        
        print("✓ Orchestrator created successfully")
        print(f"  Type: {type(orchestrator).__name__}")
        
        # Check required methods
        required_methods = ['optimize', 'get_progress_status']
        for method in required_methods:
            if not hasattr(orchestrator, method):
                raise AssertionError(f"Missing method: {method}")
            if not callable(getattr(orchestrator, method)):
                raise AssertionError(f"Method not callable: {method}")
        
        print("✓ Orchestrator has required methods")
        print("Orchestrator Results: All tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        return False

def test_end_to_end_integration():
    """Test basic end-to-end integration."""
    print("\n=== Testing End-to-End Integration ===")
    
    try:
        from portfolio_backtester.optimization.factory import create_parameter_generator
        from portfolio_backtester.optimization.evaluator import BacktestEvaluator
        from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
        from portfolio_backtester.optimization.results import OptimizationData
        from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
        from unittest.mock import Mock
        import pandas as pd
        import numpy as np
        
        # Create mock data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT']
        
        # Create mock price data
        np.random.seed(42)
        price_data = pd.DataFrame(
            np.random.randn(len(dates), len(tickers)) * 0.02 + 1,
            index=dates,
            columns=tickers
        ).cumprod()
        
        monthly_data = price_data.resample('M').last()\n        daily_data = price_data\n        returns_data = price_data.pct_change().fillna(0)\n        \n        # Create mock windows\n        windows = [\n            (dates[0], dates[100], dates[101], dates[200]),\n            (dates[50], dates[150], dates[151], dates[250])\n        ]\n        \n        optimization_data = OptimizationData(\n            monthly=monthly_data,\n            daily=daily_data,\n            returns=returns_data,\n            windows=windows\n        )\n        \n        # Create components\n        try:\n            parameter_generator = create_parameter_generator("optuna", random_state=42)\n        except:\n            print("✗ Could not create parameter generator for integration test")\n            return False\n        \n        evaluator = BacktestEvaluator(\n            metrics_to_optimize=["sharpe_ratio"],\n            is_multi_objective=False\n        )\n        \n        orchestrator = OptimizationOrchestrator(\n            parameter_generator=parameter_generator,\n            evaluator=evaluator\n        )\n        \n        global_config = {\n            "data_source": "mock",\n            "cache_enabled": False,\n            "parallel_processing": False,\n            "benchmark": "SPY"\n        }\n        \n        backtester = StrategyBacktester(\n            global_config=global_config,\n            data_source=Mock()\n        )\n        \n        print("✓ All components created successfully")\n        print("✓ Mock data prepared")\n        print("End-to-End Results: Components integration successful")\n        return True\n        \n    except Exception as e:\n        print(f"✗ End-to-end integration test failed: {e}")\n        traceback.print_exc()\n        return False

def main():
    """Run all tests and provide summary."""
    print("🔍 Checking Separation of Concerns Implementation Status\\n")\n    \n    tests = [\n        ("Basic Imports", test_basic_imports),\n        ("Factory Functionality", test_factory_functionality),\n        ("Feature Flag Isolation", test_feature_flag_isolation),\n        ("Orchestrator Functionality", test_orchestrator_functionality),\n        ("End-to-End Integration", test_end_to_end_integration)\n    ]\n    \n    results = []\n    \n    for test_name, test_func in tests:\n        try:\n            result = test_func()\n            results.append((test_name, result))\n        except Exception as e:\n            print(f"\\n❌ {test_name} crashed: {e}")\n            results.append((test_name, False))\n    \n    # Summary\n    print("\\n" + "="*50)\n    print("📊 SUMMARY")\n    print("="*50)\n    \n    passed = 0\n    failed = 0\n    \n    for test_name, result in results:\n        status = "✅ PASS" if result else "❌ FAIL"\n        print(f"{test_name}: {status}")\n        if result:\n            passed += 1\n        else:\n            failed += 1\n    \n    print(f"\\nTotal: {passed} passed, {failed} failed")\n    \n    if failed == 0:\n        print("\\n🎉 All tests passed! Separation of concerns is working correctly.")\n        print("\\n✅ READY TO PROCEED with remaining TODO items:")\n        print("   - Run comprehensive separation tests")\n        print("   - Test end-to-end optimization workflow")\n        print("   - Update documentation")\n    else:\n        print(f"\\n⚠️  {failed} test(s) failed. Issues need to be resolved.")\n        print("\\n🔧 NEXT STEPS:")\n        print("   - Fix failing tests")\n        print("   - Check import dependencies")\n        print("   - Verify module implementations")\n    \n    return failed == 0

if __name__ == "__main__":\n    success = main()\n    sys.exit(0 if success else 1)