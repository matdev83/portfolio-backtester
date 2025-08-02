#!/usr/bin/env python3
"""
Test to verify that the slow marker is working correctly for pytest.
"""

import subprocess
import sys

def test_slow_marker_exclusion():
    """Test that slow tests can be excluded from normal runs."""
    print("Testing Slow Marker Exclusion")
    print("=" * 40)
    
    try:
        # Run pytest with -m "not slow" to exclude slow tests
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/integration/optimization/test_end_to_end_optimization.py",
            "-m", "not slow",
            "-v", "--tb=short"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if slow tests were excluded
        if "test_optimization_with_timeout" not in result.stdout:
            print("✅ Timeout test was excluded (marked as slow)")
        else:
            print("❌ Timeout test was not excluded")
            return False
            
        if "test_optimization_with_early_stopping" not in result.stdout:
            print("✅ Early stopping test was excluded (marked as slow)")
        else:
            print("❌ Early stopping test was not excluded")
            return False
            
        if "test_optimization_error_handling" not in result.stdout:
            print("✅ Error handling test was excluded (marked as slow)")
        else:
            print("❌ Error handling test was not excluded")
            return False
        
        # Check that non-slow tests still run
        if "test_optuna_end_to_end_single_objective" in result.stdout:
            print("✅ Non-slow tests still run")
        else:
            print("⚠️  Non-slow tests might not be running")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Test command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def test_slow_marker_inclusion():
    """Test that slow tests can be run specifically."""
    print(f"\nTesting Slow Marker Inclusion")
    print("=" * 40)
    
    try:
        # Run pytest with -m "slow" to run only slow tests
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/integration/optimization/test_end_to_end_optimization.py",
            "-m", "slow",
            "-v", "--tb=short"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if only slow tests were included
        if "test_optimization_with_timeout" in result.stdout:
            print("✅ Timeout test was included (marked as slow)")
        else:
            print("❌ Timeout test was not included")
            return False
            
        if "test_optimization_with_early_stopping" in result.stdout:
            print("✅ Early stopping test was included (marked as slow)")
        else:
            print("❌ Early stopping test was not included")
            return False
            
        if "test_optimization_error_handling" in result.stdout:
            print("✅ Error handling test was included (marked as slow)")
        else:
            print("❌ Error handling test was not included")
            return False
        
        # Check that non-slow tests are excluded
        if "test_optuna_end_to_end_single_objective" not in result.stdout:
            print("✅ Non-slow tests were excluded")
        else:
            print("⚠️  Non-slow tests were not excluded")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Test command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def show_usage_examples():
    """Show usage examples for running tests with markers."""
    print(f"\nUsage Examples")
    print("=" * 40)
    
    examples = [
        ("Run all tests except slow ones", "pytest -m 'not slow'"),
        ("Run only slow tests", "pytest -m 'slow'"),
        ("Run integration tests except slow ones", "pytest -m 'integration and not slow'"),
        ("Run only fast tests", "pytest -m 'fast'"),
        ("Run optimization tests except slow ones", "pytest -m 'optimization and not slow'"),
        ("Run specific slow test", "pytest tests/integration/optimization/test_end_to_end_optimization.py::TestEndToEndOptimization::test_optimization_with_timeout")
    ]
    
    for description, command in examples:
        print(f"• {description}:")
        print(f"  {command}")
        print()

if __name__ == "__main__":
    print("Testing Slow Marker Configuration")
    print("=" * 50)
    
    # Test exclusion of slow tests
    exclusion_works = test_slow_marker_exclusion()
    
    # Test inclusion of only slow tests
    inclusion_works = test_slow_marker_inclusion()
    
    # Show usage examples
    show_usage_examples()
    
    # Summary
    print("=" * 50)
    print("SLOW MARKER TEST RESULTS")
    print("=" * 50)
    
    if exclusion_works and inclusion_works:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Slow marker exclusion works")
        print("✅ Slow marker inclusion works")
        print("✅ Timeout test is properly marked as slow")
        print("✅ Other slow tests are properly marked")
        
        print(f"\n📋 Recommended Usage:")
        print(f"   • Normal CI runs: pytest -m 'not slow'")
        print(f"   • Full test suite: pytest")
        print(f"   • Only slow tests: pytest -m 'slow'")
        print(f"   • Integration (not slow): pytest -m 'integration and not slow'")
        
    else:
        print("⚠️  SOME TESTS FAILED")
        print(f"Exclusion works: {exclusion_works}")
        print(f"Inclusion works: {inclusion_works}")
    
    print(f"\n🔧 Marker Status: {'VERIFIED ✅' if exclusion_works and inclusion_works else 'NEEDS REVIEW ⚠️'}")