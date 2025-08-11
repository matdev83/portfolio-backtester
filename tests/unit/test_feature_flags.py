"""
Tests for the feature flag system.

This module tests the feature flag functionality including environment variable
support, context managers for component isolation, and thread safety.
"""

import os
import threading
import time
from unittest.mock import patch

from portfolio_backtester.feature_flags import (
    FeatureFlags,
    is_new_architecture_enabled,
    should_show_migration_warnings,
)


class TestFeatureFlags:
    """Test cases for the FeatureFlags class."""

    def setup_method(self):
        """Clear any thread-local overrides before each test."""
        if hasattr(FeatureFlags._local, "overrides"):
            FeatureFlags._local.overrides = {}

    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(FeatureFlags._local, "overrides"):
            FeatureFlags._local.overrides = {}

    def test_default_flag_values(self):
        """Test that flags have correct default values."""
        with patch.dict(os.environ, {}, clear=True):
            # These flags default to True
            assert FeatureFlags.use_new_optimization_architecture()
            assert FeatureFlags.use_new_backtesting_architecture()
            assert FeatureFlags.enable_optuna_generator()
            assert FeatureFlags.enable_genetic_generator()
            assert FeatureFlags.enable_backward_compatibility()
            assert FeatureFlags.enable_deprecation_warnings()

            # These flags default to False
            assert not FeatureFlags.use_new_backtester()
            assert not FeatureFlags.use_optimization_orchestrator()

    def test_environment_variable_support(self):
        """Test that environment variables control flag values."""
        env_vars = {
            "USE_NEW_OPTIMIZATION_ARCHITECTURE": "true",
            "USE_NEW_BACKTESTER": "1",
            "USE_OPTIMIZATION_ORCHESTRATOR": "yes",
            "ENABLE_OPTUNA_GENERATOR": "false",
            "ENABLE_GENETIC_GENERATOR": "0",
            "ENABLE_BACKWARD_COMPATIBILITY": "no",
            "ENABLE_DEPRECATION_WARNINGS": "off",
        }

        with patch.dict(os.environ, env_vars):
            assert FeatureFlags.use_new_optimization_architecture()
            assert FeatureFlags.use_new_backtester()
            assert FeatureFlags.use_optimization_orchestrator()
            assert not FeatureFlags.enable_optuna_generator()
            assert not FeatureFlags.enable_genetic_generator()
            assert not FeatureFlags.enable_backward_compatibility()
            assert not FeatureFlags.enable_deprecation_warnings()

    def test_disable_optuna_context_manager(self):
        """Test the disable_optuna context manager."""
        # Initially enabled
        assert FeatureFlags.enable_optuna_generator()

        with FeatureFlags.disable_optuna():
            assert not FeatureFlags.enable_optuna_generator()

        # Restored after context
        assert FeatureFlags.enable_optuna_generator()

    def test_disable_genetic_context_manager(self):
        """Test the disable_genetic context manager."""
        # Initially enabled
        assert FeatureFlags.enable_genetic_generator()

        with FeatureFlags.disable_genetic():
            assert not FeatureFlags.enable_genetic_generator()

        # Restored after context
        assert FeatureFlags.enable_genetic_generator()

    def test_disable_all_optimizers_context_manager(self):
        """Test the disable_all_optimizers context manager."""
        # Set some flags to True initially
        with patch.dict(
            os.environ,
            {"USE_NEW_OPTIMIZATION_ARCHITECTURE": "true", "USE_OPTIMIZATION_ORCHESTRATOR": "true"},
        ):
            assert FeatureFlags.use_new_optimization_architecture()
            assert FeatureFlags.use_optimization_orchestrator()
            assert FeatureFlags.enable_optuna_generator()
            assert FeatureFlags.enable_genetic_generator()

            with FeatureFlags.disable_all_optimizers():
                assert not FeatureFlags.use_new_optimization_architecture()
                assert not FeatureFlags.use_optimization_orchestrator()
                assert not FeatureFlags.enable_optuna_generator()
                assert not FeatureFlags.enable_genetic_generator()

            # Restored after context
            assert FeatureFlags.use_new_optimization_architecture()
            assert FeatureFlags.use_optimization_orchestrator()
            assert FeatureFlags.enable_optuna_generator()
            assert FeatureFlags.enable_genetic_generator()

    def test_enable_new_architecture_context_manager(self):
        """Test the enable_new_architecture context manager."""
        # Initially disabled
        with patch.dict(os.environ, {}, clear=True):
            assert FeatureFlags.use_new_optimization_architecture()
            assert not FeatureFlags.use_new_backtester()
            assert not FeatureFlags.use_optimization_orchestrator()

            with FeatureFlags.enable_new_architecture():
                assert FeatureFlags.use_new_optimization_architecture()
                assert FeatureFlags.use_new_backtester()
                assert FeatureFlags.use_optimization_orchestrator()

            # Restored after context
            assert FeatureFlags.use_new_optimization_architecture()
            assert not FeatureFlags.use_new_backtester()
            assert not FeatureFlags.use_optimization_orchestrator()

    def test_disable_backward_compatibility_context_manager(self):
        """Test the disable_backward_compatibility context manager."""
        # Initially enabled
        assert FeatureFlags.enable_backward_compatibility()

        with FeatureFlags.disable_backward_compatibility():
            assert not FeatureFlags.enable_backward_compatibility()

        # Restored after context
        assert FeatureFlags.enable_backward_compatibility()

    def test_nested_context_managers(self):
        """Test that context managers can be nested properly."""
        assert FeatureFlags.enable_optuna_generator()
        assert FeatureFlags.enable_genetic_generator()

        with FeatureFlags.disable_optuna():
            assert not FeatureFlags.enable_optuna_generator()
            assert FeatureFlags.enable_genetic_generator()

            with FeatureFlags.disable_genetic():
                assert not FeatureFlags.enable_optuna_generator()
                assert not FeatureFlags.enable_genetic_generator()

            # Inner context restored
            assert not FeatureFlags.enable_optuna_generator()
            assert FeatureFlags.enable_genetic_generator()

        # Outer context restored
        assert FeatureFlags.enable_optuna_generator()
        assert FeatureFlags.enable_genetic_generator()

    def test_context_manager_exception_handling(self):
        """Test that context managers restore state even when exceptions occur."""
        assert FeatureFlags.enable_optuna_generator()

        try:
            with FeatureFlags.disable_optuna():
                assert not FeatureFlags.enable_optuna_generator()
                raise ValueError("Test exception")
        except ValueError:
            pass

        # State should be restored despite exception
        assert FeatureFlags.enable_optuna_generator()

    def test_get_all_flags(self):
        """Test the get_all_flags method."""
        flags = FeatureFlags.get_all_flags()

        expected_flags = {
            "use_new_optimization_architecture",
            "use_new_backtesting_architecture",
            "use_new_backtester",
            "use_optimization_orchestrator",
            "enable_optuna_generator",
            "enable_genetic_generator",
            "enable_backward_compatibility",
            "enable_deprecation_warnings",
        }

        assert set(flags.keys()) == expected_flags
        assert all(isinstance(value, bool) for value in flags.values())

    def test_thread_safety(self):
        """Test that context managers are thread-safe."""
        results = {}

        def thread1():
            with FeatureFlags.disable_optuna():
                time.sleep(0.1)  # Give other thread time to run
                results["thread1"] = FeatureFlags.enable_optuna_generator()

        def thread2():
            time.sleep(0.05)  # Start slightly after thread1
            results["thread2"] = FeatureFlags.enable_optuna_generator()

        t1 = threading.Thread(target=thread1)
        t2 = threading.Thread(target=thread2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Thread1 should see disabled flag, thread2 should see enabled
        assert not results["thread1"]
        assert results["thread2"]

    def test_override_with_existing_environment_variable(self):
        """Test that overrides work even when environment variables are set."""
        with patch.dict(os.environ, {"ENABLE_OPTUNA_GENERATOR": "true"}):
            # Environment variable says True
            assert FeatureFlags.enable_optuna_generator()

            with FeatureFlags.disable_optuna():
                # Override should take precedence
                assert not FeatureFlags.enable_optuna_generator()

            # Back to environment variable value
            assert FeatureFlags.enable_optuna_generator()


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_is_new_architecture_enabled(self):
        """Test the is_new_architecture_enabled function."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_new_architecture_enabled()

        with patch.dict(os.environ, {"USE_NEW_OPTIMIZATION_ARCHITECTURE": "true"}):
            assert is_new_architecture_enabled()

        with patch.dict(os.environ, {"USE_NEW_BACKTESTER": "true"}):
            assert is_new_architecture_enabled()

        with patch.dict(os.environ, {"USE_OPTIMIZATION_ORCHESTRATOR": "true"}):
            assert is_new_architecture_enabled()

    def test_should_show_migration_warnings(self):
        """Test the should_show_migration_warnings function."""
        # Both flags need to be True
        with patch.dict(
            os.environ,
            {"ENABLE_DEPRECATION_WARNINGS": "true", "ENABLE_BACKWARD_COMPATIBILITY": "true"},
        ):
            assert should_show_migration_warnings()

        # If either is False, should return False
        with patch.dict(
            os.environ,
            {"ENABLE_DEPRECATION_WARNINGS": "false", "ENABLE_BACKWARD_COMPATIBILITY": "true"},
        ):
            assert not should_show_migration_warnings()

        with patch.dict(
            os.environ,
            {"ENABLE_DEPRECATION_WARNINGS": "true", "ENABLE_BACKWARD_COMPATIBILITY": "false"},
        ):
            assert not should_show_migration_warnings()


class TestFeatureFlagIntegration:
    """Integration tests for feature flags with other components."""

    def test_isolation_testing_scenario(self):
        """Test a realistic scenario for component isolation testing."""
        # Test that backtester can run with all optimizers disabled
        with FeatureFlags.disable_all_optimizers():
            # In a real test, this would instantiate and run the backtester
            assert not FeatureFlags.enable_optuna_generator()
            assert not FeatureFlags.enable_genetic_generator()
            assert not FeatureFlags.use_optimization_orchestrator()
            assert not FeatureFlags.use_new_optimization_architecture()
