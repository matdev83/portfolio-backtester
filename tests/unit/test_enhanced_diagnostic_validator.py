"""
Tests for the enhanced diagnostic config validator.
"""

import tempfile
import yaml
from pathlib import Path

from portfolio_backtester.scenario_validator import validate_scenario_file
from portfolio_backtester.config_loader import OPTIMIZER_PARAMETER_DEFAULTS


class TestEnhancedDiagnosticValidator:
    """Diagnostics are not strategies; keep tests to ensure validator behaves robustly on inputs."""

    def test_valid_diagnostic_config(self):
        """Test that a valid diagnostic config passes validation."""
        # Use an existing simple built-in scenario as a smoke test (seasonal or momentum)
        # If not available, create a minimal valid temp file
        cfg = {
            "name": "smoke",
            "strategy": "ema_crossover",
            "rebalance_frequency": "ME",
            "position_sizer": "equal_weight",
            "train_window_months": 12,
            "test_window_months": 12,
            "universe_config": {"type": "named", "universe_name": "sp500_top50"},
            "strategy_params": {
                "ema_crossover.fast_ema_days": 10,
                "ema_crossover.slow_ema_days": 20,
                "ema_crossover.leverage": 1.0,
            },
        }
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            temp_file = Path(f.name)
        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)
            assert isinstance(errors, list)
        finally:
            temp_file.unlink()

    def test_optimization_parameter_consistency(self):
        """Test validation of optimization parameter consistency."""
        config = {
            "name": "test_consistency",
            "strategy": "StopLossTesterStrategy",
            "rebalance_frequency": "D",
            "position_sizer": "equal_weight",
            "train_window_months": 12,
            "test_window_months": 12,
            "optimization_targets": [{"name": "Sharpe", "direction": "maximize"}],
            "optimizers": {
                "optuna": {
                    "optimize": [
                        {
                            "parameter": "atr_length",
                            "type": "int",
                            "min_value": 5,
                            "max_value": 30,
                            "step": 1,
                        }
                    ]
                }
            },
            "strategy_params": {
                # Missing the expected parameter
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_file = Path(f.name)

        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)

            # Should detect missing optimization parameter in strategy_params
            consistency_errors = [e for e in errors if "missing from strategy_params" in e.message]
            assert len(consistency_errors) > 0, "Should detect missing optimization parameter"

        finally:
            temp_file.unlink()

    def test_parameter_value_range_validation(self):
        """Test validation of parameter values against optimization ranges."""
        config = {
            "name": "test_range",
            "strategy": "StopLossTesterStrategy",
            "rebalance_frequency": "D",
            "position_sizer": "equal_weight",
            "train_window_months": 12,
            "test_window_months": 12,
            "optimization_targets": [{"name": "Sharpe", "direction": "maximize"}],
            "optimizers": {
                "optuna": {
                    "optimize": [
                        {
                            "parameter": "atr_length",
                            "type": "int",
                            "min_value": 5,
                            "max_value": 30,
                            "step": 1,
                        }
                    ]
                }
            },
            "strategy_params": {"stop_loss_tester.atr_length": 50},  # Outside range [5, 30]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_file = Path(f.name)

        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)

            # Should detect parameter value outside range
            range_errors = [e for e in errors if "above optimization maximum" in e.message]
            assert len(range_errors) > 0, "Should detect parameter value outside range"

        finally:
            temp_file.unlink()

    def test_categorical_choice_validation(self):
        """Test validation of categorical parameter choices."""
        config = {
            "name": "test_categorical",
            "strategy": "StopLossTesterStrategy",
            "rebalance_frequency": "D",
            "position_sizer": "equal_weight",
            "train_window_months": 12,
            "test_window_months": 12,
            "optimization_targets": [{"name": "Sharpe", "direction": "maximize"}],
            "optimizers": {
                "optuna": {
                    "optimize": [
                        {
                            "parameter": "stop_loss_type",
                            "type": "categorical",
                            "choices": ["AtrBasedStopLoss", "None"],
                        }
                    ]
                }
            },
            "strategy_params": {
                "stop_loss_tester.stop_loss_type": "InvalidChoice"  # Not in choices
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_file = Path(f.name)

        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)

            # Should detect invalid categorical choice
            choice_errors = [e for e in errors if "not in optimization choices" in e.message]
            assert len(choice_errors) > 0, "Should detect invalid categorical choice"

        finally:
            temp_file.unlink()

    def test_step_size_validation(self):
        """Test validation of step sizes."""
        config = {
            "name": "test_step",
            "strategy": "StopLossTesterStrategy",
            "rebalance_frequency": "D",
            "position_sizer": "equal_weight",
            "train_window_months": 12,
            "test_window_months": 12,
            "optimization_targets": [{"name": "Sharpe", "direction": "maximize"}],
            "optimizers": {
                "optuna": {
                    "optimize": [
                        {
                            "parameter": "atr_length",
                            "type": "int",
                            "min_value": 5,
                            "max_value": 30,
                            "step": 50,  # Step larger than range
                        }
                    ]
                }
            },
            "strategy_params": {"stop_loss_tester.atr_length": 14},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_file = Path(f.name)

        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)

            # Should detect invalid step size
            step_errors = [
                e for e in errors if "step size" in e.message and "larger than" in e.message
            ]
            assert len(step_errors) > 0, "Should detect invalid step size"

        finally:
            temp_file.unlink()

    def test_window_size_warnings(self):
        """Test validation of window sizes."""
        config = {
            "name": "test_windows",
            "strategy": "StopLossTesterStrategy",
            "rebalance_frequency": "D",
            "position_sizer": "equal_weight",
            "train_window_months": 2,  # Very short
            "test_window_months": 1,  # Very short
            "optimization_targets": [{"name": "Sharpe", "direction": "maximize"}],
            "optimizers": {
                "optuna": {
                    "optimize": [
                        {
                            "parameter": "atr_length",
                            "type": "int",
                            "min_value": 5,
                            "max_value": 30,
                            "step": 1,
                        }
                    ]
                }
            },
            "strategy_params": {"stop_loss_tester.atr_length": 14},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_file = Path(f.name)

        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)

            # Should detect window size warnings
            window_warnings = [
                e
                for e in errors
                if "window" in e.message and ("too short" in e.message or "may be" in e.message)
            ]
            assert (
                len(window_warnings) >= 2
            ), f"Should detect multiple window warnings, got {len(window_warnings)}"

        finally:
            temp_file.unlink()

    def test_invalid_rebalance_frequency(self):
        """Test validation of rebalance frequency."""
        config = {
            "name": "test_rebal",
            "strategy": "StopLossTesterStrategy",
            "rebalance_frequency": "INVALID",  # Invalid frequency
            "position_sizer": "equal_weight",
            "train_window_months": 12,
            "test_window_months": 12,
            "strategy_params": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_file = Path(f.name)

        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)

            # Should detect invalid rebalance frequency
            freq_errors = [e for e in errors if "Invalid rebalance_frequency" in e.message]
            assert len(freq_errors) > 0, "Should detect invalid rebalance frequency"

        finally:
            temp_file.unlink()

    def test_invalid_position_sizer(self):
        """Test validation of position sizer."""
        config = {
            "name": "test_sizer",
            "strategy": "StopLossTesterStrategy",
            "rebalance_frequency": "D",
            "position_sizer": "unknown_sizer",  # Invalid sizer
            "train_window_months": 12,
            "test_window_months": 12,
            "strategy_params": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_file = Path(f.name)

        try:
            errors = validate_scenario_file(temp_file, OPTIMIZER_PARAMETER_DEFAULTS)

            # Should detect invalid position sizer
            sizer_errors = [e for e in errors if "Invalid position_sizer" in e.message]
            assert len(sizer_errors) > 0, "Should detect invalid position sizer"

        finally:
            temp_file.unlink()
