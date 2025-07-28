
"""Consolidated tests for backward compatibility and migration of timing configurations.

These tests ensure that:
1. Legacy configurations are properly migrated to the new timing_config structure.
2. All existing strategy configurations remain valid and produce identical behavior.
3. The migration process is robust and provides helpful warnings and errors.
"""


import pytest
import yaml
import warnings
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from src.portfolio_backtester.timing.backward_compatibility import (
    migrate_legacy_config,
    ensure_backward_compatibility,
    validate_legacy_behavior,
    get_migration_warnings,
    TimingConfigValidator,
    KNOWN_DAILY_SIGNAL_STRATEGIES
)


class TestLegacyConfigMigration:
    """Test the migration of various legacy configuration patterns."""

    def test_migrate_monthly_momentum_strategy(self):
        """Test migration of a typical monthly momentum strategy."""
        legacy_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'strategy_params': {'lookback_months': 6}
        }
        migrated = migrate_legacy_config(legacy_config)
        assert 'timing_config' in migrated
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'time_based'
        assert timing_config['rebalance_frequency'] == 'M'
        assert timing_config['rebalance_offset'] == 0

    def test_migrate_daily_uvxy_strategy(self):
        """Test migration of UVXY strategy (known daily signal strategy)."""
        legacy_config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D',
            'strategy_params': {'rsi_period': 2}
        }
        migrated = migrate_legacy_config(legacy_config)
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'signal_based'
        assert timing_config['scan_frequency'] == 'D'
        assert timing_config['min_holding_period'] == 1
        assert timing_config['max_holding_period'] == 1  # UVXY-specific default

    def test_migrate_explicit_daily_signals_flag(self):
        """Test migration of a strategy with the explicit `daily_signals` flag."""
        legacy_config = {
            'strategy': 'custom_daily',
            'daily_signals': True,
            'min_holding_period': 5
        }
        migrated = migrate_legacy_config(legacy_config)
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'signal_based'
        assert timing_config['scan_frequency'] == 'D'
        assert timing_config['min_holding_period'] == 5

    def test_migrate_already_new_format_is_unchanged(self):
        """Test that configurations already in the new format are not modified."""
        new_format_config = {
            'strategy': 'momentum',
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'Q'
            }
        }
        migrated = migrate_legacy_config(new_format_config)
        assert migrated == new_format_config

    def test_migrate_invalid_existing_timing_config_raises_error(self):
        """Test that an invalid existing timing_config raises a helpful error."""
        invalid_config = {
            'strategy': 'momentum',
            'timing_config': {'mode': 'invalid_mode'}
        }
        with pytest.raises(ValueError, match="Invalid timing mode 'invalid_mode'"):
            migrate_legacy_config(invalid_config)


class TestLegacyBehaviorAndValidation:
    """Test that the behavior of migrated configurations is identical to legacy behavior."""

    def test_validate_time_based_migration_behavior(self):
        """Test that a valid time-based migration passes behavior validation."""
        old_config = {'strategy': 'momentum', 'rebalance_frequency': 'M'}
        new_config = migrate_legacy_config(old_config)
        assert validate_legacy_behavior(old_config, new_config) is True

    def test_validate_signal_based_migration_behavior(self):
        """Test that a valid signal-based migration passes behavior validation."""
        old_config = {'strategy': 'uvxy_rsi', 'rebalance_frequency': 'D'}
        new_config = migrate_legacy_config(old_config)
        assert validate_legacy_behavior(old_config, new_config) is True

    def test_validate_behavior_mismatch_in_mode(self):
        """Test that behavior validation catches a mismatch in timing mode."""
        old_config = {'strategy': 'momentum', 'rebalance_frequency': 'M'}
        new_config = {
            'strategy': 'momentum',
            'timing_config': {'mode': 'signal_based', 'scan_frequency': 'D'}
        }
        assert validate_legacy_behavior(old_config, new_config) is False

    def test_validate_behavior_mismatch_in_frequency(self):
        """Test that behavior validation catches a mismatch in rebalance frequency."""
        old_config = {'strategy': 'momentum', 'rebalance_frequency': 'M'}
        new_config = {
            'strategy': 'momentum',
            'timing_config': {'mode': 'time_based', 'rebalance_frequency': 'Q'}
        }
        assert validate_legacy_behavior(old_config, new_config) is False


class TestExistingScenarioCompatibility:
    """Test that all existing scenarios in scenarios.yaml are compatible."""

    @pytest.fixture
    def scenarios_config(self):
        """Load scenarios configuration file."""
        config_path = Path("config/scenarios.yaml")
        if not config_path.exists():
            pytest.skip("scenarios.yaml not found")
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError:
                pytest.skip("scenarios.yaml is not a valid YAML file")

    def test_all_scenarios_migrate_successfully(self, scenarios_config):
        """Test that all scenarios in the config file can be migrated successfully."""
        backtest_scenarios = scenarios_config.get('BACKTEST_SCENARIOS', [])
        failed_scenarios = []

        for scenario in backtest_scenarios:
            scenario_name = scenario.get('name', 'unnamed')
            try:
                strategy_config = {k: v for k, v in scenario.items() if k != 'name'}
                migrated = ensure_backward_compatibility(strategy_config)
                assert 'timing_config' in migrated
                assert validate_legacy_behavior(strategy_config, migrated)
            except Exception as e:
                failed_scenarios.append((scenario_name, str(e)))

        if failed_scenarios:
            failure_report = "\n".join([f"  {name}: {error}" for name, error in failed_scenarios])
            pytest.fail(f"Failed to migrate {len(failed_scenarios)} scenarios:\n{failure_report}")


class TestNumericalAndPerformanceEquivalence:
    """Test for numerical and performance equivalence after migration."""

    def test_monthly_rebalancing_dates_identical(self):
        """Test that monthly rebalancing produces identical dates."""
        legacy_config = {'rebalance_frequency': 'M'}
        migrated_config = migrate_legacy_config(legacy_config)
        from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
        timing_controller = TimeBasedTiming(migrated_config['timing_config'])

        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        available_dates = pd.date_range(start_date, end_date, freq='B')

        rebalance_dates = timing_controller.get_rebalance_dates(start_date, end_date, available_dates, None)

        expected_dates = pd.to_datetime([
            '2023-01-31', '2023-02-28', '2023-03-31', '2023-04-28', '2023-05-31',
            '2023-06-30', '2023-07-31', '2023-08-31', '2023-09-29', '2023-10-31',
            '2023-11-30', '2023-12-29'
        ])
        pd.testing.assert_index_equal(rebalance_dates, expected_dates)

    def test_rebalance_offset_calculation_accurate(self):
        """Test that rebalance offset calculations are numerically accurate."""
        legacy_config = {'rebalance_frequency': 'M', 'rebalance_offset': 5}
        migrated_config = migrate_legacy_config(legacy_config)
        from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
        timing_controller = TimeBasedTiming(migrated_config['timing_config'])

        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-02-28')
        available_dates = pd.date_range(start_date, end_date, freq='B')

        rebalance_dates = timing_controller.get_rebalance_dates(start_date, end_date, available_dates, None)
        jan_rebalance = rebalance_dates[0]
        expected_date = pd.Timestamp('2023-02-06')  # Jan 31 + 5 business days
        assert jan_rebalance == expected_date

    def test_initialization_performance_is_maintained(self):
        """Test that timing controller initialization doesn't degrade performance."""
        import time
        configs = [{'rebalance_frequency': 'M'}, {'strategy': 'uvxy_rsi'}]
        initialization_times = []

        for config in configs:
            start_time = time.time()
            migrated_config = ensure_backward_compatibility(config)
            if migrated_config['timing_config']['mode'] == 'time_based':
                from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
                TimeBasedTiming(migrated_config['timing_config'])
            else:
                from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
                SignalBasedTiming(migrated_config['timing_config'])
            end_time = time.time()
            initialization_times.append(end_time - start_time)

        for init_time in initialization_times:
            assert init_time < 0.01, f"Initialization took too long: {init_time:.4f}s"

if __name__ == '__main__':
    pytest.main([__file__])
