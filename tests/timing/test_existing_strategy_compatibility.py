"""
Tests to ensure all existing strategy configurations remain valid
after introducing the timing system.

These tests verify that:
1. All strategies in scenarios.yaml can be migrated successfully
2. Legacy behavior produces identical results
3. No existing functionality is broken
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from src.portfolio_backtester.timing.backward_compatibility import (
    ensure_backward_compatibility,
    migrate_legacy_config,
    validate_legacy_behavior
)


class TestExistingScenarioConfigurations:
    """Test that all existing scenario configurations work with timing system."""
    
    @pytest.fixture
    def scenarios_config(self):
        """Load scenarios configuration file."""
        config_path = Path("config/scenarios.yaml")
        if not config_path.exists():
            pytest.skip("scenarios.yaml not found")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_all_scenarios_migrate_successfully(self, scenarios_config):
        """Test that all scenarios in config can be migrated."""
        backtest_scenarios = scenarios_config.get('BACKTEST_SCENARIOS', [])
        
        failed_scenarios = []
        
        for scenario in backtest_scenarios:
            scenario_name = scenario.get('name', 'unnamed')
            
            try:
                # Create strategy config from scenario
                strategy_config = {
                    'strategy': scenario.get('strategy'),
                    'rebalance_frequency': scenario.get('rebalance_frequency', 'M'),
                    'strategy_params': scenario.get('strategy_params', {}),
                    'position_sizer': scenario.get('position_sizer', 'equal_weight'),
                    'transaction_costs_bps': scenario.get('transaction_costs_bps', 10)
                }
                
                # Add any other scenario-level parameters that might affect timing
                for key in ['daily_signals', 'signal_based', 'scan_frequency', 
                           'min_holding_period', 'max_holding_period']:
                    if key in scenario:
                        strategy_config[key] = scenario[key]
                
                # Test migration
                migrated = ensure_backward_compatibility(strategy_config)
                
                # Verify timing_config was added
                assert 'timing_config' in migrated, f"No timing_config added for {scenario_name}"
                
                # Verify timing_config is valid
                timing_config = migrated['timing_config']
                assert 'mode' in timing_config, f"No mode in timing_config for {scenario_name}"
                assert timing_config['mode'] in ['time_based', 'signal_based'], \
                    f"Invalid timing mode for {scenario_name}: {timing_config['mode']}"
                
                # Verify behavior preservation
                assert validate_legacy_behavior(strategy_config, migrated), \
                    f"Behavior validation failed for {scenario_name}"
                
            except Exception as e:
                failed_scenarios.append((scenario_name, str(e)))
        
        # Report all failures at once for better debugging
        if failed_scenarios:
            failure_report = "\n".join([f"  {name}: {error}" for name, error in failed_scenarios])
            pytest.fail(f"Failed to migrate {len(failed_scenarios)} scenarios:\n{failure_report}")
    
    def test_uvxy_scenario_gets_signal_based_timing(self, scenarios_config):
        """Test that UVXY scenario gets signal-based timing."""
        backtest_scenarios = scenarios_config.get('BACKTEST_SCENARIOS', [])
        
        uvxy_scenario = None
        for scenario in backtest_scenarios:
            if scenario.get('strategy') == 'uvxy_rsi':
                uvxy_scenario = scenario
                break
        
        if uvxy_scenario is None:
            pytest.skip("UVXY scenario not found in config")
        
        strategy_config = {
            'strategy': uvxy_scenario.get('strategy'),
            'rebalance_frequency': uvxy_scenario.get('rebalance_frequency', 'D'),
            'strategy_params': uvxy_scenario.get('strategy_params', {})
        }
        
        migrated = ensure_backward_compatibility(strategy_config)
        timing_config = migrated['timing_config']
        
        assert timing_config['mode'] == 'signal_based'
        assert timing_config['scan_frequency'] == 'D'
    
    def test_monthly_strategies_get_time_based_timing(self, scenarios_config):
        """Test that monthly strategies get time-based timing."""
        backtest_scenarios = scenarios_config.get('BACKTEST_SCENARIOS', [])
        
        monthly_scenarios = [
            scenario for scenario in backtest_scenarios
            if scenario.get('rebalance_frequency') in ['M', 'ME'] and 
               scenario.get('strategy') != 'uvxy_rsi'
        ]
        
        if not monthly_scenarios:
            pytest.skip("No monthly scenarios found")
        
        for scenario in monthly_scenarios[:3]:  # Test first 3 to avoid long test times
            scenario_name = scenario.get('name', 'unnamed')
            
            strategy_config = {
                'strategy': scenario.get('strategy'),
                'rebalance_frequency': scenario.get('rebalance_frequency'),
                'strategy_params': scenario.get('strategy_params', {})
            }
            
            migrated = ensure_backward_compatibility(strategy_config)
            timing_config = migrated['timing_config']
            
            assert timing_config['mode'] == 'time_based', \
                f"Expected time_based mode for {scenario_name}"
            assert timing_config['rebalance_frequency'] in ['M', 'ME'], \
                f"Expected monthly frequency for {scenario_name}"


class TestCommonStrategyPatterns:
    """Test common strategy configuration patterns."""
    
    def test_momentum_strategy_patterns(self):
        """Test various momentum strategy configurations."""
        momentum_configs = [
            {
                'strategy': 'momentum',
                'rebalance_frequency': 'M',
                'strategy_params': {'lookback_months': 6, 'num_holdings': 10}
            },
            {
                'strategy': 'momentum',
                'rebalance_frequency': 'Q',
                'strategy_params': {'lookback_months': 12, 'num_holdings': 20}
            },
            {
                'strategy': 'sharpe_momentum',
                'rebalance_frequency': 'M',
                'strategy_params': {'rolling_window': 3}
            },
            {
                'strategy': 'sortino_momentum',
                'rebalance_frequency': 'M',
                'strategy_params': {'rolling_window': 6}
            }
        ]
        
        for config in momentum_configs:
            migrated = ensure_backward_compatibility(config)
            
            assert migrated['timing_config']['mode'] == 'time_based'
            assert migrated['timing_config']['rebalance_frequency'] == config['rebalance_frequency']
            assert validate_legacy_behavior(config, migrated)
    
    def test_factor_strategy_patterns(self):
        """Test factor strategy configurations."""
        factor_configs = [
            {
                'strategy': 'low_volatility_factor',
                'rebalance_frequency': 'M',
                'strategy_params': {'volatility_lookback_days': 252}
            },
            {
                'strategy': 'value_factor',
                'rebalance_frequency': 'Q',
                'strategy_params': {'book_to_market': True}
            }
        ]
        
        for config in factor_configs:
            migrated = ensure_backward_compatibility(config)
            
            assert migrated['timing_config']['mode'] == 'time_based'
            assert validate_legacy_behavior(config, migrated)
    
    def test_daily_signal_strategy_patterns(self):
        """Test daily signal strategy configurations."""
        daily_configs = [
            {
                'strategy': 'uvxy_rsi',
                'rebalance_frequency': 'D',
                'strategy_params': {'rsi_period': 2, 'rsi_threshold': 30}
            },
            {
                'strategy': 'custom_daily',
                'daily_signals': True,
                'scan_frequency': 'D',
                'strategy_params': {'lookback': 5}
            },
            {
                'strategy': 'breakout',
                'signal_based': True,
                'scan_frequency': 'D',
                'min_holding_period': 2,
                'max_holding_period': 10
            }
        ]
        
        for config in daily_configs:
            migrated = ensure_backward_compatibility(config)
            
            assert migrated['timing_config']['mode'] == 'signal_based'
            assert validate_legacy_behavior(config, migrated)


class TestPositionSizerCompatibility:
    """Test that position sizers work with timing system."""
    
    def test_position_sizer_configurations(self):
        """Test various position sizer configurations."""
        sizer_configs = [
            {
                'strategy': 'momentum',
                'rebalance_frequency': 'M',
                'position_sizer': 'equal_weight',
                'strategy_params': {'num_holdings': 10}
            },
            {
                'strategy': 'momentum',
                'rebalance_frequency': 'M',
                'position_sizer': 'rolling_sharpe',
                'strategy_params': {'sizer_sharpe_window': 6}
            },
            {
                'strategy': 'momentum',
                'rebalance_frequency': 'M',
                'position_sizer': 'rolling_downside_volatility',
                'strategy_params': {'target_volatility': 0.15, 'sizer_dvol_window': 12}
            }
        ]
        
        for config in sizer_configs:
            migrated = ensure_backward_compatibility(config)
            
            # Should preserve position_sizer setting
            assert migrated.get('position_sizer') == config['position_sizer']
            
            # Should have valid timing config
            assert migrated['timing_config']['mode'] == 'time_based'
            assert validate_legacy_behavior(config, migrated)


class TestOptimizationParameterCompatibility:
    """Test that optimization parameters work with timing system."""
    
    def test_optimization_parameters_preserved(self):
        """Test that optimization parameters are preserved during migration."""
        config_with_optimization = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'optimize': [
                {'parameter': 'lookback_months', 'min_value': 3, 'max_value': 12},
                {'parameter': 'num_holdings', 'min_value': 5, 'max_value': 25}
            ],
            'strategy_params': {'lookback_months': 6, 'num_holdings': 10}
        }
        
        migrated = ensure_backward_compatibility(config_with_optimization)
        
        # Optimization parameters should be preserved
        assert migrated.get('optimize') == config_with_optimization['optimize']
        
        # Should have valid timing config
        assert migrated['timing_config']['mode'] == 'time_based'
        assert validate_legacy_behavior(config_with_optimization, migrated)


class TestTransactionCostCompatibility:
    """Test that transaction cost settings work with timing system."""
    
    def test_transaction_cost_settings_preserved(self):
        """Test that transaction cost settings are preserved."""
        config_with_costs = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'transaction_costs_bps': 15,
            'strategy_params': {'lookback_months': 6}
        }
        
        migrated = ensure_backward_compatibility(config_with_costs)
        
        # Transaction costs should be preserved
        assert migrated.get('transaction_costs_bps') == 15
        
        # Should have valid timing config
        assert migrated['timing_config']['mode'] == 'time_based'
        assert validate_legacy_behavior(config_with_costs, migrated)


class TestRiskManagementCompatibility:
    """Test that risk management settings work with timing system."""
    
    def test_stop_loss_configurations(self):
        """Test stop loss configurations with timing system."""
        stop_loss_configs = [
            {
                'strategy': 'momentum',
                'rebalance_frequency': 'M',
                'stop_loss_config': {'type': 'NoStopLoss'},
                'strategy_params': {'lookback_months': 6}
            },
            {
                'strategy': 'momentum',
                'rebalance_frequency': 'M',
                'stop_loss_config': {
                    'type': 'AtrBasedStopLoss',
                    'atr_length': 14,
                    'atr_multiple': 2.5
                },
                'strategy_params': {'lookback_months': 6}
            }
        ]
        
        for config in stop_loss_configs:
            migrated = ensure_backward_compatibility(config)
            
            # Stop loss config should be preserved
            assert migrated.get('stop_loss_config') == config['stop_loss_config']
            
            # Should have valid timing config
            assert migrated['timing_config']['mode'] == 'time_based'
            assert validate_legacy_behavior(config, migrated)
    
    def test_leverage_and_smoothing_settings(self):
        """Test leverage and smoothing settings with timing system."""
        config_with_risk_params = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'strategy_params': {
                'lookback_months': 6,
                'leverage': 1.5,
                'smoothing_lambda': 0.3,
                'long_only': True
            }
        }
        
        migrated = ensure_backward_compatibility(config_with_risk_params)
        
        # Risk parameters should be preserved
        strategy_params = migrated['strategy_params']
        assert strategy_params['leverage'] == 1.5
        assert strategy_params['smoothing_lambda'] == 0.3
        assert strategy_params['long_only'] is True
        
        # Should have valid timing config
        assert migrated['timing_config']['mode'] == 'time_based'
        assert validate_legacy_behavior(config_with_risk_params, migrated)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_strategy_config(self):
        """Test handling of minimal strategy configuration."""
        minimal_config = {
            'strategy': 'momentum'
        }
        
        migrated = ensure_backward_compatibility(minimal_config)
        
        # Should get default timing config
        assert 'timing_config' in migrated
        timing_config = migrated['timing_config']
        assert timing_config['mode'] == 'time_based'
        assert timing_config['rebalance_frequency'] == 'M'  # Default
    
    def test_config_with_none_values(self):
        """Test handling of configuration with None values."""
        config_with_nones = {
            'strategy': 'momentum',
            'rebalance_frequency': None,
            'strategy_params': None
        }
        
        migrated = ensure_backward_compatibility(config_with_nones)
        
        # Should handle None values gracefully
        assert 'timing_config' in migrated
        assert migrated['timing_config']['mode'] == 'time_based'
    
    def test_config_with_unexpected_parameters(self):
        """Test handling of configuration with unexpected parameters."""
        config_with_extras = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'unexpected_param': 'value',
            'another_unexpected': 123,
            'strategy_params': {'lookback_months': 6}
        }
        
        migrated = ensure_backward_compatibility(config_with_extras)
        
        # Unexpected parameters should be preserved
        assert migrated.get('unexpected_param') == 'value'
        assert migrated.get('another_unexpected') == 123
        
        # Should still have valid timing config
        assert migrated['timing_config']['mode'] == 'time_based'


if __name__ == '__main__':
    pytest.main([__file__])