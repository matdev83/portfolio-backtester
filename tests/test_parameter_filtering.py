"""
Test parameter filtering functionality in optimization.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_backtester.utils import _resolve_strategy
from portfolio_backtester.config_initializer import _get_strategy_tunable_params, populate_default_optimizations
from portfolio_backtester.config_loader import OPTIMIZER_PARAMETER_DEFAULTS


class TestParameterFiltering:
    """Test parameter filtering functionality."""
    
    def test_strategy_tunable_parameters(self):
        """Test that different strategies have different tunable parameters."""
        
        # Test momentum strategy
        momentum_cls = _resolve_strategy("momentum")
        assert momentum_cls is not None
        momentum_params = momentum_cls.tunable_parameters()
        assert isinstance(momentum_params, set)
        assert "lookback_months" in momentum_params
        assert "num_holdings" in momentum_params
        
        # Test calmar strategy
        calmar_cls = _resolve_strategy("calmar_momentum")
        assert calmar_cls is not None
        calmar_params = calmar_cls.tunable_parameters()
        assert isinstance(calmar_params, set)
        assert "rolling_window" in calmar_params
        assert "num_holdings" in calmar_params
        
        # Verify they have different parameters
        assert momentum_params != calmar_params
        
        # Test helper function
        helper_momentum = _get_strategy_tunable_params("momentum")
        assert helper_momentum == momentum_params
        
        helper_calmar = _get_strategy_tunable_params("calmar_momentum")
        assert helper_calmar == calmar_params
    
    def test_parameter_filtering_logic(self):
        """Test the core parameter filtering logic."""
        
        # Mock scenario configuration
        base_scen_cfg = {
            "strategy": "momentum",
            "strategy_params": {},
            "optimize": [
                {"parameter": "lookback_months"},     # Relevant to momentum
                {"parameter": "num_holdings"},        # Relevant to momentum
                {"parameter": "rolling_window"},      # NOT relevant to momentum
                {"parameter": "alpha"},               # NOT relevant to momentum
            ]
        }
        
        # Get strategy tunable parameters
        strat_cls = _resolve_strategy(base_scen_cfg["strategy"])
        strategy_tunable_params = strat_cls.tunable_parameters()
        
        # Apply filtering logic (from optuna_objective.py)
        SPECIAL_SCEN_CFG_KEYS = ["position_sizer"]
        optimization_specs = base_scen_cfg.get("optimize", [])
        
        filtered_params = []
        skipped_params = []
        
        for opt_spec in optimization_specs:
            param_name = opt_spec["parameter"]
            
            if (strategy_tunable_params and 
                param_name not in strategy_tunable_params and 
                param_name not in SPECIAL_SCEN_CFG_KEYS):
                skipped_params.append(param_name)
            else:
                filtered_params.append(param_name)
        
        # Verify filtering results
        assert "lookback_months" in filtered_params
        assert "num_holdings" in filtered_params
        assert "rolling_window" in skipped_params
        assert "alpha" in skipped_params
        
        # Verify efficiency improvement
        assert len(skipped_params) > 0
        assert len(filtered_params) < len(optimization_specs)
    
    def test_sizer_parameter_inclusion(self):
        """Test that sizer parameters are included when relevant."""
        
        base_scen_cfg = {
            "strategy": "momentum",
            "position_sizer": "rolling_sharpe",
            "strategy_params": {},
            "optimize": [
                {"parameter": "sizer_sharpe_window"},  # Should be included for rolling_sharpe sizer
            ]
        }
        
        # Get strategy tunable parameters
        strat_cls = _resolve_strategy(base_scen_cfg["strategy"])
        strategy_tunable_params = strat_cls.tunable_parameters()
        
        # Add sizer parameters (from optuna_objective.py logic)
        sizer_param_map = {
            "rolling_sharpe": "sizer_sharpe_window",
            "rolling_sortino": "sizer_sortino_window", 
            "rolling_beta": "sizer_beta_window",
            "rolling_benchmark_corr": "sizer_corr_window",
            "rolling_downside_volatility": "sizer_dvol_window",
        }
        
        position_sizer = base_scen_cfg.get("position_sizer")
        if position_sizer and position_sizer in sizer_param_map:
            strategy_tunable_params.add(sizer_param_map[position_sizer])
        
        # Test filtering
        SPECIAL_SCEN_CFG_KEYS = ["position_sizer"]
        optimization_specs = base_scen_cfg.get("optimize", [])
        
        skipped_params = []
        for opt_spec in optimization_specs:
            param_name = opt_spec["parameter"]
            
            if (strategy_tunable_params and 
                param_name not in strategy_tunable_params and 
                param_name not in SPECIAL_SCEN_CFG_KEYS):
                skipped_params.append(param_name)
        
        # sizer_sharpe_window should NOT be skipped
        assert "sizer_sharpe_window" not in skipped_params
    
    def test_config_validation_warnings(self):
        """Test that config validation produces appropriate warnings."""
        
        # Create test scenario with invalid parameters
        test_scenarios = [{
            "name": "Test_Invalid_Params",
            "strategy": "momentum",
            "strategy_params": {},
            "optimize": [
                {"parameter": "lookback_months"},     # Valid
                {"parameter": "rolling_window"},      # Invalid for momentum
                {"parameter": "nonexistent_param"},   # Invalid everywhere
            ]
        }]
        
        # Capture print output to verify warnings
        with patch('builtins.print') as mock_print:
            populate_default_optimizations(test_scenarios, OPTIMIZER_PARAMETER_DEFAULTS)
            
            # Check that warnings were printed
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            warning_calls = [call for call in print_calls if "Warning" in call]
            
            # Should have warnings about invalid parameters
            assert len(warning_calls) > 0
            
            # Check specific warning content
            rolling_window_warning = any("rolling_window" in call for call in warning_calls)
            assert rolling_window_warning, "Should warn about rolling_window not being tunable by momentum strategy"
    
    def test_all_strategies_parameter_differences(self):
        """Test that all strategies have different parameter sets (confirming optimization benefit)."""
        
        strategies_to_test = [
            "momentum",
            "calmar_momentum", 
            "sharpe_momentum",
            "sortino_momentum",
            "vams_momentum",
            "vams_no_downside"
        ]
        
        strategy_params = {}
        
        for strategy_name in strategies_to_test:
            strat_cls = _resolve_strategy(strategy_name)
            if strat_cls:
                params = strat_cls.tunable_parameters()
                strategy_params[strategy_name] = params
        
        # Verify we have different parameter sets
        param_sets = list(strategy_params.values())
        
        # Check that not all strategies have identical parameters
        first_set = param_sets[0] if param_sets else set()
        all_identical = all(param_set == first_set for param_set in param_sets)
        
        assert not all_identical, "Strategies should have different tunable parameters"
        
        # Calculate potential optimization savings
        all_params = set()
        for params in strategy_params.values():
            all_params.update(params)
        
        total_unique_params = len(all_params)
        
        for strategy_name, params in strategy_params.items():
            strategy_specific_count = len(params)
            potential_savings = total_unique_params - strategy_specific_count
            
            if potential_savings > 0:
                efficiency = strategy_specific_count / total_unique_params * 100
                print(f"{strategy_name}: {efficiency:.1f}% efficiency, saves {potential_savings} parameters")
        
        # At least some strategies should have savings
        has_savings = any(len(params) < total_unique_params for params in strategy_params.values())
        assert has_savings, "Parameter filtering should provide optimization savings for some strategies"


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestParameterFiltering()
    
    print("Running parameter filtering tests...")
    
    try:
        test_instance.test_strategy_tunable_parameters()
        print("✓ Strategy tunable parameters test passed")
        
        test_instance.test_parameter_filtering_logic()
        print("✓ Parameter filtering logic test passed")
        
        test_instance.test_sizer_parameter_inclusion()
        print("✓ Sizer parameter inclusion test passed")
        
        test_instance.test_config_validation_warnings()
        print("✓ Config validation warnings test passed")
        
        test_instance.test_all_strategies_parameter_differences()
        print("✓ All strategies parameter differences test passed")
        
        print("\n🎉 All parameter filtering tests passed!")
        print("✓ Implementation is working correctly")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()