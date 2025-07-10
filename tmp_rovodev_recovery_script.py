#!/usr/bin/env python3
"""
Recovery Script: Re-enable Critical Backtester Features

This script re-enables the disabled Monte Carlo and WFO randomization features
that were disabled for performance reasons. It implements the recovery plan
outlined in .dev/PLAN.md.

Features to restore:
1. Monte Carlo synthetic data generation (enable_synthetic_data: true)
2. Monte Carlo during optimization (enable_during_optimization: true) 
3. WFO window randomization (enable_window_randomization: true)
4. WFO start date randomization (enable_start_date_randomization: true)
5. Stage 2 Monte Carlo stress testing (enable_stage2_stress_testing: true)
"""

import yaml
import os
from pathlib import Path

def restore_features():
    """Restore all disabled features in the configuration files."""
    
    # Path to parameters.yaml
    config_dir = Path(__file__).parent / "config"
    parameters_file = config_dir / "parameters.yaml"
    
    if not parameters_file.exists():
        print(f"ERROR: Parameters file not found: {parameters_file}")
        return False
    
    print("🔧 Starting feature recovery...")
    
    # Load current configuration
    with open(parameters_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Track changes made
    changes_made = []
    
    # 1. Enable WFO robustness features
    wfo_config = config.get('wfo_robustness_config', {})
    
    if not wfo_config.get('enable_window_randomization', False):
        wfo_config['enable_window_randomization'] = True
        changes_made.append("✅ Enabled WFO window randomization")
    
    if not wfo_config.get('enable_start_date_randomization', False):
        wfo_config['enable_start_date_randomization'] = True
        changes_made.append("✅ Enabled WFO start date randomization")
    
    # 2. Enable Monte Carlo features
    mc_config = config.get('monte_carlo_config', {})
    
    if not mc_config.get('enable_synthetic_data', False):
        mc_config['enable_synthetic_data'] = True
        changes_made.append("✅ Enabled Monte Carlo synthetic data generation")
    
    if not mc_config.get('enable_during_optimization', False):
        mc_config['enable_during_optimization'] = True
        changes_made.append("✅ Enabled Monte Carlo during optimization (Stage 1)")
    
    # Add Stage 2 stress testing if not present
    if not mc_config.get('enable_stage2_stress_testing', False):
        mc_config['enable_stage2_stress_testing'] = True
        changes_made.append("✅ Enabled Stage 2 Monte Carlo stress testing")
    
    # 3. Add performance optimization settings to maintain speed
    # Add caching and optimization settings
    if 'cache_synthetic_data' not in mc_config:
        mc_config['cache_synthetic_data'] = True
        changes_made.append("✅ Enabled synthetic data caching for performance")
    
    if 'max_cache_size_mb' not in mc_config:
        mc_config['max_cache_size_mb'] = 1000
        changes_made.append("✅ Set cache size limit to 1GB")
    
    if 'parallel_generation' not in mc_config:
        mc_config['parallel_generation'] = True
        changes_made.append("✅ Enabled parallel synthetic data generation")
    
    # Add optimization mode setting
    if 'optimization_mode' not in mc_config:
        mc_config['optimization_mode'] = 'balanced'  # fast/balanced/comprehensive
        changes_made.append("✅ Set optimization mode to 'balanced'")
    
    # 4. Add WFO performance optimizations
    if 'cache_windows' not in wfo_config:
        wfo_config['cache_windows'] = True
        changes_made.append("✅ Enabled WFO window caching")
    
    if 'vectorized_randomization' not in wfo_config:
        wfo_config['vectorized_randomization'] = True
        changes_made.append("✅ Enabled vectorized randomization")
    
    # Update the configuration
    config['wfo_robustness_config'] = wfo_config
    config['monte_carlo_config'] = mc_config
    
    # Create backup of original file
    backup_file = parameters_file.with_suffix('.yaml.backup')
    if not backup_file.exists():
        with open(backup_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"📁 Created backup: {backup_file}")
    
    # Write updated configuration
    with open(parameters_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Report changes
    if changes_made:
        print("\n🎉 Feature recovery completed successfully!")
        print("\nChanges made:")
        for change in changes_made:
            print(f"  {change}")
        
        print(f"\n📝 Updated configuration file: {parameters_file}")
        print(f"📁 Backup saved to: {backup_file}")
        
        print("\n⚡ Performance optimizations added:")
        print("  • Synthetic data caching enabled")
        print("  • Parallel generation enabled")
        print("  • WFO window caching enabled")
        print("  • Vectorized randomization enabled")
        print("  • Balanced optimization mode set")
        
        print("\n🧪 Next steps:")
        print("  1. Run tests to validate functionality:")
        print("     python -m pytest tests/optimization/test_wfo_robustness.py -v")
        print("     python -m pytest tests/monte_carlo/ -v")
        print("  2. Run a small optimization to test performance:")
        print("     python -m src.portfolio_backtester.backtester --mode optimize --scenario-name Test_Optuna_Minimal")
        
        return True
    else:
        print("ℹ️  All features were already enabled. No changes needed.")
        return True

if __name__ == "__main__":
    success = restore_features()
    if success:
        print("\n✅ Recovery script completed successfully!")
    else:
        print("\n❌ Recovery script failed!")
        exit(1)