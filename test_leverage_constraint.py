#!/usr/bin/env python3
"""
Test script for EMA crossover strategy with 3x leverage to test constraint enforcement.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test EMA crossover strategy with 3x leverage and volatility constraint."""
    
    # Use the loaded configuration
    global_config = GLOBAL_CONFIG.copy()
    scenarios = BACKTEST_SCENARIOS.copy()
    
    # Find the EMA_Crossover_Test scenario
    ema_scenario = None
    for scenario in scenarios:
        if scenario['name'] == 'EMA_Crossover_Test':
            ema_scenario = scenario.copy()
            break
    
    if not ema_scenario:
        logger.error("EMA_Crossover_Test scenario not found!")
        return
    
    # Print constraint configuration
    constraints = ema_scenario.get('optimization_constraints', [])
    logger.info(f"Found {len(constraints)} optimization constraints:")
    for constraint in constraints:
        logger.info(f"  - {constraint}")
    
    # Print current strategy params
    strategy_params = ema_scenario.get('strategy_params', {})
    logger.info(f"Strategy parameters: {strategy_params}")
    
    # Create mock args for testing
    class MockArgs:
        def __init__(self):
            self.mode = "optimize"
            self.scenario_filter = "EMA_Crossover_Test"
            self.scenario_name = "EMA_Crossover_Test"
            self.optimizer = "optuna"
            self.optuna_trials = 20  # Small number for testing
            self.optuna_timeout_sec = None
            self.study_name = "leverage_constraint_test"
            self.storage_url = None
            self.pruning_enabled = False
            self.pruning_n_startup_trials = 5
            self.pruning_n_warmup_steps = 10
            self.pruning_interval_steps = 1
            self.n_jobs = 1
            self.early_stop_patience = 5
            self.random_seed = 42
    
    args = MockArgs()
    
    # Initialize backtester
    backtester = Backtester(global_config, [ema_scenario], args, random_state=42)
    
    # Run optimization
    logger.info("Starting EMA crossover leverage constraint test...")
    backtester.run()
    
    # Check results
    results = backtester.results
    if results:
        for name, result in results.items():
            logger.info(f"\nResults for {name}:")
            if 'returns' in result and not result['returns'].empty:
                # Calculate metrics to verify constraint
                from src.portfolio_backtester.reporting.performance_metrics import calculate_metrics
                
                # Get benchmark returns
                benchmark_data = backtester.daily_data_ohlc[global_config['benchmark']]['Close']
                benchmark_returns = benchmark_data.pct_change(fill_method=None).fillna(0)
                
                # Calculate metrics
                strategy_returns = result['returns']
                aligned_benchmark = benchmark_returns.reindex(strategy_returns.index).fillna(0)
                
                metrics = calculate_metrics(
                    strategy_returns, 
                    aligned_benchmark, 
                    global_config['benchmark']
                )
                
                # Check constraint compliance
                ann_vol = metrics.get('Ann. Vol', np.nan)
                sortino = metrics.get('Sortino', np.nan)
                max_dd = metrics.get('Max Drawdown', np.nan)
                
                logger.info(f"  Annualized Volatility: {ann_vol:.4f} (Constraint: ≤ 0.15)")
                logger.info(f"  Sortino Ratio: {sortino:.4f}")
                logger.info(f"  Max Drawdown: {max_dd:.4f}")
                
                # Check optimal parameters
                if 'optimal_params' in result:
                    logger.info(f"  Optimal parameters: {result['optimal_params']}")
                
                # Verify constraint
                if not np.isnan(ann_vol):
                    constraint_satisfied = ann_vol <= 0.15
                    logger.info(f"  Volatility constraint satisfied: {constraint_satisfied}")
                    if not constraint_satisfied:
                        logger.warning(f"  ⚠️  CONSTRAINT VIOLATED! Volatility {ann_vol:.4f} > 0.15")
                        logger.info("  This suggests the constraint enforcement may not be working properly!")
                    else:
                        logger.info(f"  ✅ Constraint satisfied: {ann_vol:.4f} ≤ 0.15")
                else:
                    logger.warning("  ⚠️  Could not calculate annualized volatility")
            else:
                logger.warning(f"  No returns data found for {name}")
    else:
        logger.warning("No optimization results found!")

if __name__ == "__main__":
    main() 