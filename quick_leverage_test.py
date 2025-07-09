#!/usr/bin/env python3
"""
Quick test for EMA crossover strategy with leverage to verify constraint enforcement.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Quick test EMA crossover strategy with leverage and volatility constraint."""
    
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
    
    # Create mock args for testing with very few trials
    class MockArgs:
        def __init__(self):
            self.mode = "optimize"
            self.scenario_filter = "EMA_Crossover_Test"
            self.scenario_name = "EMA_Crossover_Test"
            self.optimizer = "optuna"
            self.optuna_trials = 5  # Very small number for quick testing
            self.optuna_timeout_sec = None
            self.study_name = "quick_leverage_test"
            self.storage_url = None
            self.pruning_enabled = False
            self.pruning_n_startup_trials = 2
            self.pruning_n_warmup_steps = 5
            self.pruning_interval_steps = 1
            self.n_jobs = 1
            self.early_stop_patience = 3
            self.random_seed = 42
    
    args = MockArgs()
    
    # Initialize backtester
    backtester = Backtester(global_config, [ema_scenario], args, random_state=42)
    
    # Run optimization
    logger.info("Starting quick EMA crossover leverage constraint test (5 trials only)...")
    backtester.run()
    
    logger.info("Quick test completed!")

if __name__ == "__main__":
    main() 