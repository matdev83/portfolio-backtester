import cProfile
import sys
import tempfile
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure src on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if src_path.is_dir():
    sys.path.insert(0, str(src_path))
else:
    sys.path.insert(0, str(project_root))

from line_profiler import LineProfiler
from portfolio_backtester.optimization.generators.fixed_genetic_generator import FixedGeneticParameterGenerator
from portfolio_backtester.optimization.population_evaluator import PopulationEvaluator
from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.optimization.results import OptimizationData
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.strategies._core.strategy_factory_impl import StrategyFactory
from portfolio_backtester.optimization.performance.deduplication_factory import DeduplicationFactory


def create_dummy_data():
    """Create dummy data for profiling"""
    # Create a simple date range
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    
    # Create dummy daily data
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    daily_data = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)),
        index=dates,
        columns=tickers
    )
    
    # Create dummy monthly data
    monthly_dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='ME')
    monthly_data = pd.DataFrame(
        np.random.randn(len(monthly_dates), len(tickers)),
        index=monthly_dates,
        columns=tickers
    )
    
    # Create dummy returns data
    returns_data = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)) * 0.01,  # 1% daily returns
        index=dates,
        columns=tickers
    )
    
    # Create WFO windows
    windows = [
        WFOWindow(
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2021-06-30'),
            test_start=pd.Timestamp('2021-07-01'),
            test_end=pd.Timestamp('2021-12-31'),
            evaluation_frequency='M',
            strategy_name='Window1'
        ),
        WFOWindow(
            train_start=pd.Timestamp('2020-07-01'),
            train_end=pd.Timestamp('2021-12-31'),
            test_start=pd.Timestamp('2022-01-01'),
            test_end=pd.Timestamp('2022-06-30'),
            evaluation_frequency='M',
            strategy_name='Window2'
        ),
    ]
    
    return OptimizationData(
        daily=daily_data,
        monthly=monthly_data,
        returns=returns_data,
        windows=windows
    )


def profile_ga_optimizer():
    """Run a GA optimization and capture cProfile and line_profiler outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--ga-population-size", type=int, default=50)
    parser.add_argument("--ga-max-generations", type=int, default=5)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.1)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.8)
    parser.add_argument("--joblib-batch-size", type=str, default="auto")
    parser.add_argument("--joblib-pre-dispatch", type=str, default="3*n_jobs")
    parser.add_argument("--use-persistent-cache", action="store_true", help="Enable persistent cache")
    args = parser.parse_args()
    
    # Create dummy data
    data = create_dummy_data()
    
    # Create scenario config
    scenario_config = {
        "strategy": {
            "name": "DummyStrategy",
            "params": {
                "lookback": 20,
                "threshold": 0.5
            }
        },
        "universe": {
            "tickers": ["AAPL", "MSFT", "GOOG", "AMZN"]
        },
        "backtest": {
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 100000,
            "position_size": 0.1
        }
    }
    
    # Create parameter space
    parameter_space = {
        "lookback": {"low": 10, "high": 50, "type": "int"},
        "threshold": {"low": 0.1, "high": 0.9, "type": "float"}
    }
    
    # Create backtester and evaluator
    strategy_params = {"lookback": 20, "threshold": 0.5}
    strategy = StrategyFactory.create_strategy("DummySignalStrategy", strategy_params)
    backtester = StrategyBacktester(strategy)
    window_evaluator = WindowEvaluator(backtester)
    evaluator = BacktestEvaluator(window_evaluator)
    
    # Create population evaluator with joblib settings
    population_evaluator = PopulationEvaluator(
        evaluator,
        n_jobs=args.n_jobs,
        joblib_batch_size=args.joblib_batch_size,
        joblib_pre_dispatch=args.joblib_pre_dispatch
    )
    
    # Create GA optimizer
    deduplicator_config = {
        "enable_deduplication": True,
        "use_persistent_cache": args.use_persistent_cache
    }
    
    # Create genetic parameter generator
    optimizer = FixedGeneticParameterGenerator(random_seed=42)
    optimizer.parameter_space = parameter_space
    optimizer.population_size = args.ga_population_size
    optimizer.max_generations = args.ga_max_generations
    optimizer.mutation_rate = args.ga_mutation_rate
    optimizer.crossover_rate = args.ga_crossover_rate
    optimizer.population = optimizer._create_initial_population()
    
    # Set up line profiler
    line_profiler = LineProfiler()
    line_profiler.add_function(StrategyBacktester.backtest_strategy)
    line_profiler.add_function(WindowEvaluator.evaluate_window)
    line_profiler.add_function(PopulationEvaluator.evaluate_population)
    line_profiler.add_function(PopulationEvaluator._evaluate_parallel)
    
    # Set up cProfile
    c_profiler = cProfile.Profile()
    
    print("Running GA optimizer profiler...")
    
    # Define optimization function
    def run_optimization():
        # Simulate the optimization loop
        for generation in range(optimizer.max_generations):
            print(f"\nGeneration {generation+1}/{optimizer.max_generations}")
            population = optimizer.suggest_population()
            
            # Evaluate the population
            results = population_evaluator.evaluate_population(
                population=population,
                scenario_config=scenario_config,
                data=data,
                backtester=backtester
            )
            
            # Report results back to the optimizer
            optimizer.report_population_results(population, results)
            
            # Get the best result so far
            best_result = optimizer.get_best_result()
            print(f"Best value: {best_result.best_objective_value}")
        
        return optimizer.get_best_result()
    
    try:
        c_profiler.enable()
        
        result = line_profiler.runcall(run_optimization)
        
        # Print some stats
        print(f"\nOptimization completed with {len(result.trials)} trials")
        print(f"Best objective value: {result.best_objective_value}")
        print(f"Best parameters: {result.best_parameters}")
        
        # Get deduplication stats
        if hasattr(optimizer, "deduplicator") and hasattr(optimizer.deduplicator, "get_stats"):
            dedup_stats = optimizer.deduplicator.get_stats()
            print("\nDeduplication stats:")
            for key, value in dedup_stats.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
    finally:
        c_profiler.disable()
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = tempfile.gettempdir()
        
        cprofile_file = Path(temp_dir) / f"ga_optimizer_direct_cprofile_{timestamp}.pstats"
        c_profiler.dump_stats(cprofile_file)
        print(f"cProfile results saved to: {cprofile_file}")
        
        lprofile_file = Path(temp_dir) / f"ga_optimizer_direct_lprofile_{timestamp}.txt"
        with open(lprofile_file, "w", encoding="utf-8") as f:
            line_profiler.print_stats(f)
        print(f"line_profiler results saved to: {lprofile_file}")


if __name__ == "__main__":
    profile_ga_optimizer()
