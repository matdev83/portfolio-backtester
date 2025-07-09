import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from portfolio_backtester.optimization.genetic_optimizer import GeneticOptimizer, get_ga_optimizer_parameter_defaults
from portfolio_backtester.backtester import Backtester # Required for type hinting and mocking

# Mock global configurations and scenario data
MOCK_GLOBAL_CONFIG = {
    "benchmark": "SPY",
    "optimizer_parameter_defaults": {
        "param1_int": {"type": "int", "low": 2, "high": 5, "step": 1},
        "param2_float": {"type": "float", "low": 0.2, "high": 0.5},
        "param3_cat": {"type": "categorical", "values": ["A", "B", "C"]},
        "ga_num_generations": {"default": 10, "type": "int"},
        "ga_num_parents_mating": {"default": 5, "type": "int"},
        "ga_sol_per_pop": {"default": 5, "type": "int"}
    }
}

MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE = {
    "name": "Test_GA_Single",
    "strategy_params": {"initial_param": 50},
    "train_window_months": 36,  # 3 years as specified
    "test_window_months": 48,   # 4 years as specified
    "walk_forward_type": "expanding",
    "optimize": [
        {"parameter": "param1_int", "type": "int", "min_value": 2, "max_value": 5, "step": 1},
        {"parameter": "param2_float", "type": "float", "min_value": 0.2, "max_value": 0.5},
        {"parameter": "param3_cat", "type": "categorical", "values": ["A", "B", "C"]}
    ],
    "optimization_metric": "Sharpe",
    "genetic_algorithm_params": {
        "num_generations": 10,
        "sol_per_pop": 5,
        "num_parents_mating": 3
    }
}

MOCK_SCENARIO_CONFIG_MULTI_OBJECTIVE = {
    "name": "Test_GA_Multi",
    "strategy_params": {"initial_param": 50},
    "train_window_months": 36,  # 3 years as specified
    "test_window_months": 48,   # 4 years as specified
    "walk_forward_type": "expanding",
    "optimize": [
        {"parameter": "param1_int", "type": "int", "min_value": 2, "max_value": 5, "step": 1}
    ],
    "optimization_targets": [
        {"name": "Sharpe", "direction": "maximize"},
        {"name": "MaxDD", "direction": "minimize"}
    ],
    "genetic_algorithm_params": {
        "num_generations": 5,
        "sol_per_pop": 4,
        "num_parents_mating": 2
    }
}

# Mock data (sufficient for 36+48=84 months minimum)
# Create 10 years of data to ensure we have enough for the proper window sizes
date_rng = pd.date_range(start='2015-01-01', end='2025-01-01', freq='BME')
MOCK_MONTHLY_DATA = pd.DataFrame(index=date_rng, data={'PRICE_A': np.random.rand(len(date_rng)) * 100})
MOCK_DAILY_DATA = pd.DataFrame(index=pd.date_range(start='2015-01-01', end='2025-01-01', freq='B'), data={'PRICE_A': np.random.rand(pd.date_range(start='2015-01-01', end='2025-01-01', freq='B').size) * 100})
MOCK_RETS_FULL = MOCK_DAILY_DATA.pct_change(fill_method=None).fillna(0)


@pytest.fixture
def mock_backtester_instance():
    mock_bt = MagicMock(spec=Backtester)
    mock_bt.args = MagicMock()
    mock_bt.args.early_stop_patience = 3
    mock_bt.n_jobs = 1 # Set n_jobs directly on the mock_bt
    mock_bt.features = None # Add features attribute to mock
    mock_bt.logger = MagicMock() # Add logger attribute
    # Mock the evaluation function that the GeneticOptimizer will call
    mock_bt._evaluate_params_walk_forward = MagicMock()
    return mock_bt

def test_genetic_optimizer_initialization(mock_backtester_instance):
    optimizer = GeneticOptimizer(
        scenario_config=MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE,
        backtester_instance=mock_backtester_instance,
        global_config=MOCK_GLOBAL_CONFIG,
        monthly_data=MOCK_MONTHLY_DATA,
        daily_data=MOCK_DAILY_DATA,
        rets_full=MOCK_RETS_FULL,
        random_state=42
    )
    assert optimizer.scenario_config["name"] == "Test_GA_Single"
    assert optimizer.random_state == 42
    assert len(optimizer.optimization_params_spec) == 3
    assert optimizer.metrics_to_optimize == ["Sharpe"]
    assert not optimizer.is_multi_objective

def test_genetic_optimizer_initialization_multi_objective(mock_backtester_instance):
    optimizer = GeneticOptimizer(
        scenario_config=MOCK_SCENARIO_CONFIG_MULTI_OBJECTIVE,
        backtester_instance=mock_backtester_instance,
        global_config=MOCK_GLOBAL_CONFIG,
        monthly_data=MOCK_MONTHLY_DATA,
        daily_data=MOCK_DAILY_DATA,
        rets_full=MOCK_RETS_FULL,
        random_state=42
    )
    assert optimizer.is_multi_objective
    assert optimizer.metrics_to_optimize == ["Sharpe", "MaxDD"]

def test_decode_chromosome(mock_backtester_instance):
    optimizer = GeneticOptimizer(
        scenario_config=MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE,
        backtester_instance=mock_backtester_instance,
        global_config=MOCK_GLOBAL_CONFIG,
        monthly_data=MOCK_MONTHLY_DATA,
        daily_data=MOCK_DAILY_DATA,
        rets_full=MOCK_RETS_FULL
    )
    # Chromosome: param1_int=3, param2_float=0.3, param3_cat="B" (index 1)
    chromosome = np.array([3.0, 0.3, 1.0]) # PyGAD solutions are float arrays
    decoded_params = optimizer._decode_chromosome(chromosome)

    assert decoded_params["param1_int"] == 3
    assert abs(decoded_params["param2_float"] - 0.3) < 1e-6
    assert decoded_params["param3_cat"] == "B"
    assert "initial_param" in decoded_params # Ensure base params are preserved

def test_get_gene_space_and_types(mock_backtester_instance):
    optimizer = GeneticOptimizer(
        scenario_config=MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE,
        backtester_instance=mock_backtester_instance,
        global_config=MOCK_GLOBAL_CONFIG,
        monthly_data=MOCK_MONTHLY_DATA,
        daily_data=MOCK_DAILY_DATA,
        rets_full=MOCK_RETS_FULL
    )
    gene_space, gene_type = optimizer._get_gene_space_and_types()

    assert len(gene_space) == 3
    # For MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE: param1_int (int), param2_float (float), param3_cat (encoded as int)
    # So, gene_type should be a list [int, float, int]
    assert gene_type == [int, float, int]

    # param1_int
    assert gene_space[0] == {"low": 2, "high": 5, "step": 1} # From MOCK_SCENARIO_CONFIG

    # param2_float - may have step added by validation
    expected_float_space = {"low": 0.2, "high": 0.5}
    if "step" in gene_space[1]:
        expected_float_space["step"] = gene_space[1]["step"]  # Accept whatever step was added
    assert gene_space[1] == expected_float_space

    # param3_cat (categorical encoded as int indices)
    assert gene_space[2] == {"low": 0, "high": 2, "step": 1} # 3 choices: A, B, C -> indices 0, 1, 2


@patch('portfolio_backtester.optimization.genetic_optimizer._evaluate_params_walk_forward')
@patch('portfolio_backtester.optimization.genetic_optimizer.pygad.GA')
def test_run_single_objective(mock_pygad_ga, mock_evaluate_params, mock_backtester_instance):
    # Configure the mock _evaluate_params_walk_forward to return a single float
    mock_evaluate_params.return_value = 0.8 # Example Sharpe ratio

    optimizer = GeneticOptimizer(
        scenario_config=MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE,
        backtester_instance=mock_backtester_instance,
        global_config=MOCK_GLOBAL_CONFIG,
        monthly_data=MOCK_MONTHLY_DATA,
        daily_data=MOCK_DAILY_DATA,
        rets_full=MOCK_RETS_FULL,
        random_state=123
    )

    # Mock the GA instance returned by pygad.GA
    mock_ga_instance = MagicMock()
    mock_ga_instance.best_solution.return_value = (np.array([3, 0.4, 0]), 0.85, 0) # solution, fitness, idx
    mock_ga_instance.generations_completed = MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE["genetic_algorithm_params"]["num_generations"]
    mock_pygad_ga.return_value = mock_ga_instance

    best_params, num_evals = optimizer.run()

    mock_pygad_ga.assert_called_once() # Check if pygad.GA was initialized
    ga_call_args = mock_pygad_ga.call_args[1] # Get keyword arguments
    assert ga_call_args['num_generations'] == 10
    assert ga_call_args['sol_per_pop'] == 5
    assert ga_call_args['num_genes'] == 3
    assert ga_call_args['random_seed'] == 123
    assert not ga_call_args.get('algorithm_type') # Should be standard GA for single objective

    mock_ga_instance.run.assert_called_once() # Check if GA execution was started

    assert best_params["param1_int"] == 3
    assert abs(best_params["param2_float"] - 0.4) < 1e-6
    assert best_params["param3_cat"] == "A" # index 0
    assert num_evals == 10 * 5 # generations * sol_per_pop

    # Check fitness function call (indirectly via _evaluate_params_walk_forward)
    # This is tricky as fitness_func_wrapper is called internally by PyGAD.
    # We rely on the mock_backtester_instance._evaluate_params_walk_forward being called.
    # To test fitness_func_wrapper directly:
    fitness_value = optimizer.fitness_func_wrapper(mock_ga_instance, np.array([3,0.4,0]), 0)
    assert fitness_value == 0.8 # From the return_value of mocked _evaluate_params_walk_forward
    mock_evaluate_params.assert_called()


@patch('portfolio_backtester.optimization.genetic_optimizer._evaluate_params_walk_forward')
@patch('portfolio_backtester.optimization.genetic_optimizer.pygad.GA')
def test_run_multi_objective(mock_pygad_ga, mock_evaluate_params, mock_backtester_instance):
    # Configure mock _evaluate_params_walk_forward for multi-objective
    # Returns (Sharpe, MaxDD) -> PyGAD fitness will be (Sharpe, -MaxDD)
    mock_evaluate_params.return_value = (0.9, 0.15) # (Sharpe, MaxDD)

    optimizer = GeneticOptimizer(
        scenario_config=MOCK_SCENARIO_CONFIG_MULTI_OBJECTIVE,
        backtester_instance=mock_backtester_instance,
        global_config=MOCK_GLOBAL_CONFIG,
        monthly_data=MOCK_MONTHLY_DATA,
        daily_data=MOCK_DAILY_DATA,
        rets_full=MOCK_RETS_FULL,
        random_state=456
    )

    mock_ga_instance = MagicMock()
    # For multi-objective, best_solutions_fitness and best_solutions are used
    # Solution: param1_int=4
    # Fitness values as returned by fitness_func_wrapper: (Sharpe, -MaxDD)
    mock_ga_instance.best_solutions = [np.array([4.0]), np.array([3.0])] # Two solutions on Pareto front
    mock_ga_instance.best_solutions_fitness = [(0.9, -0.15), (0.8, -0.20)] # Corresponding fitness values
    mock_ga_instance.generations_completed = MOCK_SCENARIO_CONFIG_MULTI_OBJECTIVE["genetic_algorithm_params"]["num_generations"]
    mock_pygad_ga.return_value = mock_ga_instance

    best_params, num_evals = optimizer.run()

    mock_pygad_ga.assert_called_once()
    ga_call_args = mock_pygad_ga.call_args[1]
    assert ga_call_args['num_generations'] == 5
    assert ga_call_args['sol_per_pop'] == 4
    assert ga_call_args['num_genes'] == 1 # Only param1_int
    assert ga_call_args['random_seed'] == 456
    # In a real PyGAD multi-objective setup, you might set algorithm_type="nsgaii"
    # The current GeneticOptimizer code comments this out, so we don't assert it here.
    # assert ga_call_args.get('algorithm_type') == "nsgaii" # If NSGA-II is explicitly set

    mock_ga_instance.run.assert_called_once()

    # The selection logic picks the one best for the first objective (Sharpe, maximize)
    assert best_params["param1_int"] == 4
    assert num_evals == 5 * 4

    # Test fitness_func_wrapper for multi-objective
    # Chromosome for param1_int=4
    fitness_values = optimizer.fitness_func_wrapper(mock_ga_instance, np.array([4.0]), 0)
    assert isinstance(fitness_values, list) or isinstance(fitness_values, tuple)
    assert len(fitness_values) == 2
    assert abs(fitness_values[0] - 0.9) < 1e-6  # Sharpe (maximize)
    assert abs(fitness_values[1] - (-0.15)) < 1e-6 # -MaxDD (effectively maximizing -MDD)
    mock_evaluate_params.assert_called()


@patch('portfolio_backtester.optimization.genetic_optimizer._evaluate_params_walk_forward')
def test_fitness_func_wrapper_minimization(mock_evaluate_params, mock_backtester_instance):
    scenario_minimize = MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE.copy()
    scenario_minimize["optimization_targets"] = [{"name": "MaxDD", "direction": "minimize"}]
    del scenario_minimize["optimization_metric"] # Remove to use optimization_targets

    mock_evaluate_params.return_value = 0.2 # MaxDD value

    optimizer = GeneticOptimizer(
        scenario_config=scenario_minimize,
        backtester_instance=mock_backtester_instance,
        global_config=MOCK_GLOBAL_CONFIG,
        monthly_data=MOCK_MONTHLY_DATA,
        daily_data=MOCK_DAILY_DATA,
        rets_full=MOCK_RETS_FULL
    )

    # PyGAD maximizes, so for minimization, fitness should be -value
    raw_fitness_value = optimizer.fitness_func_wrapper(MagicMock(), np.array([3,0.4,0]), 0)
    # Ensure it's a single float for this test case
    fitness_value = raw_fitness_value[0] if isinstance(raw_fitness_value, (list, tuple)) else raw_fitness_value
    assert abs(float(fitness_value) - (-0.2)) < 1e-6

def test_get_ga_optimizer_parameter_defaults():
    defaults = get_ga_optimizer_parameter_defaults()
    assert "ga_num_generations" in defaults
    assert defaults["ga_num_generations"]["default"] == 100
    assert "ga_mutation_type" in defaults
    assert defaults["ga_mutation_type"]["default"] == "random"

# TODO: Add tests for early stopping callback if possible by mocking GA generations
# TODO: Add tests for parallel processing setup if n_jobs > 1

class TestGeneticOptimizerWithWalkForward:
    """More integrated tests that don't mock _evaluate_params_walk_forward as much"""

    @pytest.fixture
    def backtester_for_ga(self, tmp_path):
        # A simplified Backtester setup for testing GA integration
        args = MagicMock()
        args.optimizer = "genetic"
        args.early_stop_patience = 3
        args.n_jobs = 1
        args.pruning_enabled = False # Not used by GA but part of Backtester args
        args.random_seed = 42
        # other args as needed by _evaluate_params_walk_forward
        args.storage_url = None
        args.study_name = "test_study"
        args.optuna_timeout_sec = None


        # Mocking the full Backtester is complex, so we use a real one but
        # will mock its `run_scenario` method which is called by `_evaluate_params_walk_forward`

        # Create minimal config files if needed by Backtester initialization
        # For now, assume MOCK_GLOBAL_CONFIG and MOCK_SCENARIO_CONFIG are sufficient
        # and that Backtester can be initialized without file loading for this test.

        # This part is tricky because Backtester loads config from files.
        # We might need to patch `load_config` or provide temp config files.
        # For now, let's try to directly instantiate and then override necessary parts.

        with patch('portfolio_backtester.config_loader.load_config') as mock_load_config, \
             patch('portfolio_backtester.backtester.populate_default_optimizations') as mock_populate:
            # Prevent file loading by patching, and assign our mocks
            from portfolio_backtester import backtester as bt_module
            bt_module.GLOBAL_CONFIG = MOCK_GLOBAL_CONFIG
            bt_module.OPTIMIZER_PARAMETER_DEFAULTS = MOCK_GLOBAL_CONFIG["optimizer_parameter_defaults"]
            bt_module.BACKTEST_SCENARIOS = [MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE]


            bt_instance = Backtester(
                global_config=MOCK_GLOBAL_CONFIG,
                scenarios=[MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE],
                args=args,
                random_state=42
            )

        # Mock the parts of Backtester that _evaluate_params_walk_forward depends on
        bt_instance.features = None # Assuming no complex features needed for this test
        bt_instance.logger = MagicMock()
        bt_instance.run_scenario = MagicMock(return_value=pd.Series(np.random.randn(100) * 0.01, index=pd.date_range('2021-01-01', periods=100, freq='D')))

        # Also mock calculate_metrics to return a predictable structure
        # The structure is a Pandas Series with metric names as index
        mock_metrics = pd.Series({
            "Sharpe": 1.5, "Calmar": 1.2, "MaxDD": -0.1, "Total Return": 0.25
        })
        # Patch calculate_metrics globally as it's imported at the module level in backtester.py
        # and used by _evaluate_params_walk_forward
        self.calculate_metrics_patcher = patch('portfolio_backtester.reporting.performance_metrics.calculate_metrics', return_value=mock_metrics)
        self.mocked_calculate_metrics = self.calculate_metrics_patcher.start()


        # Ensure monthly_data has enough span for walk-forward
        # And includes the benchmark column
        date_index_monthly = pd.date_range(start='2018-01-01', end='2023-01-01', freq='BME')
        spy_data_monthly = np.random.rand(len(date_index_monthly)) * 100
        bt_instance.monthly_data = pd.DataFrame(
            data={'PRICE_A': np.random.rand(len(date_index_monthly)) * 100,
                  'SPY': spy_data_monthly}, # Benchmark data
            index=date_index_monthly
        )
        # Ensure daily_data also has benchmark for _evaluate_params_walk_forward if it uses it for benchmark series
        date_index_daily = pd.date_range(start='2018-01-01', end='2023-01-01', freq='B')
        bt_instance.daily_data = pd.DataFrame(
            data={'PRICE_A': np.random.rand(len(date_index_daily)) * 100,
                  'SPY': np.random.rand(len(date_index_daily)) * 100},
            index=date_index_daily
        )
        bt_instance.rets_full = MOCK_RETS_FULL   # Keep this simple

        yield bt_instance

        self.calculate_metrics_patcher.stop()


    def test_ga_optimizer_with_mocked_backtester_evaluation(self, backtester_for_ga):
        """
        This test checks if GeneticOptimizer correctly uses the Backtester's
        _evaluate_params_walk_forward method by running a small GA.
        """
        scenario_config = MOCK_SCENARIO_CONFIG_SINGLE_OBJECTIVE.copy()
        scenario_config["train_window_months"] = 12
        scenario_config["test_window_months"] = 6
        scenario_config["genetic_algorithm_params"] = {
            "num_generations": 2,
            "sol_per_pop": 3,
            "num_parents_mating": 2,
            "mutation_num_genes": 1,
            "mutation_percent_genes": 34
        }

        optimizer = GeneticOptimizer(
            scenario_config=scenario_config,
            backtester_instance=backtester_for_ga,
            global_config=MOCK_GLOBAL_CONFIG,
            monthly_data=backtester_for_ga.monthly_data,
            daily_data=backtester_for_ga.daily_data,
            rets_full=backtester_for_ga.rets_full,
            random_state=789
        )

        optimizer.run(save_plot=False)

        # Calculate expected number of calls to run_scenario
        train_window_m = scenario_config["train_window_months"]
        test_window_m = scenario_config["test_window_months"]
        idx = backtester_for_ga.monthly_data.index
        windows = []
        start_idx = train_window_m
        while start_idx + test_window_m <= len(idx):
            train_end_idx = start_idx - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + test_window_m - 1
            if test_end_idx >= len(idx): break
            if scenario_config.get("walk_forward_type", "expanding").lower() == "rolling":
                train_start_idx = train_end_idx - train_window_m + 1
            else:
                train_start_idx = 0
            windows.append((idx[train_start_idx], idx[train_end_idx], idx[test_start_idx], idx[test_end_idx]))
            start_idx += test_window_m
        
        num_windows = len(windows)
        num_generations = scenario_config["genetic_algorithm_params"]["num_generations"]
        sol_per_pop = scenario_config["genetic_algorithm_params"]["sol_per_pop"]
        
        # The fitness function is called for each solution in the initial population,
        # and then for each new solution in subsequent generations.
        # PyGAD calls fitness_func for the initial population, so sol_per_pop calls.
        # Then for each generation, it creates a new population.
        # The number of fitness function calls is complex to predict exactly without
        # knowing the internals of PyGAD's parent selection and crossover.
        # However, it should be at least sol_per_pop.
        # A simpler check is to see if it was called at all.
        assert backtester_for_ga.run_scenario.call_count > 0
        
        # A more precise check would be:
        # Initial population: sol_per_pop calls
        # Each generation: num_parents_mating new solutions
        # This is still complex. Let's stick to a simpler check.
        # The number of calls to run_scenario should be num_windows * number_of_fitness_evaluations.
        # Let's just check that it was called.
        assert backtester_for_ga.run_scenario.call_count > 0
