import pygad
import numpy as np
import logging # Import logging
import optuna

# Setup logger for this module
logger = logging.getLogger(__name__)

# Assuming utils.py is in the parent directory relative to optimization directory
# For example, if utils.py is in src/portfolio_backtester/ and this is src/portfolio_backtester/optimization/
# Adjust the import path as necessary based on your project structure and how it's run.
# If this file is run as part of the package, `from ..utils import INTERRUPTED` should work.
try:
    from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
except ImportError:
    # Fallback if running standalone or structure is different
    # This is not ideal for production, implies a structure issue or direct script execution
    # For robustness in development, we can define a local flag.
    logger.warning("Could not import CENTRAL_INTERRUPTED_FLAG from ..utils. Using a local dummy flag for GeneticOptimizer.")
    CENTRAL_INTERRUPTED_FLAG = False


class GeneticOptimizer:
    def __init__(self, scenario_config, backtester_instance, global_config, monthly_data, daily_data, rets_full, random_state=None):
        self.scenario_config = scenario_config
        self.backtester = backtester_instance
        self.global_config = global_config
        self.monthly_data = monthly_data
        self.daily_data = daily_data
        self.rets_full = rets_full
        self.random_state = random_state

        self.optimization_params_spec = scenario_config.get("optimize", [])
        if not self.optimization_params_spec:
            raise ValueError("Genetic optimizer requires 'optimize' specifications in the scenario config.")

        self.metrics_to_optimize = [t["name"] for t in scenario_config.get("optimization_targets", [])] or \
                                   [scenario_config.get("optimization_metric", "Calmar")]
        self.is_multi_objective = len(self.metrics_to_optimize) > 1
        if self.is_multi_objective:
            # For PyGAD multi-objective, fitness function needs to return a list/tuple of objectives
            # and we need to define weights for each objective if using a single fitness value approach,
            # or rely on PyGAD's NSGA-II support.
            # For simplicity, let's assume the first metric is the primary one if pygad doesn't directly support multi-objective in the same way Optuna does.
            # However, PyGAD does support multi-objective optimization with algorithms like NSGA-II.
            # We will need to adjust the fitness function accordingly.
            logger.info(f"Multi-objective optimization for: {self.metrics_to_optimize}")


        self.ga_instance = None

    def _decode_chromosome(self, solution):
        """Decodes a chromosome (solution) into a dictionary of parameters."""
        params = self.scenario_config["strategy_params"].copy()
        idx = 0
        for spec in self.optimization_params_spec:
            pname = spec["parameter"]
            ptype = self.global_config.get("optimizer_parameter_defaults", {}).get(pname, {}).get("type", spec.get("type"))

            if ptype == "int":
                # Ensure the gene value is within the min/max bounds and respects step if any
                # PyGAD handles gene_space for this, so direct mapping is fine.
                params[pname] = int(round(solution[idx]))
            elif ptype == "float":
                params[pname] = solution[idx]
            elif ptype == "categorical":
                # Categorical: the gene value will be an index into the choices list
                choices = spec.get("values", self.global_config.get("optimizer_parameter_defaults", {}).get(pname, {}).get("values"))
                # Ensure index is within bounds
                cat_idx = int(round(solution[idx]))
                params[pname] = choices[min(max(cat_idx, 0), len(choices) - 1)]
            idx += 1
        return params

    def fitness_func_wrapper(self, ga_instance, solution, solution_idx):
        """Wrapper for the fitness function to be used by PyGAD."""
        current_params = self._decode_chromosome(solution)

        trial_scenario_config = self.scenario_config.copy()
        trial_scenario_config["strategy_params"] = current_params

        # Use the same walk-forward evaluation as Optuna
        # The _evaluate_params_walk_forward method needs to be accessible.
        # It might be better to refactor _evaluate_params_walk_forward out of Backtester
        # or pass the Backtester instance to this optimizer.
        # For now, assuming backtester_instance has this method.

        # Mock Optuna trial object for _evaluate_params_walk_forward
        class MockTrial:
            def __init__(self, params, study=None):
                self.params = params
                self.user_attrs = {}
                self.study = study # PyGAD doesn't have a direct equivalent of Optuna's study direction here in fitness_func

            def suggest_int(self, name, low, high, step=1): return self.params.get(name)
            def suggest_float(self, name, low, high, step=None, log=False): return self.params.get(name)
            def suggest_categorical(self, name, choices): return self.params.get(name)
            def report(self, value, step): pass # PyGAD handles progress differently
            def should_prune(self): return False # Pruning is specific to Optuna's architecture
            def set_user_attr(self, key, value): self.user_attrs[key] = value

        # PyGAD doesn't have a direct 'study' object in the fitness function to get directions.
        # We need to define how to handle maximization/minimization for PyGAD.
        # PyGAD maximizes the fitness function by default. If we need to minimize, we should return -value.
        # For multi-objective, PyGAD's NSGA-II handles this.

        mock_study = type('MockStudy', (), {'directions': [optuna.study.StudyDirection.MAXIMIZE for _ in self.metrics_to_optimize]})()
        mock_trial = MockTrial(current_params, mock_study)


        # Extract walk-forward windows logic from Backtester or duplicate here
        # For now, assume self.backtester has _evaluate_params_walk_forward
        # and it's adapted or callable in this context.
        # This is a simplification. The _evaluate_params_walk_forward method is complex
        # and tightly coupled with Optuna's trial object and pruning.
        # A more robust solution would be to refactor that evaluation logic.

        # Simplified call:
        # This is a placeholder for the complex walk-forward evaluation.
        # We need to replicate or adapt the logic from `_evaluate_params_walk_forward`.
        # This part is CRITICAL and needs careful implementation.

        # For now, let's assume we have a way to get windows. This part needs to be robust.
        train_window_m = self.scenario_config.get("train_window_months", 24)
        test_window_m = self.scenario_config.get("test_window_months", 12)
        wf_type = self.scenario_config.get("walk_forward_type", "expanding").lower()
        idx = self.monthly_data.index
        windows = []
        start_idx = train_window_m
        while start_idx + test_window_m <= len(idx):
            train_end_idx = start_idx - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + test_window_m - 1
            if test_end_idx >= len(idx): break
            if wf_type == "rolling":
                train_start_idx = train_end_idx - train_window_m + 1
            else:
                train_start_idx = 0
            windows.append((idx[train_start_idx], idx[train_end_idx], idx[test_start_idx], idx[test_end_idx]))
            start_idx += test_window_m

        if not windows:
            logger.error("Not enough data for walk-forward windows in GeneticOptimizer.")
            return -np.inf if not self.is_multi_objective else [-np.inf] * len(self.metrics_to_optimize)


        # This is where the call to the evaluation function that mirrors
        # `_evaluate_params_walk_forward` would go.
        # It needs access to `self.backtester.run_scenario`, `calculate_metrics`, etc.
        # This is a significant dependency.
        # For now, we'll use the passed `backtester_instance`
        objectives_values = self.backtester._evaluate_params_walk_forward(
            mock_trial, # This mock trial might not be fully compatible
            trial_scenario_config,
            windows,
            self.monthly_data,
            self.daily_data,
            self.rets_full,
            self.metrics_to_optimize,
            self.is_multi_objective
        )

        if self.is_multi_objective:
            # PyGAD's NSGA-II expects a list/tuple of objective values.
            # We need to ensure the directions (max/min) are handled correctly.
            # NSGA-II typically maximizes all objectives. If minimization is needed,
            # the fitness value for that objective should be negated.
            processed_objectives = []
            optimization_targets_config = self.scenario_config.get("optimization_targets", [])
            # Ensure we have directions for all metrics
            directions = [t.get("direction", "maximize").lower() for t in optimization_targets_config]
            if not directions and len(self.metrics_to_optimize) == 1 : # Single objective, default to maximize
                 directions = ["maximize"]
            elif len(directions) != len(self.metrics_to_optimize): # Fallback if mismatch
                 directions = ["maximize"] * len(self.metrics_to_optimize)


            for i, val in enumerate(objectives_values):
                if directions[i] == "minimize":
                    processed_objectives.append(-val if np.isfinite(val) else np.inf) # PyGAD maximizes, so negate for minimization
                else:
                    processed_objectives.append(val if np.isfinite(val) else -np.inf)
            logger.debug(f"Solution {solution_idx} params: {current_params}, objectives: {processed_objectives}")
            return processed_objectives
        else:
            # Single objective
            fitness_value = objectives_values
            # PyGAD maximizes. If original objective was minimize, negate it.
            opt_target_config = self.scenario_config.get("optimization_targets")
            direction = "maximize" # Default
            if opt_target_config and isinstance(opt_target_config, list) and len(opt_target_config) > 0:
                direction = opt_target_config[0].get("direction", "maximize").lower()
            elif "optimization_metric" in self.scenario_config: # Legacy single objective
                 # No explicit direction, assume maximize for metrics like Calmar, Sharpe
                 pass


            if direction == "minimize":
                logger.debug(f"Solution {solution_idx} params: {current_params}, raw_fitness: {fitness_value}, adjusted_fitness: {-fitness_value if np.isfinite(fitness_value) else np.inf}")
                return -fitness_value if np.isfinite(fitness_value) else np.inf # Return positive infinity for bad minimization value
            else:
                logger.debug(f"Solution {solution_idx} params: {current_params}, fitness: {fitness_value if np.isfinite(fitness_value) else -np.inf}")
                return fitness_value if np.isfinite(fitness_value) else -np.inf # Return negative infinity for bad maximization value


    def _get_gene_space_and_types(self):
        gene_space = []
        gene_type = []

        for spec in self.optimization_params_spec:
            pname = spec["parameter"]
            opt_def = self.global_config.get("optimizer_parameter_defaults", {}).get(pname, {})
            ptype = opt_def.get("type", spec.get("type"))

            low = spec.get("min_value", opt_def.get("low"))
            high = spec.get("max_value", opt_def.get("high"))
            step = spec.get("step", opt_def.get("step", 1 if ptype == "int" else None))

            if ptype == "int":
                gene_space.append({"low": int(low), "high": int(high), "step": int(step) if step else 1})
                gene_type.append(int)
            elif ptype == "float":
                gene_space.append({"low": float(low), "high": float(high)})
                gene_type.append(float)
            elif ptype == "categorical":
                choices = spec.get("values", opt_def.get("values"))
                if not choices: raise ValueError(f"Categorical parameter {pname} has no choices.")
                gene_space.append({"low": 0, "high": len(choices) - 1, "step": 1})
                gene_type.append(int) # Categorical genes are represented by integer indices
            else:
                raise ValueError(f"Unsupported PTYPE {ptype} for parameter {pname} in genetic optimizer.")
        return gene_space, gene_type # Return the list of types

    def run(self):
        logger.info("Setting up Genetic Algorithm...")

        gene_space, gene_types_for_pygad = self._get_gene_space_and_types()
        num_genes = len(self.optimization_params_spec)

        # GA Parameters (these could be configurable)
        ga_params_config = self.scenario_config.get("genetic_algorithm_params", {})
        num_generations = ga_params_config.get("num_generations", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_num_generations", {}).get("default", 100))
        num_parents_mating = ga_params_config.get("num_parents_mating", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_num_parents_mating", {}).get("default", 10))
        sol_per_pop = ga_params_config.get("sol_per_pop", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_sol_per_pop", {}).get("default", 50))
        parent_selection_type = ga_params_config.get("parent_selection_type", "sss") # steady-state selection
        crossover_type = ga_params_config.get("crossover_type", "single_point")
        mutation_type = ga_params_config.get("mutation_type", "random")
        mutation_percent_genes = ga_params_config.get("mutation_percent_genes", "default") # PyGAD's default

        # PyGAD seed
        pygad_seed = self.random_state if self.random_state is not None else np.random.randint(0, 10000)


        on_generation_callback = None
        if self.backtester.args.early_stop_patience > 0:
            self.zero_fitness_streak = 0
            self.best_fitness_so_far = -np.inf # Assuming maximization

            def on_gen(ga_instance):
                current_best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
                logger.debug(f"Generation {ga_instance.generations_completed}, Best fitness: {current_best_fitness}")

                # Check for zero returns equivalent (stagnation or very poor performance)
                # This is a simplified check. A more robust one would look at the variance of fitness values.
                if abs(current_best_fitness - self.best_fitness_so_far) < 1e-6 : # Could be configurable tolerance
                    self.zero_fitness_streak +=1
                else:
                    self.zero_fitness_streak = 0

                self.best_fitness_so_far = max(self.best_fitness_so_far, current_best_fitness)

                if self.zero_fitness_streak >= self.backtester.args.early_stop_patience:
                    logger.info(f"Early stopping GA due to {self.zero_fitness_streak} generations with no improvement.")
                    return "stop"

                if CENTRAL_INTERRUPTED_FLAG:
                    logger.warning("Genetic Algorithm optimization interrupted by user via central flag.")
                    return "stop" # PyGAD expects "stop" to terminate
            on_generation_callback = on_gen


        self.ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=self.fitness_func_wrapper,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            parent_selection_type=parent_selection_type,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes,
            random_seed=pygad_seed,
            on_generation=on_generation_callback,
            parallel_processing=['thread', self.backtester.n_jobs] if self.backtester.n_jobs > 1 else None,
            # For multi-objective, specify algorithm, e.g., NSGA-II
            # This requires PyGAD >= 2.18.0
            # algorithm_type="nsgaii" if self.is_multi_objective else "ga", # Conditional algorithm type
        )

        # Check PyGAD version for algorithm_type if needed, or handle it more gracefully.
        # For now, let's assume standard GA. If multi-objective is used, fitness_func_wrapper
        # would need to return a single weighted value or this needs adjustment.
        # The above `algorithm_type` is commented out as it might need specific PyGAD version.
        # If using NSGA-II, the fitness function should return a list/tuple of objectives.
        # The current fitness_func_wrapper attempts to do this if self.is_multi_objective is True.

        if self.is_multi_objective:
            # Ensure PyGAD version supports NSGA-II or similar if this path is taken.
            # This might require setting `self.ga_instance.algorithm_type = "nsgaii"` if not in constructor
            logger.info("Attempting to use multi-objective optimization with Genetic Algorithm.")


        logger.info("Running Genetic Algorithm...")
        self.ga_instance.run()

        try:
            if CENTRAL_INTERRUPTED_FLAG:
                logger.warning("Genetic Algorithm was interrupted. Attempting to retrieve best solution so far.")
                # No specific action needed here, the code below will try to get the best solution.
                # If it fails due to severe interruption, the except block will catch it.

            num_evaluations = self.ga_instance.generations_completed * sol_per_pop

            if self.is_multi_objective:
                pareto_solutions_fitness = self.ga_instance.best_solutions_fitness
                pareto_chromosomes = self.ga_instance.best_solutions

                if not pareto_chromosomes: # Check if any solutions are available
                    logger.error("GA (multi-objective): No solutions found on Pareto front (possibly due to interruption or poor performance).")
                    return self.scenario_config["strategy_params"].copy(), num_evaluations # Return default params

                # Simplified: pick the first solution from the Pareto front if available
                solution = pareto_chromosomes[0]
                solution_fitness = pareto_solutions_fitness[0]
                log_msg = f"GA multi-objective: Selected first solution from Pareto front. Fitness: {solution_fitness}."
            else: # Single objective
                # Check if a best solution exists
                if self.ga_instance.best_solution_generation == -1 and not CENTRAL_INTERRUPTED_FLAG: # No solution found and not interrupted
                     logger.error("GA (single-objective): No best solution found and not due to interruption.")
                     return self.scenario_config["strategy_params"].copy(), num_evaluations

                # If interrupted, best_solution might raise error if no solutions were ever found.
                # However, PyGAD's best_solution() is designed to return the best one found so far.
                # If ga_instance.run() was stopped very early, generations_completed could be 0.
                # We rely on the try-except to catch issues if PyGAD cannot provide a best_solution.
                solution, solution_fitness, _ = self.ga_instance.best_solution(pop_fitness=self.ga_instance.last_generation_fitness)
                log_msg = f"GA single-objective: Best fitness: {solution_fitness}."

            logger.info(log_msg)
            if self.ga_instance.generations_completed > 0 and solution is not None:
                 self.ga_instance.plot_fitness(title="GA Fitness Value vs. Generation", save_dir=f"plots/ga_fitness_{self.scenario_config['name']}.png")
                 optimal_params = self.scenario_config["strategy_params"].copy()
                 optimal_params.update(self._decode_chromosome(solution))
                 logger.info(f"Best parameters found by GA: {optimal_params}")
                 print(f"Genetic Optimizer - Best parameters found: {optimal_params}")
                 return optimal_params, num_evaluations
            else: # Interrupted very early or no solution found
                logger.warning("GA: No valid solution to decode, returning default parameters.")
                return self.scenario_config["strategy_params"].copy(), num_evaluations

        except Exception as e:
            logger.error(f"Error during GA solution processing (possibly due to interruption or no solution found): {e}")
            # Return original base parameters and calculated evaluations up to interruption (or 0 if error before run)
            num_evals_on_error = self.ga_instance.generations_completed * sol_per_pop if hasattr(self.ga_instance, 'generations_completed') else 0
            return self.scenario_config["strategy_params"].copy(), num_evals_on_error

def get_ga_optimizer_parameter_defaults():
    """Returns default parameters for GA specific settings."""
    return {
        "ga_num_generations": {"default": 100, "type": "int", "low": 10, "high": 1000, "help": "Number of generations for GA."},
        "ga_num_parents_mating": {"default": 10, "type": "int", "low": 2, "high": 50, "help": "Number of parents to mate in GA."},
        "ga_sol_per_pop": {"default": 50, "type": "int", "low": 10, "high": 200, "help": "Solutions per population in GA."},
        "ga_parent_selection_type": {"default": "sss", "type": "categorical", "values": ["sss", "rws", "sus", "rank", "random", "tournament"], "help": "Parent selection type for GA."},
        "ga_crossover_type": {"default": "single_point", "type": "categorical", "values": ["single_point", "two_points", "uniform", "scattered"], "help": "Crossover type for GA."},
        "ga_mutation_type": {"default": "random", "type": "categorical", "values": ["random", "swap", "inversion", "scramble", "adaptive"], "help": "Mutation type for GA."},
        "ga_mutation_percent_genes": {"default": "default", "type": "str", "help": "Percentage of genes to mutate (e.g., 'default', 10 for 10%)."} # PyGAD uses "default" or float/int
    }

if __name__ == '__main__':
    # This is a placeholder for testing the optimizer standalone
    # You would need to mock scenario_config, backtester_instance, etc.
    print("Genetic Optimizer Module")
    # Example:
    # mock_global_config = {"optimizer_parameter_defaults": get_ga_optimizer_parameter_defaults()}
    # mock_scenario = {
    #     "name": "test_ga",
    #     "strategy_params": {"param1": 10},
    #     "optimize": [
    #         {"parameter": "param1", "type": "int", "min_value": 1, "max_value": 20},
    #         {"parameter": "param2", "type": "float", "min_value": 0.1, "max_value": 1.0},
    #         {"parameter": "param3", "type": "categorical", "values": ["A", "B", "C"]}
    #     ],
    #     "optimization_metric": "Sharpe",
    #     # ... other necessary configs
    # }
    # mock_backtester = ... (needs a mock Backtester with _evaluate_params_walk_forward)
    # optimizer = GeneticOptimizer(mock_scenario, mock_backtester, mock_global_config, ...)
    # best_params, trials = optimizer.run()
    # print(f"Best params: {best_params}, Trials: {trials}")

    # Example of how defaults might be used or exposed:
    # defaults = get_ga_optimizer_parameter_defaults()
    # print("GA Parameter Defaults:", defaults)
    pass
