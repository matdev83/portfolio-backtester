import pygad
import numpy as np
import logging # Import logging
import optuna
# _evaluate_params_walk_forward is now a method of the Backtester class
from ..utils import generate_randomized_wfo_windows

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


from ..optimization.trial_evaluator import TrialEvaluator

class GeneticOptimizer:
    def __init__(self, scenario_config, backtester_instance, global_config, monthly_data, daily_data, rets_full, random_state=None):
        self.scenario_config = scenario_config
        self.backtester = backtester_instance
        self.backtester_instance = backtester_instance  # For backward compatibility
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
            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f"Multi-objective optimization for: {self.metrics_to_optimize}")

        self.ga_instance = None

        # Early stopping variables
        self.zero_fitness_streak = 0
        self.best_fitness_so_far = -np.inf

    def _validate_gene_space(self, gene_space):
        """Validate gene space to prevent PyGAD errors."""
        for i, space in enumerate(gene_space):
            if isinstance(space, dict):
                low = space.get("low", 0)
                high = space.get("high", 1)
                step = space.get("step", 1)
                
                # Ensure valid range
                if low >= high:
                    raise ValueError(f"Gene {i}: low ({low}) must be less than high ({high})")
                
                # For integer genes, ensure at least one valid value exists
                if step > 0 and (high - low) < step:
                    if logger.isEnabledFor(logging.WARNING):

                        logger.warning(f"Gene {i}: range too small for step size. Adjusting step to 1.")
                    space["step"] = 1
                    
        return gene_space

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
        try:
            current_params = self._decode_chromosome(solution)

            trial_scenario_config = self.scenario_config.copy()
            trial_scenario_config["strategy_params"] = current_params

            # Mock Optuna trial object for _evaluate_params_walk_forward
            class MockTrial:
                def __init__(self, params, study=None, number=0):
                    self.params = params
                    self.user_attrs = {}
                    self.study = study
                    self.number = number

                def suggest_int(self, name, low, high, step=1): return self.params.get(name)
                def suggest_float(self, name, low, high, step=None, log=False): return self.params.get(name)
                def suggest_categorical(self, name, choices): return self.params.get(name)
                def report(self, value, step): pass
                def should_prune(self): return False
                def set_user_attr(self, key, value): self.user_attrs[key] = value

            mock_study = type('MockStudy', (), {'directions': [optuna.study.StudyDirection.MAXIMIZE for _ in self.metrics_to_optimize]})()
            mock_trial = MockTrial(current_params, mock_study, solution_idx)

            # Generate walk-forward windows using centralized function
            windows = generate_randomized_wfo_windows(
                self.monthly_data.index,
                self.scenario_config,
                self.global_config,
                self.random_state
            )

            if not windows:
                logger.error("Not enough data for walk-forward windows in GeneticOptimizer.")
                return -np.inf if not self.is_multi_objective else [-np.inf] * len(self.metrics_to_optimize)

            evaluator = TrialEvaluator(self.backtester, self.scenario_config, self.monthly_data, self.daily_data, self.rets_full, self.metrics_to_optimize, self.is_multi_objective, windows)
            objectives_values = evaluator.evaluate(mock_trial)

            if self.is_multi_objective:
                # PyGAD's NSGA-II expects a list/tuple of objective values.
                processed_objectives = []
                optimization_targets_config = self.scenario_config.get("optimization_targets", [])
                directions = [t.get("direction", "maximize").lower() for t in optimization_targets_config]
                if not directions and len(self.metrics_to_optimize) == 1:
                     directions = ["maximize"]
                elif len(directions) != len(self.metrics_to_optimize):
                     directions = ["maximize"] * len(self.metrics_to_optimize)

                # Ensure objectives_values is a flat list of scalars
                if isinstance(objectives_values, np.ndarray):
                    objectives_iter = objectives_values.flatten().tolist()
                elif not isinstance(objectives_values, (list, tuple)):
                    objectives_iter = [objectives_values]
                else:
                    objectives_iter = list(objectives_values)

                for i, val in enumerate(objectives_iter):
                    if directions[i] == "minimize":
                        processed_objectives.append(float(-val) if np.isfinite(val) else np.inf)
                    else:
                        processed_objectives.append(float(val) if np.isfinite(val) else -np.inf)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Solution {solution_idx} params: {current_params}, objectives: {processed_objectives}")
                return processed_objectives
            else:
                # Single objective
                fitness_value = objectives_values[0] if isinstance(objectives_values, (list, tuple)) else objectives_values
                
                # PyGAD maximizes. If original objective was minimize, negate it.
                opt_target_config = self.scenario_config.get("optimization_targets")
                direction = "maximize"
                if opt_target_config and isinstance(opt_target_config, list) and len(opt_target_config) > 0:
                    direction = opt_target_config[0].get("direction", "maximize").lower()

                if direction == "minimize":
                    result = -fitness_value if np.isfinite(fitness_value) else np.inf
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Solution {solution_idx} params: {current_params}, raw_fitness: {fitness_value}, adjusted_fitness: {result}")
                    return result
                else:
                    result = fitness_value if np.isfinite(fitness_value) else -np.inf
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Solution {solution_idx} params: {current_params}, fitness: {result}")
                    return result

        except Exception as e:
            logger.error(f"Error in fitness function for solution {solution_idx}: {e}")
            # Return worst possible fitness for this solution
            if self.is_multi_objective:
                return [-np.inf] * len(self.metrics_to_optimize)
            else:
                return -np.inf

    def _get_gene_space_and_types(self):
        gene_space = []
        gene_type = []

        for spec in self.optimization_params_spec:
            pname = spec["parameter"]
            opt_def = self.global_config.get("optimizer_parameter_defaults", {}).get(pname, {})
            ptype = spec.get("type", opt_def.get("type"))

            low = spec.get("min_value", opt_def.get("low"))
            high = spec.get("max_value", opt_def.get("high"))
            step = spec.get("step", opt_def.get("step", 1 if ptype == "int" else None))

            if ptype == "int":
                # Validate that we have valid bounds
                if low is None or high is None:
                    raise ValueError(f"Parameter {pname}: min_value and max_value must be specified")
                    
                low, high = int(low), int(high)
                step = int(step) if step else 1
                
                # Ensure valid range
                if low >= high:
                    raise ValueError(f"Parameter {pname}: min_value ({low}) must be less than max_value ({high})")
                
                gene_space.append({"low": low, "high": high, "step": step})
                gene_type.append(int)
            elif ptype == "float":
                # Validate that we have valid bounds
                if low is None or high is None:
                    raise ValueError(f"Parameter {pname}: min_value and max_value must be specified")
                    
                low, high = float(low), float(high)
                
                # Ensure valid range
                if low >= high:
                    raise ValueError(f"Parameter {pname}: min_value ({low}) must be less than max_value ({high})")
                    
                gene_space.append({"low": low, "high": high})
                gene_type.append(float)
            elif ptype == "categorical":
                choices = spec.get("values", opt_def.get("values"))
                if not choices:
                    raise ValueError(f"Categorical parameter {pname} has no choices.")
                if len(choices) < 2:
                    raise ValueError(f"Categorical parameter {pname} must have at least 2 choices.")
                    
                # For categorical parameters, we don't use min_value/max_value, we use the number of choices
                gene_space.append({"low": 0, "high": len(choices) - 1, "step": 1})
                gene_type.append(int)
            else:
                raise ValueError(f"Unsupported parameter type {ptype} for parameter {pname} in genetic optimizer.")
        
        # Validate the gene space
        gene_space = self._validate_gene_space(gene_space)
        return gene_space, gene_type

    def run(self, save_plot=True):
        logger.debug("Setting up Genetic Algorithm...")
        
        # Initialize variables that might be referenced in exception handling
        ga_params_config = self.scenario_config.get("genetic_algorithm_params", {})
        sol_per_pop = ga_params_config.get("sol_per_pop", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_sol_per_pop", {}).get("default", 50))

        try:
            gene_space, gene_types_for_pygad = self._get_gene_space_and_types()
            num_genes = len(self.optimization_params_spec)

            if num_genes == 0:
                logger.error("No optimization parameters specified.")
                return self.scenario_config["strategy_params"].copy(), 0

            # GA Parameters
            num_generations = ga_params_config.get("num_generations", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_num_generations", {}).get("default", 100))
            num_parents_mating = ga_params_config.get("num_parents_mating", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_num_parents_mating", {}).get("default", 10))
            parent_selection_type = ga_params_config.get("parent_selection_type", "sss")
            crossover_type = ga_params_config.get("crossover_type", "single_point")
            mutation_type = ga_params_config.get("mutation_type", "random")
            mutation_percent_genes = ga_params_config.get("mutation_percent_genes", "default")

            # Ensure minimum population size
            if sol_per_pop < 4:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Population size {sol_per_pop} too small, setting to 4")
                sol_per_pop = 4
            
            if num_parents_mating >= sol_per_pop:
                num_parents_mating = max(2, sol_per_pop // 2)
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"num_parents_mating adjusted to {num_parents_mating}")

            # PyGAD seed
            pygad_seed = self.random_state if self.random_state is not None else np.random.randint(0, 10000)

            on_generation_callback = None
            if self.backtester.args.early_stop_patience > 0:
                self.zero_fitness_streak = 0
                self.best_fitness_so_far = -np.inf

                def on_gen(ga_instance):
                    try:
                        current_best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Generation {ga_instance.generations_completed}, Best fitness: {current_best_fitness}")

                        if np.isfinite(current_best_fitness) and np.isfinite(self.best_fitness_so_far):
                            if abs(current_best_fitness - self.best_fitness_so_far) < 1e-6:
                                self.zero_fitness_streak += 1
                            else:
                                self.zero_fitness_streak = 0
                        elif current_best_fitness == self.best_fitness_so_far:
                            self.zero_fitness_streak += 1
                        else:
                            self.zero_fitness_streak = 0

                        self.best_fitness_so_far = max(self.best_fitness_so_far, current_best_fitness)

                        if self.zero_fitness_streak >= self.backtester.args.early_stop_patience:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Early stopping GA due to {self.zero_fitness_streak} generations with no improvement.")
                            return "stop"

                        if CENTRAL_INTERRUPTED_FLAG:
                            logger.warning("Genetic Algorithm optimization interrupted by user via central flag.")
                            return "stop"
                    except Exception as e:
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(f"Error in generation callback: {e}")
                        
                on_generation_callback = on_gen

            # Configure GA parameters
            ga_kwargs = {
                "num_generations": num_generations,
                "num_parents_mating": num_parents_mating,
                "fitness_func": self.fitness_func_wrapper,
                "sol_per_pop": sol_per_pop,
                "num_genes": num_genes,
                "gene_space": gene_space,
                "parent_selection_type": parent_selection_type,
                "crossover_type": crossover_type,
                "mutation_type": mutation_type,
                "mutation_percent_genes": mutation_percent_genes,
                "random_seed": pygad_seed,
                "on_generation": on_generation_callback,
            }

            # Add parallel processing if available
            if self.backtester.n_jobs > 1:
                ga_kwargs["parallel_processing"] = ['thread', self.backtester.n_jobs]

            # For multi-objective optimization, try to use NSGA-II if available
            if self.is_multi_objective:
                try:
                    # Check if PyGAD supports NSGA-II
                    import pygad.nsga2 as nsga2_module
                    ga_kwargs["algorithm_type"] = "nsga2"
                    logger.debug("Using NSGA-II algorithm for multi-objective optimization.")
                except (ImportError, AttributeError):
                    logger.warning("NSGA-II not available in this PyGAD version. Using standard GA with weighted objectives.")

            self.ga_instance = pygad.GA(**ga_kwargs)

            logger.debug("Running Genetic Algorithm...")
            self.ga_instance.run()

            # Process results
            num_evaluations = self.ga_instance.generations_completed * sol_per_pop

            if self.is_multi_objective and hasattr(self.ga_instance, 'best_solutions'):
                # Multi-objective results
                pareto_solutions_fitness = getattr(self.ga_instance, 'best_solutions_fitness', [])
                pareto_chromosomes = getattr(self.ga_instance, 'best_solutions', [])

                if not pareto_chromosomes:
                    logger.error("GA (multi-objective): No solutions found on Pareto front.")
                    return self.scenario_config["strategy_params"].copy(), num_evaluations

                # Select the first solution from the Pareto front as representative
                solution = pareto_chromosomes[0]
                solution_fitness = pareto_solutions_fitness[0] if pareto_solutions_fitness else "Unknown"
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"GA multi-objective: Selected first solution from Pareto front. Fitness: {solution_fitness}"
                    )
            else:
                # Single objective results
                if self.ga_instance.best_solution_generation == -1 and not CENTRAL_INTERRUPTED_FLAG:
                     logger.error("GA (single-objective): No best solution found and not due to interruption.")
                     return self.scenario_config["strategy_params"].copy(), num_evaluations

                solution, solution_fitness, _ = self.ga_instance.best_solution(pop_fitness=self.ga_instance.last_generation_fitness)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"GA single-objective: Best fitness: {solution_fitness}")

            if self.ga_instance.generations_completed > 0 and solution is not None:
                if save_plot:
                    try:
                        # Ensure plots directory exists
                        import os
                        os.makedirs("plots", exist_ok=True)
                        plot_path = f"plots/ga_fitness_{self.scenario_config['name']}.png"

                        # Switch to non-interactive backend when interactive mode is disabled
                        try:
                            import matplotlib
                            if not getattr(self.backtester.args, "interactive", False):
                                matplotlib.use("Agg", force=True)
                        except Exception as e:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Could not set matplotlib backend: {e}")

                        self.ga_instance.plot_fitness(title="GA Fitness Value vs. Generation", save_dir=plot_path)

                        # Close the figure to free resources and avoid blocking even in Agg backend
                        try:
                            import matplotlib.pyplot as plt
                            plt.close('all')
                        except Exception:
                            pass
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"GA fitness plot saved to {plot_path}")
                    except Exception as e:
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(f"Could not save GA fitness plot: {e}")
                        
                optimal_params = self.scenario_config["strategy_params"].copy()
                optimal_params.update(self._decode_chromosome(solution))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Best parameters found by GA: {optimal_params}")
                print(f"Genetic Optimizer - Best parameters found: {optimal_params}")
                return optimal_params, num_evaluations
            else:
                logger.warning("GA: No valid solution to decode, returning default parameters.")
                return self.scenario_config["strategy_params"].copy(), 0

        except Exception as e:
            logger.error(f"Error during GA optimization: {e}")
            # Return original base parameters and calculated evaluations up to interruption
            num_evals_on_error = getattr(self.ga_instance, 'generations_completed', 0) * sol_per_pop if hasattr(self, 'ga_instance') else 0
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
