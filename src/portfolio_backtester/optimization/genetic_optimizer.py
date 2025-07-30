import logging
import time

import numpy as np
import optuna
import pygad

# new import for adaptive controllers
from .adaptive_parameters import (
    DiversityCalculator,
    AdaptiveMutationController,
    AdaptiveCrossoverController,
)

# new import for advanced crossover operators
from .advanced_crossover import get_crossover_operator, CROSSOVER_OPERATORS

# _evaluate_params_walk_forward is now a method of the Backtester class
from ..optimization.trial_evaluator import TrialEvaluator
from ..utils import generate_randomized_wfo_windows
from .elite_archive import EliteArchive

from .base_optimizer import BaseOptimizer

# Set up logger
logger = logging.getLogger(__name__)

class GeneticOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ga_instance = None
        self.zero_fitness_streak = 0
        self.best_fitness_so_far = -np.inf
        self.start_time = None

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
            objectives_values = evaluator.evaluate(current_params)

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

    def optimize(self, save_plot=True):
        logger.debug("Setting up Genetic Algorithm...")
        
        # Initialize variables that might be referenced in exception handling
        ga_params_config = self.scenario_config.get("genetic_algorithm_params", {})
        sol_per_pop = ga_params_config.get("sol_per_pop", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_sol_per_pop", {}).get("default", 50))

        try:
            gene_space, gene_types_for_pygad = self._get_gene_space_and_types()
            num_genes = len(self.optimization_params_spec)

            if num_genes == 0:
                logger.error("No optimization parameters specified.")
                return self.scenario_config["strategy_params"].copy(), 0, None

            # ------------------------------------------------------------------
            # Adaptive parameter control setup
            # ------------------------------------------------------------------
            adaptive_cfg = ga_params_config.get("adaptive_mutation", {})
            adaptive_enabled = adaptive_cfg.get("enabled", False)
            if adaptive_enabled:
                # Diversity calculator requires gene_space to compute normalised ranges.
                diversity_calc = DiversityCalculator(gene_space)
                mutation_controller = AdaptiveMutationController(
                    base_rate=adaptive_cfg.get("base_rate", 0.1),
                    min_rate=adaptive_cfg.get("min_rate", 0.01),
                    max_rate=adaptive_cfg.get("max_rate", 0.5),
                    diversity_threshold=adaptive_cfg.get("diversity_threshold", 0.3),
                    max_generations=ga_params_config.get("num_generations", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_num_generations", {}).get("default", 100)),
                )
                crossover_controller = AdaptiveCrossoverController(
                    base_rate=adaptive_cfg.get("base_crossover_rate", 0.8),
                    min_rate=adaptive_cfg.get("min_crossover_rate", 0.6),
                    max_rate=adaptive_cfg.get("max_crossover_rate", 0.95),
                )
            else:
                diversity_calc = None
                mutation_controller = None
                crossover_controller = None

            # ------------------------------------------------------------------
            # Elite preservation setup
            # ------------------------------------------------------------------
            elite_cfg = ga_params_config.get("elite_preservation", {})
            elite_enabled = elite_cfg.get("enabled", False)
            if elite_enabled:
                archive = EliteArchive(
                    max_size=elite_cfg.get("max_archive_size", 50),
                    aging_factor=elite_cfg.get("aging_factor", 0.95),
                )
                injection_strategy = elite_cfg.get("injection_strategy", "direct")
                injection_frequency = elite_cfg.get("injection_frequency", 5)
                min_elites = elite_cfg.get("min_elites", 2)
                max_elites = elite_cfg.get("max_elites", 5)
            else:
                archive = None
                injection_strategy = "direct"
                injection_frequency = 0
                min_elites = 0
                max_elites = 0

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

            self.start_time = time.time()

            on_generation_callback = None
            if (self.backtester.args.early_stop_patience > 0 or self.backtester.args.timeout is not None) or adaptive_enabled:
                self.zero_fitness_streak = 0
                self.best_fitness_so_far = -np.inf

                def on_gen(ga_instance):
                    """Callback executed by PyGAD after each generation."""
                    try:
                        # Snapshot population and fitness arrays immediately
                        population = ga_instance.population
                        fitness_arr = np.asarray(ga_instance.last_generation_fitness, dtype=float)
                        generation_no = ga_instance.generations_completed

                        # -----------------------------
                        # Timeout handling
                        # -----------------------------
                        if self.backtester.args.timeout is not None and self.start_time is not None:
                            elapsed_time = time.time() - self.start_time
                            if elapsed_time > self.backtester.args.timeout:
                                logger.warning(
                                    f"Genetic Algorithm optimization timed out after {elapsed_time:.2f} seconds."
                                )
                                return "stop"

                        # -----------------------------
                        # Early stopping based on fitness stagnation
                        # -----------------------------
                        current_best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
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

                        if (
                            self.backtester.args.early_stop_patience > 0
                            and self.zero_fitness_streak >= self.backtester.args.early_stop_patience
                        ):
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    f"Early stopping GA due to {self.zero_fitness_streak} generations with no improvement."
                                )
                            return "stop"

                        # -----------------------------
                        # Adaptive parameter control
                        # -----------------------------
                        if adaptive_enabled and diversity_calc is not None and mutation_controller is not None and crossover_controller is not None:
                            pop_diversity = diversity_calc.phenotypic_diversity(population)
                            fitness_var = float(np.var(fitness_arr))
                            
                            new_mut_rate = mutation_controller.rate(pop_diversity, fitness_var, generation_no)
                            new_cx_rate = crossover_controller.rate(pop_diversity, 1.0 - pop_diversity)

                            # Map probability -> percent of genes for PyGAD (0-1 -> 0-100)
                            ga_instance.mutation_percent_genes = int(new_mut_rate * 100)
                            # If PyGAD exposes mutation_probability or crossover_probability attributes, set them too.
                            if hasattr(ga_instance, "mutation_probability"):
                                ga_instance.mutation_probability = new_mut_rate
                            if hasattr(ga_instance, "crossover_probability"):
                                ga_instance.crossover_probability = new_cx_rate

                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    f"[Gen {generation_no}] Diversity={pop_diversity:.3f}, MutationRate={new_mut_rate:.3f}, CrossoverRate={new_cx_rate:.3f}"
                                )

                        # -----------------------------
                        # Elite archive maintenance & injection
                        # -----------------------------
                        if elite_enabled and archive is not None:
                            # Add current generation elites (top min_elites by fitness)
                            elite_count = max(min_elites, 1)
                            top_idx = np.argsort(fitness_arr)[-elite_count:][::-1]
                            for idx in top_idx:
                                archive.add(population[idx], float(fitness_arr[idx]), generation_no)

                            # Periodic injection
                            if injection_frequency > 0 and generation_no % injection_frequency == 0:
                                population, fitness_arr = archive.inject(
                                    population,
                                    fitness_arr,
                                    strategy=injection_strategy,
                                    num_elites=max_elites,
                                )
                                # Update GA instance arrays directly
                                ga_instance.population[:] = population
                                ga_instance.last_generation_fitness[:] = fitness_arr

                        # Log current generation stats
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Generation {ga_instance.generations_completed}: BestFitness={current_best_fitness}"
                            )

                        # User interrupt flag (if exists in surrounding scope)
                        if "CENTRAL_INTERRUPTED_FLAG" in globals() and globals()["CENTRAL_INTERRUPTED_FLAG"]:
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
                "mutation_type": mutation_type,
                "mutation_percent_genes": mutation_percent_genes,
                "random_seed": pygad_seed,
                "on_generation": on_generation_callback,
            }

            # Handle advanced crossover operators
            crossover_operator_name = ga_params_config.get("advanced_crossover_type")
            if crossover_operator_name and crossover_operator_name in CROSSOVER_OPERATORS:
                custom_crossover_func = get_crossover_operator(crossover_operator_name)
                if custom_crossover_func:
                    ga_kwargs["crossover_type"] = custom_crossover_func
                    # Store additional parameters for the crossover operator
                    if crossover_operator_name == "simulated_binary":
                        ga_kwargs["sbx_distribution_index"] = ga_params_config.get("sbx_distribution_index", 20.0)
                    elif crossover_operator_name == "multi_point":
                        ga_kwargs["num_crossover_points"] = ga_params_config.get("num_crossover_points", 3)
                    elif crossover_operator_name == "uniform_variant":
                        ga_kwargs["uniform_crossover_bias"] = ga_params_config.get("uniform_crossover_bias", 0.5)
                    logger.debug(f"Using advanced crossover operator: {crossover_operator_name}")
            else:
                # If no advanced crossover specified, use the standard PyGAD crossover types
                ga_kwargs["crossover_type"] = crossover_type

            # Add parallel processing if available
            #if self.backtester.n_jobs > 1:
            #    ga_kwargs["parallel_processing"] = ['thread', self.backtester.n_jobs]

            # For multi-objective optimization, try to use NSGA-II if available
            if self.is_multi_objective:
                try:
                    # Check if PyGAD supports NSGA-II
                    import pygad.nsga2 as nsga2_module
                    ga_kwargs["algorithm_type"] = "nsga2"
                    logger.debug("Using NSGA-II algorithm for multi-objective optimization.")
                except (ImportError, AttributeError):
                    logger.warning("NSGA-II not available in this PyGAD version. Using standard GA with weighted objectives.")

            keep_elitism = ga_params_config.get("keep_elitism", 2)
            ga_kwargs["keep_elitism"] = keep_elitism

            self.ga_instance = pygad.GA(**ga_kwargs)

            logger.debug("Running Genetic Algorithm...")
            self.ga_instance.run()

            # Process results
            num_evaluations = self.ga_instance.generations_completed * sol_per_pop
            solution = None
            solution_fitness = None

            if self.is_multi_objective and hasattr(self.ga_instance, 'best_solutions'):
                # Multi-objective results
                pareto_solutions_fitness = getattr(self.ga_instance, 'best_solutions_fitness', [])
                pareto_chromosomes = getattr(self.ga_instance, 'best_solutions', [])

                if not pareto_chromosomes:
                    logger.error("GA (multi-objective): No solutions found on Pareto front.")
                    return self.scenario_config["strategy_params"].copy(), num_evaluations, None

                # Select the first solution from the Pareto front as representative
                solution = pareto_chromosomes[0]
                solution_fitness = pareto_solutions_fitness[0] if pareto_solutions_fitness else "Unknown"
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"GA multi-objective: Selected first solution from Pareto front. Fitness: {solution_fitness}"
                    )
            else:
                # Single objective results
                if self.ga_instance.best_solution_generation == -1 and not globals().get("CENTRAL_INTERRUPTED_FLAG", False):
                     logger.error("GA (single-objective): No best solution found and not due to interruption.")
                     return self.scenario_config["strategy_params"].copy(), num_evaluations, None

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
                return optimal_params, num_evaluations, solution
            else:
                logger.warning("GA: No valid solution to decode, returning default parameters.")
                return self.scenario_config["strategy_params"].copy(), 0, None

        except Exception as e:
            logger.error(f"Error during GA optimization: {e}")
            # Return original base parameters and calculated evaluations up to interruption
            sol_per_pop_safe = ga_params_config.get("sol_per_pop", self.global_config.get("optimizer_parameter_defaults", {}).get("ga_sol_per_pop", {}).get("default", 50))
            num_evals_on_error = getattr(self.ga_instance, 'generations_completed', 0) * sol_per_pop_safe if hasattr(self, 'ga_instance') else 0
            return self.scenario_config["strategy_params"].copy(), num_evals_on_error, None

def get_ga_optimizer_parameter_defaults():
    """Returns default parameters for GA specific settings."""
    return {
        "ga_num_generations": {"default": 100, "type": "int", "low": 10, "high": 1000, "help": "Number of generations for GA."},
        "ga_num_parents_mating": {"default": 10, "type": "int", "low": 2, "high": 50, "help": "Number of parents to mate in GA."},
        "ga_sol_per_pop": {"default": 50, "type": "int", "low": 10, "high": 200, "help": "Solutions per population in GA."},
        "ga_parent_selection_type": {"default": "sss", "type": "categorical", "values": ["sss", "rws", "sus", "rank", "random", "tournament"], "help": "Parent selection type for GA."},
        "ga_crossover_type": {"default": "single_point", "type": "categorical", "values": ["single_point", "two_points", "uniform", "scattered"], "help": "Crossover type for GA."},
        "ga_advanced_crossover_type": {"default": None, "type": "categorical", "values": [None, "simulated_binary", "multi_point", "uniform_variant", "arithmetic"], "help": "Advanced crossover operator for GA."},
        "ga_sbx_distribution_index": {"default": 20.0, "type": "float", "low": 1.0, "high": 100.0, "help": "Distribution index for Simulated Binary Crossover."},
        "ga_num_crossover_points": {"default": 3, "type": "int", "low": 2, "high": 10, "help": "Number of crossover points for Multi-point crossover."},
        "ga_uniform_crossover_bias": {"default": 0.5, "type": "float", "low": 0.1, "high": 0.9, "help": "Bias parameter for Uniform crossover variant."},
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
