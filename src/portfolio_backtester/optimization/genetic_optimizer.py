import logging
import pygad
import multiprocessing

from ..optimization.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


# Helper function for multiprocessing
def _evaluate_solution_parallel(args):
    """Helper function to run a single backtest trial in a separate process."""
    backtester_instance, scenario_config, params = args
    return backtester_instance.evaluate_trial_parameters(scenario_config, params)


class GeneticOptimizer(BaseOptimizer):
    """
    A Genetic Algorithm optimizer that uses PyGAD and parallel processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ga_instance = None
        self.pool = None

    def _batch_fitness_function(self, ga_instance, solutions, solutions_indices):
        """
        Fitness function that evaluates a batch of solutions in parallel.
        """
        if not self.pool:
            self.pool = multiprocessing.Pool()

        args_list = []
        for sol in solutions:
            params = self._decode_solution(sol)
            # For multi-categorical, we need to create a separate trial for each combination
            if any(p.get("type") == "multi-categorical" for p in self.optimization_params_spec):
                # This is a simplified approach. A more robust implementation would handle
                # multiple multi-categorical parameters.
                multi_cat_param = next(
                    p for p in self.optimization_params_spec if p.get("type") == "multi-categorical"
                )
                param_name = multi_cat_param["parameter"]
                choices = multi_cat_param["values"]
                for i in range(1, 1 << len(choices)):
                    temp_params = params.copy()
                    selected_values = []
                    for j, choice in enumerate(choices):
                        if (i >> j) & 1:
                            selected_values.append(choice)
                    temp_params[param_name] = list[str](
                        selected_values
                    )  # Cast to list[str] for type checker
                    args_list.append((self.backtester, self.scenario_config, temp_params))
            else:
                args_list.append((self.backtester, self.scenario_config, params))

        results = self.pool.map(_evaluate_solution_parallel, args_list)

        # Extract the desired metric for fitness
        fitness_values = [res.get(self.metrics_to_optimize[0], 0) for res in results]
        return fitness_values

    def _decode_solution(self, solution):
        """Converts a GA solution (chromosome) to a parameter dictionary."""
        params: dict[str, int | float | str | list[str]] = {}
        for i, param_spec in enumerate(self.optimization_params_spec):
            param_name = param_spec["parameter"]
            value = solution[i]
            param_type = param_spec.get("type")

            if param_type == "int":
                params[param_name] = int(round(value))
            elif param_type == "categorical":
                choices = param_spec["values"]
                params[param_name] = choices[int(round(value))]
            elif param_type == "multi-categorical":
                choices = param_spec["values"]
                # The solution for a multi-categorical parameter is a bitmask
                mask = int(round(value))
                selected_values = []
                for i, choice in enumerate(choices):
                    if (mask >> i) & 1:
                        selected_values.append(choice)
                params[param_name] = selected_values
            else:
                # If type is specified as 'int', ensure the value is an int.
                # This handles cases where param_type might not be one of the explicitly checked string types.
                if param_spec.get("type") == "int":
                    params[param_name] = int(round(value))
                else:
                    params[param_name] = float(value)
        return params

    def optimize(self):
        """
        Runs the entire genetic algorithm optimization and returns the best result.
        """
        gene_space = []
        for param_spec in self.optimization_params_spec:
            space = {"low": param_spec["min_value"], "high": param_spec["max_value"]}
            if "step" in param_spec:
                space["step"] = param_spec["step"]
            gene_space.append(space)

        ga_params = self.scenario_config.get("genetic_algorithm_params", {})

        self.ga_instance = pygad.GA(
            num_generations=ga_params.get("ga_num_generations", 10),
            num_parents_mating=ga_params.get("ga_num_parents_mating", 5),
            fitness_func=self._batch_fitness_function,
            sol_per_pop=ga_params.get("ga_sol_per_pop", 10),
            num_genes=len(self.optimization_params_spec),
            gene_space=gene_space,
            random_seed=self.random_state,
            crossover_type=ga_params.get("ga_crossover_type", "single_point"),
            mutation_type=ga_params.get("ga_mutation_type", "random"),
            mutation_percent_genes=ga_params.get("ga_mutation_percent_genes", 10),
            keep_elitism=ga_params.get("keep_elitism", 2),
            on_generation=lambda ga_instance: logger.info(
                f"Generation {ga_instance.generations_completed} complete. Best fitness: {ga_instance.best_solution()[1]}"
            ) if logger.isEnabledFor(logging.INFO) else None,
        )

        self.ga_instance.run()

        if self.pool:
            self.pool.close()
            self.pool.join()

        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        best_params = self._decode_solution(solution)

        logger.info(f"Genetic Algorithm finished. Best solution fitness: {solution_fitness}")

        num_trials = self.ga_instance.num_generations * self.ga_instance.sol_per_pop
        return best_params, num_trials, self.ga_instance.best_solution()
