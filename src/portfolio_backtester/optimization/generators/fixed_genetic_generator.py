import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np

from ..parameter_generator import PopulationBasedParameterGenerator, validate_parameter_space
from ..results import EvaluationResult, OptimizationResult

logger = logging.getLogger(__name__)


class FixedGeneticParameterGenerator(PopulationBasedParameterGenerator):
    """A fixed genetic algorithm implementation."""

    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.parameter_space: Dict[str, Any] = {}
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.population: Optional[List[Dict[str, Any]]] = None
        self.generation_count = 0
        self.max_generations = 10
        self._is_finished = False
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Dict[str, Any]] = None
        self.best_fitness: float = -1e9

    def initialize(
        self, scenario_config: Dict[str, Any], optimization_config: Dict[str, Any]
    ) -> None:
        """Initialize the generator with configuration."""
        self.parameter_space = scenario_config.get("parameter_space", {})
        validate_parameter_space(self.parameter_space)

        ga_config = scenario_config.get("ga_settings", optimization_config.get("ga_settings", {}))
        self.population_size = ga_config.get("population_size", 50)
        self.max_generations = ga_config.get("max_generations", 10)
        self.mutation_rate = ga_config.get("mutation_rate", 0.1)
        self.crossover_rate = ga_config.get("crossover_rate", 0.8)

        self.population = self._create_initial_population()

    def _create_initial_population(self) -> List[Dict[str, Any]]:
        """Create the initial population."""
        return [self._create_individual() for _ in range(self.population_size)]

    def _create_individual(self) -> Dict[str, Any]:
        """Create a single random individual."""
        individual = {}
        for param_name, param_config in self.parameter_space.items():
            param_type = param_config["type"]
            if param_type == "int":
                individual[param_name] = self.rng.integers(
                    param_config["low"], param_config["high"], endpoint=True
                )
            elif param_type == "float":
                individual[param_name] = self.rng.uniform(param_config["low"], param_config["high"])
            elif param_type == "categorical":
                individual[param_name] = self.rng.choice(param_config["choices"])
        return individual

    def suggest_population(self) -> List[Dict[str, Any]]:
        """Suggest an entire population of parameter sets to evaluate."""
        if self.population is None:
            raise RuntimeError("Generator has not been initialized.")
        return self.population

    def report_population_results(
        self, population: List[Dict[str, Any]], results: List[EvaluationResult]
    ) -> None:
        """Report the results for an entire population."""
        # --- Fitness Extraction ---
        fitness_values = []
        for res in results:
            # For now, handle single-objective only. Multi-objective will be more complex.
            if isinstance(res.objective_value, (int, float)):
                fitness_values.append(res.objective_value)
            else:
                # Simple way to handle multi-objective: use the first objective value
                # A better approach (e.g., Pareto dominance) could be implemented later.
                if isinstance(res.objective_value, list) and res.objective_value:
                    fitness_values.append(res.objective_value[0])
                else:
                    fitness_values.append(-1e9)  # Penalize invalid results

        self.population = self._evolve_population(population, fitness_values)

        self.generation_count += 1
        if self.generation_count >= self.max_generations:
            self._is_finished = True

    def _evolve_population(
        self, population: List[Dict[str, Any]], fitness_values: List[float]
    ) -> List[Dict[str, Any]]:
        """Evolve the population to the next generation."""
        new_population = []

        # --- Elitism ---
        # Carry over the best individual from the current generation.
        best_index = np.argmax(fitness_values)
        new_population.append(population[best_index])

        # --- Update Best Individual ---
        if fitness_values[best_index] > self.best_fitness:
            self.best_fitness = fitness_values[best_index]
            self.best_individual = population[best_index]

        # --- Selection, Crossover, and Mutation ---
        while len(new_population) < self.population_size:
            parent1 = self._selection(population, fitness_values)
            parent2 = self._selection(population, fitness_values)

            child = self._crossover(parent1, parent2)
            mutated_child = self._mutation(child)

            new_population.append(mutated_child)

        return new_population

    def _selection(
        self,
        population: List[Dict[str, Any]],
        fitness_values: List[float],
        tournament_size: int = 3,
    ) -> Dict[str, Any]:
        """Select an individual using tournament selection."""
        tournament_indices = self.rng.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index_in_tournament = np.argmax(tournament_fitness)
        winner_index_in_population = tournament_indices[winner_index_in_tournament]
        return cast(Dict[str, Any], population[winner_index_in_population])

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform uniform crossover."""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy()

        child = {}
        for key in parent1.keys():
            if self.rng.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual."""
        if self.rng.random() > self.mutation_rate:
            return individual

        mutated_individual = individual.copy()
        param_to_mutate = self.rng.choice(list(self.parameter_space.keys()))

        param_config = self.parameter_space[param_to_mutate]
        param_type = param_config["type"]

        if param_type == "int":
            mutated_individual[param_to_mutate] = self.rng.integers(
                param_config["low"], param_config["high"], endpoint=True
            )
        elif param_type == "float":
            mutated_individual[param_to_mutate] = self.rng.uniform(
                param_config["low"], param_config["high"]
            )
        elif param_type == "categorical":
            mutated_individual[param_to_mutate] = self.rng.choice(param_config["choices"])

        return mutated_individual

    def is_finished(self) -> bool:
        """Check whether optimization should continue."""
        return self._is_finished

    def get_best_result(self) -> "OptimizationResult":
        """Get the best optimization result found so far."""
        if self.best_individual is None:
            return OptimizationResult(
                best_parameters={},
                best_value=-1e9,
                n_evaluations=self.generation_count * self.population_size,
                optimization_history=self.optimization_history,
            )

        return OptimizationResult(
            best_parameters=self.best_individual,
            best_value=self.best_fitness,
            n_evaluations=self.generation_count * self.population_size,
            optimization_history=self.optimization_history,
        )

    def get_best_parameters(self) -> Dict[str, Any]:
        """Return the best parameter dictionary discovered so far."""
        return self.best_individual if self.best_individual is not None else {}
