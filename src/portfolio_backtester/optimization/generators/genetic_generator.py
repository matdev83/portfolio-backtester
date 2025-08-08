"""
Genetic Algorithm Parameter Generator using PyGAD.

This module implements a genetic algorithm parameter generator that conforms to the
ParameterGenerator interface. It uses PyGAD's external fitness function capability
to integrate with the optimization orchestrator while maintaining clean separation
of concerns.

The generator handles population management, generation advancement, and chromosome
encoding/decoding for parameter dictionaries. It completely removes dependency on
the MockTrial class and provides proper PyGAD integration.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import pygad

from ..results import EvaluationResult, OptimizationResult
from ..parameter_generator import (
    ParameterGenerator,
    ParameterGeneratorNotInitializedError,
    ParameterGeneratorFinishedError,
    InvalidParameterSpaceError,
    validate_parameter_space,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneticTrial:
    """Represents a single genetic algorithm trial/individual.

    This class stores information about a single chromosome evaluation,
    including the parameter values, fitness score, and generation number.

    Attributes:
        number: Sequential trial number
        parameters: Parameter dictionary decoded from chromosome
        chromosome: Raw chromosome representation
        fitness: Fitness value for this individual
        generation: Generation number when this individual was evaluated
    """

    number: int
    parameters: Dict[str, Any]
    chromosome: np.ndarray
    fitness: float
    generation: int


class GeneticParameterGenerator(ParameterGenerator):
    """Genetic Algorithm parameter generator using PyGAD.

    This class implements the ParameterGenerator interface using PyGAD's
    genetic algorithm with external fitness function capability. It handles
    population initialization, generation advancement, and chromosome
    encoding/decoding.

    The generator maintains complete separation from backtesting logic and
    provides a clean interface for genetic algorithm optimization.

    Attributes:
        random_state: Random seed for reproducible results
        ga_instance: PyGAD GA instance
        parameter_space: Dictionary defining the parameter space
        gene_space: PyGAD gene space configuration
        gene_types: List of parameter types for each gene
        param_names: List of parameter names in order
        is_multi_objective: Whether this is multi-objective optimization
        metrics_to_optimize: List of metrics to optimize
        optimization_directions: List of optimization directions (maximize/minimize)
        current_generation: Current generation number
        current_evaluation: Current evaluation counter
        trials: List of all trials performed
        best_trial: Best trial found so far
        optimization_history: History of all evaluations
        max_evaluations: Maximum number of evaluations
        max_generations: Maximum number of generations
        population_size: Size of the population
        is_initialized: Whether the generator has been initialized
        is_finished_flag: Whether optimization is complete
    """

    def __init__(self, random_state: Optional[int] = None):
        """Initialize the genetic parameter generator.

        Args:
            random_state: Random seed for reproducible results
        """

        self.random_state = random_state
        self.ga_instance = None

        # Parameter space configuration
        self.parameter_space: Dict[str, Any] = {}
        self.gene_space: List[Dict[str, Any]] = []
        self.gene_type: List[Any] = []
        self.gene_types: List[Any] = []
        self.param_names: List[str] = []

        # Optimization configuration
        self.is_multi_objective: bool = False
        self.metrics_to_optimize: List[str] = []
        self.optimization_directions: List[str] = []

        # State tracking
        self.current_generation: int = 0
        self.current_evaluation: int = 0
        self.trials: List[GeneticTrial] = []
        self.best_trial: Optional[GeneticTrial] = None
        self.optimization_history: List[Dict[str, Any]] = []

        # Configuration parameters
        self.max_evaluations: int = 1000
        self.max_generations: int = 100
        self.population_size: int = 50
        self.num_parents_mating: int = 10
        self.parent_selection_type: str = "sss"
        self.crossover_type: str = "single_point"
        self.mutation_type: str = "random"
        self.mutation_percent_genes: int = 10
        self.keep_elitism: int = 2

        # Initialization state
        self.is_initialized: bool = False
        self.is_finished_flag: bool = False

        # External fitness tracking
        self._pending_evaluations: Dict[str, Dict[str, Any]] = {}  # chromosome_hash -> parameters
        self._completed_evaluations: Dict[str, float] = {}  # chromosome_hash -> aggregated fitness

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"GeneticParameterGenerator initialized with random_state={random_state}")

    def initialize(
        self, scenario_config: Dict[str, Any], optimization_config: Dict[str, Any]
    ) -> None:
        """Initialize the parameter generator with configuration.

        Args:
            scenario_config: Scenario configuration including strategy settings
            optimization_config: Optimization-specific configuration

        Raises:
            InvalidParameterSpaceError: If parameter space is invalid
            ValueError: If configuration is invalid
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("GeneticParameterGenerator.initialize() called")

        # Extract and validate parameter space
        self.parameter_space = optimization_config.get("parameter_space", {})
        if not self.parameter_space:
            raise InvalidParameterSpaceError("Parameter space cannot be empty")

        validate_parameter_space(self.parameter_space)

        # Extract optimization configuration
        self.metrics_to_optimize = optimization_config.get("metrics_to_optimize", ["sharpe_ratio"])
        self.is_multi_objective = len(self.metrics_to_optimize) > 1

        # Extract optimization directions
        optimization_targets = optimization_config.get("optimization_targets", [])
        if optimization_targets:
            self.optimization_directions = [
                target.get("direction", "maximize").lower() for target in optimization_targets
            ]
        else:
            # Default to maximize for all metrics
            self.optimization_directions = ["maximize"] * len(self.metrics_to_optimize)

        # Ensure directions match metrics
        if len(self.optimization_directions) != len(self.metrics_to_optimize):
            self.optimization_directions = ["maximize"] * len(self.metrics_to_optimize)

        # Extract GA-specific configuration
        ga_config = optimization_config.get("genetic_algorithm_params", {})
        self.max_evaluations = optimization_config.get("max_evaluations", 1000)
        self.max_generations = ga_config.get("num_generations", 100)
        self.population_size = ga_config.get("sol_per_pop", 50)
        self.num_parents_mating = ga_config.get("num_parents_mating", 10)
        self.parent_selection_type = ga_config.get("parent_selection_type", "sss")
        self.crossover_type = ga_config.get("crossover_type", "single_point")
        self.mutation_type = ga_config.get("mutation_type", "random")
        self.mutation_percent_genes = ga_config.get("mutation_percent_genes", 10)
        self.keep_elitism = ga_config.get("keep_elitism", 2)

        # Validate population parameters
        if self.population_size < 4:
            logger.warning(f"Population size {self.population_size} too small, setting to 4")
            self.population_size = 4

        if self.num_parents_mating >= self.population_size:
            self.num_parents_mating = max(2, self.population_size // 2)
            logger.warning(f"num_parents_mating adjusted to {self.num_parents_mating}")

        # Build gene space and parameter mapping
        self._build_gene_space()

        # Reset state
        self.current_generation = 0
        self.current_evaluation = 0
        self.trials = []
        self.best_trial = None
        self.optimization_history = []
        self.is_finished_flag = False
        self._pending_evaluations = {}
        self._completed_evaluations = {}
        self.last_multi_fitness: Optional[List[float]] = None

        # Initialize PyGAD instance
        self._initialize_pygad()

        self.is_initialized = True

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"GeneticParameterGenerator initialized: "
                f"parameter_space={list(self.parameter_space.keys())}, "
                f"population_size={self.population_size}, "
                f"max_generations={self.max_generations}"
            )

    def _build_gene_space(self) -> None:
        """Build PyGAD gene space from parameter space configuration.

        This method converts the parameter space configuration into PyGAD's
        gene space format, handling integer, float, and categorical parameters.
        It includes comprehensive validation and support for step sizes and
        discrete value sets.
        """
        self.gene_space = []
        self.gene_types = []
        self.param_names = []

        for param_name, param_config in self.parameter_space.items():
            param_type = param_config.get("type", "float")
            self.param_names.append(param_name)

            if param_type == "int":
                low = param_config.get("low")
                high = param_config.get("high")
                step = param_config.get("step", 1)

                # Validate required fields
                if low is None or high is None:
                    raise InvalidParameterSpaceError(
                        f"Integer parameter '{param_name}' must have 'low' and 'high' bounds"
                    )

                # Convert to integers
                try:
                    low = int(low)
                    high = int(high)
                    step = int(step)
                except (ValueError, TypeError):
                    raise InvalidParameterSpaceError(
                        f"Integer parameter '{param_name}' bounds and step must be integers"
                    )

                # Validate range
                if low >= high:
                    raise InvalidParameterSpaceError(
                        f"Parameter '{param_name}': low ({low}) must be less than high ({high})"
                    )

                # Validate step size
                if step <= 0:
                    raise InvalidParameterSpaceError(
                        f"Parameter '{param_name}': step ({step}) must be positive"
                    )

                # Ensure at least one valid value exists
                if (high - low) < step:
                    logger.warning(
                        f"Parameter '{param_name}': range ({high - low}) smaller than step ({step}). "
                        f"Adjusting step to 1."
                    )
                    step = 1

                gene_config = {"low": int(low), "high": int(high), "step": int(step)}
                self.gene_space.append(gene_config)
                self.gene_types.append(int)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Integer parameter '{param_name}': {gene_config}")

            elif param_type == "float":
                low = param_config.get("low")
                high = param_config.get("high")
                step = param_config.get("step")  # Optional for floats

                # Validate required fields
                if low is None or high is None:
                    raise InvalidParameterSpaceError(
                        f"Float parameter '{param_name}' must have 'low' and 'high' bounds"
                    )

                # Convert to floats
                try:
                    low = float(low)
                    high = float(high)
                    if step is not None:
                        step = float(step)
                except (ValueError, TypeError):
                    raise InvalidParameterSpaceError(
                        f"Float parameter '{param_name}' bounds must be numeric"
                    )

                # Validate range
                if low >= high:
                    raise InvalidParameterSpaceError(
                        f"Parameter '{param_name}': low ({low}) must be less than high ({high})"
                    )

                # Check for infinite or NaN values
                if not (np.isfinite(low) and np.isfinite(high)):
                    raise InvalidParameterSpaceError(
                        f"Parameter '{param_name}': bounds must be finite numbers"
                    )

                # Build gene configuration
                float_gene: Dict[str, Union[float, int]] = {"low": float(low), "high": float(high)}
                if step is not None:
                    if step <= 0:
                        raise InvalidParameterSpaceError(
                            f"Parameter '{param_name}': step ({step}) must be positive"
                        )
                    float_gene["step"] = float(step)

                self.gene_space.append(float_gene)
                self.gene_types.append(float)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Float parameter '{param_name}': {float_gene}")

            elif param_type == "categorical":
                choices = param_config.get("choices")

                # Validate choices
                if choices is None:
                    raise InvalidParameterSpaceError(
                        f"Categorical parameter '{param_name}' must have 'choices'"
                    )

                if not isinstance(choices, (list, tuple)):
                    raise InvalidParameterSpaceError(
                        f"Parameter '{param_name}': choices must be a list or tuple"
                    )

                if len(choices) < 2:
                    raise InvalidParameterSpaceError(
                        f"Categorical parameter '{param_name}' must have at least 2 choices"
                    )

                # Check for duplicate choices
                if len(set(choices)) != len(choices):
                    logger.warning(
                        f"Parameter '{param_name}': duplicate choices detected. "
                        f"This may affect optimization performance."
                    )

                # For categorical parameters, use indices into choices list
                cat_gene = {"low": 0, "high": len(choices) - 1, "step": 1}
                self.gene_space.append(cat_gene)
                self.gene_types.append(int)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Categorical parameter '{param_name}': {len(choices)} choices -> {cat_gene}"
                    )

            elif param_type == "multi-categorical":
                choices = param_config.get("values", [])
                if not choices:
                    raise InvalidParameterSpaceError(
                        f"Multi-categorical parameter '{param_name}' must have 'values' defined."
                    )
                # For multi-categorical, we use a single integer gene that acts as a bitmask
                self.gene_space.append({"low": 0, "high": (1 << len(choices)) - 1})
                self.gene_type.append("int")

        # Validate the complete gene space
        self._validate_gene_space()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Built gene space with {len(self.gene_space)} genes: {self.gene_space}")

    def _validate_gene_space(self) -> None:
        """Validate the constructed gene space for PyGAD compatibility.

        This method performs additional validation on the complete gene space
        to ensure it will work correctly with PyGAD and catch potential issues
        early.
        """
        if not self.gene_space:
            raise InvalidParameterSpaceError("Gene space cannot be empty")

        for i, gene_config in enumerate(self.gene_space):
            param_name = self.param_names[i] if i < len(self.param_names) else f"gene_{i}"

            # Validate gene configuration structure
            if not isinstance(gene_config, dict):
                raise InvalidParameterSpaceError(
                    f"Gene {i} ({param_name}): configuration must be a dictionary"
                )

            # Check required fields
            if "low" not in gene_config or "high" not in gene_config:
                raise InvalidParameterSpaceError(
                    f"Gene {i} ({param_name}): must have 'low' and 'high' bounds"
                )

            low = gene_config["low"]
            high = gene_config["high"]
            step = gene_config.get("step")

            # Validate bounds
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise InvalidParameterSpaceError(f"Gene {i} ({param_name}): bounds must be numeric")

            if low >= high:
                raise InvalidParameterSpaceError(
                    f"Gene {i} ({param_name}): low ({low}) must be less than high ({high})"
                )

            # Validate step
            if step is not None:
                if not isinstance(step, (int, float)) or step <= 0:
                    raise InvalidParameterSpaceError(
                        f"Gene {i} ({param_name}): step must be a positive number"
                    )

                # For integer genes with step, ensure at least one valid value
                if isinstance(step, int) and (high - low) < step:
                    raise InvalidParameterSpaceError(
                        f"Gene {i} ({param_name}): range ({high - low}) smaller than step ({step})"
                    )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Gene space validation passed for {len(self.gene_space)} genes")

    def _initialize_pygad(self) -> None:
        """Initialize the PyGAD GA instance with external fitness function.

        This method sets up PyGAD to use an external fitness function that
        coordinates with the optimization orchestrator for parameter evaluation.
        """
        # Create PyGAD configuration
        num_genes = len(self.gene_space)
        ga_kwargs = {
            "num_generations": self.max_generations,
            "num_parents_mating": self.num_parents_mating,
            "fitness_func": self._external_fitness_function,
            "sol_per_pop": self.population_size,
            "num_genes": num_genes,
            "gene_space": self.gene_space,
            "parent_selection_type": self.parent_selection_type,
            "crossover_type": self.crossover_type,
            "mutation_type": self.mutation_type,
            "keep_elitism": self.keep_elitism,
            "on_generation": self._on_generation_callback,
            "random_seed": self.random_state,
        }
        # Fix mutation warnings for small gene spaces
        if num_genes < 10:
            ga_kwargs["mutation_percent_genes"] = None
            ga_kwargs["mutation_num_genes"] = 1
        else:
            ga_kwargs["mutation_percent_genes"] = self.mutation_percent_genes
        # Handle multi-objective optimization if supported
        if self.is_multi_objective:
            try:
                if (
                    pygad is not None
                    and hasattr(pygad, "GA")
                    and hasattr(pygad.GA, "__init__")
                    and "algorithm_type" in pygad.GA.__init__.__code__.co_varnames
                ):
                    ga_kwargs["algorithm_type"] = "nsga2"
                    logger.debug("Using NSGA-II algorithm for multi-objective optimization")
                else:
                    raise AttributeError("algorithm_type not supported")
            except (ImportError, AttributeError):
                logger.warning(
                    "NSGA-II not available in this PyGAD version. Using standard GA with weighted objectives."
                )

        # Create GA instance
        self.ga_instance = pygad.GA(**ga_kwargs)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("PyGAD instance initialized")

    def _external_fitness_function(self, ga_instance, solution: np.ndarray, solution_idx: int):
        """External fitness function for PyGAD.

        This function is called by PyGAD for each individual in the population.
        Instead of evaluating fitness directly, it coordinates with the external
        optimization orchestrator through the suggest_parameters/report_result cycle.

        Args:
            ga_instance: PyGAD GA instance
            solution: Chromosome/solution array
            solution_idx: Index of solution in population

        Returns:
            Fitness value for the solution
        """
        # Convert chromosome to parameter dictionary
        parameters = self._decode_chromosome(solution)

        # Create a hash for this chromosome to track evaluations
        chromosome_hash = self._hash_chromosome(solution)

        # Check if we already have fitness for this chromosome
        if chromosome_hash in self._completed_evaluations:
            fitness = self._completed_evaluations[chromosome_hash]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using cached fitness for solution {solution_idx}: {fitness}")
            return fitness

        # Store pending evaluation
        self._pending_evaluations[chromosome_hash] = parameters

        # Return a placeholder fitness - this will be updated when report_result is called
        # For now, return a neutral fitness value
        return 0.0

    def _on_generation_callback(self, ga_instance):
        """Callback function called after each generation.

        This function updates the generation counter and handles any
        generation-specific logic like early stopping or progress reporting.

        Args:
            ga_instance: PyGAD GA instance
        """
        self.current_generation = ga_instance.generations_completed

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generation {self.current_generation} completed")

        # Check for early stopping conditions
        if self.current_evaluation >= self.max_evaluations:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Stopping due to max evaluations ({self.max_evaluations})")
            return "stop"

    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest the next parameter set to evaluate.

        This method generates random parameters within the defined parameter space
        for genetic algorithm optimization. It uses a simplified approach that
        generates parameters on-demand rather than managing PyGAD's population directly.

        Returns:
            Dictionary of parameter names and values to evaluate

        Raises:
            ParameterGeneratorNotInitializedError: If not initialized
            ParameterGeneratorFinishedError: If optimization is complete
        """
        if not self.is_initialized:
            raise ParameterGeneratorNotInitializedError(
                "Generator must be initialized before suggesting parameters"
            )

        if self.is_finished():
            raise ParameterGeneratorFinishedError(
                "Generator is finished, cannot suggest more parameters"
            )

        if self.ga_instance is None:
            raise ParameterGeneratorNotInitializedError("GA instance not initialized")

        pop = getattr(self.ga_instance, "population", None)
        if pop is None or not np.any(pop):
            if self.ga_instance is not None and hasattr(
                self.ga_instance, "create_initial_population"
            ):
                try:
                    self.ga_instance.create_initial_population()
                    pop = (
                        self.ga_instance.population
                        if self.ga_instance.population is not None
                        else np.array([])
                    )
                except AttributeError:
                    pop = np.array([])
            else:
                pop = np.array([])

        population = pop
        chromosome_idx = self.current_evaluation % self.population_size
        chromosome = population[chromosome_idx]

        # Decode the chromosome to parameters
        parameters = self._decode_chromosome(chromosome)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Suggesting parameters: {parameters}")

        return parameters

    def report_result(self, parameters: Dict[str, Any], result: EvaluationResult) -> None:
        """Report the result of evaluating a parameter set.

        This method receives evaluation results and updates the genetic algorithm's
        internal tracking. It stores the results and updates the best solution found.

        Args:
            parameters: The parameter set that was evaluated
            result: The evaluation result containing objective values and metrics

        Raises:
            ParameterGeneratorNotInitializedError: If not initialized
        """
        if not self.is_initialized:
            raise ParameterGeneratorNotInitializedError(
                "Generator must be initialized before reporting results"
            )

        self.current_evaluation += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Reporting result for evaluation {self.current_evaluation}: "
                f"objective_value={result.objective_value}"
            )

        # Convert objective value to fitness
        fitness = self._aggregate_fitness(
            self._convert_objective_to_fitness(result.objective_value)
        )

        # Create trial record
        chromosome = self._encode_parameters(parameters)
        trial = GeneticTrial(
            number=self.current_evaluation - 1,
            parameters=parameters.copy(),
            chromosome=chromosome,
            fitness=float(fitness),
            generation=self.current_generation,
        )

        self.trials.append(trial)

        # Update best trial
        if self._is_better_trial(trial):
            self.best_trial = trial
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"New best trial: fitness={trial.fitness}")

        # Add to optimization history
        history_entry = {
            "evaluation": self.current_evaluation,
            "generation": self.current_generation,
            "parameters": parameters.copy(),
            "objective_value": result.objective_value,
            "fitness": fitness,
            "metrics": result.metrics.copy(),
        }
        self.optimization_history.append(history_entry)

        # Update generation counter periodically
        if self.current_evaluation % self.population_size == 0:
            self.current_generation += 1

    def _convert_objective_to_fitness(
        self, objective_value: Union[float, List[float]]
    ) -> Union[float, List[float]]:
        """Convert objective value to fitness value for PyGAD.

        PyGAD maximizes fitness, so we need to handle minimize objectives
        by negating them.

        Args:
            objective_value: Objective value(s) from evaluation

        Returns:
            Fitness value(s) for PyGAD
        """
        if self.is_multi_objective:
            if isinstance(objective_value, list):
                fitness_values = []
                for i, (value, direction) in enumerate(
                    zip(objective_value, self.optimization_directions)
                ):
                    if direction == "minimize":
                        fitness_values.append(-value if np.isfinite(value) else -1e9)
                    else:
                        fitness_values.append(value if np.isfinite(value) else -1e9)
                return fitness_values
            else:
                # Single value for multi-objective - replicate with direction handling
                fitness_values = []
                for direction in self.optimization_directions:
                    if direction == "minimize":
                        fitness_values.append(
                            -objective_value if np.isfinite(objective_value) else -1e9
                        )
                    else:
                        fitness_values.append(
                            objective_value if np.isfinite(objective_value) else -1e9
                        )
                return fitness_values
        else:
            # Single objective
            if isinstance(objective_value, list):
                value = float(objective_value[0])
            else:
                value = float(objective_value)

            direction = (
                self.optimization_directions[0] if self.optimization_directions else "maximize"
            )
            if direction == "minimize":
                return -value if np.isfinite(value) else -1e9
            else:
                return value if np.isfinite(value) else -1e9

    def _aggregate_fitness(self, fitness: Union[float, List[float]]) -> float:
        if isinstance(fitness, list):
            self.last_multi_fitness = fitness
            agg = 0.0
            for val in fitness:
                agg += float(val)
            return agg
        return float(fitness)

    def _advance_generation(self) -> None:
        """Advance PyGAD to the next generation.

        This method triggers PyGAD to evolve the population to the next
        generation using the fitness values from completed evaluations.
        """
        if self.ga_instance is None:
            return

        # Update fitness values in PyGAD
        if self.ga_instance is not None and self.ga_instance.population is not None:
            population = self.ga_instance.population
        else:
            population = np.array([])
        fitness_values = []

        for chromosome in population:
            chromosome_hash = self._hash_chromosome(chromosome)
            if chromosome_hash in self._completed_evaluations:
                fitness_values.append(self._completed_evaluations[chromosome_hash])
            else:
                # This shouldn't happen if we're managing evaluations correctly
                logger.warning(f"Missing fitness for chromosome: {chromosome_hash}")
                if self.is_multi_objective:
                    fitness_values.append([-1e9] * len(self.metrics_to_optimize))
                else:
                    fitness_values.append(-1e9)

        # Set fitness values and evolve population
        self.ga_instance.last_generation_fitness = np.array(fitness_values)

        # Evolve to next generation if not at the end
        if self.current_generation < self.max_generations - 1:
            # This will trigger the evolution process
            self.ga_instance.cal_pop_fitness()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Advanced to generation {self.current_generation + 1}")

    def is_finished(self) -> bool:
        """Check whether optimization should continue.

        Returns:
            True if optimization is complete, False if it should continue
        """
        if self.is_finished_flag:
            return True

        # Check if we've reached max evaluations
        if self.current_evaluation >= self.max_evaluations:
            self.is_finished_flag = True
            return True

        # Check if we've reached max generations and no pending evaluations
        if self.current_generation >= self.max_generations and not self._pending_evaluations:
            self.is_finished_flag = True
            return True

        return False

    def get_best_result(self) -> OptimizationResult:
        """Get the best optimization result found so far.

        Returns:
            OptimizationResult containing the best parameters and objective value

        Raises:
            ParameterGeneratorNotInitializedError: If not initialized
        """
        if not self.is_initialized:
            raise ParameterGeneratorNotInitializedError(
                "Generator must be initialized before getting results"
            )

        if self.best_trial is None:
            # No trials completed yet - return empty result
            if self.is_multi_objective:
                best_value = -1e9
            else:
                best_value = -1e9

            return OptimizationResult(
                best_parameters={},
                best_value=best_value,
                n_evaluations=self.current_evaluation,
                optimization_history=self.optimization_history.copy(),
                best_trial=None,
            )

        # Convert fitness back to objective value for reporting
        direction = self.optimization_directions[0] if self.optimization_directions else "maximize"
        fv = float(self.best_trial.fitness)
        best_value = -fv if direction == "minimize" else fv

        return OptimizationResult(
            best_parameters=self.best_trial.parameters.copy(),
            best_value=best_value,
            n_evaluations=self.current_evaluation,
            optimization_history=self.optimization_history.copy(),
            best_trial=self.best_trial,
        )

    def get_best_parameters(self) -> Dict[str, Any]:
        """Return best parameter dictionary discovered so far."""
        return self.get_best_result().best_parameters

    def _is_better_trial(self, trial: GeneticTrial) -> bool:
        """Check if the given trial is better than the current best.

        Args:
            trial: Trial to compare

        Returns:
            True if this trial is better than the current best
        """
        if self.best_trial is None:
            return True

        # Compare aggregated float fitness (higher is better)
        return float(trial.fitness) > float(self.best_trial.fitness)

    def _hash_chromosome(self, chromosome: np.ndarray) -> str:
        """Create a hash string for a chromosome.

        Args:
            chromosome: Chromosome array

        Returns:
            Hash string for the chromosome
        """
        # Convert to string representation for hashing
        return str(chromosome.tolist())

    # Override optional methods from ParameterGenerator base class

    def supports_multi_objective(self) -> bool:
        """Check if this generator supports multi-objective optimization.

        Returns:
            True if PyGAD supports NSGA-II, False otherwise
        """
        try:
            return True
        except (ImportError, AttributeError):
            return False

    def supports_pruning(self) -> bool:
        """Check if this generator supports early pruning of poor trials.

        Returns:
            False since genetic algorithms don't typically support pruning
        """
        return False

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the complete optimization history.

        Returns:
            List of dictionaries containing the history of all parameter
            evaluations performed
        """
        return self.optimization_history.copy()

    def get_parameter_importance(self) -> Optional[Dict[str, float]]:
        """Get parameter importance scores if available.

        Returns:
            None since genetic algorithms don't typically provide parameter importance
        """
        return None

    def set_random_state(self, random_state: Optional[int]) -> None:
        """Set the random state for reproducible results.

        Args:
            random_state: Random seed for reproducible optimization runs
        """
        self.random_state = random_state

        # If already initialized, update PyGAD instance
        if self.ga_instance is not None:
            self.ga_instance.random_seed = random_state

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Random state set to: {random_state}")

    def get_current_evaluation_count(self) -> int:
        """Get the number of parameter evaluations performed so far.

        Returns:
            Number of parameter sets that have been evaluated
        """
        return self.current_evaluation

    def get_trials(self) -> List[GeneticTrial]:
        """Get all trials performed so far.

        Returns:
            List of all GeneticTrial objects
        """
        return self.trials.copy()

    def get_current_generation(self) -> int:
        """Get the current generation number.

        Returns:
            Current generation number
        """
        return self.current_generation

    def get_population_size(self) -> int:
        """Get the population size.

        Returns:
            Size of the genetic algorithm population
        """
        return self.population_size

    def _decode_chromosome(self, chromosome: np.ndarray) -> Dict[str, Any]:
        """Decode a chromosome into a parameter dictionary.

        This method converts a PyGAD chromosome (array of gene values) into
        a dictionary of parameter names and values, handling type conversion
        and bounds checking for different parameter types. It ensures
        deterministic encoding/decoding behavior.

        Args:
            chromosome: PyGAD chromosome array

        Returns:
            Dictionary mapping parameter names to their decoded values

        Raises:
            ValueError: If chromosome length doesn't match parameter space
        """
        if len(chromosome) != len(self.parameter_space):
            raise ValueError(
                f"Chromosome length ({len(chromosome)}) doesn't match "
                f"parameter space size ({len(self.parameter_space)})"
            )

        parameters: Dict[str, Any] = {}

        for i, (param_name, param_config) in enumerate(self.parameter_space.items()):
            param_type = param_config.get("type", "float")
            gene_value = chromosome[i]

            # Handle NaN or infinite values
            if not np.isfinite(gene_value):
                logger.warning(f"Non-finite gene value for {param_name}: {gene_value}")
                # Use midpoint of range as fallback
                if param_type == "categorical":
                    gene_value = len(param_config["choices"]) // 2
                else:
                    gene_value = (param_config["low"] + param_config["high"]) / 2

            if param_type == "int":
                # Ensure integer value within bounds
                low = int(param_config["low"])
                high = int(param_config["high"])
                step = int(param_config.get("step", 1))

                # Convert to integer and apply step
                int_value = int(round(gene_value))

                # Apply step constraint
                if step > 1:
                    # Adjust value to nearest step boundary
                    steps_from_low = round((int_value - low) / step)
                    int_value = low + steps_from_low * step

                # Clamp to bounds
                int_value = max(low, min(high, int_value))
                parameters[param_name] = int_value

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Decoded {param_name}: {gene_value} -> {int_value} (int, bounds=[{low}, {high}], step={step})"
                    )

            elif param_type == "float":
                # Ensure float value within bounds
                low_f = float(param_config["low"])
                high_f = float(param_config["high"])
                step_v = param_config.get("step")

                float_value = float(gene_value)

                if step_v is not None:
                    step_f = float(step_v)
                    steps_from_low = round((float_value - low_f) / step_f)
                    float_value = low_f + steps_from_low * step_f

                float_value = max(low_f, min(high_f, float_value))
                parameters[param_name] = float_value

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Decoded {param_name}: {gene_value} -> {float_value} (float, bounds=[{low_f}, {high_f}])"
                    )

            elif param_type == "categorical":
                # Convert gene index to categorical choice
                choices = param_config["choices"]
                choice_idx = int(round(gene_value))

                # Clamp to valid indices (bounds clamping for categorical parameters)
                choice_idx = max(0, min(len(choices) - 1, choice_idx))
                value = choices[choice_idx]
                parameters[param_name] = value

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Decoded {param_name}: {gene_value} -> {choice_idx} -> '{value}' (categorical)"
                    )

            else:
                # This shouldn't happen if gene space was built correctly
                logger.error(f"Unknown parameter type '{param_type}' for {param_name}")
                parameters[param_name] = gene_value

        return parameters

    def _encode_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Encode a parameter dictionary into a chromosome.

        This method converts a dictionary of parameter names and values into
        a PyGAD chromosome (array of gene values), handling type conversion
        for different parameter types. It ensures deterministic encoding/decoding
        behavior and proper bounds checking.

        Args:
            parameters: Dictionary mapping parameter names to values

        Returns:
            PyGAD chromosome array

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if len(parameters) != len(self.parameter_space):
            logger.warning(
                f"Parameter count mismatch: expected {len(self.parameter_space)}, "
                f"got {len(parameters)}"
            )

        chromosome = np.zeros(len(self.parameter_space), dtype=float)

        for i, (param_name, param_config) in enumerate(self.parameter_space.items()):
            if param_name not in parameters:
                # Use default value (midpoint of range)
                param_type = param_config.get("type", "float")
                if param_type == "categorical":
                    default_value = len(param_config["choices"]) // 2
                else:
                    default_value = (param_config["low"] + param_config["high"]) / 2

                logger.warning(
                    f"Parameter '{param_name}' not found in parameters dict, "
                    f"using default value: {default_value}"
                )
                chromosome[i] = default_value
                continue

            param_type = param_config.get("type", "float")
            param_value = parameters[param_name]

            if param_type == "int":
                try:
                    int_value = int(param_value)
                    # Ensure value is within bounds
                    low = int(param_config["low"])
                    high = int(param_config["high"])
                    int_value = max(low, min(high, int_value))
                    chromosome[i] = float(int_value)

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Encoded {param_name}: {param_value} -> {int_value} (int)")

                except (ValueError, TypeError):
                    logger.error(f"Invalid integer value for {param_name}: {param_value}")
                    # Use midpoint as fallback
                    chromosome[i] = (param_config["low"] + param_config["high"]) / 2

            elif param_type == "float":
                try:
                    float_value = float(param_value)
                    # Ensure value is within bounds
                    low_f = float(param_config["low"])
                    high_f = float(param_config["high"])
                    float_value = max(low_f, min(high_f, float_value))
                    chromosome[i] = float_value

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Encoded {param_name}: {param_value} -> {float_value} (float)"
                        )

                except (ValueError, TypeError):
                    logger.error(f"Invalid float value for {param_name}: {param_value}")
                    # Use midpoint as fallback
                    chromosome[i] = (param_config["low"] + param_config["high"]) / 2

            elif param_type == "categorical":
                # Convert categorical choice to index
                choices = param_config["choices"]
                try:
                    choice_idx = choices.index(param_value)
                    chromosome[i] = float(choice_idx)

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Encoded {param_name}: '{param_value}' -> {choice_idx} (categorical)"
                        )

                except ValueError:
                    # If value not found, use first choice
                    logger.warning(
                        f"Value '{param_value}' not found in choices for '{param_name}'. "
                        f"Available choices: {choices}. Using first choice."
                    )
                    chromosome[i] = 0.0

            else:
                # Fallback for unknown types - treat as float
                try:
                    chromosome[i] = float(param_value)
                    logger.warning(
                        f"Unknown parameter type '{param_type}' for '{param_name}', treating as float"
                    )
                except (ValueError, TypeError):
                    logger.error(f"Cannot convert value for {param_name}: {param_value}")
                    chromosome[i] = 0.0

        return chromosome
