"""
Population diversity management for genetic algorithms.

This module provides utilities to measure, monitor, and enforce diversity
in genetic algorithm populations, reducing wasted evaluations on similar individuals.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import logging
from .performance.deduplication_factory import DeduplicationFactory

logger = logging.getLogger(__name__)


class PopulationDiversityManager:
    """
    Manages population diversity in genetic algorithms.

    This class helps maintain genetic diversity by detecting nearly identical individuals,
    implementing diversity preservation strategies, and providing metrics on population
    health.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        min_diversity_ratio: float = 0.5,
        enforce_diversity: bool = True,
        parameter_space: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the population diversity manager.

        Args:
            similarity_threshold: Threshold above which two individuals are considered too similar
                (range: 0.0-1.0, where 1.0 means identical)
            min_diversity_ratio: Minimum ratio of unique individuals in the population
                (range: 0.0-1.0)
            enforce_diversity: Whether to actively enforce diversity or just monitor it
            parameter_space: Parameter space configuration (optional at init, can set later)
        """
        self.similarity_threshold = similarity_threshold
        self.min_diversity_ratio = min_diversity_ratio
        self.enforce_diversity = enforce_diversity
        self.parameter_space = parameter_space

        # Internal state
        self._parameter_types: Dict[str, str] = {}
        self._parameter_ranges: Dict[str, Tuple[float, float]] = {}
        self._parameter_choices: Dict[str, List[Any]] = {}
        self._seen_hashes: Set[Tuple[str, ...]] = set()

        # Metrics
        self.duplicate_count = 0
        self.near_duplicate_count = 0
        self.diversity_ratio = 1.0

        # Deduplication integration
        self._deduplicator = DeduplicationFactory.create_deduplicator(
            optimizer_type="genetic", config={"enable_deduplication": True}
        )

    def set_parameter_space(self, parameter_space: Dict[str, Any]) -> None:
        """
        Set or update the parameter space configuration.

        Args:
            parameter_space: Parameter space configuration dictionary
        """
        self.parameter_space = parameter_space

        # Extract parameter metadata for similarity calculations
        if parameter_space:
            for param_name, config in parameter_space.items():
                param_type = config["type"]
                self._parameter_types[param_name] = param_type

                if param_type in ("int", "float"):
                    self._parameter_ranges[param_name] = (config["low"], config["high"])
                elif param_type == "categorical":
                    self._parameter_choices[param_name] = config["choices"]

    def compute_similarity(self, individual1: Dict[str, Any], individual2: Dict[str, Any]) -> float:
        """
        Compute similarity between two individuals.

        Args:
            individual1: First parameter set
            individual2: Second parameter set

        Returns:
            Similarity score (0.0-1.0, where 1.0 means identical)
        """
        if not self.parameter_space:
            raise ValueError("Parameter space not set. Call set_parameter_space() first.")

        param_similarities = []

        # Compare each parameter between individuals
        for param_name, type_info in self._parameter_types.items():
            val1 = individual1.get(param_name)
            val2 = individual2.get(param_name)

            # Skip if either value is missing
            if val1 is None or val2 is None:
                continue

            # Calculate similarity based on parameter type
            if type_info == "categorical":
                # For categorical: 1.0 if same, 0.0 if different
                param_sim = 1.0 if val1 == val2 else 0.0
            else:  # numerical (int, float)
                # For numerical: distance-based similarity
                low, high = self._parameter_ranges[param_name]
                range_size = max(high - low, 1e-10)  # Avoid division by zero

                # Normalize distance to 0.0-1.0 range and convert to similarity
                normalized_distance = abs(val1 - val2) / range_size
                param_sim = 1.0 - min(normalized_distance, 1.0)

            param_similarities.append(param_sim)

        # Overall similarity is the average of parameter similarities
        return sum(param_similarities) / max(len(param_similarities), 1)

    def is_too_similar(self, individual: Dict[str, Any], population: List[Dict[str, Any]]) -> bool:
        """
        Check if an individual is too similar to any in the existing population.

        Args:
            individual: The individual to check
            population: The existing population to compare against

        Returns:
            True if the individual is too similar to any existing one
        """
        for existing in population:
            similarity = self.compute_similarity(individual, existing)
            if similarity >= self.similarity_threshold:
                self.near_duplicate_count += 1
                return True
        return False

    def analyze_population_diversity(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze diversity metrics for a population.

        Args:
            population: The population to analyze

        Returns:
            Dictionary of diversity metrics
        """
        if not population:
            return {"diversity_ratio": 0.0, "duplicate_count": 0, "near_duplicate_count": 0}

        # Reset metrics
        self._seen_hashes = set()
        self.duplicate_count = 0
        self.near_duplicate_count = 0

        # Count exact duplicates using parameter hashes
        for individual in population:
            param_hash = self._get_parameter_hash(individual)
            if param_hash in self._seen_hashes:
                self.duplicate_count += 1
            else:
                self._seen_hashes.add(param_hash)

        # Calculate diversity ratio
        unique_count = len(self._seen_hashes)
        self.diversity_ratio = unique_count / len(population)

        # Compute pairwise similarities for more detailed analysis
        similarity_matrix = np.zeros((len(population), len(population)))
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                sim = self.compute_similarity(population[i], population[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # Average similarity across the population
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(population), k=1)])

        return {
            "diversity_ratio": self.diversity_ratio,
            "duplicate_count": self.duplicate_count,
            "near_duplicate_count": self.near_duplicate_count,
            "unique_count": unique_count,
            "population_size": len(population),
            "average_similarity": float(avg_similarity),
            "is_diverse_enough": self.diversity_ratio >= self.min_diversity_ratio,
        }

    def _get_parameter_hash(self, params: Dict[str, Any]) -> Tuple[str, ...]:
        """Create a deterministic, hashable key for a parameter dict."""
        # Use repr for stable representation across ints/floats/strings and sort by key
        return tuple(f"{k}={repr(params[k])}" for k in sorted(params.keys()))

    def diversify_individual(
        self, individual: Dict[str, Any], rng: np.random.Generator
    ) -> Dict[str, Any]:
        """
        Diversify an individual to make it more unique.

        Args:
            individual: The individual to diversify
            rng: NumPy random number generator instance

        Returns:
            A diversified version of the individual
        """
        if not self.parameter_space:
            raise ValueError("Parameter space not set. Call set_parameter_space() first.")

        diversified = individual.copy()

        # Select a random subset of parameters to mutate (25-75% of parameters)
        param_names = list(self.parameter_space.keys())
        num_params_to_mutate = max(
            1,
            rng.integers(
                low=max(1, len(param_names) // 4), high=max(2, (len(param_names) * 3) // 4)
            ),
        )

        params_to_mutate = rng.choice(param_names, size=num_params_to_mutate, replace=False)

        # Apply stronger mutation to selected parameters
        for param_name in params_to_mutate:
            param_config = self.parameter_space[param_name]
            param_type = param_config["type"]

            if param_type == "int":
                # For integers: either shift significantly or pick new random value
                if rng.random() < 0.7:  # 70% chance of significant shift
                    current = diversified[param_name]
                    low, high = param_config["low"], param_config["high"]
                    range_size = high - low
                    # Shift by 20-50% of range in random direction
                    shift = rng.integers(
                        low=max(1, int(range_size * 0.2)), high=max(2, int(range_size * 0.5))
                    )
                    if rng.random() < 0.5:  # 50% chance of negative shift
                        shift = -shift
                    # Apply shift with bounds checking
                    # Fix for potential integer overflow/infinite loop issues
                    new_val = float(current) + float(shift)
                    new_val = max(float(low), min(float(high), new_val))
                    # Convert back to int if needed
                    if isinstance(current, (int, np.integer)):
                        new_val = int(new_val)
                    diversified[param_name] = new_val
                else:  # 30% chance of completely random value
                    diversified[param_name] = rng.integers(
                        param_config["low"], param_config["high"], endpoint=True
                    )

            elif param_type == "float":
                # For floats: similar strategy as integers
                if rng.random() < 0.7:
                    current = diversified[param_name]
                    low, high = param_config["low"], param_config["high"]
                    range_size = high - low
                    # Shift by 20-50% of range in random direction
                    shift = rng.uniform(range_size * 0.2, range_size * 0.5)
                    if rng.random() < 0.5:
                        shift = -shift
                    # Apply shift with bounds checking
                    # Fix for potential integer overflow/infinite loop issues
                    new_val = float(current) + float(shift)
                    new_val = max(float(low), min(float(high), new_val))
                    # Convert back to int if needed
                    if isinstance(current, (int, np.integer)):
                        new_val = int(new_val)
                    diversified[param_name] = new_val
                else:
                    diversified[param_name] = rng.uniform(param_config["low"], param_config["high"])

            elif param_type == "categorical":
                # For categorical: pick a different value than current
                current = diversified[param_name]
                choices = param_config["choices"]
                if len(choices) > 1:
                    other_choices = [c for c in choices if c != current]
                    if other_choices:
                        diversified[param_name] = rng.choice(other_choices)

        return diversified

    def diversify_population(
        self, population: List[Dict[str, Any]], rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        """
        Ensure population diversity by diversifying individuals as needed.

        Args:
            population: The population to diversify
            rng: NumPy random number generator instance

        Returns:
            Diversified population
        """
        if not self.enforce_diversity:
            return population

        diversity_metrics = self.analyze_population_diversity(population)

        # If diversity is already good, return as-is
        if diversity_metrics["is_diverse_enough"] or not self.enforce_diversity:
            return population

        logger.debug(f"Diversifying population with diversity ratio: {self.diversity_ratio:.4f}")

        # Create a new population, starting with unique individuals
        unique_individuals: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        diversified_population: List[Dict[str, Any]] = []

        # First pass: collect unique individuals
        for individual in population:
            param_hash = self._get_parameter_hash(individual)
            if param_hash not in unique_individuals:
                unique_individuals[param_hash] = individual
                diversified_population.append(individual)

        # Second pass: fill remaining slots with diversified copies
        # Add a safety counter to prevent infinite loops
        attempts = 0
        max_attempts = len(population) * 10  # Reasonable limit
        
        while len(diversified_population) < len(population) and attempts < max_attempts:
            # Pick a random base individual to diversify
            base_individual = rng.choice(list(unique_individuals.values()))
            candidate = self.diversify_individual(base_individual, rng)

            # Check if the diversified individual is unique enough
            if not self.is_too_similar(candidate, diversified_population):
                diversified_population.append(candidate)
            
            # Increment attempts counter
            attempts += 1

        # If we hit max attempts, fill any remaining slots with random individuals
        while len(diversified_population) < len(population):
            logger.warning("Diversity manager hit max attempts, adding random individuals")
            diversified_population.append(self._create_random_individual())
            
        # Update metrics after diversification
        final_metrics = self.analyze_population_diversity(diversified_population)
        logger.debug(
            f"Population diversity increased: {self.diversity_ratio:.4f} -> "
            f"{final_metrics['diversity_ratio']:.4f}"
        )

        return diversified_population

    def _create_random_individual(self) -> Dict[str, Any]:
        """
        Create a completely random individual from parameter space.
        Used as a fallback when diversification attempts are exhausted.
        
        Returns:
            A randomly generated parameter set
        """
        if not self.parameter_space:
            raise ValueError("Parameter space not set. Call set_parameter_space() first.")
            
        individual = {}
        for param_name, config in self.parameter_space.items():
            param_type = config["type"]
            
            if param_type == "int":
                individual[param_name] = np.random.randint(config["low"], config["high"] + 1)
            elif param_type == "float":
                individual[param_name] = np.random.uniform(config["low"], config["high"])
            elif param_type == "categorical":
                individual[param_name] = np.random.choice(config["choices"])
        
        return individual
    
    def get_stats(self) -> Dict[str, Any]:
        """Get diversity statistics."""
        return {
            "diversity_ratio": self.diversity_ratio,
            "duplicate_count": self.duplicate_count,
            "near_duplicate_count": self.near_duplicate_count,
            "min_diversity_ratio": self.min_diversity_ratio,
            "similarity_threshold": self.similarity_threshold,
            "enforce_diversity": self.enforce_diversity,
        }
