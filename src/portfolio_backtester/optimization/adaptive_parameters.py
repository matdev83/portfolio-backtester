from __future__ import annotations

"""Adaptive parameter control utilities for the genetic optimizer.

This module provides simple reference implementations of:

1. DiversityCalculator – computes population diversity metrics that can be
   used by adaptive controllers.
2. AdaptiveMutationController – adjusts the mutation rate during the GA run
   based on diversity, fitness variance and generation progress.
3. AdaptiveCrossoverController – adjusts the crossover probability based on
   diversity and convergence rate.

The algorithms implemented here intentionally stay relatively lightweight and
heuristic-based to avoid adding large external dependencies or significantly
increasing the optimisation overhead.  The defaults are conservative so that
behaviour of the GA with *adaptive_* configuration disabled remains identical
(see GeneticOptimizer integration).
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np

# Small epsilon to avoid division by zero
_EPS = 1e-12


class DiversityCalculator:
    """Utility that computes diversity metrics for a GA population.

    The current implementation focuses on *genotypic* diversity computed as the
    average pair-wise Euclidean distance between chromosomes after normalising
    every gene to the range \[0, 1].  This is fast to compute with NumPy and
    provides a reasonable proxy for the overall search-space coverage.
    """

    def __init__(self, gene_space: Sequence[dict]):
        # Pre-compute normalisation constants so that we can vectorise the
        # diversity computation during the GA run.
        lows = []
        highs = []
        for space in gene_space:
            low = space.get("low", 0)
            high = space.get("high", 1)
            if high <= low:
                # Prevent zero division – treat as a very small range.
                high = low + 1.0
            lows.append(low)
            highs.append(high)
        self._lows = np.asarray(lows, dtype=float)
        self._ranges = np.asarray(highs, dtype=float) - self._lows
        # Replace any zero ranges with 1 to avoid NaNs.
        self._ranges[self._ranges == 0] = 1.0
        # Maximum possible distance after normalisation (diagonal of unit
        # hyper-cube).
        self._max_dist = np.sqrt(len(gene_space))

    def phenotypic_diversity(self, population: np.ndarray) -> float:
        """Average pair-wise distance in \[0, 1] normalised gene space."""
        if population.size == 0:
            return 0.0
        # Normalise population into 0-1 cube.
        norm_pop = (population - self._lows) / self._ranges
        # Compute pair-wise differences with broadcasting.
        diff = norm_pop[:, None, :] - norm_pop[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Average over the upper triangle to avoid double counting.
        # We divide by 2 because every distance appears twice in diff.
        mean_dist = np.sum(dists) / (2 * max(population.shape[0] * (population.shape[0] - 1) / 2, 1))
        # Normalise to 0-1 by dividing by the maximum possible distance.
        return float(mean_dist / (self._max_dist + _EPS))

    def fitness_diversity(self, fitness_scores: np.ndarray) -> float:
        """Coefficient of variation of the fitness scores.

        Returns 0 when the population is perfectly converged (zero variance).
        """
        if fitness_scores.size <= 1:
            return 0.0
        std = float(np.std(fitness_scores))
        rng = float(np.max(fitness_scores) - np.min(fitness_scores) + _EPS)
        return std / rng


@dataclass
class AdaptiveMutationController:
    base_rate: float = 0.1
    min_rate: float = 0.01
    max_rate: float = 0.5
    diversity_threshold: float = 0.3
    max_generations: int = 100

    def rate(self, population_diversity: float, fitness_variance: float, generation: int) -> float:
        """Compute an adaptive mutation probability.

        The heuristic increases mutation when diversity is low or when fitness
        variance stalls, and gradually decreases it as generations progress.
        """
        # Diversity factor – boost when below threshold, dampen otherwise.
        diversity_factor = 1.5 if population_diversity < self.diversity_threshold else 0.8

        # Fitness variance factor – higher variance indicates exploration.
        fitness_factor = 1.0
        if fitness_variance < 1e-8:
            fitness_factor = 1.3  # Encourage exploration when fitness flat.
        elif fitness_variance > 1.0:
            fitness_factor = 0.9  # Reduce mutation when fitness widely spread.

        # Generation decay – linearly decay mutation to half over run.
        generation_factor = max(0.5, 1.0 - (generation / max(self.max_generations, 1)) * 0.5)

        rate = self.base_rate * diversity_factor * fitness_factor * generation_factor
        return float(max(self.min_rate, min(self.max_rate, rate)))


@dataclass
class AdaptiveCrossoverController:
    base_rate: float = 0.8
    min_rate: float = 0.6
    max_rate: float = 0.95

    def rate(self, population_diversity: float, convergence_measure: float) -> float:
        """Compute an adaptive crossover probability.

        The heuristic raises crossover probability when diversity is low and
        lowers it when the population appears to have converged.
        """
        rate = self.base_rate
        if population_diversity < 0.3:
            rate *= 1.2  # Promote recombination to introduce new genes.
        if convergence_measure > 0.8:
            rate *= 0.8  # Reduce crossover when already converged.
        return float(max(self.min_rate, min(self.max_rate, rate)))
