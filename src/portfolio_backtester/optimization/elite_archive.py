from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from ..api_stability import api_stable

"""Elite preservation utilities for the Genetic Optimizer.

The archive stores the best chromosomes seen throughout the run and supports
reinserting them back into the population to prevent loss of high-quality
solutions due to genetic drift.

All functionality is encapsulated in this module so that the main
`genetic_optimizer.py` file remains focused on orchestration while keeping its
length manageable.
"""

__all__ = [
    "EliteSolution",
    "EliteArchive",
]


@dataclass
class EliteSolution:
    chromosome: np.ndarray
    fitness_score: float
    generation: int
    timestamp: datetime
    metadata: Dict[str, Any]


class EliteArchive:
    """Fixed-size archive that maintains the globally best chromosomes."""

    def __init__(self, max_size: int = 50, aging_factor: float = 0.95):
        self.max_size = max_size
        self.aging_factor = aging_factor
        self.elites: List[EliteSolution] = []

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @api_stable(version="1.0", strict_params=True, strict_return=False)
    def add(self, chromosome: np.ndarray, fitness: float, generation: int) -> None:
        """Attempt to add a new elite to the archive."""
        if not np.isfinite(fitness):
            return  # ignore invalid fitness
        elite = EliteSolution(
            chromosome=chromosome.copy(),
            fitness_score=float(fitness),
            generation=generation,
            timestamp=datetime.utcnow(),
            metadata={},
        )
        # Insert keeping the archive sorted (descending fitness)
        self.elites.append(elite)
        self.elites.sort(key=lambda e: e.fitness_score, reverse=True)
        # Drop worst if above capacity
        if len(self.elites) > self.max_size:
            self.elites.pop(-1)

    def age(self, current_generation: int) -> None:
        """Update aged fitness values (not used currently but future-proof)."""
        for e in self.elites:
            age = current_generation - e.generation
            e.metadata["age"] = age
            e.metadata["aged_fitness"] = e.fitness_score * (self.aging_factor**age)

    # ------------------------------------------------------------------
    # Injection strategies
    # ------------------------------------------------------------------
    def inject_direct(
        self, population: np.ndarray, fitness: np.ndarray, num_elites: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.elites:
            return population, fitness
        num_elites = min(num_elites, len(self.elites))
        # Replace worst individuals (ascending fitness)
        worst_idx = np.argsort(fitness)[:num_elites]
        for i, idx in enumerate(worst_idx):
            population[idx] = self.elites[i].chromosome.copy()
            fitness[idx] = self.elites[i].fitness_score
        return population, fitness

    def inject_tournament(
        self, population: np.ndarray, fitness: np.ndarray, tournament_size: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.elites:
            return population, fitness

        # Create pool of current population + elites
        pool = list(zip(population, fitness)) + [
            (e.chromosome, e.fitness_score) for e in self.elites
        ]

        num_replace = max(1, len(population) // 10)

        # Use tournament selection to choose which individuals to replace
        for _ in range(num_replace):
            # Tournament selection for replacement position
            tournament_indices = np.random.choice(
                len(population), size=tournament_size, replace=False
            )
            worst_idx = min(tournament_indices, key=lambda i: fitness[i])

            # Select best from elite pool to inject
            if pool:
                best_elite = max(pool, key=lambda x: x[1])
                population[worst_idx] = best_elite[0].copy()
                fitness[worst_idx] = best_elite[1]
                # Remove used elite to avoid duplicates
                pool.remove(best_elite)

        return population, fitness

    # Generic dispatcher
    def inject(self, population: np.ndarray, fitness: np.ndarray, strategy: str, **kwargs):
        if strategy == "direct":
            return self.inject_direct(population, fitness, **kwargs)
        if strategy == "tournament":
            return self.inject_tournament(population, fitness, **kwargs)
        # Unknown -> no-op
        return population, fitness
