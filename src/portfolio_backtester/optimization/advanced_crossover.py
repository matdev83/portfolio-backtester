"""Advanced crossover operators for the genetic optimizer.

This module implements several advanced crossover operators that can be used
with the PyGAD genetic algorithm framework:

1. Simulated Binary Crossover (SBX)
2. Multi-point crossover
3. Uniform crossover variant
4. Problem-specific crossover operators

These operators provide more sophisticated recombination strategies than the
basic crossover types available in PyGAD.
"""

import numpy as np
from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)

def simulated_binary_crossover(parents: np.ndarray, 
                              offspring_size: Tuple[int, int], 
                              ga_instance: Any) -> np.ndarray:
    """Simulated Binary Crossover (SBX) implementation.
    
    SBX is designed to mimic the behavior of traditional crossover operators
    used in continuous optimization. It creates offspring that are close to
    the parents, with the spread controlled by a distribution index.
    
    Args:
        parents: Array of parent solutions with shape (num_parents, num_genes)
        offspring_size: Tuple specifying (num_offspring, num_genes)
        ga_instance: PyGAD GA instance (used for accessing gene_space info)
        
    Returns:
        Array of offspring solutions with shape matching offspring_size
    """
    # Distribution index - controls how close offspring are to parents
    # Higher values create offspring closer to parents
    eta_c = getattr(ga_instance, 'sbx_distribution_index', 20.0)
    
    offspring = []
    idx = 0
    
    while len(offspring) < offspring_size[0]:
        # Select two parents
        parent1_idx = idx % parents.shape[0]
        parent2_idx = (idx + 1) % parents.shape[0]
        
        parent1 = parents[parent1_idx, :].copy()
        parent2 = parents[parent2_idx, :].copy()
        
        # Apply SBX to create two offspring
        for i in range(len(parent1)):
            # Apply SBX transformation with probability 0.5
            if abs(parent1[i] - parent2[i]) > 1e-14 and np.random.random() <= 0.5:
                # Calculate beta
                if parent1[i] < parent2[i]:
                    y1, y2 = parent1[i], parent2[i]
                else:
                    y1, y2 = parent2[i], parent1[i]
                
                # Calculate beta_q
                rand = np.random.random()
                if rand <= 0.5:
                    beta = 2.0 * rand
                else:
                    beta = 1.0 / (2.0 * (1.0 - rand))
                
                beta = pow(beta, 1.0 / (eta_c + 1.0))
                
                # Calculate offspring values
                c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                
                # Ensure offspring are within bounds
                if hasattr(ga_instance, 'gene_space') and ga_instance.gene_space:
                    gene_space = ga_instance.gene_space[i] if isinstance(ga_instance.gene_space, list) else ga_instance.gene_space
                    if isinstance(gene_space, dict):
                        low = gene_space.get('low', -np.inf)
                        high = gene_space.get('high', np.inf)
                        c1 = max(low, min(high, c1))
                        c2 = max(low, min(high, c2))
                
                # Assign values
                if parent1[i] < parent2[i]:
                    parent1[i], parent2[i] = c1, c2
                else:
                    parent1[i], parent2[i] = c2, c1
        
        # Add offspring to population
        offspring.append(parent1)
        if len(offspring) < offspring_size[0]:
            offspring.append(parent2)
        
        idx += 2
    
    # Trim to exact offspring size
    offspring = np.array(offspring[:offspring_size[0]])

    # Final clamping: ensure all offspring genes are within bounds
    if hasattr(ga_instance, 'gene_space') and ga_instance.gene_space:
        for i in range(offspring.shape[0]):
            for j in range(offspring.shape[1]):
                gene_space = ga_instance.gene_space[j] if isinstance(ga_instance.gene_space, list) else ga_instance.gene_space
                if isinstance(gene_space, dict):
                    low = gene_space.get('low', -np.inf)
                    high = gene_space.get('high', np.inf)
                    offspring[i, j] = max(low, min(high, offspring[i, j]))

    return offspring

def multi_point_crossover(parents: np.ndarray,
                         offspring_size: Tuple[int, int],
                         ga_instance: Any) -> np.ndarray:
    """Multi-point crossover with configurable number of crossover points.
    
    Args:
        parents: Array of parent solutions with shape (num_parents, num_genes)
        offspring_size: Tuple specifying (num_offspring, num_genes)
        ga_instance: PyGAD GA instance
        
    Returns:
        Array of offspring solutions with shape matching offspring_size
    """
    # Get number of crossover points (default to 3)
    num_points = getattr(ga_instance, 'num_crossover_points', 3)
    
    offspring = []
    idx = 0
    
    while len(offspring) < offspring_size[0]:
        # Select two parents
        parent1_idx = idx % parents.shape[0]
        parent2_idx = (idx + 1) % parents.shape[0]
        
        parent1 = parents[parent1_idx, :].copy()
        parent2 = parents[parent2_idx, :].copy()
        
        # Generate crossover points
        chromosome_length = len(parent1)
        if num_points >= chromosome_length:
            num_points = chromosome_length - 1
            
        crossover_points = np.sort(np.random.choice(
            np.arange(1, chromosome_length), 
            size=min(num_points, chromosome_length - 1), 
            replace=False
        ))
        
        # Add start and end points
        points = np.concatenate([[0], crossover_points, [chromosome_length]])
        
        # Alternate between parents for segments
        offspring1 = np.empty_like(parent1)
        offspring2 = np.empty_like(parent2)
        
        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            if i % 2 == 0:
                # Take from parent1
                offspring1[start:end] = parent1[start:end]
                offspring2[start:end] = parent2[start:end]
            else:
                # Take from parent2
                offspring1[start:end] = parent2[start:end]
                offspring2[start:end] = parent1[start:end]
        
        # Add offspring to population
        offspring.append(offspring1)
        if len(offspring) < offspring_size[0]:
            offspring.append(offspring2)
        
        idx += 2
    
    # Trim to exact offspring size
    return np.array(offspring[:offspring_size[0]])

def uniform_crossover_variant(parents: np.ndarray,
                             offspring_size: Tuple[int, int],
                             ga_instance: Any) -> np.ndarray:
    """Uniform crossover variant with bias parameter.
    
    Unlike standard uniform crossover which selects genes with 50% probability,
    this variant allows for a bias parameter to favor one parent over the other.
    
    Args:
        parents: Array of parent solutions with shape (num_parents, num_genes)
        offspring_size: Tuple specifying (num_offspring, num_genes)
        ga_instance: PyGAD GA instance
        
    Returns:
        Array of offspring solutions with shape matching offspring_size
    """
    # Bias parameter - probability of selecting from first parent (default 0.5)
    bias = getattr(ga_instance, 'uniform_crossover_bias', 0.5)
    
    offspring = []
    idx = 0
    
    while len(offspring) < offspring_size[0]:
        # Select two parents
        parent1_idx = idx % parents.shape[0]
        parent2_idx = (idx + 1) % parents.shape[0]
        
        parent1 = parents[parent1_idx, :].copy()
        parent2 = parents[parent2_idx, :].copy()
        
        # Create offspring by selecting genes based on bias
        mask = np.random.random(len(parent1)) < bias
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        
        # Add offspring to population
        offspring.append(offspring1)
        if len(offspring) < offspring_size[0]:
            offspring.append(offspring2)
        
        idx += 2
    
    # Trim to exact offspring size
    return np.array(offspring[:offspring_size[0]])

def arithmetic_crossover(parents: np.ndarray,
                        offspring_size: Tuple[int, int],
                        ga_instance: Any) -> np.ndarray:
    """Arithmetic crossover for continuous optimization problems.
    
    Creates offspring as weighted averages of parents, suitable for continuous
    parameter optimization.
    
    Args:
        parents: Array of parent solutions with shape (num_parents, num_genes)
        offspring_size: Tuple specifying (num_offspring, num_genes)
        ga_instance: PyGAD GA instance
        
    Returns:
        Array of offspring solutions with shape matching offspring_size
    """
    offspring = []
    idx = 0
    
    while len(offspring) < offspring_size[0]:
        # Select two parents
        parent1_idx = idx % parents.shape[0]
        parent2_idx = (idx + 1) % parents.shape[0]
        
        parent1 = parents[parent1_idx, :].copy()
        parent2 = parents[parent2_idx, :].copy()
        
        # Generate weight for arithmetic combination
        alpha = np.random.random()
        
        # Create offspring as weighted averages
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = (1 - alpha) * parent1 + alpha * parent2
        
        # Ensure offspring are within bounds if gene_space is available
        if hasattr(ga_instance, 'gene_space') and ga_instance.gene_space:
            for i in range(len(offspring1)):
                gene_space = ga_instance.gene_space[i] if isinstance(ga_instance.gene_space, list) else ga_instance.gene_space
                if isinstance(gene_space, dict):
                    low = gene_space.get('low', -np.inf)
                    high = gene_space.get('high', np.inf)
                    offspring1[i] = max(low, min(high, offspring1[i]))
                    offspring2[i] = max(low, min(high, offspring2[i]))
        
        # Add offspring to population
        offspring.append(offspring1)
        if len(offspring) < offspring_size[0]:
            offspring.append(offspring2)
        
        idx += 2
    
    # Trim to exact offspring size
    return np.array(offspring[:offspring_size[0]])

# Crossover operator selection strategies
def adaptive_crossover_selector(ga_instance, generation: int) -> str:
    """Select crossover operator based on generation and diversity.
    
    Args:
        ga_instance: PyGAD GA instance
        generation: Current generation number
        
    Returns:
        Name of crossover operator to use
    """
    # This would be called from the on_generation callback
    # For now, we'll implement a simple generation-based selection
    if generation < ga_instance.num_generations * 0.3:
        return "simulated_binary"  # Exploration phase
    elif generation < ga_instance.num_generations * 0.7:
        return "multi_point"  # Exploitation phase
    else:
        return "arithmetic"  # Fine-tuning phase

# Registry of available crossover operators
CROSSOVER_OPERATORS = {
    "simulated_binary": simulated_binary_crossover,
    "multi_point": multi_point_crossover,
    "uniform_variant": uniform_crossover_variant,
    "arithmetic": arithmetic_crossover,
}

def get_crossover_operator(name: str):
    """Get crossover operator function by name.
    
    Args:
        name: Name of the crossover operator
        
    Returns:
        Crossover operator function
    """
    return CROSSOVER_OPERATORS.get(name, None)