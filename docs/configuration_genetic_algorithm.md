### Genetic Algorithm Enhancements (v3)

The **Genetic** optimizer now includes three major optional subsystems designed to
speed-up convergence and prevent the loss of valuable solutions:

1. **Adaptive Parameter Control** – mutation and crossover probabilities are
   adjusted *during* the run based on population diversity, fitness variance and
   generation progress.
2. **Elite Preservation System** – the best chromosomes discovered across
   generations are kept in a fixed-size archive and can be periodically
   reinjected into the population to avoid genetic drift.
3. **Advanced Crossover Operators** – sophisticated recombination strategies that
   provide better exploration and exploitation capabilities than standard PyGAD
   crossover types.

All features are fully backward-compatible and *disabled by default*.  Your
existing scenarios will work unchanged.

### Adaptive Parameter Control

| Controller | Behaviour |
|------------|-----------|
| `DiversityCalculator` | Computes average pair-wise distance between chromosomes (0-1). |
| `AdaptiveMutationController` | • Increases mutation when diversity < threshold or fitness variance stalls.<br>• Gradually decays mutation as generations progress.<br>• Respects `min_rate`/`max_rate` bounds. |
| `AdaptiveCrossoverController` | • Raises crossover probability when diversity is low.<br>• Lowers it when the population appears converged. |

#### YAML Configuration Example

```yaml
# inside genetic_algorithm_params section
adaptive_mutation:
  enabled: true        # <-- master switch
  base_rate: 0.10      # starting mutation probability
  min_rate: 0.01
  max_rate: 0.50
  diversity_threshold: 0.30   # below this => boost mutation
  # optional crossover overrides
  base_crossover_rate: 0.80
  min_crossover_rate: 0.60
  max_crossover_rate: 0.95
```

### Elite Preservation System

A lightweight archive stores the top-`N` chromosomes (by fitness) seen so far.
These elites can be reinserted every few generations, ensuring the optimizer
never "forgets" its best discoveries.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable elite preservation | `false` |
| `max_archive_size` | Maximum elites stored globally | `50` |
| `aging_factor` | Fitness decay per generation (prevents stagnation) | `0.95` |
| `injection_strategy` | `direct` (replace worst) | `tournament` | `direct` |
| `injection_frequency` | How often (in generations) to inject elites | `5` |
| `min_elites` / `max_elites` | Range of elites to inject each time | `2` / `5` |

#### YAML Configuration Example

```yaml
elite_preservation:
  enabled: true
  max_archive_size: 50
  aging_factor: 0.95
  injection_strategy: "direct"      # or "tournament"
  injection_frequency: 5            # generations
  min_elites: 2
  max_elites: 5
```

### Advanced Crossover Operators

The genetic optimizer now supports four advanced crossover operators that provide
more sophisticated recombination strategies than the basic crossover types
available in PyGAD. These operators can significantly improve convergence speed
and solution quality for different types of optimization problems.

#### Available Operators

1. **Simulated Binary Crossover (SBX)** – Designed for continuous optimization
   problems, creates offspring that are close to parents with spread controlled
   by a distribution index.
2. **Multi-point Crossover** – Creates offspring by selecting multiple crossover
   points and alternating between parents for chromosome segments.
3. **Uniform Crossover Variant** – Allows for a bias parameter to favor one
   parent over another, unlike standard uniform crossover.
4. **Arithmetic Crossover** – Creates offspring as weighted averages of parents,
   suitable for continuous parameter optimization.

#### YAML Configuration

To use advanced crossover operators, specify the `advanced_crossover_type`
parameter in your genetic algorithm configuration:

```yaml
genetic_algorithm_params:
  # Select which advanced crossover operator to use
  advanced_crossover_type: simulated_binary  # or multi_point, uniform_variant, arithmetic
  
  # Operator-specific parameters (optional, shown with defaults)
  sbx_distribution_index: 20.0    # For SBX - controls offspring spread (1.0-100.0)
  num_crossover_points: 3         # For Multi-point - number of crossover points (2-10)
  uniform_crossover_bias: 0.5     # For Uniform variant - probability of selecting from first parent (0.1-0.9)
```

#### Usage Examples

```yaml
# Example 1: Simulated Binary Crossover for continuous optimization
genetic_algorithm_params:
  advanced_crossover_type: simulated_binary
  sbx_distribution_index: 15.0

# Example 2: Multi-point Crossover for linked gene problems
genetic_algorithm_params:
  advanced_crossover_type: multi_point
  num_crossover_points: 4

# Example 3: Uniform Crossover Variant with bias
genetic_algorithm_params:
  advanced_crossover_type: uniform_variant
  uniform_crossover_bias: 0.3

# Example 4: Arithmetic Crossover for weighted averaging
genetic_algorithm_params:
  advanced_crossover_type: arithmetic
```

#### Benefits

* **Better Exploration**: SBX and Multi-point operators provide more diverse
  recombination strategies
* **Fine-tuned Control**: Configurable parameters allow optimization for specific
  problem types
* **Performance Optimized**: Minimal computational overhead compared to standard
  PyGAD operators
* **Backward Compatible**: Standard PyGAD crossover types still work when
  `advanced_crossover_type` is not specified

### Full GA Configuration Snippet

Below is a minimal but complete `genetic_algorithm_params` block that activates
all new subsystems while keeping all previous parameters unchanged:

```yaml
genetic_algorithm_params:
  num_generations: 150
  sol_per_pop: 40
  num_parents_mating: 20
  parent_selection_type: tournament
  crossover_type: single_point
  mutation_type: random
  keep_elitism: 2  # basic PyGAD elitism (still supported)

  # New ✨
  adaptive_mutation:
    enabled: true
    base_rate: 0.12
    min_rate: 0.02
    max_rate: 0.40
    diversity_threshold: 0.25
  elite_preservation:
    enabled: true
    max_archive_size: 100
    injection_strategy: tournament
    injection_frequency: 3
    min_elites: 3
    max_elites: 7
  # Advanced Crossover Operators
  advanced_crossover_type: simulated_binary
  sbx_distribution_index: 15.0
```
