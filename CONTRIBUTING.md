# Contributing to Portfolio Backtester

First off, thank you for considering contributing to Portfolio Backtester! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/your-username/portfolio-backtester/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If this is something you think you can fix, then [fork Portfolio Backtester](https://github.com/your-username/portfolio-backtester/fork) and create a branch with a descriptive name.

A good branch name would be (where issue #33 is the ticket you're working on):

```bash
git checkout -b 33-add-new-optimization-metric
```

### Get the code

```bash
# Clone your fork to your local machine
git clone https://github.com/your-username/portfolio-backtester.git

# Go to the project directory
cd portfolio-backtester

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Run the tests

To make sure everything is working as expected, run the tests:

```bash
python -m pytest
```

## Property-Based Testing with Hypothesis

Portfolio Backtester uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. This approach generates random inputs to test invariants and properties of our code, which helps catch edge cases and improve test coverage.

### Running Hypothesis Tests

To run all Hypothesis tests:

```bash
# Run all Hypothesis tests
python -m pytest -k hypothesis

# Run with CI profile (more examples, longer timeouts)
$env:HYPOTHESIS_PROFILE='ci'; python -m pytest -k hypothesis
# On Linux/macOS: HYPOTHESIS_PROFILE=ci python -m pytest -k hypothesis
```

### Profiles

We have two Hypothesis profiles:

- **dev** (default): Faster feedback with fewer examples
  - `max_examples=100`
  - `deadline=2000ms`
- **ci**: More thorough testing with more examples
  - `max_examples=300`
  - `deadline=4000ms`

### Writing Property Tests

When adding new features or fixing bugs, consider adding property tests. Here's a basic workflow:

1. **Identify invariants**: What properties should always hold for your function?
   - Example: "Normalized weights should always sum to 1.0 or 0.0"
   - Example: "ATR values should always be non-negative"

2. **Create a strategy**: Define how to generate test inputs
   ```python
   @st.composite
   def weights_and_leverage(draw):
       n_assets = draw(st.integers(min_value=2, max_value=10))
       weights = draw(hnp.arrays(dtype=float, shape=n_assets, elements=st.floats(...)))
       leverage = draw(st.floats(min_value=0.1, max_value=3.0))
       return weights, leverage
   ```

3. **Write the test**: Use the `@given` decorator with your strategy
   ```python
   @given(weights_and_leverage())
   @settings(deadline=None)  # For slow functions
   def test_normalize_weights_properties(data):
       weights, leverage = data
       normalized = _normalize_weights(weights, leverage)
       assert np.isclose(normalized.sum(), leverage if weights.sum() != 0 else 0)
   ```

4. **Handle degenerate cases**: Use `assume()` to filter out inputs that make the property meaningless
   ```python
   assume(np.std(returns) > 1e-10)  # Skip cases with zero variance
   ```

5. **Add examples**: When Hypothesis finds a failure, add it as an example
   ```python
   @example(np.array([0.0, 0.0]))  # Known edge case
   @given(st.arrays(...))
   def test_function(data):
       # ...
   ```

### Common Strategies

We have reusable strategies in `tests/strategies/common_strategies.py`:

- `price_dataframes()`: Generate OHLCV price data
- `return_series()`: Generate return series
- `return_matrices()`: Generate matrices of returns
- `weights_and_leverage()`: Generate portfolio weights and leverage
- `frequencies()`: Generate valid pandas frequency strings
- `timestamps()`: Generate pandas timestamps
- `parameter_spaces()`: Generate valid parameter spaces for optimization
- `parameter_values()`: Generate parameter values based on a parameter space
- `populations()`: Generate populations of parameter dictionaries
- `evaluation_results()`: Generate evaluation results for optimization
- `optimization_configs()`: Generate optimization configuration dictionaries

### Optimization Property Tests

We have property tests for the optimization components in the following files:

- `tests/unit/optimization/test_genetic_optimizer_properties.py`: Tests for the genetic algorithm optimizer
- `tests/unit/optimization/test_population_evaluator_properties.py`: Tests for the population evaluator
- `tests/unit/optimization/test_parameter_generator_properties.py`: Tests for the parameter generator
- `tests/unit/optimization/test_deduplication_properties.py`: Tests for deduplication mechanisms
- `tests/integration/optimization/test_ga_optimization_flow.py`: Integration tests for the full GA optimization flow

These tests verify important properties of the optimization components, such as:

- Parameter space validation
- Population generation and evolution
- Deduplication of identical parameter sets
- Correct caching of evaluation results
- End-to-end optimization flow

### Tips for Effective Property Testing

1. **Start simple**: Test basic properties first before complex ones
2. **Use appropriate settings**: Use `@settings(deadline=None)` for slow functions
3. **Narrow search space**: Constrain your strategies to reasonable values
4. **Add regression tests**: When Hypothesis finds a bug, add an `@example`
5. **Run with CI profile**: Before submitting PRs, run with the CI profile

### Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :)

### Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with Portfolio Backtester's master branch:

```bash
git remote add upstream https://github.com/your-username/portfolio-backtester.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```bash
git checkout 33-add-new-optimization-metric
git rebase master
git push --set-upstream origin 33-add-new-optimization-metric
```

Finally, go to GitHub and [make a Pull Request](https://github.com/your-username/portfolio-backtester/compare) :D

### Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing and merging, check out this guide: [About Git rebase](https://docs.github.com/en/get-started/using-git/about-git-rebase).