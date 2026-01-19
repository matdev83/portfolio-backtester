import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Wrapper class for Monte Carlo simulation functionality.
    Provides a clean interface for the existing Monte Carlo functions.
    """

    def __init__(
        self,
        n_simulations=1000,
        n_years=10,
        initial_capital=1.0,
        block_size: int | None = None,
    ):
        """
        Initialize the Monte Carlo simulator.

        Args:
            n_simulations: Number of simulation paths
            n_years: Number of years to simulate
            initial_capital: Starting capital
            block_size: Block length (in months) for bootstrap sampling
        """
        self.n_simulations = n_simulations
        self.n_years = n_years
        self.initial_capital = initial_capital
        self.block_size = block_size

    def run_simulation(self, strategy_returns):
        """
        Run Monte Carlo simulation on strategy returns.

        Args:
            strategy_returns: Historical returns of the strategy

        Returns:
            DataFrame with simulation results
        """
        return run_monte_carlo_simulation(
            strategy_returns,
            n_simulations=self.n_simulations,
            n_years=self.n_years,
            initial_capital=self.initial_capital,
            block_size=self.block_size,
        )

    def plot_results(
        self,
        simulation_results,
        title="Monte Carlo Simulation",
        scenario_name=None,
        params=None,
        output_dir="data/pnl_charts",
        interactive=False,
    ):
        """
        Plot simulation results.

        Args:
            simulation_results: Results from run_simulation
            title: Plot title
            scenario_name: Scenario name for filename
            params: Parameters for filename
            output_dir: Directory to save plot
            interactive: Whether to display plot
        """
        plot_monte_carlo_results(
            simulation_results,
            title=title,
            scenario_name=scenario_name,
            params=params,
            output_dir=output_dir,
            interactive=interactive,
        )


def _coerce_returns_to_monthly(returns: pd.Series) -> pd.Series:
    """Ensure returns are monthly by compounding within each month."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        return returns

    if len(returns) < 3:
        return returns

    try:
        freq = pd.infer_freq(returns.index)
    except (ValueError, TypeError):
        freq = None

    if freq and (freq.startswith("M") or freq.startswith("ME") or freq.startswith("MS")):
        return returns

    # Treat daily/weekly or unknown frequency as sub-monthly.
    monthly = (1 + returns).resample("ME").prod() - 1
    return monthly.dropna()


def _block_bootstrap_series(
    values: np.ndarray,
    n_samples: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a block-bootstrap sample preserving short-term autocorrelation."""
    if n_samples <= 0 or len(values) == 0:
        return np.array([])

    block_size = max(1, int(block_size))
    samples: list[float] = []
    n_values = len(values)

    while len(samples) < n_samples:
        start = int(rng.integers(0, n_values))
        end = start + block_size
        if end <= n_values:
            block = values[start:end]
        else:
            wrap = end - n_values
            block = np.concatenate([values[start:], values[:wrap]])
        samples.extend(block.tolist())

    return np.asarray(samples[:n_samples], dtype=float)


def run_monte_carlo_simulation(
    strategy_returns,
    n_simulations=1000,
    n_years=10,
    initial_capital=1.0,
    block_size: int | None = None,
):
    """
    Runs a Monte Carlo simulation on a given strategy's returns.

    Args:
        strategy_returns (pd.Series): Historical returns of the strategy.
        n_simulations (int): Number of simulation paths to generate.
        n_years (int): Number of years to simulate into the future.
        initial_capital (float): The starting capital for the simulation.
        block_size (int | None): Block length in months for bootstrap sampling.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated portfolio values for each path.
    """
    returns_series: pd.Series
    if isinstance(strategy_returns, pd.DataFrame):
        if strategy_returns.shape[1] >= 1:
            returns_series = strategy_returns.iloc[:, 0]
        else:
            returns_series = pd.Series(dtype=float)
    elif isinstance(strategy_returns, pd.Series):
        returns_series = strategy_returns
    else:
        returns_series = pd.Series(strategy_returns)

    returns_series = returns_series.dropna()
    monthly_returns = _coerce_returns_to_monthly(returns_series)
    if monthly_returns.empty:
        return pd.DataFrame([initial_capital])

    n_months = int(n_years * 12)
    block_len = block_size or max(1, min(12, len(monthly_returns) // 4 or 1))
    rng = np.random.default_rng()

    all_sim_results = np.empty((n_months + 1, n_simulations), dtype=float)
    all_sim_results[0, :] = float(initial_capital)

    values = monthly_returns.to_numpy(dtype=float)
    for sim in range(n_simulations):
        sampled = _block_bootstrap_series(values, n_months, block_len, rng)
        compounded = np.cumprod(1 + sampled)
        all_sim_results[1:, sim] = float(initial_capital) * compounded

    return pd.DataFrame(all_sim_results)


def plot_monte_carlo_results(
    simulation_results,
    title="Monte Carlo Simulation",
    scenario_name=None,
    params=None,
    output_dir="data/pnl_charts",
    interactive=False,
):
    """
    Plots the results of a Monte Carlo simulation.

    Args:
        simulation_results (pd.DataFrame): The results from the MC simulation.
        title (str): The title for the plot.
        scenario_name (str): Scenario name for filename.
        params (dict): Parameters to serialize for filename.
        output_dir (str): Directory to save the plot.
        interactive (bool): If True, display the plot, else only save.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot all simulation paths
    ax.plot(simulation_results, color="gray", alpha=0.1, lw=0.5)

    # Highlight key statistics
    median_path = simulation_results.median(axis=1)
    percentile_5 = simulation_results.quantile(0.05, axis=1)
    percentile_95 = simulation_results.quantile(0.95, axis=1)

    ax.plot(median_path, color="red", lw=2, label="Median Outcome")
    ax.plot(percentile_5, color="blue", lw=2, linestyle="--", label="5th Percentile")
    ax.plot(percentile_95, color="blue", lw=2, linestyle="--", label="95th Percentile")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Months into Future", fontsize=12)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()

    # --- Save to file logic ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Serialize params for filename
    if params:
        try:
            param_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
            param_str = (
                param_str.replace('"', "").replace(":", "").replace(",", "_").replace(" ", "")
            )
        except Exception:
            param_str = "params"
    else:
        param_str = "params"
    scenario_str = scenario_name if scenario_name else "scenario"
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{scenario_str}_{param_str}_{unique_id}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Monte Carlo P&L chart saved to: {filepath}")

    if interactive:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)
