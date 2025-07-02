import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import track
import os
import uuid
import json

def run_monte_carlo_simulation(
    strategy_returns, 
    n_simulations=1000, 
    n_years=10, 
    initial_capital=1.0
):
    """
    Runs a Monte Carlo simulation on a given strategy's returns.

    Args:
        strategy_returns (pd.Series): Historical returns of the strategy.
        n_simulations (int): Number of simulation paths to generate.
        n_years (int): Number of years to simulate into the future.
        initial_capital (float): The starting capital for the simulation.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated portfolio values for each path.
    """
    # Use monthly returns, assuming CAL_FACTOR is 12
    monthly_returns = strategy_returns
    n_months = n_years * 12
    
    # Calculate historical statistics
    mean_return = monthly_returns.mean()
    std_dev = monthly_returns.std()

    # Prepare an array to hold all simulation results
    all_sim_results = np.zeros((n_months + 1, n_simulations))
    all_sim_results[0, :] = initial_capital

    # Run the simulation
    for i in track(range(n_simulations), description="Running Monte Carlo Simulation..."):
        portfolio_value = initial_capital
        for t in range(1, n_months + 1):
            # Generate a random return based on historical stats
            random_return = np.random.normal(mean_return, std_dev)
            portfolio_value *= (1 + random_return)
            all_sim_results[t, i] = portfolio_value
            
    return pd.DataFrame(all_sim_results)

def plot_monte_carlo_results(simulation_results, title="Monte Carlo Simulation", scenario_name=None, params=None, output_dir="data/pnl_charts", interactive=False):
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
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot all simulation paths
    ax.plot(simulation_results, color='gray', alpha=0.1, lw=0.5)
    
    # Highlight key statistics
    median_path = simulation_results.median(axis=1)
    percentile_5 = simulation_results.quantile(0.05, axis=1)
    percentile_95 = simulation_results.quantile(0.95, axis=1)

    ax.plot(median_path, color='red', lw=2, label='Median Outcome')
    ax.plot(percentile_5, color='blue', lw=2, linestyle='--', label='5th Percentile')
    ax.plot(percentile_95, color='blue', lw=2, linestyle='--', label='95th Percentile')
    
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
            param_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
            param_str = param_str.replace('"', '').replace(':', '').replace(',', '_').replace(' ', '')
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
