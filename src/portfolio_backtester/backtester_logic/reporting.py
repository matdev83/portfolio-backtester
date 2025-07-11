import matplotlib.pyplot as plt
import os
import pandas as pd
from rich.console import Console
from rich.table import Table
import os
from datetime import datetime
import seaborn as sns # Added for better plotting
import numpy as np # Added for numerical operations
import optuna # Added for stability measures plotting
import pandas as pd # Added for parameter stability plotting

from ..reporting.performance_metrics import calculate_metrics

def display_results(self, daily_data_for_display):
    logger = self.logger
    logger.info("Generating performance report.")
    console = Console()

    if not self.results:
        logger.warning("No results available to display. Skipping performance report generation.")
        return
    
    # Check if optimization reports should be generated
    advanced_reporting_config = self.global_config.get('advanced_reporting_config', {})
    enable_optimization_reports = advanced_reporting_config.get('enable_optimization_reports', True)

    benchmark_ticker = self.global_config["benchmark"]
    if isinstance(daily_data_for_display.columns, pd.MultiIndex):
        if benchmark_ticker in daily_data_for_display.columns.get_level_values(0):
            if ('Close' in daily_data_for_display[benchmark_ticker].columns):
                bench_prices = daily_data_for_display[(benchmark_ticker, 'Close')]
            elif ('Adj Close' in daily_data_for_display[benchmark_ticker].columns):
                 bench_prices = daily_data_for_display[(benchmark_ticker, 'Adj Close')]
            else:
                logger.error(f"Could not find 'Close' or 'Adj Close' for benchmark {benchmark_ticker} in multi-indexed data.")
                bench_prices = pd.Series(dtype=float)
        elif benchmark_ticker in daily_data_for_display.columns.get_level_values(1):
            if ('Close', benchmark_ticker) in daily_data_for_display.columns:
                bench_prices = daily_data_for_display[('Close', benchmark_ticker)]
            elif ('Adj Close', benchmark_ticker) in daily_data_for_display.columns:
                bench_prices = daily_data_for_display[('Adj Close', benchmark_ticker)]
            else:
                logger.error(f"Could not find ('Close'/'Adj Close', {benchmark_ticker}) in multi-indexed data.")
                bench_prices = pd.Series(dtype=float)
        else:
            logger.error(f"Benchmark ticker {benchmark_ticker} not found in multi-indexed columns.")
            bench_prices = pd.Series(dtype=float)
    else:
        if benchmark_ticker in daily_data_for_display.columns:
            bench_prices = daily_data_for_display[benchmark_ticker]
        else:
            logger.error(f"Benchmark ticker {benchmark_ticker} not found in single-level column data.")
            bench_prices = pd.Series(dtype=float)

    bench_rets_full = bench_prices.pct_change(fill_method=None).fillna(0)

    first_result_key = list(self.results.keys())[0]
    train_end_date = self.results[first_result_key].get("train_end_date")

    periods_data = []
    num_trials_full = {name: res.get("num_trials_for_dsr", 1) for name, res in self.results.items()}

    if train_end_date:
        periods_data.append({
            "title": "In-Sample Performance (Net of Costs)",
            "returns_map": {name: res["returns"][res["returns"].index <= train_end_date] for name, res in self.results.items()},
            "bench_returns": bench_rets_full[bench_rets_full.index <= train_end_date],
            "num_trials_map": {name: 1 for name in self.results.keys()}
        })
        periods_data.append({
            "title": "Out-of-Sample Performance (Net of Costs)",
            "returns_map": {name: res["returns"][res["returns"].index > train_end_date] for name, res in self.results.items()},
            "bench_returns": bench_rets_full[bench_rets_full.index > train_end_date],
            "num_trials_map": num_trials_full
        })

    periods_data.append({
        "title": "Full Period Performance (Net of Costs)",
        "returns_map": {name: res["returns"] for name, res in self.results.items()},
        "bench_returns": bench_rets_full,
        "num_trials_map": num_trials_full
    })

    # Check for constraint violations and display warnings
    constraint_violations_found = False
    for name, result_data in self.results.items():
        if result_data.get("constraint_status") == "VIOLATED":
            constraint_violations_found = True
            break
    
    if constraint_violations_found:
        _display_constraint_violation_warning(self, console)
    
    for period_data in periods_data:
        _generate_performance_table(
            self,
            console,
            period_data["returns_map"],
            period_data["bench_returns"],
            period_data["title"],
            period_data["num_trials_map"]
        )
    
    logger.info("Performance tables displayed.")
    _plot_performance_summary(self, bench_rets_full, train_end_date)

    # Plot stability measures if available and optimization reports are enabled
    if enable_optimization_reports:
        for name, result_data in self.results.items():
            if "best_trial_obj" in result_data and result_data["best_trial_obj"] is not None:
                _plot_stability_measures(self, name, result_data["best_trial_obj"], result_data["returns"])
                
                # Create Monte Carlo robustness analysis if we have optimal parameters
                if "optimal_params" in result_data and result_data["optimal_params"] is not None:
                    # Find the original scenario config
                    scenario_config = None
                    for scenario in self.scenarios:
                        if scenario["name"] in name:
                            scenario_config = scenario
                            break
                    
                    if scenario_config:
                        _plot_monte_carlo_robustness_analysis(
                            self, 
                            name, 
                            scenario_config, 
                            result_data["optimal_params"],
                            self.monthly_data,
                            daily_data_for_display,
                            self.rets_full
                        )
    else:
        logger.info("Optimization reports are disabled. Skipping advanced stability measures and Monte Carlo analysis.")

def _generate_performance_table(self, console: Console, period_returns: dict,
                                  bench_period_rets: pd.Series, title: str,
                                  num_trials_map: dict):
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)

    bench_metrics = calculate_metrics(bench_period_rets, bench_period_rets, self.global_config["benchmark"], name=self.global_config["benchmark"], num_trials=1)

    all_period_metrics = {self.global_config["benchmark"]: bench_metrics}

    for name in period_returns.keys():
        display_name = self.results[name]["display_name"]
        table.add_column(display_name, style="magenta")
    table.add_column(self.global_config["benchmark"], style="green")

    for name, rets in period_returns.items():
        display_name = self.results[name]["display_name"]
        strategy_num_trials = num_trials_map.get(name, 1)
        metrics = calculate_metrics(rets, bench_period_rets, self.global_config["benchmark"], name=display_name, num_trials=strategy_num_trials)
        all_period_metrics[display_name] = metrics

    percentage_metrics = ["Total Return", "Ann. Return"]
    high_precision_metrics = ["ADF p-value"]

    if not bench_metrics.empty:
        for metric_name in bench_metrics.index:
            row_values = [metric_name]
            for strategy_name_key in period_returns.keys():
                display_name = self.results[strategy_name_key]["display_name"]
                value = all_period_metrics[display_name].loc[metric_name]
                if metric_name in percentage_metrics:
                    row_values.append(f"{value:.2%}")
                elif metric_name in high_precision_metrics:
                    row_values.append(f"{value:.6f}")
                else:
                    row_values.append(f"{value:.4f}")
            
            bench_value = bench_metrics.loc[metric_name]
            if metric_name in percentage_metrics:
                row_values.append(f"{bench_value:.2%}")
            elif metric_name in high_precision_metrics:
                row_values.append(f"{bench_value:.6f}")
            else:
                row_values.append(f"{bench_value:.4f}")
            table.add_row(*row_values)
    
    console.print(table)

    for name in period_returns.keys():
        result_data = self.results.get(name, {})
        optimal_params = result_data.get("optimal_params")
        display_name = result_data.get("display_name", name)
        constraint_status = result_data.get("constraint_status", "UNKNOWN")

        if optimal_params:
            # Add constraint status to the title if there are violations
            title_suffix = ""
            if constraint_status == "VIOLATED":
                title_suffix = " ⚠️ CONSTRAINT VIOLATION"
            elif constraint_status == "ADJUSTED":
                title_suffix = " ✅ CONSTRAINT ADJUSTED"
            elif constraint_status == "FALLBACK_OK":
                title_suffix = " ⚠️ FALLBACK USED"
            
            params_table = Table(
                title=f"Optimal Parameters for {display_name}{title_suffix}", 
                show_header=True, 
                header_style="bold magenta"
            )
            params_table.add_column("Parameter", style="cyan")
            params_table.add_column("Value", style="green")
            
            for param, value in optimal_params.items():
                params_table.add_row(str(param), str(value))
            
            # Add constraint status row if there are constraints
            constraints_config = result_data.get("constraints_config", [])
            if constraints_config:
                params_table.add_row("", "")  # Empty row for spacing
                params_table.add_row("Constraint Status", constraint_status)
                
                constraint_message = result_data.get("constraint_message", "")
                if constraint_message:
                    # Split long messages into multiple rows
                    if len(constraint_message) > 50:
                        words = constraint_message.split()
                        lines = []
                        current_line = []
                        for word in words:
                            if len(" ".join(current_line + [word])) <= 50:
                                current_line.append(word)
                            else:
                                lines.append(" ".join(current_line))
                                current_line = [word]
                        if current_line:
                            lines.append(" ".join(current_line))
                        
                        for i, line in enumerate(lines):
                            if i == 0:
                                params_table.add_row("Constraint Details", line)
                            else:
                                params_table.add_row("", line)
                    else:
                        params_table.add_row("Constraint Details", constraint_message)
            
            console.print(params_table)

def _plot_performance_summary(self, bench_rets_full: pd.Series, train_end_date: pd.Timestamp | None):
    logger = self.logger
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    ax1.set_title("Cumulative Returns (Net of Costs)", fontsize=16)
    ax1.set_ylabel("Cumulative Returns (Log Scale)", fontsize=12)
    ax1.set_yscale('log')
    
    all_cumulative_returns_plotting = []
    for name, result_data in self.results.items():
        cumulative_strategy_returns = (1 + result_data["returns"]).cumprod()
        cumulative_strategy_returns.plot(ax=ax1, label=result_data["display_name"])
        all_cumulative_returns_plotting.append(cumulative_strategy_returns)
        
        # Add green dots for new all-time highs (ATH)
        ath_mask = cumulative_strategy_returns.expanding().max() == cumulative_strategy_returns
        ath_dates = cumulative_strategy_returns.index[ath_mask]
        ath_values = cumulative_strategy_returns[ath_mask]
        ax1.scatter(ath_dates, ath_values, color='green', s=20, alpha=0.7, zorder=5)
    
    cumulative_bench_returns = (1 + bench_rets_full).cumprod()
    cumulative_bench_returns.plot(ax=ax1, label=self.global_config["benchmark"], linestyle='--')
    all_cumulative_returns_plotting.append(cumulative_bench_returns)
    
    # Add green dots for benchmark ATH as well
    bench_ath_mask = cumulative_bench_returns.expanding().max() == cumulative_bench_returns
    bench_ath_dates = cumulative_bench_returns.index[bench_ath_mask]
    bench_ath_values = cumulative_bench_returns[bench_ath_mask]
    ax1.scatter(bench_ath_dates, bench_ath_values, color='green', s=20, alpha=0.7, zorder=5)

    if all_cumulative_returns_plotting:
        combined_cumulative_returns = pd.concat(all_cumulative_returns_plotting)
        max_val = combined_cumulative_returns.max().max()
        min_val = combined_cumulative_returns.min().min()
        ax1.set_ylim(bottom=min(0.9, min_val * 0.9) if min_val > 0 else 0.1, top=max_val * 1.1)

    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)

    def calculate_drawdown(returns_series):
        cumulative = (1 + returns_series).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return drawdown

    for name, result_data in self.results.items():
        drawdown = calculate_drawdown(result_data["returns"])
        drawdown.plot(ax=ax2, label=result_data["display_name"])
    
    bench_drawdown = calculate_drawdown(bench_rets_full)
    bench_drawdown.plot(ax=ax2, label=self.global_config["benchmark"], linestyle='--')

    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.fill_between(bench_drawdown.index, 0, bench_drawdown, color='gray', alpha=0.2)

    plt.tight_layout()
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = self.args.scenario_name if self.args.scenario_name else list(self.results.keys())[0]
    scenario_name_for_filename = base_filename.replace(" ", "_").replace("(", "").replace(")", "")

    filename = f"performance_summary_{scenario_name_for_filename}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    
    plt.savefig(filepath)
    logger.info(f"Performance plot saved to: {filepath}")

    if getattr(self.args, 'interactive', False):
        plt.show(block=False)
        logger.info("Performance plots displayed interactively.")
    else:
        plt.close(fig)
        logger.info("Performance plots generated and saved.")
    
    # Close all figures to prevent memory warnings
    plt.close('all')

def _plot_stability_measures(self, scenario_name: str, best_trial_obj, optimal_returns: pd.Series):
    """
    Create a Monte Carlo-style visualization showing P&L curves from all optimization trials.
    
    Args:
        scenario_name: Name of the scenario
        best_trial_obj: Best trial object from Optuna containing the study
        optimal_returns: Returns series from the final optimized strategy
    """
    logger = self.logger
    
    try:
        # Check if we have the study object
        if not hasattr(best_trial_obj, 'study') or best_trial_obj.study is None:
            logger.warning("No study object found in best_trial_obj. Cannot create trial P&L visualization.")
            return
            
        study = best_trial_obj.study
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) < 2:
            logger.warning(f"Only {len(completed_trials)} completed trials found. Need at least 2 for meaningful visualization.")
            return
            
        # Extract trial returns data
        trial_returns_data = []
        for trial in completed_trials:
            if 'trial_returns' in trial.user_attrs:
                try:
                    returns_dict = trial.user_attrs['trial_returns']
                    dates = pd.to_datetime(returns_dict['dates'])
                    returns = pd.Series(returns_dict['returns'], index=dates)
                    
                    # Handle multi-objective optimization
                    try:
                        trial_value = trial.value
                    except RuntimeError:
                        # Multi-objective optimization: use first objective value
                        trial_value = trial.values[0] if trial.values else 0.0
                    
                    trial_returns_data.append({
                        'trial_number': trial.number,
                        'returns': returns,
                        'params': trial.user_attrs.get('trial_params', trial.params),
                        'value': trial_value
                    })
                except Exception as e:
                    logger.debug(f"Failed to extract returns for trial {trial.number}: {e}")
                    continue
        
        if len(trial_returns_data) < 2:
            logger.warning(f"Only {len(trial_returns_data)} trials have stored returns data. Cannot create visualization.")
            return
            
        logger.info(f"Creating Monte Carlo-style trial P&L visualization with {len(trial_returns_data)} trials...")
        
        # Set up the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot all trial P&L curves as light gray lines
        all_cumulative_returns = []
        trial_values = []
        
        for trial_data in trial_returns_data:
            returns = trial_data['returns']
            cumulative_returns = (1 + returns).cumprod()
            all_cumulative_returns.append(cumulative_returns)
            trial_values.append(trial_data['value'])
            
            # Plot trial curve in light gray
            ax.plot(cumulative_returns.index, cumulative_returns.values, 
                   color='lightgray', alpha=0.3, linewidth=0.8, zorder=1)
        
        # Plot the final optimized strategy as a bold black line
        if optimal_returns is not None and not optimal_returns.empty:
            optimal_cumulative = (1 + optimal_returns).cumprod()
            ax.plot(optimal_cumulative.index, optimal_cumulative.values,
                   color='black', linewidth=2.5, label='Optimized Strategy', zorder=3)
        
        # Calculate and plot statistics
        if len(all_cumulative_returns) >= 5:
            # Find common date range
            common_start = max(cr.index.min() for cr in all_cumulative_returns)
            common_end = min(cr.index.max() for cr in all_cumulative_returns)
            
            # Align all series to common date range
            aligned_series = []
            for cr in all_cumulative_returns:
                aligned = cr.loc[common_start:common_end]
                if len(aligned) > 10:  # Only include if sufficient data
                    aligned_series.append(aligned)
            
            if len(aligned_series) >= 5:
                # Create a DataFrame with all aligned series
                # Reset index to avoid duplicate labels
                aligned_data = {}
                for i, series in enumerate(aligned_series):
                    # Reset index to avoid duplicate labels
                    series_reset = series.reset_index(drop=True)
                    aligned_data[f'trial_{i}'] = series_reset
                
                aligned_df = pd.DataFrame(aligned_data)
                
                # Calculate percentiles
                percentile_5 = aligned_df.quantile(0.05, axis=1)
                percentile_95 = aligned_df.quantile(0.95, axis=1)
                median = aligned_df.median(axis=1)
                
                # Create a common index for plotting
                common_index = aligned_series[0].index[:len(aligned_df)]
                
                # Plot confidence bands
                ax.fill_between(common_index, percentile_5.values, percentile_95.values,
                               alpha=0.2, color='blue', label='90% Confidence Band', zorder=2)
                
                # Plot median line
                ax.plot(common_index, median.values, 
                       color='blue', linewidth=1.5, linestyle='--', 
                       label='Median Trial Performance', zorder=2)
        
        # Formatting
        ax.set_title(f'Optimization Trial P&L Curves: {scenario_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Returns', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Add statistics text box
        stats_text = f"""Trial Statistics:
Total Trials: {len(trial_returns_data)}
Best Trial Value: {max(trial_values):.3f}
Worst Trial Value: {min(trial_values):.3f}
Median Trial Value: {np.median(trial_values):.3f}
Std Dev of Values: {np.std(trial_values):.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10, fontfamily='monospace')
        
        # Save the plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trial_pnl_curves_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        os.makedirs("plots", exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Trial P&L curves plot saved to: {filepath}")
        
        # Check if advanced parameter analysis is enabled
        advanced_reporting_config = self.global_config.get('advanced_reporting_config', {})
        if advanced_reporting_config.get('enable_advanced_parameter_analysis', False):
            # Create comprehensive parameter impact visualizations
            _plot_parameter_impact_analysis(self, scenario_name, best_trial_obj, timestamp)
        else:
            logger.info("Advanced parameter analysis is disabled. Skipping hyperparameter correlation/sensitivity analysis.")
        
    except Exception as e:
        logger.error(f"Error creating trial P&L visualization: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _plot_parameter_impact_analysis(self, scenario_name: str, best_trial_obj, timestamp: str):
    """
    Create comprehensive parameter impact and stability visualizations.
    
    Args:
        scenario_name: Name of the scenario
        best_trial_obj: Best trial object from Optuna containing the study
        timestamp: Timestamp for file naming
    """
    logger = self.logger
    
    try:
        if not hasattr(best_trial_obj, 'study') or best_trial_obj.study is None:
            logger.warning("No study object found. Cannot create parameter impact analysis.")
            return
            
        study = best_trial_obj.study
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) < 10:
            logger.warning(f"Only {len(completed_trials)} completed trials. Need at least 10 for meaningful parameter analysis.")
            return
            
        logger.info(f"Creating parameter impact analysis with {len(completed_trials)} trials...")
        
        # Extract parameter and performance data
        param_data = []
        for trial in completed_trials:
            trial_params = trial.params.copy()
            
            # Handle multi-objective optimization
            try:
                trial_value = trial.value
            except RuntimeError:
                # Multi-objective optimization: use first objective value
                trial_value = trial.values[0] if trial.values else 0.0
            
            trial_params['objective_value'] = trial_value
            trial_params['trial_number'] = trial.number
            param_data.append(trial_params)
        
        df = pd.DataFrame(param_data)
        
        # Get parameter names (exclude objective_value and trial_number)
        param_names = [col for col in df.columns if col not in ['objective_value', 'trial_number']]
        
        if len(param_names) < 2:
            logger.warning("Need at least 2 parameters for meaningful analysis.")
            return
            
        # Create parameter heatmaps
        _create_parameter_heatmaps(self, df, param_names, scenario_name, timestamp)
        
        # Create parameter sensitivity analysis
        _create_parameter_sensitivity_analysis(self, df, param_names, scenario_name, timestamp)
        
        # Create parameter stability analysis
        _create_parameter_stability_analysis(self, df, param_names, scenario_name, timestamp)
        
        # Create parameter correlation analysis
        _create_parameter_correlation_analysis(self, df, param_names, scenario_name, timestamp)
        
        # Create parameter importance ranking
        _create_parameter_importance_ranking(self, df, param_names, scenario_name, timestamp)
        
        # Create parameter robustness analysis
        _create_parameter_robustness_analysis(self, df, param_names, scenario_name, timestamp)
        
        logger.info("Parameter impact analysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Error creating parameter impact analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _plot_monte_carlo_robustness_analysis(self, scenario_name: str, scenario_config: dict, 
                                        optimal_params: dict, monthly_data, daily_data, rets_full):
    """
    STAGE 2 MONTE CARLO: Comprehensive stress testing after optimization completes.
    
    This creates Monte Carlo robustness analysis by running the optimized strategy with different levels
    of synthetic data replacement to assess strategy robustness to market conditions.
    
    MONTE CARLO TWO-STAGE PROCESS:
    1. Stage 1 (DURING OPTIMIZATION): Lightweight MC during WFO test phases
       - Single replacement percentage for parameter robustness testing
       - Applied in evaluation_logic.py during optimization trials
       
    2. Stage 2 (HERE): Full MC stress testing for comprehensive strategy assessment
       - Multiple replacement levels (5%, 7.5%, 10%, 12.5%, 15%) 
       - Multiple simulations per level (20+ per level)
       - Generates comprehensive robustness analysis plots
       - Only runs ONCE after optimization finds optimal parameters
    
    Args:
        scenario_name: Name of the scenario
        scenario_config: Original scenario configuration
        optimal_params: Best parameters found during optimization
        monthly_data: Monthly data for backtesting
        daily_data: Daily data for backtesting
        rets_full: Full returns data
    """
    logger = self.logger
    
    try:
        logger.info(f"Stage 2 MC: Starting comprehensive stress testing for {scenario_name}...")
        
        # Check if synthetic data generation is available
        monte_carlo_config = self.global_config.get('monte_carlo_config', {})
        if not monte_carlo_config.get('enable_synthetic_data', False):
            logger.warning("Stage 2 MC: Synthetic data generation is disabled. Cannot create robustness analysis.")
            return
            
        # Check if Stage 2 stress testing is enabled
        if not monte_carlo_config.get('enable_stage2_stress_testing', True):
            logger.info("Stage 2 MC: Stage 2 stress testing is disabled for faster optimization. Skipping robustness analysis.")
            return
            
        from ..monte_carlo.asset_replacement import AssetReplacementManager
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        
        # Replacement percentages to test (based on industry best practices for portfolio stress testing)
        replacement_percentages = [0.05, 0.075, 0.10, 0.125, 0.15]  # 5%, 7.5%, 10%, 12.5%, 15%
        num_simulations_per_level = 20  # Number of simulations per replacement level
        
        # Colors for different replacement levels
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
        
        # Create optimized scenario config
        optimized_scenario = scenario_config.copy()
        optimized_scenario["strategy_params"] = optimal_params
        
        # Store results for each replacement level
        simulation_results = {}
        
        # Calculate total work for progress bar
        total_simulations = len(replacement_percentages) * num_simulations_per_level
        
        # Create progress bar for Stage 2 Monte Carlo
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]Stage 2 Monte Carlo Stress Testing...", total=total_simulations)
            
            for i, replacement_pct in enumerate(replacement_percentages):
                logger.debug(f"Stage 2 MC: Running simulations with {replacement_pct:.1%} synthetic data replacement...")
                
                # Configure asset replacement manager for this level
                mc_config = monte_carlo_config.copy()
                mc_config['replacement_percentage'] = replacement_pct
                mc_config['stage1_optimization'] = False  # Full validation for Stage 2
                
                # Enable full validation and multiple attempts for Stage 2 stress testing
                mc_config['generation_config'] = {
                    'buffer_multiplier': 1.2,
                    'max_attempts': 3,
                    'validation_tolerance': 0.3
                }
                
                mc_config['validation_config'] = {
                    'enable_validation': True,
                    'tolerance': 0.4,
                    'ks_test_pvalue_threshold': 0.05,
                    'autocorr_max_deviation': 0.15,
                    'volatility_clustering_threshold': 0.02,
                    'tail_index_tolerance': 0.3
                }
                
                level_results = []
                
                # Update progress description for current replacement level
                progress.update(task, description=f"[cyan]Stage 2 MC: {replacement_pct:.1%} replacement ({i+1}/{len(replacement_percentages)})")
            
            for sim_num in range(num_simulations_per_level):
                try:
                    # Create asset replacement manager with different seed for each simulation
                    asset_replacement_manager = AssetReplacementManager(mc_config)
                    
                    # Generate synthetic data for this simulation
                    universe = scenario_config.get('universe', self.global_config.get('universe', []))
                    
                    # Convert daily data to expected format
                    daily_data_dict = {}
                    logger.debug(f"Daily data columns: {daily_data.columns}")
                    logger.debug(f"Daily data shape: {daily_data.shape}")
                    
                    if isinstance(daily_data.columns, pd.MultiIndex):
                        logger.debug(f"MultiIndex levels: {daily_data.columns.names}")
                        logger.debug(f"Level 0 values: {daily_data.columns.get_level_values(0).unique()}")
                        logger.debug(f"Level 1 values: {daily_data.columns.get_level_values(1).unique()}")
                        
                        # Handle MultiIndex columns (Ticker, Field) structure
                        for ticker in universe:
                            if ticker in daily_data.columns.get_level_values(0):  # Check first level
                                ticker_data = daily_data.xs(ticker, level=0, axis=1, drop_level=True)
                                if not ticker_data.empty:
                                    daily_data_dict[ticker] = ticker_data
                                    logger.debug(f"Stage 2 MC: Added {ticker} data with shape: {ticker_data.shape}")
                            elif ticker in daily_data.columns.get_level_values(1):  # Check second level
                                # Handle (Field, Ticker) structure
                                ticker_columns = [col for col in daily_data.columns if col[1] == ticker]
                                if ticker_columns:
                                    ticker_data = daily_data[ticker_columns]
                                    ticker_data.columns = [col[0] for col in ticker_columns]  # Keep only field names
                                    daily_data_dict[ticker] = ticker_data
                                    logger.debug(f"Stage 2 MC: Added {ticker} data with shape: {ticker_data.shape}")
                    else:
                        # Handle simple column structure - assume it's close prices
                        for ticker in universe:
                            if ticker in daily_data.columns:
                                ticker_data = pd.DataFrame({
                                    'Open': daily_data[ticker],
                                    'High': daily_data[ticker],
                                    'Low': daily_data[ticker],
                                    'Close': daily_data[ticker]
                                }, index=daily_data.index)
                                daily_data_dict[ticker] = ticker_data
                                logger.debug(f"Stage 2 MC: Added {ticker} data with shape: {ticker_data.shape}")
                    
                    logger.debug(f"Stage 2 MC: Created daily_data_dict with {len(daily_data_dict)} assets")
                    
                    # Generate synthetic data for full historical period
                    # Use all available data for stress testing, but keep minimum historical data for parameter estimation
                    total_days = len(daily_data)
                    min_historical_days = monte_carlo_config.get('min_historical_observations', 252)  # 1 year minimum
                    
                    # Start the test period after minimum historical data requirement
                    test_start_idx = min_historical_days
                    test_start = daily_data.index[test_start_idx]
                    test_end = daily_data.index[-1]
                    test_period_days = total_days - test_start_idx
                    
                    logger.debug(f"Stage 2 MC: Monte Carlo stress test period: {test_start} to {test_end} ({test_period_days} days, full historical data after {min_historical_days} day parameter estimation period)")
                    synthetic_data, replacement_info = asset_replacement_manager.create_monte_carlo_dataset(
                        original_data=daily_data_dict,
                        universe=universe,
                        test_start=test_start,
                        test_end=test_end,
                        run_id=f"robustness_{replacement_pct:.0%}_{sim_num}",
                        random_seed=42 + i * 100 + sim_num  # Deterministic but different seeds
                    )
                    
                    # Apply synthetic data to returns
                    modified_rets = rets_full.copy()
                    for ticker in replacement_info.selected_assets:
                        if ticker in synthetic_data and ticker in modified_rets.columns:
                            synthetic_returns = synthetic_data[ticker]['Close'].pct_change(fill_method=None).fillna(0)
                            # Replace with synthetic returns where available
                            common_dates = modified_rets.index.intersection(synthetic_returns.index)
                            if len(common_dates) > 0:
                                modified_rets.loc[common_dates, ticker] = synthetic_returns.loc[common_dates]
                    
                    # Run backtest with synthetic data
                    sim_returns = self.run_scenario(
                        optimized_scenario, 
                        monthly_data, 
                        daily_data, 
                        modified_rets, 
                        verbose=False
                    )
                    
                    if sim_returns is not None and not sim_returns.empty:
                        level_results.append(sim_returns)
                        logger.debug(f"Stage 2 MC: Simulation {sim_num} completed for {replacement_pct:.1%} replacement")
                        
                except Exception as e:
                    logger.warning(f"Stage 2 MC: Simulation {sim_num} failed for {replacement_pct:.1%} replacement: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                finally:
                    # Update progress bar regardless of success/failure
                    progress.advance(task)
            
                if level_results:
                    simulation_results[replacement_pct] = level_results
                    logger.info(f"Stage 2 MC: Completed {len(level_results)} simulations for {replacement_pct:.1%} replacement level")
                else:
                    logger.warning(f"Stage 2 MC: No successful simulations for {replacement_pct:.1%} replacement level")
        
        # Create the Monte Carlo plot
        if simulation_results:
            _create_monte_carlo_robustness_plot(self, scenario_name, simulation_results, 
                                              replacement_percentages, colors, optimal_params)
        else:
            logger.warning("Stage 2 MC: No simulation results available for Monte Carlo robustness plot")
            
    except Exception as e:
        logger.error(f"Stage 2 MC: Error in Monte Carlo robustness analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _create_monte_carlo_robustness_plot(self, scenario_name: str, simulation_results: dict, 
                                      replacement_percentages: list, colors: list, optimal_params: dict):
    """
    Create the actual Monte Carlo robustness plot with different colors for each replacement level.
    
    Args:
        scenario_name: Name of the scenario
        simulation_results: Dictionary of simulation results by replacement percentage
        replacement_percentages: List of replacement percentages
        colors: List of colors for each replacement level
        optimal_params: Optimal parameters used
    """
    logger = self.logger
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax, ax_params) = plt.subplots(2, 1, figsize=(16, 12), 
                                           gridspec_kw={'height_ratios': [4, 1]})
        
        # Plot results for each replacement level
        for i, replacement_pct in enumerate(replacement_percentages):
            if replacement_pct not in simulation_results:
                continue
                
            level_results = simulation_results[replacement_pct]
            color = colors[i % len(colors)]
            
            # Plot each simulation in this level
            for j, sim_returns in enumerate(level_results):
                cumulative_returns = (1 + sim_returns).cumprod()
                
                # Use different alpha for different simulations within the same level
                alpha = 0.3 if j > 0 else 0.8  # First simulation more opaque
                
                # Label only the first simulation of each level
                label = f"{replacement_pct:.0%} Synthetic Replacement" if j == 0 else None
                
                ax.plot(cumulative_returns.index, cumulative_returns.values, 
                       color=color, alpha=alpha, linewidth=1.5 if j == 0 else 1.0, label=label)
        
        # Formatting
        ax.set_title(f"Monte Carlo Robustness Analysis: {scenario_name}\n"
                    f"Strategy Performance with Progressive Synthetic Data Replacement", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Returns", fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Create parameter information table in the bottom subplot
        ax_params.axis('off')  # Turn off axis for the parameter subplot
        
        # Create parameter text
        param_text = "Optimal Strategy Parameters:\n"
        for param, value in optimal_params.items():
            if isinstance(value, float):
                param_text += f"  {param}: {value:.3f}\n"
            else:
                param_text += f"  {param}: {value}\n"
        
        # Add statistics summary to parameter text
        param_text += "\nSimulation Summary:\n"
        for replacement_pct in replacement_percentages:
            if replacement_pct in simulation_results:
                count = len(simulation_results[replacement_pct])
                param_text += f"  {replacement_pct:.0%} Replacement: {count} simulations\n"
        
        # Display parameter information in bottom subplot
        ax_params.text(0.02, 0.98, param_text, transform=ax_params.transAxes, 
                      verticalalignment='top', fontsize=11, fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        # Adjust layout and save the plot
        plt.tight_layout()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monte_carlo_robustness_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        os.makedirs("plots", exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Monte Carlo robustness plot saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error creating Monte Carlo robustness plot: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _create_parameter_heatmaps(self, df: pd.DataFrame, param_names: list, scenario_name: str, timestamp: str):
    """Create 2D parameter heatmaps showing performance across parameter combinations."""
    logger = self.logger
    
    try:
        # Create heatmaps for all parameter pairs
        num_params = len(param_names)
        if num_params < 2:
            return
            
        # Calculate number of subplots needed
        num_pairs = min(6, num_params * (num_params - 1) // 2)  # Limit to 6 most important pairs
        
        if num_pairs == 0:
            return
            
        # Create figure with subplots
        cols = min(3, num_pairs)
        rows = (num_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get parameter pairs sorted by importance
        param_pairs = []
        for i in range(num_params):
            for j in range(i + 1, num_params):
                param1, param2 = param_names[i], param_names[j]
                # Calculate correlation with objective as a proxy for importance
                corr1 = abs(df[param1].corr(df['objective_value']))
                corr2 = abs(df[param2].corr(df['objective_value']))
                importance = corr1 + corr2
                param_pairs.append((param1, param2, importance))
        
        # Sort by importance and take top pairs
        param_pairs.sort(key=lambda x: x[2], reverse=True)
        param_pairs = param_pairs[:num_pairs]
        
        for idx, (param1, param2, _) in enumerate(param_pairs):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Create pivot table for heatmap
            try:
                # Bin parameters if they have too many unique values
                param1_values = df[param1].copy()
                param2_values = df[param2].copy()
                
                if len(param1_values.unique()) > 10:
                    param1_values = pd.cut(param1_values, bins=10, precision=2)
                if len(param2_values.unique()) > 10:
                    param2_values = pd.cut(param2_values, bins=10, precision=2)
                
                # Create pivot table
                pivot_data = df.groupby([param1_values, param2_values], observed=True)['objective_value'].mean().unstack(fill_value=np.nan)
                
                # Create heatmap
                im = sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                               ax=ax, cbar_kws={'label': 'Objective Value'})
                
                ax.set_title(f'{param1} vs {param2}', fontsize=12, fontweight='bold')
                ax.set_xlabel(param2, fontsize=10)
                ax.set_ylabel(param1, fontsize=10)
                
                # Rotate labels if needed
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)
                
            except Exception as e:
                logger.debug(f"Failed to create heatmap for {param1} vs {param2}: {e}")
                ax.text(0.5, 0.5, f'Heatmap failed\nfor {param1} vs {param2}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        # Hide unused subplots
        for idx in range(num_pairs, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            ax.set_visible(False)
        
        plt.suptitle(f'Parameter Heatmaps: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"parameter_heatmaps_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Parameter heatmaps saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error creating parameter heatmaps: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _create_parameter_sensitivity_analysis(self, df: pd.DataFrame, param_names: list, scenario_name: str, timestamp: str):
    """Create parameter sensitivity analysis showing how performance changes with each parameter."""
    logger = self.logger
    
    try:
        num_params = len(param_names)
        if num_params == 0:
            return
            
        # Create subplots
        cols = min(3, num_params)
        rows = (num_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, param in enumerate(param_names):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            try:
                # Create scatter plot with trend line
                x = df[param]
                y = df['objective_value']
                
                # Scatter plot
                ax.scatter(x, y, alpha=0.6, s=30, color='steelblue')
                
                # Add trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, linewidth=2)
                
                # Add moving average if enough points
                if len(x) > 20:
                    sorted_data = df[[param, 'objective_value']].sort_values(param)
                    window_size = max(5, len(sorted_data) // 10)
                    rolling_mean = sorted_data['objective_value'].rolling(window=window_size, center=True).mean()
                    ax.plot(sorted_data[param], rolling_mean, color='orange', linewidth=2, 
                           label=f'Moving Average (window={window_size})')
                
                # Calculate and display correlation
                correlation = x.corr(y)
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(param, fontsize=10)
                ax.set_ylabel('Objective Value', fontsize=10)
                ax.set_title(f'Sensitivity: {param}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if len(x) > 20:
                    ax.legend()
                
            except Exception as e:
                logger.debug(f"Failed to create sensitivity plot for {param}: {e}")
                ax.text(0.5, 0.5, f'Sensitivity plot failed\nfor {param}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        # Hide unused subplots
        for idx in range(num_params, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            ax.set_visible(False)
        
        plt.suptitle(f'Parameter Sensitivity Analysis: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"parameter_sensitivity_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Parameter sensitivity analysis saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error creating parameter sensitivity analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _create_parameter_stability_analysis(self, df: pd.DataFrame, param_names: list, scenario_name: str, timestamp: str):
    """Create parameter stability analysis showing stable vs unstable parameter regions."""
    logger = self.logger
    
    try:
        num_params = len(param_names)
        if num_params == 0:
            return
            
        # Create figure with two types of plots
        fig = plt.figure(figsize=(16, 10))
        
        # Top subplot: Parameter stability over trials
        ax1 = plt.subplot(2, 2, (1, 2))
        
        # Plot parameter values over trial numbers
        for param in param_names[:6]:  # Limit to 6 parameters for readability
            values = df[param].values
            trial_numbers = df['trial_number'].values
            ax1.plot(trial_numbers, values, marker='o', markersize=3, alpha=0.7, label=param)
        
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('Parameter Values (Normalized)', fontsize=12)
        ax1.set_title('Parameter Evolution Over Trials', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Bottom left: Parameter variance analysis
        ax2 = plt.subplot(2, 2, 3)
        
        param_variances = []
        param_means = []
        for param in param_names:
            variance = df[param].var()
            mean = df[param].mean()
            param_variances.append(variance)
            param_means.append(mean)
        
        # Normalize variances for comparison
        max_var = max(param_variances) if param_variances else 1
        normalized_variances = [v / max_var for v in param_variances]
        
        bars = ax2.bar(range(len(param_names)), normalized_variances, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Parameters', fontsize=12)
        ax2.set_ylabel('Normalized Variance', fontsize=12)
        ax2.set_title('Parameter Stability (Lower = More Stable)', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(param_names)))
        ax2.set_xticklabels(param_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, var) in enumerate(zip(bars, normalized_variances)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{var:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Bottom right: Performance stability regions
        ax3 = plt.subplot(2, 2, 4)
        
        # Calculate performance quartiles
        q1 = df['objective_value'].quantile(0.25)
        q3 = df['objective_value'].quantile(0.75)
        
        # Color points based on performance quartiles
        colors = []
        for val in df['objective_value']:
            if val >= q3:
                colors.append('green')  # Top quartile
            elif val <= q1:
                colors.append('red')    # Bottom quartile
            else:
                colors.append('orange') # Middle quartiles
        
        # Use first two most important parameters for 2D plot
        if len(param_names) >= 2:
            param1, param2 = param_names[0], param_names[1]
            scatter = ax3.scatter(df[param1], df[param2], c=colors, alpha=0.6, s=50)
            ax3.set_xlabel(param1, fontsize=12)
            ax3.set_ylabel(param2, fontsize=12)
            ax3.set_title('Parameter Stability Regions', fontsize=12, fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Top Quartile'),
                Patch(facecolor='orange', label='Middle Quartiles'),
                Patch(facecolor='red', label='Bottom Quartile')
            ]
            ax3.legend(handles=legend_elements, loc='upper right')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Need at least 2 parameters\nfor stability regions', 
                    transform=ax3.transAxes, ha='center', va='center')
        
        plt.suptitle(f'Parameter Stability Analysis: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"parameter_stability_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Parameter stability analysis saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error creating parameter stability analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _create_parameter_correlation_analysis(self, df: pd.DataFrame, param_names: list, scenario_name: str, timestamp: str):
    """Create parameter correlation analysis showing how parameters interact with each other."""
    logger = self.logger
    
    try:
        if len(param_names) < 2:
            return
            
        # Create correlation matrix
        correlation_data = df[param_names + ['objective_value']].corr()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Full correlation matrix
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Right plot: Correlation with objective value
        obj_correlations = correlation_data['objective_value'].drop('objective_value')
        obj_correlations = obj_correlations.sort_values(key=abs, ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in obj_correlations.values]
        bars = ax2.barh(range(len(obj_correlations)), obj_correlations.values, color=colors, alpha=0.7)
        
        ax2.set_yticks(range(len(obj_correlations)))
        ax2.set_yticklabels(obj_correlations.index)
        ax2.set_xlabel('Correlation with Objective Value', fontsize=12)
        ax2.set_title('Parameter Importance by Correlation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, obj_correlations.values)):
            ax2.text(val + (0.01 if val > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)
        
        plt.suptitle(f'Parameter Correlation Analysis: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"parameter_correlation_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Parameter correlation analysis saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error creating parameter correlation analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _create_parameter_importance_ranking(self, df: pd.DataFrame, param_names: list, scenario_name: str, timestamp: str):
    """Create parameter importance ranking showing which parameters matter most."""
    logger = self.logger
    
    try:
        if len(param_names) == 0:
            return
            
        # Calculate multiple importance metrics
        importance_metrics = {}
        
        for param in param_names:
            # Correlation with objective
            correlation = abs(df[param].corr(df['objective_value']))
            
            # Variance (normalized)
            variance = df[param].var()
            max_var = df[param_names].var().max()
            normalized_variance = variance / max_var if max_var > 0 else 0
            
            # Range (normalized)
            param_range = df[param].max() - df[param].min()
            max_range = max([df[p].max() - df[p].min() for p in param_names])
            normalized_range = param_range / max_range if max_range > 0 else 0
            
            # Mutual information approximation (binned correlation)
            try:
                from scipy.stats import entropy
                # Bin the parameter values
                param_binned = pd.cut(df[param], bins=min(10, len(df[param].unique())), duplicates='drop')
                obj_binned = pd.cut(df['objective_value'], bins=10, duplicates='drop')
                
                # Calculate mutual information approximation
                joint_counts = pd.crosstab(param_binned, obj_binned)
                mutual_info = 0
                for i in range(joint_counts.shape[0]):
                    for j in range(joint_counts.shape[1]):
                        if joint_counts.iloc[i, j] > 0:
                            p_xy = joint_counts.iloc[i, j] / joint_counts.sum().sum()
                            p_x = joint_counts.iloc[i, :].sum() / joint_counts.sum().sum()
                            p_y = joint_counts.iloc[:, j].sum() / joint_counts.sum().sum()
                            if p_x > 0 and p_y > 0:
                                mutual_info += p_xy * np.log2(p_xy / (p_x * p_y))
                
                mutual_info = max(0, mutual_info)  # Ensure non-negative
            except:
                mutual_info = 0
            
            importance_metrics[param] = {
                'correlation': correlation,
                'variance': normalized_variance,
                'range': normalized_range,
                'mutual_info': mutual_info
            }
        
        # Create composite importance score
        composite_scores = {}
        for param in param_names:
            metrics = importance_metrics[param]
            # Weighted combination of metrics
            composite_score = (
                0.4 * metrics['correlation'] +
                0.2 * metrics['variance'] +
                0.2 * metrics['range'] +
                0.2 * metrics['mutual_info']
            )
            composite_scores[param] = composite_score
        
        # Sort by composite score
        sorted_params = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Composite importance ranking
        params, scores = zip(*sorted_params)
        colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
        bars1 = ax1.bar(range(len(params)), scores, color=colors, alpha=0.8)
        ax1.set_xlabel('Parameters', fontsize=12)
        ax1.set_ylabel('Composite Importance Score', fontsize=12)
        ax1.set_title('Parameter Importance Ranking', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(params)))
        ax1.set_xticklabels(params, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars1, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Correlation importance
        correlations = [importance_metrics[param]['correlation'] for param in params]
        bars2 = ax2.bar(range(len(params)), correlations, color='steelblue', alpha=0.8)
        ax2.set_xlabel('Parameters', fontsize=12)
        ax2.set_ylabel('Absolute Correlation', fontsize=12)
        ax2.set_title('Parameter Importance by Correlation', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(params)))
        ax2.set_xticklabels(params, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Variance importance
        variances = [importance_metrics[param]['variance'] for param in params]
        bars3 = ax3.bar(range(len(params)), variances, color='orange', alpha=0.8)
        ax3.set_xlabel('Parameters', fontsize=12)
        ax3.set_ylabel('Normalized Variance', fontsize=12)
        ax3.set_title('Parameter Importance by Variance', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(params)))
        ax3.set_xticklabels(params, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Radar chart of top 5 parameters
        top_5_params = params[:5]
        categories = ['Correlation', 'Variance', 'Range', 'Mutual Info']
        
        # Prepare data for radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        colors_radar = plt.cm.Set3(np.linspace(0, 1, len(top_5_params)))
        
        for i, param in enumerate(top_5_params):
            metrics = importance_metrics[param]
            values = [metrics['correlation'], metrics['variance'], metrics['range'], metrics['mutual_info']]
            values += values[:1]  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=param, color=colors_radar[i])
            ax4.fill(angles, values, alpha=0.25, color=colors_radar[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Top 5 Parameters - Multi-Metric Comparison', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Parameter Importance Analysis: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"parameter_importance_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Parameter importance analysis saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error creating parameter importance analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _create_parameter_robustness_analysis(self, df: pd.DataFrame, param_names: list, scenario_name: str, timestamp: str):
    """Create parameter robustness analysis showing stable vs unstable parameter regions."""
    logger = self.logger
    
    try:
        if len(param_names) == 0:
            return
            
        # Calculate performance statistics
        mean_performance = df['objective_value'].mean()
        std_performance = df['objective_value'].std()
        
        # Define robustness regions
        high_performance = mean_performance + 0.5 * std_performance
        low_performance = mean_performance - 0.5 * std_performance
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Robustness heatmap (top subplot)
        ax1 = plt.subplot(2, 2, (1, 2))
        
        if len(param_names) >= 2:
            # Use two most important parameters
            param1, param2 = param_names[0], param_names[1]
            
            # Create grid for interpolation
            x = df[param1].values
            y = df[param2].values
            z = df['objective_value'].values
            
            # Create regular grid
            xi = np.linspace(x.min(), x.max(), 50)
            yi = np.linspace(y.min(), y.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate
            from scipy.interpolate import griddata
            Zi = griddata((x, y), z, (Xi, Yi), method='cubic', fill_value=np.nan)
            
            # Create contour plot
            contour = ax1.contourf(Xi, Yi, Zi, levels=20, cmap='viridis', alpha=0.8)
            plt.colorbar(contour, ax=ax1, label='Objective Value')
            
            # Overlay scatter points
            scatter = ax1.scatter(x, y, c=z, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Add robustness regions
            ax1.contour(Xi, Yi, Zi, levels=[high_performance], colors='red', linewidths=2, linestyles='--')
            ax1.contour(Xi, Yi, Zi, levels=[low_performance], colors='blue', linewidths=2, linestyles='--')
            
            ax1.set_xlabel(param1, fontsize=12)
            ax1.set_ylabel(param2, fontsize=12)
            ax1.set_title(f'Parameter Robustness Landscape: {param1} vs {param2}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Need at least 2 parameters\nfor robustness landscape', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=14)
        
        # 2. Robustness by parameter value (bottom left)
        ax2 = plt.subplot(2, 2, 3)
        
        # Calculate robustness for each parameter
        robustness_scores = {}
        for param in param_names:
            # Calculate local standard deviation
            param_values = df[param].values
            obj_values = df['objective_value'].values
            
            # Sort by parameter value
            sorted_indices = np.argsort(param_values)
            sorted_param = param_values[sorted_indices]
            sorted_obj = obj_values[sorted_indices]
            
            # Calculate rolling standard deviation
            window_size = max(5, len(sorted_obj) // 10)
            rolling_std = pd.Series(sorted_obj).rolling(window=window_size, center=True).std()
            
            # Average rolling standard deviation as robustness score (lower = more robust)
            robustness_score = rolling_std.mean()
            robustness_scores[param] = robustness_score
        
        # Sort by robustness (lower is better)
        sorted_robustness = sorted(robustness_scores.items(), key=lambda x: x[1])
        params_rob, scores_rob = zip(*sorted_robustness)
        
        # Create bar chart
        colors = ['green' if score < std_performance else 'red' for score in scores_rob]
        bars = ax2.bar(range(len(params_rob)), scores_rob, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Parameters', fontsize=12)
        ax2.set_ylabel('Robustness Score (Lower = More Robust)', fontsize=12)
        ax2.set_title('Parameter Robustness Ranking', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(params_rob)))
        ax2.set_xticklabels(params_rob, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add threshold line
        ax2.axhline(y=std_performance, color='orange', linestyle='--', linewidth=2, 
                   label=f'Performance Std: {std_performance:.3f}')
        ax2.legend()
        
        # 3. Parameter stability over performance quantiles (bottom right)
        ax3 = plt.subplot(2, 2, 4)
        
        # Divide data into performance quantiles
        quantiles = [0.2, 0.4, 0.6, 0.8, 1.0]
        quantile_labels = ['Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%']
        
        # Calculate parameter stability for each quantile
        if len(param_names) >= 1:
            param_to_analyze = param_names[0]  # Use most important parameter
            
            quantile_stabilities = []
            for i, q in enumerate(quantiles):
                if i == 0:
                    mask = df['objective_value'] <= df['objective_value'].quantile(q)
                else:
                    mask = ((df['objective_value'] > df['objective_value'].quantile(quantiles[i-1])) & 
                           (df['objective_value'] <= df['objective_value'].quantile(q)))
                
                if mask.sum() > 0:
                    stability = df[mask][param_to_analyze].std()
                    quantile_stabilities.append(stability)
                else:
                    quantile_stabilities.append(0)
            
            # Create bar chart
            colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(quantile_labels)))
            bars = ax3.bar(range(len(quantile_labels)), quantile_stabilities, color=colors, alpha=0.8)
            
            ax3.set_xlabel('Performance Quantiles', fontsize=12)
            ax3.set_ylabel(f'{param_to_analyze} Stability (Std Dev)', fontsize=12)
            ax3.set_title(f'Parameter Stability Across Performance Levels', fontsize=12, fontweight='bold')
            ax3.set_xticks(range(len(quantile_labels)))
            ax3.set_xticklabels(quantile_labels, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, stability in zip(bars, quantile_stabilities):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{stability:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No parameters available\nfor stability analysis', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=14)
        
        plt.suptitle(f'Parameter Robustness Analysis: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"parameter_robustness_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Parameter robustness analysis saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error creating parameter robustness analysis: {e}")
        import traceback
        logger.debug(traceback.format_exc())

def _display_constraint_violation_warning(self, console: Console):
    """Display a prominent constraint violation warning."""
    from rich.panel import Panel
    from rich.text import Text
    
    warning_text = Text()
    warning_text.append("🚨 CONSTRAINT VIOLATION DETECTED! 🚨\n\n", style="bold red")
    
    for name, result_data in self.results.items():
        constraint_status = result_data.get("constraint_status", "UNKNOWN")
        if constraint_status == "VIOLATED":
            warning_text.append(f"Strategy: {result_data.get('display_name', name)}\n", style="bold yellow")
            warning_text.append(f"Status: {constraint_status}\n", style="red")
            warning_text.append(f"Issue: {result_data.get('constraint_message', 'Unknown constraint violation')}\n", style="red")
            
            violations = result_data.get("constraint_violations", [])
            if violations:
                warning_text.append("Violations:\n", style="bold red")
                for i, violation in enumerate(violations, 1):
                    warning_text.append(f"  {i}. {violation}\n", style="red")
            
            warning_text.append("\n")
    
    warning_text.append("💡 RECOMMENDATIONS:\n", style="bold cyan")
    warning_text.append("• Relax the constraint limits (e.g., increase max volatility)\n", style="cyan")
    warning_text.append("• Modify strategy parameters to reduce risk\n", style="cyan")
    warning_text.append("• Consider different optimization targets\n", style="cyan")
    warning_text.append("• Review strategy configuration and universe\n", style="cyan")
    
    panel = Panel(
        warning_text,
        title="⚠️  OPTIMIZATION CONSTRAINT FAILURE ⚠️",
        title_align="center",
        border_style="red",
        expand=False
    )
    
    console.print(panel)
    console.print()  # Add spacing