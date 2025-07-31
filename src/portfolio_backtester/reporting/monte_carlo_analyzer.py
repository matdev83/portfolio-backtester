import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from portfolio_backtester.optimization.results import OptimizationResult

# Import the parameter analysis function
from .parameter_analysis import _plot_parameter_impact_analysis

logger = logging.getLogger(__name__)

def plot_stability_measures(backtester, scenario_name: str, optimization_result: OptimizationResult, optimal_returns: pd.Series):
    """
    Create a Monte Carlo-style visualization showing P&L curves from all optimization trials.
    
    Args:
        scenario_name: Name of the scenario
        optimization_result: The result object from the optimization run
        optimal_returns: Returns series from the final optimized strategy
    """
    logger = backtester.logger
    
    try:
        optimization_history = optimization_result.optimization_history
        if not optimization_history or len(optimization_history) < 2:
            logger.warning(f"Only {len(optimization_history)} completed trials found. Need at least 2 for meaningful visualization.")
            return
            
        trial_returns_data = []
        for trial in optimization_history:
            if 'metrics' in trial and 'trial_returns' in trial['metrics']:
                try:
                    returns_dict = trial['metrics']['trial_returns']
                    dates = pd.to_datetime(returns_dict['dates'])
                    returns = pd.Series(returns_dict['returns'], index=dates)
                    
                    trial_value = trial['objective_value']
                    
                    trial_returns_data.append({
                        'trial_number': trial['evaluation'],
                        'returns': returns,
                        'params': trial['parameters'],
                        'value': trial_value
                    })
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Failed to extract returns for trial {trial['evaluation']}: {e}")
                    continue
        
        if len(trial_returns_data) < 2:
            if logger.isEnabledFor(logging.WARNING):

                logger.warning(f"Only {len(trial_returns_data)} trials have stored returns data. Cannot create visualization.")
            return
            
        if logger.isEnabledFor(logging.INFO):
            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f"Creating Monte Carlo-style trial P&L visualization with {len(trial_returns_data)} trials...")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        all_cumulative_returns = []
        trial_values = []
        
        for trial_data in trial_returns_data:
            returns = trial_data['returns']
            cumulative_returns = (1 + returns).cumprod()
            all_cumulative_returns.append(cumulative_returns)
            trial_values.append(trial_data['value'])
            
            ax.plot(cumulative_returns.index, cumulative_returns.values, 
                   color='lightgray', alpha=0.3, linewidth=0.8, zorder=1)
        
        if optimal_returns is not None and not optimal_returns.empty:
            optimal_cumulative = (1 + optimal_returns).cumprod()
            ax.plot(optimal_cumulative.index, optimal_cumulative.values,
                   color='black', linewidth=2.5, label='Optimized Strategy', zorder=3)
        
        if len(all_cumulative_returns) >= 5:
            common_start = max(cr.index.min() for cr in all_cumulative_returns)
            common_end = min(cr.index.max() for cr in all_cumulative_returns)
            
            aligned_series = []
            for cr in all_cumulative_returns:
                aligned = cr.loc[common_start:common_end]
                if len(aligned) > 10:
                    aligned_series.append(aligned)
            
            if len(aligned_series) >= 5:
                aligned_data = {}
                for i, series in enumerate(aligned_series):
                    series_reset = series.reset_index(drop=True)
                    aligned_data[f'trial_{i}'] = series_reset
                
                aligned_df = pd.DataFrame(aligned_data)
                
                percentile_5 = aligned_df.quantile(0.05, axis=1)
                percentile_95 = aligned_df.quantile(0.95, axis=1)
                median = aligned_df.median(axis=1)
                
                common_index = aligned_series[0].index[:len(aligned_df)]
                
                ax.fill_between(common_index, percentile_5.values, percentile_95.values,
                               alpha=0.2, color='blue', label='90% Confidence Band', zorder=2)
                
                ax.plot(common_index, median.values, 
                       color='blue', linewidth=1.5, linestyle='--', 
                       label='Median Trial Performance', zorder=2)
        
        ax.set_title(f'Optimization Trial P&L Curves: {scenario_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Returns', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        stats_text = f"""Trial Statistics:
Total Trials: {len(trial_returns_data)}
Best Trial Value: {max(trial_values):.3f}
Worst Trial Value: {min(trial_values):.3f}
Median Trial Value: {np.median(trial_values):.3f}
Std Dev of Values: {np.std(trial_values):.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10, fontfamily='monospace')
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trial_pnl_curves_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        os.makedirs("plots", exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if logger.isEnabledFor(logging.INFO):
            if logger.isEnabledFor(logging.INFO):

                logger.info(f"Trial P&L curves plot saved to: {filepath}")
        
        advanced_reporting_config = backtester.global_config.get('advanced_reporting_config', {})
        if advanced_reporting_config.get('enable_advanced_parameter_analysis', False):
            _plot_parameter_impact_analysis(backtester, scenario_name, best_trial_obj, timestamp)
        else:
            logger.info("Advanced parameter analysis is disabled. Skipping hyperparameter correlation/sensitivity analysis.")
        
    except Exception as e:
        logger.error(f"Error creating trial P&L visualization: {e}")
        import traceback
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())
