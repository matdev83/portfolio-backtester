"""
Optimizer Report Generator

This module creates comprehensive markdown reports for optimization results,
including performance metrics analysis, parameter importance, and professional
interpretation of results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class OptimizerReportGenerator:
    """Generates comprehensive markdown reports for optimization results."""
    
    def __init__(self, base_reports_dir: str = "data/reports"):
        self.base_reports_dir = Path(base_reports_dir)
        self.base_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metric interpretations and ranges
        self.metric_interpretations = {
            'Sharpe': {
                'ranges': [
                    (float('-inf'), 0, 'Poor', 'Negative risk-adjusted returns indicate the strategy underperforms risk-free assets'),
                    (0, 0.5, 'Below Average', 'Low risk-adjusted returns suggest inadequate compensation for risk taken'),
                    (0.5, 1.0, 'Average', 'Moderate risk-adjusted returns indicate reasonable performance'),
                    (1.0, 1.5, 'Good', 'Strong risk-adjusted returns demonstrate effective risk management'),
                    (1.5, 2.0, 'Very Good', 'Excellent risk-adjusted returns indicate superior strategy performance'),
                    (2.0, float('inf'), 'Exceptional', 'Outstanding risk-adjusted returns suggest exceptional strategy design')
                ],
                'description': 'Sharpe Ratio measures risk-adjusted returns by comparing excess returns to volatility',
                'formula': '(Portfolio Return - Risk-Free Rate) / Portfolio Volatility',
                'interpretation_note': 'Higher values indicate better risk-adjusted performance'
            },
            'Calmar': {
                'ranges': [
                    (float('-inf'), 0, 'Poor', 'Negative Calmar ratio indicates poor drawdown management'),
                    (0, 0.5, 'Below Average', 'Low Calmar ratio suggests inadequate returns relative to maximum drawdown'),
                    (0.5, 1.0, 'Average', 'Moderate Calmar ratio indicates reasonable drawdown-adjusted returns'),
                    (1.0, 2.0, 'Good', 'Strong Calmar ratio demonstrates effective drawdown control'),
                    (2.0, 3.0, 'Very Good', 'Excellent Calmar ratio indicates superior downside risk management'),
                    (3.0, float('inf'), 'Exceptional', 'Outstanding Calmar ratio suggests exceptional risk control')
                ],
                'description': 'Calmar Ratio measures return relative to maximum drawdown',
                'formula': 'Annualized Return / Maximum Drawdown',
                'interpretation_note': 'Higher values indicate better drawdown-adjusted performance'
            },
            'Sortino': {
                'ranges': [
                    (float('-inf'), 0, 'Poor', 'Negative Sortino ratio indicates poor downside risk management'),
                    (0, 0.5, 'Below Average', 'Low Sortino ratio suggests inadequate returns relative to downside risk'),
                    (0.5, 1.0, 'Average', 'Moderate Sortino ratio indicates reasonable downside-adjusted returns'),
                    (1.0, 1.5, 'Good', 'Strong Sortino ratio demonstrates effective downside risk control'),
                    (1.5, 2.0, 'Very Good', 'Excellent Sortino ratio indicates superior downside risk management'),
                    (2.0, float('inf'), 'Exceptional', 'Outstanding Sortino ratio suggests exceptional downside protection')
                ],
                'description': 'Sortino Ratio measures return relative to downside deviation',
                'formula': '(Portfolio Return - Target Return) / Downside Deviation',
                'interpretation_note': 'Higher values indicate better downside risk-adjusted performance'
            },
            'Max_Drawdown': {
                'ranges': [
                    (float('-inf'), -0.5, 'Severe', 'Extreme drawdowns indicate very high risk and poor risk management'),
                    (-0.5, -0.3, 'High', 'Large drawdowns suggest significant risk and potential for substantial losses'),
                    (-0.3, -0.2, 'Moderate', 'Moderate drawdowns indicate manageable risk levels'),
                    (-0.2, -0.1, 'Low', 'Small drawdowns demonstrate good risk control'),
                    (-0.1, -0.05, 'Very Low', 'Minimal drawdowns indicate excellent risk management'),
                    (-0.05, 0, 'Exceptional', 'Negligible drawdowns suggest outstanding risk control')
                ],
                'description': 'Maximum Drawdown measures the largest peak-to-trough decline',
                'formula': '(Trough Value - Peak Value) / Peak Value',
                'interpretation_note': 'Values closer to zero indicate better risk control (expressed as negative percentages)'
            },
            'Volatility': {
                'ranges': [
                    (0, 0.05, 'Very Low', 'Very low volatility may indicate conservative positioning or limited opportunities'),
                    (0.05, 0.1, 'Low', 'Low volatility suggests stable returns with moderate risk'),
                    (0.1, 0.15, 'Moderate', 'Moderate volatility indicates balanced risk-return profile'),
                    (0.15, 0.25, 'High', 'High volatility suggests significant risk and potential for large swings'),
                    (0.25, 0.4, 'Very High', 'Very high volatility indicates substantial risk and potential instability'),
                    (0.4, float('inf'), 'Extreme', 'Extreme volatility suggests very high risk and potential for severe losses')
                ],
                'description': 'Volatility measures the standard deviation of returns',
                'formula': 'Standard Deviation of Portfolio Returns (annualized)',
                'interpretation_note': 'Lower values generally indicate more stable returns'
            },
            'Win_Rate': {
                'ranges': [
                    (0, 0.3, 'Poor', 'Low win rate indicates frequent losses and potential strategy issues'),
                    (0.3, 0.4, 'Below Average', 'Below average win rate suggests room for improvement'),
                    (0.4, 0.5, 'Average', 'Average win rate indicates balanced performance'),
                    (0.5, 0.6, 'Good', 'Good win rate demonstrates consistent positive performance'),
                    (0.6, 0.7, 'Very Good', 'Very good win rate indicates strong strategy effectiveness'),
                    (0.7, 1.0, 'Exceptional', 'Exceptional win rate suggests highly effective strategy')
                ],
                'description': 'Win Rate measures the percentage of profitable periods',
                'formula': 'Number of Profitable Periods / Total Periods',
                'interpretation_note': 'Higher values indicate more consistent profitability'
            }
        }
    
    def create_unique_run_directory(self, strategy_name: str, run_id: Optional[str] = None) -> Path:
        """Create a unique directory for this optimization run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_id:
            dir_name = f"{strategy_name}_{run_id}_{timestamp}"
        else:
            dir_name = f"{strategy_name}_{timestamp}"
        
        run_dir = self.base_reports_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "data").mkdir(exist_ok=True)
        
        return run_dir
    
    def interpret_metric(self, metric_name: str, value: float) -> Tuple[str, str]:
        """Interpret a performance metric value and provide explanation."""
        if metric_name not in self.metric_interpretations:
            return "Unknown", "No interpretation available for this metric"
        
        metric_info = self.metric_interpretations[metric_name]
        
        for min_val, max_val, rating, explanation in metric_info['ranges']:
            if min_val <= value < max_val:
                return rating, explanation
        
        return "Unknown", "Value outside expected range"
    
    def generate_performance_summary_table(self, metrics: Dict[str, float]) -> str:
        """Generate a markdown table with performance metrics and interpretations."""
        table_rows = []
        table_rows.append("| Metric | Value | Rating | Interpretation |")
        table_rows.append("|--------|-------|--------|----------------|")
        
        for metric_name, value in metrics.items():
            if metric_name in self.metric_interpretations:
                rating, interpretation = self.interpret_metric(metric_name, value)
                
                # Format value based on metric type
                if metric_name == 'Max_Drawdown':
                    formatted_value = f"{value:.2%}"
                elif metric_name in ['Volatility', 'Win_Rate']:
                    formatted_value = f"{value:.2%}" if metric_name == 'Win_Rate' else f"{value:.2%}"
                else:
                    formatted_value = f"{value:.3f}"
                
                table_rows.append(f"| {metric_name.replace('_', ' ')} | {formatted_value} | {rating} | {interpretation} |")
        
        return "\n".join(table_rows)
    
    def generate_metric_descriptions(self) -> str:
        """Generate detailed descriptions of all performance metrics."""
        descriptions = []
        descriptions.append("## Performance Metrics Glossary\n")
        
        for metric_name, info in self.metric_interpretations.items():
            descriptions.append(f"### {metric_name.replace('_', ' ')}")
            descriptions.append(f"**Description:** {info['description']}")
            descriptions.append(f"**Formula:** `{info['formula']}`")
            descriptions.append(f"**Interpretation:** {info['interpretation_note']}")
            descriptions.append("")
        
        return "\n".join(descriptions)
    
    def save_optimization_data(self, run_dir: Path, optimization_data: Dict[str, Any]) -> None:
        """Save optimization data to JSON files for future reference."""
        data_dir = run_dir / "data"
        
        # Save main optimization results
        with open(data_dir / "optimization_results.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_data = self._make_json_serializable(optimization_data)
            json.dump(serializable_data, f, indent=2)
        
        # Save parameter importance if available
        if 'parameter_importance' in optimization_data:
            with open(data_dir / "parameter_importance.json", 'w') as f:
                json.dump(optimization_data['parameter_importance'], f, indent=2)
        
        # Save trial data if available
        if 'trials_data' in optimization_data:
            trials_df = pd.DataFrame(optimization_data['trials_data'])
            trials_df.to_csv(data_dir / "trials_data.csv", index=False)
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def move_plots_to_run_directory(self, run_dir: Path, plots_source_dir: str = "plots") -> Dict[str, str]:
        """Move generated plots from the plots directory to the run directory."""
        plots_dir = run_dir / "plots"
        source_plots_dir = Path(plots_source_dir)
        
        moved_plots = {}  # Dictionary mapping plot type to filename
        
        if source_plots_dir.exists():
            for plot_file in source_plots_dir.glob("*.png"):
                destination = plots_dir / plot_file.name
                try:
                    plot_file.rename(destination)
                    # Categorize plots by type for better organization
                    plot_type = self._categorize_plot(plot_file.name)
                    moved_plots[plot_type] = plot_file.name
                    logger.info(f"Moved plot {plot_file.name} to {destination}")
                except Exception as e:
                    logger.warning(f"Failed to move plot {plot_file.name}: {e}")
        
        return moved_plots
    
    def _categorize_plot(self, filename: str) -> str:
        """Categorize plot files by their content type."""
        filename_lower = filename.lower()
        
        if 'performance_summary' in filename_lower or 'cumulative' in filename_lower:
            return 'performance_summary'
        elif 'parameter_importance' in filename_lower:
            return 'parameter_importance'
        elif 'parameter_correlation' in filename_lower:
            return 'parameter_correlation'
        elif 'parameter_heatmap' in filename_lower:
            return 'parameter_heatmaps'
        elif 'parameter_sensitivity' in filename_lower:
            return 'parameter_sensitivity'
        elif 'parameter_stability' in filename_lower:
            return 'parameter_stability'
        elif 'parameter_robustness' in filename_lower:
            return 'parameter_robustness'
        elif 'stability_measures' in filename_lower or 'trial_pnl' in filename_lower:
            return 'stability_measures'
        elif 'monte_carlo' in filename_lower or 'robustness' in filename_lower:
            return 'monte_carlo_robustness'
        elif 'ga_fitness' in filename_lower or 'fitness' in filename_lower:
            return 'optimization_progress'
        else:
            return 'other_analysis'
    
    def generate_markdown_report(
        self, 
        run_dir: Path,
        strategy_name: str,
        optimization_results: Dict[str, Any],
        performance_metrics: Dict[str, float],
        optimal_parameters: Dict[str, Any],
        moved_plots: Dict[str, str],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a comprehensive markdown report."""
        
        report_lines = []
        
        # Header
        report_lines.append(f"# Optimization Report: {strategy_name}")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Run Directory:** `{run_dir.name}`")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        # Get overall performance assessment
        sharpe_rating = "Unknown"
        calmar_rating = "Unknown"
        if 'Sharpe' in performance_metrics:
            sharpe_rating, _ = self.interpret_metric('Sharpe', performance_metrics['Sharpe'])
        if 'Calmar' in performance_metrics:
            calmar_rating, _ = self.interpret_metric('Calmar', performance_metrics['Calmar'])
        
        report_lines.append(f"The {strategy_name} strategy optimization has been completed with **{sharpe_rating}** risk-adjusted performance (Sharpe) and **{calmar_rating}** drawdown-adjusted performance (Calmar).")
        report_lines.append("")
        
        # Winning Strategy Configuration - Enhanced Section
        report_lines.append("## Winning Strategy Configuration")
        report_lines.append("")
        report_lines.append("The optimization process identified the following optimal parameter set that achieved the best performance:")
        report_lines.append("")
        
        # Create a more prominent parameters table
        report_lines.append("### Optimal Parameters")
        report_lines.append("")
        report_lines.append("| Parameter | Optimal Value | Description | Impact |")
        report_lines.append("|-----------|---------------|-------------|---------|")
        
        # Get parameter importance if available
        param_importance = optimization_results.get('parameter_importance', {})
        
        for param, value in optimal_parameters.items():
            description = self._get_parameter_description(param)
            importance = param_importance.get(param, 0.0)
            impact = self._get_parameter_impact_description(importance)
            
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
                
            report_lines.append(f"| **{param}** | `{formatted_value}` | {description} | {impact} |")
        
        report_lines.append("")
        
        # Add parameter summary if importance data is available
        if param_importance:
            report_lines.append("### Parameter Importance Summary")
            report_lines.append("")
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            
            report_lines.append("**Most Influential Parameters:**")
            for i, (param, importance) in enumerate(sorted_params[:3]):
                report_lines.append(f"{i+1}. **{param}**: {importance:.1%} influence on performance")
            
            report_lines.append("")
        
        # Strategy Performance Visualization - Prominently display main performance chart
        if 'performance_summary' in moved_plots:
            report_lines.append("## Strategy Performance Overview")
            report_lines.append("")
            plot_file = moved_plots['performance_summary']
            report_lines.append(f"![Strategy Performance Summary](plots/{plot_file})")
            report_lines.append("*Cumulative returns and drawdown analysis showing strategy performance over time with all-time high markers*")
            report_lines.append("")
            report_lines.append("""**How to interpret this chart:**
- **Top panel**: Cumulative returns on logarithmic scale
  - **Strategy line**: Your optimized strategy performance
  - **Benchmark line**: Comparison benchmark (usually SPY)
  - **Green dots**: All-time high markers showing when new peaks were reached
  - **Log scale**: Allows comparison of percentage gains across different time periods
- **Bottom panel**: Drawdown analysis
  - **Drawdown**: Percentage decline from previous peak (always negative or zero)
  - **Gray shading**: Benchmark drawdown periods for comparison
  - **Deeper drawdowns**: Indicate higher risk periods
- **Key insights**: Look for consistent upward trend with controlled drawdowns compared to benchmark""")
            report_lines.append("")
        
        # Performance Metrics
        report_lines.append("## Performance Analysis")
        report_lines.append("")
        report_lines.append(self.generate_performance_summary_table(performance_metrics))
        report_lines.append("")
        
        # Optimization Statistics
        if additional_info:
            report_lines.append("## Optimization Statistics")
            report_lines.append("")
            
            if 'num_trials' in additional_info:
                report_lines.append(f"**Total Trials:** {additional_info['num_trials']}")
            if 'optimization_time' in additional_info:
                report_lines.append(f"**Optimization Time:** {additional_info['optimization_time']}")
            if 'best_trial_number' in additional_info:
                report_lines.append(f"**Best Trial Number:** {additional_info['best_trial_number']}")
            
            report_lines.append("")
        
        # Comprehensive Analysis and Visualizations - All plots embedded
        if moved_plots:
            report_lines.append("## Comprehensive Analysis")
            report_lines.append("")
            report_lines.append("The optimization process generated detailed visualizations to support strategy analysis and validation:")
            report_lines.append("")
            
            # Define plot categories with detailed descriptions and interpretation guides
            plot_categories = {
                'optimization_progress': {
                    'title': 'Optimization Progress',
                    'description': 'Shows how the optimization algorithm improved performance over time',
                    'interpretation': '''**How to interpret:** 
- **Y-axis**: Best performance metric found so far (higher is better for most metrics)
- **X-axis**: Trial/generation number showing algorithm progress
- **Trend**: Should show general upward trend indicating algorithm is finding better solutions
- **Convergence**: Flattening curve suggests algorithm has found optimal region
- **Key insight**: Steep initial improvements followed by gradual refinement indicates healthy optimization'''
                },
                'parameter_importance': {
                    'title': 'Parameter Importance Analysis',
                    'description': 'Ranks parameters by their impact on strategy performance to identify key drivers',
                    'interpretation': '''**How to interpret:**
- **Bar height**: Indicates how much each parameter influences performance
- **High importance**: Parameters that significantly affect strategy results (focus optimization here)
- **Low importance**: Parameters with minimal impact (can use wider ranges or defaults)
- **Key insight**: Focus on top 2-3 parameters for manual tuning and deeper analysis'''
                },
                'parameter_correlation': {
                    'title': 'Parameter Correlation Matrix',
                    'description': 'Reveals relationships between optimization parameters and their interactions',
                    'interpretation': '''**How to interpret:**
- **Color scale**: Red = negative correlation, Blue = positive correlation, White = no correlation
- **Strong correlations (|r| > 0.7)**: Parameters that move together (may be redundant)
- **Negative correlations**: Parameters that work in opposite directions
- **Objective correlation**: Bottom row/column shows which parameters most affect performance
- **Key insight**: Highly correlated parameters may indicate over-parameterization'''
                },
                'parameter_heatmaps': {
                    'title': 'Parameter Performance Heatmaps',
                    'description': 'Two-dimensional performance landscapes showing optimal parameter combinations',
                    'interpretation': '''**How to interpret:**
- **Color intensity**: Darker/brighter colors indicate better performance regions
- **Hot spots**: Areas of high performance (optimal parameter combinations)
- **Gradients**: Smooth transitions suggest stable parameter regions
- **Cliffs**: Sharp changes indicate sensitive parameter boundaries
- **Key insight**: Look for broad high-performance regions for robust parameter selection'''
                },
                'parameter_sensitivity': {
                    'title': 'Parameter Sensitivity Analysis',
                    'description': 'Shows how strategy performance changes with individual parameter variations',
                    'interpretation': '''**How to interpret:**
- **Scatter points**: Each point represents one optimization trial
- **Trend line**: Shows general relationship between parameter and performance
- **Slope**: Steeper slopes indicate higher parameter sensitivity
- **Spread**: Wide scatter suggests parameter interacts with others
- **Key insight**: Flat relationships suggest robust parameters, steep slopes need careful tuning'''
                },
                'parameter_stability': {
                    'title': 'Parameter Stability Analysis',
                    'description': 'Assesses parameter robustness and identifies stable vs. unstable regions',
                    'interpretation': '''**How to interpret:**
- **Top plot**: Parameter evolution over trials (should converge for stable parameters)
- **Bottom left**: Parameter variance (lower bars = more stable parameters)
- **Bottom right**: Performance stability across parameter ranges
- **Convergence**: Parameters that settle to consistent values are more reliable
- **Key insight**: Stable parameters are safer for live trading implementation'''
                },
                'parameter_robustness': {
                    'title': 'Parameter Robustness Assessment',
                    'description': 'Comprehensive analysis showing parameter reliability across different market conditions',
                    'interpretation': '''**How to interpret:**
- **Robustness landscape**: Shows performance stability across parameter space
- **Contour lines**: Connect regions of similar robustness
- **Robustness ranking**: Bar chart showing most to least robust parameters
- **Quantile analysis**: How parameters behave in different performance scenarios
- **Key insight**: Robust parameters maintain performance across varying market conditions'''
                },
                'stability_measures': {
                    'title': 'Trial Performance Analysis',
                    'description': 'Monte Carlo-style visualization showing distribution of all optimization trial results',
                    'interpretation': '''**How to interpret:**
- **Gray lines**: Individual trial performance curves (each represents one parameter set)
- **Blue band**: 90% confidence interval showing typical performance range
- **Blue dashed line**: Median performance across all trials
- **Black line**: Final optimized strategy performance
- **Key insight**: Final strategy should be in upper performance range, wide bands indicate high variability'''
                },
                'monte_carlo_robustness': {
                    'title': 'Monte Carlo Robustness Testing',
                    'description': 'Stress testing with progressive synthetic data replacement to assess strategy robustness',
                    'interpretation': '''**How to interpret:**
- **Different colors**: Each color represents different levels of synthetic data (5%, 7.5%, 10%, etc.)
- **Multiple lines per color**: Multiple simulations at each replacement level
- **Performance degradation**: How much performance drops as more synthetic data is used
- **Consistency**: Similar performance across simulations indicates robust strategy
- **Key insight**: Robust strategies maintain performance even with significant market data changes'''
                },
                'other_analysis': {
                    'title': 'Additional Analysis',
                    'description': 'Supplementary visualizations providing additional insights into strategy behavior',
                    'interpretation': '''**How to interpret:**
- **Context-dependent**: Interpretation depends on specific analysis type
- **Supplementary insights**: Additional perspectives on strategy characteristics
- **Cross-validation**: May show out-of-sample or alternative validation results
- **Key insight**: Provides additional confidence in optimization results'''
                }
            }
            
            # Display plots by category - embed all plots in the report with detailed interpretations
            for category, info in plot_categories.items():
                if category in moved_plots:
                    plot_file = moved_plots[category]
                    report_lines.append(f"### {info['title']}")
                    report_lines.append(f"![{info['title']}](plots/{plot_file})")
                    report_lines.append(f"*{info['description']}*")
                    report_lines.append("")
                    report_lines.append(info['interpretation'])
                    report_lines.append("")
            
            # List any additional plots not categorized
            categorized_plots = set(plot_categories.keys())
            additional_plots = [plot for category, plot in moved_plots.items() if category not in categorized_plots]
            
            if additional_plots:
                report_lines.append("### Additional Visualizations")
                report_lines.append("")
                for plot_file in additional_plots:
                    plot_name = plot_file.replace('.png', '').replace('_', ' ').title()
                    report_lines.append(f"![{plot_name}](plots/{plot_file})")
                    report_lines.append(f"*{plot_name} - Additional analysis visualization*")
                    report_lines.append("")
        
        # Risk Assessment
        report_lines.append("## Risk Assessment")
        report_lines.append("")
        
        risk_assessment = self._generate_risk_assessment(performance_metrics)
        report_lines.extend(risk_assessment)
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        recommendations = self._generate_recommendations(performance_metrics, optimal_parameters)
        report_lines.extend(recommendations)
        report_lines.append("")
        
        # Appendix with metric descriptions
        report_lines.append("---")
        report_lines.append("")
        report_lines.append(self.generate_metric_descriptions())
        
        # Save the report
        report_content = "\n".join(report_lines)
        report_file = run_dir / "optimization_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Generated comprehensive optimization report: {report_file}")
        return str(report_file)
    
    def _get_parameter_description(self, param_name: str) -> str:
        """Get description for optimization parameters."""
        descriptions = {
            'lookback_months': 'Number of months to look back for momentum calculation',
            'skip_months': 'Number of recent months to skip in momentum calculation',
            'num_holdings': 'Number of assets to hold in the portfolio',
            'top_decile_fraction': 'Fraction of top-performing assets to consider',
            'smoothing_lambda': 'Exponential smoothing factor for position transitions',
            'leverage': 'Portfolio leverage multiplier',
            'long_only': 'Whether to allow only long positions',
            'sizer_dvol_window': 'Window size for downside volatility calculation',
            'sizer_target_volatility': 'Target volatility for position sizing',
            'sizer_max_leverage': 'Maximum leverage allowed by position sizer'
        }
        return descriptions.get(param_name, 'Strategy parameter')
    
    def _get_parameter_impact_description(self, importance: float) -> str:
        """Get impact description based on parameter importance."""
        if importance >= 0.3:
            return "High Impact"
        elif importance >= 0.15:
            return "Medium Impact"
        elif importance >= 0.05:
            return "Low Impact"
        else:
            return "Minimal Impact"
    
    def _generate_risk_assessment(self, metrics: Dict[str, float]) -> List[str]:
        """Generate risk assessment based on performance metrics."""
        assessment = []
        
        # Analyze drawdown risk
        if 'Max_Drawdown' in metrics:
            dd_rating, dd_explanation = self.interpret_metric('Max_Drawdown', metrics['Max_Drawdown'])
            assessment.append(f"**Drawdown Risk:** {dd_rating}")
            assessment.append(f"- {dd_explanation}")
            
            if metrics['Max_Drawdown'] < -0.3:
                assessment.append("- WARNING: Large drawdowns may indicate excessive risk-taking")
            elif metrics['Max_Drawdown'] > -0.1:
                assessment.append("- POSITIVE: Well-controlled drawdowns indicate good risk management")
        
        # Analyze volatility risk
        if 'Volatility' in metrics:
            vol_rating, vol_explanation = self.interpret_metric('Volatility', metrics['Volatility'])
            assessment.append(f"**Volatility Risk:** {vol_rating}")
            assessment.append(f"- {vol_explanation}")
            
            if metrics['Volatility'] > 0.25:
                assessment.append("- WARNING: High volatility - consider position sizing adjustments")
        
        # Analyze consistency
        if 'Win_Rate' in metrics:
            wr_rating, wr_explanation = self.interpret_metric('Win_Rate', metrics['Win_Rate'])
            assessment.append(f"**Consistency:** {wr_rating}")
            assessment.append(f"- {wr_explanation}")
        
        return assessment
    
    def _generate_recommendations(self, metrics: Dict[str, float], parameters: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Risk-adjusted performance recommendations
        if 'Sharpe' in metrics:
            if metrics['Sharpe'] < 0.5:
                recommendations.append("- **Consider Strategy Revision:** Low Sharpe ratio suggests inadequate risk-adjusted returns")
                recommendations.append("  - Review parameter ranges for optimization")
                recommendations.append("  - Consider alternative position sizing methods")
                recommendations.append("  - Evaluate additional risk management techniques")
            elif metrics['Sharpe'] > 1.5:
                recommendations.append("- **Strong Performance:** Excellent Sharpe ratio indicates effective strategy design")
                recommendations.append("  - Consider increasing position sizes if risk tolerance allows")
                recommendations.append("  - Monitor for potential overfitting in optimization")
        
        # Drawdown management recommendations
        if 'Max_Drawdown' in metrics:
            if metrics['Max_Drawdown'] < -0.2:
                recommendations.append("- **Enhance Risk Management:** Large drawdowns suggest need for better risk controls")
                recommendations.append("  - Implement stop-loss mechanisms")
                recommendations.append("  - Consider dynamic position sizing based on volatility")
                recommendations.append("  - Evaluate portfolio diversification")
        
        # Parameter-specific recommendations
        if 'leverage' in parameters:
            if parameters['leverage'] > 1.5:
                recommendations.append("- **Leverage Monitoring:** High leverage requires careful risk monitoring")
                recommendations.append("  - Implement real-time risk monitoring")
                recommendations.append("  - Consider leverage limits based on market conditions")
        
        # General recommendations
        recommendations.append("- **Regular Monitoring:** Implement ongoing performance tracking")
        recommendations.append("- **Reoptimization:** Consider periodic reoptimization as market conditions change")
        recommendations.append("- **Out-of-Sample Testing:** Validate results on unseen data before live deployment")
        
        return recommendations

def create_optimization_report(
    strategy_name: str,
    optimization_results: Dict[str, Any],
    performance_metrics: Dict[str, float],
    optimal_parameters: Dict[str, Any],
    plots_source_dir: str = "plots",
    run_id: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Main function to create a comprehensive optimization report.
    
    Args:
        strategy_name: Name of the optimized strategy
        optimization_results: Full optimization results data
        performance_metrics: Dictionary of performance metrics (Sharpe, Calmar, etc.)
        optimal_parameters: Dictionary of optimal parameter values
        plots_source_dir: Directory containing generated plots
        run_id: Optional unique identifier for this run
        additional_info: Additional information to include in the report
    
    Returns:
        Path to the generated report file
    """
    generator = OptimizerReportGenerator()
    
    # Create unique run directory
    run_dir = generator.create_unique_run_directory(strategy_name, run_id)
    
    # Save optimization data
    generator.save_optimization_data(run_dir, optimization_results)
    
    # Move plots to run directory
    moved_plots = generator.move_plots_to_run_directory(run_dir, plots_source_dir)
    
    # Generate comprehensive markdown report
    report_path = generator.generate_markdown_report(
        run_dir=run_dir,
        strategy_name=strategy_name,
        optimization_results=optimization_results,
        performance_metrics=performance_metrics,
        optimal_parameters=optimal_parameters,
        moved_plots=moved_plots,
        additional_info=additional_info
    )
    
    return report_path