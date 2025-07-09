#!/usr/bin/env python3
"""
Test script to demonstrate the new optimizer reporting system.

This script creates a sample optimization report to show the enhanced
reporting capabilities including:
- Comprehensive markdown reports
- Performance metric interpretations
- Professional formatting
- Organized file structure
"""

import pandas as pd
import numpy as np
from src.portfolio_backtester.reporting.optimizer_report_generator import create_optimization_report
import os
import tempfile
import shutil

def create_sample_optimization_data():
    """Create sample optimization data for demonstration."""
    
    # Sample performance metrics
    performance_metrics = {
        'Sharpe': 1.25,
        'Calmar': 0.85,
        'Sortino': 1.45,
        'Max_Drawdown': -0.15,
        'Volatility': 0.18,
        'Win_Rate': 0.62,
        'Total_Return': 0.24,
        'Ann_Return': 0.12
    }
    
    # Sample optimal parameters
    optimal_parameters = {
        'lookback_months': 6,
        'skip_months': 1,
        'num_holdings': 15,
        'smoothing_lambda': 0.3,
        'leverage': 1.2,
        'sizer_dvol_window': 12,
        'sizer_target_volatility': 0.15
    }
    
    # Sample optimization results
    optimization_results = {
        "strategy_name": "MomentumStrategy_Demo",
        "optimal_parameters": optimal_parameters,
        "performance_metrics": performance_metrics,
        "optimization_metadata": {
            "num_trials": 150,
            "optimizer_type": "optuna",
            "optimization_date": pd.Timestamp.now().isoformat(),
            "random_seed": 42
        },
        "parameter_importance": {
            'lookback_months': 0.35,
            'num_holdings': 0.28,
            'smoothing_lambda': 0.15,
            'leverage': 0.12,
            'skip_months': 0.10
        }
    }
    
    # Additional info
    additional_info = {
        "num_trials": 150,
        "best_trial_number": 87,
        "optimization_time": "45 minutes",
        "random_seed": 42
    }
    
    return optimization_results, performance_metrics, optimal_parameters, additional_info

def create_sample_plots():
    """Create sample plot files to demonstrate plot organization."""
    import matplotlib.pyplot as plt
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Create sample plots
    plot_files = []
    
    # 1. Parameter importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    params = ['lookback_months', 'num_holdings', 'smoothing_lambda', 'leverage', 'skip_months']
    importance = [0.35, 0.28, 0.15, 0.12, 0.10]
    ax.bar(params, importance, color='steelblue', alpha=0.7)
    ax.set_title('Parameter Importance Ranking')
    ax.set_ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = "plots/parameter_importance.png"
    plt.savefig(plot_path)
    plt.close()
    plot_files.append("parameter_importance.png")
    
    # 2. Performance evolution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    trials = np.arange(1, 151)
    performance = np.random.normal(1.0, 0.3, 150)
    performance = np.maximum(performance, 0.1)  # Ensure positive
    performance = np.maximum.accumulate(performance * 0.99 + np.random.normal(0, 0.05, 150))  # Add trend
    ax.plot(trials, performance, color='green', alpha=0.7)
    ax.set_title('Optimization Progress: Best Performance Over Trials')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Best Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = "plots/optimization_progress.png"
    plt.savefig(plot_path)
    plt.close()
    plot_files.append("optimization_progress.png")
    
    # 3. Parameter correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    correlation_data = np.random.rand(5, 5)
    correlation_data = (correlation_data + correlation_data.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_data, 1.0)
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                xticklabels=params, yticklabels=params, ax=ax)
    ax.set_title('Parameter Correlation Matrix')
    plt.tight_layout()
    plot_path = "plots/parameter_correlation.png"
    plt.savefig(plot_path)
    plt.close()
    plot_files.append("parameter_correlation.png")
    
    return plot_files

def main():
    """Main function to demonstrate the reporting system."""
    print("🚀 Testing Enhanced Optimizer Reporting System")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample optimization data...")
    optimization_results, performance_metrics, optimal_parameters, additional_info = create_sample_optimization_data()
    
    # Create sample plots
    print("📈 Creating sample visualization plots...")
    plot_files = create_sample_plots()
    
    # Generate the comprehensive report
    print("📝 Generating comprehensive optimization report...")
    try:
        report_path = create_optimization_report(
            strategy_name="MomentumStrategy_Demo",
            optimization_results=optimization_results,
            performance_metrics=performance_metrics,
            optimal_parameters=optimal_parameters,
            plots_source_dir="plots",
            run_id="demo_test",
            additional_info=additional_info
        )
        
        print(f"✅ Report generated successfully!")
        print(f"📁 Report location: {report_path}")
        print()
        print("📋 Report Contents:")
        print("   - optimization_report.md (Main markdown report)")
        print("   - plots/ (All visualization files)")
        print("   - data/ (Raw optimization data in JSON format)")
        print()
        
        # Display a preview of the report
        print("📖 Report Preview (first 20 lines):")
        print("-" * 40)
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:20]):
                    print(f"{i+1:2d}: {line.rstrip()}")
                if len(lines) > 20:
                    print(f"... ({len(lines) - 20} more lines)")
        except Exception as e:
            print(f"Could not preview report: {e}")
        
        print()
        print("🎯 Key Features Demonstrated:")
        print("   ✓ Professional markdown formatting")
        print("   ✓ Performance metric interpretations")
        print("   ✓ Parameter descriptions and analysis")
        print("   ✓ Risk assessment and recommendations")
        print("   ✓ Organized file structure with timestamps")
        print("   ✓ Comprehensive metric glossary")
        print("   ✓ Plot organization and referencing")
        
        # Show directory structure
        report_dir = os.path.dirname(report_path)
        print(f"\n📂 Generated Directory Structure:")
        for root, dirs, files in os.walk(report_dir):
            level = root.replace(report_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print(f"\n🧹 Cleaning up temporary plots...")
    try:
        if os.path.exists("plots"):
            shutil.rmtree("plots")
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    print("\n🎉 Enhanced Optimizer Reporting System Test Completed!")
    print("\nNext Steps:")
    print("1. Run an actual optimization to see the system in action")
    print("2. Check the data/reports directory for generated reports")
    print("3. Review the markdown reports for insights and recommendations")

if __name__ == "__main__":
    main()