#!/usr/bin/env python3
"""
Synthetic Data Generator Comparison Script

This script generates comprehensive comparison charts between historical and synthetic
financial data for selected tickers. It demonstrates the improved GARCH-based
synthetic data generator with visual inspection capabilities.

Usage:
    python generate_synthetic_data_comparison.py

Output:
    - Individual comparison charts for each ticker
    - Statistical analysis reports
    - Summary comparison tables
"""

import sys
import os
from pathlib import Path
import logging
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from portfolio_backtester.monte_carlo.visual_inspection import SyntheticDataVisualInspector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def download_historical_data(tickers, period='2y'):
    """
    Download historical data for specified tickers.
    
    Args:
        tickers: List of ticker symbols
        period: Time period for data (default: 2 years)
        
    Returns:
        Dictionary of ticker -> OHLC DataFrame
    """
    historical_data = {}
    
    logger.info(f"Downloading historical data for {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            logger.info(f"Downloading data for {ticker}")
            
            # Download data using yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                continue
            
            # Ensure we have OHLC columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing OHLC data for {ticker}")
                continue
            
            # Clean the data
            data = data[required_columns].dropna()
            
            if len(data) < 252:  # Need at least 1 year of data
                logger.warning(f"Insufficient data for {ticker}: {len(data)} observations")
                continue
            
            historical_data[ticker] = data
            logger.info(f"Successfully downloaded {len(data)} observations for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            continue
    
    return historical_data

def create_monte_carlo_config():
    """
    Create configuration for Monte Carlo synthetic data generation.
    
    Returns:
        Configuration dictionary
    """
    return {
        'enable_synthetic_data': True,
        'replacement_percentage': 0.10,
        'min_historical_observations': 252,
        'random_seed': 42,
        
        # GARCH model configuration
        'garch_config': {
            'model_type': 'GARCH',
            'p': 1,
            'q': 1,
            'distribution': 'studentt',
            'bounds': {
                'omega': [1e-6, 1.0],
                'alpha': [0.01, 0.3],
                'beta': [0.5, 0.99],
                'nu': [2.1, 30.0]
            }
        },
        
        # Generation configuration
        'generation_config': {
            'buffer_multiplier': 1.2,
            'max_attempts': 3,
            'validation_tolerance': 0.3
        },
        
        # Validation configuration
        'validation_config': {
            'enable_validation': True,
            'tolerance': 0.4,  # 40% tolerance for statistics
            'ks_test_pvalue_threshold': 0.05,
            'autocorr_max_deviation': 0.15,
            'volatility_clustering_threshold': 0.02,
            'tail_index_tolerance': 0.3
        }
    }

def main():
    """
    Main function to generate synthetic data comparison charts.
    """
    # Target tickers for analysis
    tickers = ['AAPL', 'AMZN', 'TSLA', 'NVDA', 'CAT']
    
    # Number of synthetic paths to generate per ticker
    num_synthetic_paths = 10
    
    # Output directory for charts
    output_dir = project_root / 'plots' / 'synthetic_data_analysis'
    
    logger.info("Starting synthetic data comparison analysis")
    logger.info(f"Target tickers: {tickers}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Download historical data
        historical_data = download_historical_data(tickers, period='3y')
        
        if not historical_data:
            logger.error("No historical data available. Exiting.")
            return
        
        logger.info(f"Successfully downloaded data for {len(historical_data)} tickers")
        
        # Create Monte Carlo configuration
        config = create_monte_carlo_config()
        
        # Initialize visual inspector
        inspector = SyntheticDataVisualInspector(config)
        
        # Generate comparison reports
        logger.info("Generating synthetic data and comparison charts...")
        
        results = inspector.generate_comparison_report(
            historical_data=historical_data,
            tickers=list(historical_data.keys()),
            num_synthetic_paths=num_synthetic_paths,
            output_dir=str(output_dir)
        )
        
        # Print summary results
        print("\n" + "="*80)
        print("SYNTHETIC DATA GENERATION SUMMARY")
        print("="*80)
        
        for ticker, analysis in results.items():
            print(f"\n{ticker}:")
            print("-" * 20)
            
            hist_stats = analysis['historical_stats']
            synth_stats = analysis['synthetic_stats']
            
            if synth_stats:
                avg_synth_stats = {
                    key: np.mean([s[key] for s in synth_stats if key in s and isinstance(s[key], (int, float))])
                    for key in ['mean', 'std', 'skewness', 'kurtosis']
                }
                
                print(f"Historical - Mean: {hist_stats['mean']:.4f}, Std: {hist_stats['std']:.4f}")
                print(f"Synthetic  - Mean: {avg_synth_stats['mean']:.4f}, Std: {avg_synth_stats['std']:.4f}")
                print(f"Historical - Skew: {hist_stats['skewness']:.4f}, Kurt: {hist_stats['kurtosis']:.4f}")
                print(f"Synthetic  - Skew: {avg_synth_stats['skewness']:.4f}, Kurt: {avg_synth_stats['kurtosis']:.4f}")
                
                # Statistical test results
                if 'statistical_tests' in analysis and analysis['statistical_tests']:
                    path_1_tests = analysis['statistical_tests'].get('path_1', {})
                    if 'ks_test' in path_1_tests:
                        ks_pvalue = path_1_tests['ks_test']['pvalue']
                        print(f"KS Test p-value: {ks_pvalue:.4f} ({'PASS' if ks_pvalue > 0.05 else 'FAIL'})")
            else:
                print("No synthetic data generated successfully")
        
        print(f"\nDetailed charts saved to: {output_dir}")
        print("\nGenerated files:")
        if output_dir.exists():
            for file in sorted(output_dir.glob("*.png")):
                print(f"  - {file.name}")
        
        logger.info("Synthetic data comparison analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 