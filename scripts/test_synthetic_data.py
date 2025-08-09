#!/usr/bin/env python3
"""
Synthetic Data Quality Testing Tool

This script tests the quality of synthetic financial data generation by comparing
synthetic data with real historical data for a given stock symbol.

Usage:
    python scripts/test_synthetic_data.py AAPL
    python scripts/test_synthetic_data.py MSFT --paths 5 --output reports/
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import modules directly to avoid package initialization issues
import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import required modules
visual_inspection = import_module_from_path(
    "visual_inspection", 
    src_path / "portfolio_backtester" / "monte_carlo" / "visual_inspection.py"
)
synthetic_data_generator = import_module_from_path(
    "synthetic_data_generator",
    src_path / "portfolio_backtester" / "monte_carlo" / "synthetic_data_generator.py"
)
hybrid_data_source = import_module_from_path(
    "hybrid_data_source",
    src_path / "portfolio_backtester" / "data_sources" / "hybrid_data_source.py"
)

SyntheticDataVisualInspector = visual_inspection.SyntheticDataVisualInspector
SyntheticDataGenerator = synthetic_data_generator.SyntheticDataGenerator
HybridDataSource = hybrid_data_source.HybridDataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_historical_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical data for a given symbol using the built-in hybrid data source.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        period: Time period ('1y', '2y', '5y', etc.)
        
    Returns:
        DataFrame with OHLC data
    """
    logger.info(f"Fetching historical data for {symbol} using hybrid data source...")
    
    try:
        # Calculate date range based on period
        end_date = datetime.now()
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            # Default to 2 years
            start_date = end_date - timedelta(days=730)
            
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Initialize hybrid data source
        data_source = HybridDataSource(cache_expiry_hours=24, prefer_stooq=True)
        
        # Fetch data
        raw_data = data_source.get_data([symbol], start_date_str, end_date_str)
        
        if raw_data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
            
        # Extract ticker data using the normalizer
        normalizer = OHLCNormalizerFactory.create_normalizer(raw_data)
        ticker_data = normalizer.extract_ticker_data(raw_data, symbol)
        
        if ticker_data.empty:
            raise ValueError(f"No valid data extracted for symbol {symbol}")
            
        logger.info(f"Successfully fetched {len(ticker_data)} days of data for {symbol}")
        return ticker_data
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        raise


def create_synthetic_data_config(symbol: str) -> Dict[str, Any]:
    """
    Create configuration for synthetic data generation.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Configuration dictionary
    """
    return {
        "random_seed": 42,
        "garch_params": {
            "p": 1,
            "q": 1,
            "mean_model": "constant"
        },
        "validation": {
            "enabled": True,
            "statistical_tests": True
        },
        "asset_name": symbol
    }


def print_analysis_summary(results: Dict[str, Any], symbol: str):
    """
    Print a summary of the analysis results.
    
    Args:
        results: Analysis results from visual inspector
        symbol: Stock symbol
    """
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DATA QUALITY ANALYSIS FOR {symbol}")
    print(f"{'='*60}")
    
    if symbol not in results:
        print(f"❌ No analysis results available for {symbol}")
        return
        
    analysis = results[symbol]
    
    # Historical statistics
    hist_stats = analysis["historical_stats"]
    print(f"\n📊 HISTORICAL DATA STATISTICS:")
    print(f"   Mean Return:     {hist_stats['mean']:.6f}")
    print(f"   Volatility:      {hist_stats['std']:.6f}")
    print(f"   Skewness:        {hist_stats['skewness']:.4f}")
    print(f"   Kurtosis:        {hist_stats['kurtosis']:.4f}")
    print(f"   VaR (95%):       {hist_stats['var_95']:.6f}")
    print(f"   VaR (99%):       {hist_stats['var_99']:.6f}")
    
    # Synthetic statistics (average across paths)
    if analysis["synthetic_stats"]:
        synth_stats = analysis["synthetic_stats"]
        avg_stats = {
            key: sum(stat[key] for stat in synth_stats if key in stat) / len(synth_stats)
            for key in ['mean', 'std', 'skewness', 'kurtosis', 'var_95', 'var_99']
            if all(key in stat for stat in synth_stats)
        }
        
        print(f"\n🎲 SYNTHETIC DATA STATISTICS (Average across {len(synth_stats)} paths):")
        print(f"   Mean Return:     {avg_stats.get('mean', 0):.6f}")
        print(f"   Volatility:      {avg_stats.get('std', 0):.6f}")
        print(f"   Skewness:        {avg_stats.get('skewness', 0):.4f}")
        print(f"   Kurtosis:        {avg_stats.get('kurtosis', 0):.4f}")
        print(f"   VaR (95%):       {avg_stats.get('var_95', 0):.6f}")
        print(f"   VaR (99%):       {avg_stats.get('var_99', 0):.6f}")
        
        # Quality assessment
        print(f"\n✅ QUALITY ASSESSMENT:")
        mean_diff = abs(avg_stats.get('mean', 0) - hist_stats['mean'])
        vol_diff = abs(avg_stats.get('std', 0) - hist_stats['std'])
        skew_diff = abs(avg_stats.get('skewness', 0) - hist_stats['skewness'])
        
        print(f"   Mean Difference:     {mean_diff:.6f} {'✅' if mean_diff < 0.001 else '⚠️'}")
        print(f"   Volatility Diff:     {vol_diff:.6f} {'✅' if vol_diff < 0.01 else '⚠️'}")
        print(f"   Skewness Diff:       {skew_diff:.4f} {'✅' if skew_diff < 0.5 else '⚠️'}")
    
    # Statistical tests
    if "statistical_tests" in analysis and analysis["statistical_tests"]:
        print(f"\n🧪 STATISTICAL TESTS (p-values):")
        
        for path_name, tests in analysis["statistical_tests"].items():
            if "path_1" in path_name:  # Show results for first path
                ks_pvalue = tests.get("ks_test", {}).get("pvalue", 0)
                mw_pvalue = tests.get("mann_whitney", {}).get("pvalue", 0)
                lev_pvalue = tests.get("levene_test", {}).get("pvalue", 0)
                
                print(f"   Kolmogorov-Smirnov:  {ks_pvalue:.4f} {'✅' if ks_pvalue > 0.05 else '⚠️'}")
                print(f"   Mann-Whitney U:      {mw_pvalue:.4f} {'✅' if mw_pvalue > 0.05 else '⚠️'}")
                print(f"   Levene (variance):   {lev_pvalue:.4f} {'✅' if lev_pvalue > 0.05 else '⚠️'}")
                break
    
    # GARCH analysis
    if "garch_analysis" in analysis:
        garch = analysis["garch_analysis"]
        if "historical" in garch and garch["historical"].get("garch_params"):
            print(f"\n📈 GARCH ANALYSIS:")
            hist_garch = garch["historical"]["garch_params"]
            print(f"   Historical volatility clustering: {garch['historical'].get('volatility_clustering', 'N/A')}")
            
            if garch.get("synthetic") and len(garch["synthetic"]) > 0:
                synth_garch = garch["synthetic"][0]  # First synthetic path
                print(f"   Synthetic volatility clustering:  {synth_garch.get('volatility_clustering', 'N/A')}")
    
    print(f"\n{'='*60}")
    print("Legend: ✅ = Good match, ⚠️ = Potential issue")
    print("Note: p-values > 0.05 indicate synthetic data is statistically similar to historical data")
    print(f"{'='*60}\n")


def main():
    """Main function to run synthetic data quality testing."""
    parser = argparse.ArgumentParser(
        description="Test synthetic data quality against real market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_synthetic_data.py AAPL
  python scripts/test_synthetic_data.py MSFT --paths 5 --output reports/
  python scripts/test_synthetic_data.py GOOGL --period 1y --paths 10
        """
    )
    
    parser.add_argument(
        "symbol",
        help="Stock symbol to analyze (e.g., AAPL, MSFT, GOOGL)"
    )
    
    parser.add_argument(
        "--paths",
        type=int,
        default=3,
        help="Number of synthetic paths to generate (default: 3)"
    )
    
    parser.add_argument(
        "--period",
        default="2y",
        help="Historical data period (default: 2y)"
    )
    
    parser.add_argument(
        "--output",
        help="Output directory for plots and reports (optional)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (faster execution)"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate symbol
        symbol = args.symbol.upper()
        logger.info(f"Starting synthetic data quality analysis for {symbol}")
        
        # Fetch historical data
        historical_data = fetch_historical_data(symbol, args.period)
        
        # Create configuration
        config = create_synthetic_data_config(symbol)
        
        # Initialize visual inspector
        inspector = SyntheticDataVisualInspector(config)
        
        # Prepare data dictionary
        historical_data_dict = {symbol: historical_data}
        
        # Generate comparison report
        logger.info(f"Generating {args.paths} synthetic paths and running analysis...")
        
        results = inspector.generate_comparison_report(
            historical_data=historical_data_dict,
            tickers=[symbol],
            num_synthetic_paths=args.paths,
            output_dir=args.output
        )
        
        # Print summary
        print_analysis_summary(results, symbol)
        
        if args.output:
            output_path = Path(args.output)
            logger.info(f"Analysis complete! Reports and plots saved to: {output_path.absolute()}")
        else:
            logger.info("Analysis complete! (Use --output to save plots and reports)")
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()