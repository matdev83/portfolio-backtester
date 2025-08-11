#!/usr/bin/env python3
"""
Synthetic Data Quality Testing Tool - Working Version

This script tests the quality of synthetic financial data generation by comparing
synthetic data with real historical data using the project's hybrid data source.

Usage:
    python scripts/test_synthetic_data_working.py AAPL
    python scripts/test_synthetic_data_working.py MSFT --paths 3
"""

import argparse
import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set PYTHONPATH to include src directory
project_root = Path(__file__).parent.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set environment variable for imports
os.environ["PYTHONPATH"] = src_path


def fetch_historical_data_with_hybrid_source(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical data using the project's hybrid data source.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        period: Time period ('1y', '2y', '5y', etc.)

    Returns:
        DataFrame with OHLC data
    """
    logger.info(f"Fetching historical data for {symbol} using hybrid data source...")

    try:
        # Import the hybrid data source
        from portfolio_backtester.data_sources.hybrid_data_source import HybridDataSource
        from portfolio_backtester.interfaces.ohlc_normalizer import OHLCNormalizerFactory

        # Calculate date range
        end_date = datetime.now()
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
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

    except ImportError as e:
        logger.warning(f"Could not import hybrid data source: {e}")
        logger.info("Falling back to sample data generation...")
        return create_sample_ohlc_data(symbol, 500)
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        logger.info("Falling back to sample data generation...")
        return create_sample_ohlc_data(symbol, 500)


def create_sample_ohlc_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Create sample OHLC data for testing purposes."""
    logger.info(f"Creating sample OHLC data for {symbol} ({days} days)")

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate realistic price movements
    np.random.seed(42)
    initial_price = 100.0
    daily_returns = np.random.normal(0.0004, 0.012, len(dates))
    price_series = initial_price * np.exp(np.cumsum(daily_returns))

    # Generate OHLC
    opens = price_series.copy()
    closes = price_series.copy()
    intraday_vol = 0.005
    highs = closes * (1 + np.abs(np.random.normal(0, intraday_vol, len(dates))))
    lows = closes * (1 - np.abs(np.random.normal(0, intraday_vol, len(dates))))

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    volumes = np.random.lognormal(15, 0.5, len(dates)).astype(int)

    data = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes}, index=dates
    )

    logger.info(
        f"Generated sample data: {len(data)} days, price range ${data['Close'].min():.2f} - ${data['Close'].max():.2f}"
    )
    return data


def generate_synthetic_data_paths(
    historical_data: pd.DataFrame, symbol: str, num_paths: int = 3
) -> List[pd.DataFrame]:
    """
    Generate multiple synthetic data paths using GARCH-like properties.
    """
    logger.info(f"Generating {num_paths} synthetic data paths for {symbol}")

    # Calculate historical returns
    hist_returns = historical_data["Close"].pct_change().dropna()

    # Estimate parameters
    mean_return = hist_returns.mean()
    base_vol = hist_returns.std()

    synthetic_paths = []

    for path_id in range(num_paths):
        logger.info(f"Generating synthetic path {path_id + 1}/{num_paths}")

        # Use different seed for each path
        np.random.seed(123 + path_id)
        n_days = len(historical_data)

        synthetic_returns = []
        current_vol = base_vol

        # Generate returns with volatility clustering
        for i in range(n_days):
            if i > 0:
                # Volatility clustering: vol depends on previous return
                vol_adjustment = 1 + 0.1 * abs(synthetic_returns[i - 1]) / base_vol
                current_vol = base_vol * vol_adjustment * 0.9 + current_vol * 0.1

            # Generate return
            return_val = np.random.normal(mean_return, current_vol)
            synthetic_returns.append(return_val)

        # Convert to price series
        synthetic_returns = pd.Series(synthetic_returns, index=historical_data.index)
        initial_price = historical_data["Close"].iloc[0]
        synthetic_prices = initial_price * (1 + synthetic_returns).cumprod()

        # Generate OHLC from synthetic prices
        intraday_vol = 0.003
        synthetic_opens = synthetic_prices.copy()
        synthetic_closes = synthetic_prices.copy()

        # Add intraday movements
        highs = synthetic_closes * (
            1 + np.abs(np.random.normal(0, intraday_vol, len(synthetic_closes)))
        )
        lows = synthetic_closes * (
            1 - np.abs(np.random.normal(0, intraday_vol, len(synthetic_closes)))
        )

        # Ensure OHLC consistency
        highs = np.maximum(highs, np.maximum(synthetic_opens, synthetic_closes))
        lows = np.minimum(lows, np.minimum(synthetic_opens, synthetic_closes))

        synthetic_data = pd.DataFrame(
            {
                "Open": synthetic_opens,
                "High": highs,
                "Low": lows,
                "Close": synthetic_closes,
                "Volume": historical_data["Volume"],
            },
            index=historical_data.index,
        )

        synthetic_paths.append(synthetic_data)

        logger.info(
            f"Path {path_id + 1} generated: price range ${synthetic_data['Close'].min():.2f} - ${synthetic_data['Close'].max():.2f}"
        )

    return synthetic_paths


def calculate_statistics(returns: pd.Series, label: str) -> Dict:
    """Calculate comprehensive statistics for return series."""
    return {
        "label": label,
        "mean": returns.mean(),
        "std": returns.std(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "min": returns.min(),
        "max": returns.max(),
        "var_95": returns.quantile(0.05),
        "var_99": returns.quantile(0.01),
        "autocorr_lag1": returns.autocorr(lag=1) if len(returns) > 1 else 0,
        "count": len(returns),
    }


def perform_statistical_tests(hist_returns: pd.Series, synth_returns: pd.Series) -> Dict:
    """Perform comprehensive statistical tests comparing historical and synthetic returns."""
    from scipy import stats

    tests = {}

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(hist_returns, synth_returns)
    tests["ks_test"] = {"statistic": ks_stat, "pvalue": ks_pvalue}

    # Mann-Whitney U test
    mw_stat, mw_pvalue = stats.mannwhitneyu(hist_returns, synth_returns, alternative="two-sided")
    tests["mann_whitney"] = {"statistic": float(mw_stat), "pvalue": float(mw_pvalue)}

    # Levene test for equal variances
    lev_stat, lev_pvalue = stats.levene(hist_returns, synth_returns)
    tests["levene_test"] = {"statistic": float(lev_stat), "pvalue": float(lev_pvalue)}

    return tests


def perform_advanced_validation(hist_returns: pd.Series, synth_returns: pd.Series) -> Dict:
    """Perform advanced validation tests including volatility clustering and extreme values."""
    validation = {}

    # Volatility clustering validation (autocorrelation in squared returns)
    hist_squared = hist_returns**2
    synth_squared = synth_returns**2

    hist_vol_autocorr = hist_squared.autocorr(lag=1) if len(hist_squared) > 1 else 0
    synth_vol_autocorr = synth_squared.autocorr(lag=1) if len(synth_squared) > 1 else 0

    validation["volatility_clustering"] = {
        "historical": hist_vol_autocorr,
        "synthetic": synth_vol_autocorr,
        "difference": abs(hist_vol_autocorr - synth_vol_autocorr),
        "passed": abs(hist_vol_autocorr - synth_vol_autocorr) < 0.1,
    }

    # Rolling volatility validation
    window = 22  # 22 trading days
    if len(hist_returns) >= window and len(synth_returns) >= window:
        hist_rolling_vol = hist_returns.rolling(window=window).std().dropna()
        synth_rolling_vol = synth_returns.rolling(window=window).std().dropna()

        if len(hist_rolling_vol) > 0 and len(synth_rolling_vol) > 0:
            hist_mean_vol = hist_rolling_vol.mean()
            synth_mean_vol = synth_rolling_vol.mean()
            vol_diff = (
                abs(synth_mean_vol - hist_mean_vol) / hist_mean_vol if hist_mean_vol > 0 else 0
            )

            validation["rolling_volatility"] = {
                "historical_mean": hist_mean_vol,
                "synthetic_mean": synth_mean_vol,
                "relative_difference": vol_diff,
                "passed": vol_diff < 0.3,  # 30% tolerance
            }

    # Extreme values validation
    if len(hist_returns) >= 100 and len(synth_returns) >= 100:
        hist_percentiles = np.percentile(hist_returns.dropna(), [1, 5, 95, 99])
        synth_percentiles = np.percentile(synth_returns.dropna(), [1, 5, 95, 99])

        percentile_diffs = np.abs(hist_percentiles - synth_percentiles)
        max_diff = np.max(percentile_diffs)

        validation["extreme_values"] = {
            "historical_percentiles": {
                "1%": float(hist_percentiles[0]),
                "5%": float(hist_percentiles[1]),
                "95%": float(hist_percentiles[2]),
                "99%": float(hist_percentiles[3]),
            },
            "synthetic_percentiles": {
                "1%": float(synth_percentiles[0]),
                "5%": float(synth_percentiles[1]),
                "95%": float(synth_percentiles[2]),
                "99%": float(synth_percentiles[3]),
            },
            "max_difference": max_diff,
            "passed": max_diff < 0.5,  # Absolute difference tolerance
        }

    # Fat tails validation using kurtosis
    hist_kurtosis = hist_returns.kurtosis()
    synth_kurtosis = synth_returns.kurtosis()

    kurtosis_diff = abs(hist_kurtosis - synth_kurtosis)

    validation["fat_tails"] = {
        "historical_kurtosis": hist_kurtosis,
        "synthetic_kurtosis": synth_kurtosis,
        "difference": kurtosis_diff,
        "passed": kurtosis_diff < 1.0,  # Kurtosis tolerance
    }

    return validation


def print_analysis_summary(
    historical_data: pd.DataFrame, synthetic_paths: List[pd.DataFrame], symbol: str
):
    """Print comprehensive analysis summary."""
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DATA QUALITY ANALYSIS FOR {symbol}")
    print(f"{'='*60}")

    # Calculate historical statistics
    hist_returns = historical_data["Close"].pct_change().dropna()
    hist_stats = calculate_statistics(hist_returns, "Historical")

    print(f"\n📊 HISTORICAL DATA STATISTICS:")
    print(f"   Mean Return:     {hist_stats['mean']:.6f}")
    print(f"   Volatility:      {hist_stats['std']:.6f}")
    print(f"   Skewness:        {hist_stats['skewness']:.4f}")
    print(f"   Kurtosis:        {hist_stats['kurtosis']:.4f}")
    print(f"   VaR (95%):       {hist_stats['var_95']:.6f}")
    print(f"   VaR (99%):       {hist_stats['var_99']:.6f}")
    print(f"   Autocorr (lag1): {hist_stats['autocorr_lag1']:.4f}")

    # Calculate synthetic statistics (average across paths)
    synth_stats_list = []
    for i, synth_data in enumerate(synthetic_paths):
        synth_returns = synth_data["Close"].pct_change().dropna()
        synth_stats = calculate_statistics(synth_returns, f"Synthetic {i+1}")
        synth_stats_list.append(synth_stats)

    # Average statistics
    avg_stats = {}
    for key in ["mean", "std", "skewness", "kurtosis", "var_95", "var_99", "autocorr_lag1"]:
        avg_stats[key] = sum(stats[key] for stats in synth_stats_list) / len(synth_stats_list)

    print(f"\n🎲 SYNTHETIC DATA STATISTICS (Average across {len(synthetic_paths)} paths):")
    print(f"   Mean Return:     {avg_stats['mean']:.6f}")
    print(f"   Volatility:      {avg_stats['std']:.6f}")
    print(f"   Skewness:        {avg_stats['skewness']:.4f}")
    print(f"   Kurtosis:        {avg_stats['kurtosis']:.4f}")
    print(f"   VaR (95%):       {avg_stats['var_95']:.6f}")
    print(f"   VaR (99%):       {avg_stats['var_99']:.6f}")
    print(f"   Autocorr (lag1): {avg_stats['autocorr_lag1']:.4f}")

    # Quality assessment
    print(f"\n✅ QUALITY ASSESSMENT:")
    mean_diff = abs(avg_stats["mean"] - hist_stats["mean"])
    vol_diff = abs(avg_stats["std"] - hist_stats["std"])
    skew_diff = abs(avg_stats["skewness"] - hist_stats["skewness"])
    autocorr_diff = abs(avg_stats["autocorr_lag1"] - hist_stats["autocorr_lag1"])

    print(f"   Mean Difference:     {mean_diff:.6f} {'✅' if mean_diff < 0.001 else '⚠️'}")
    print(f"   Volatility Diff:     {vol_diff:.6f} {'✅' if vol_diff < 0.01 else '⚠️'}")
    print(f"   Skewness Diff:       {skew_diff:.4f} {'✅' if skew_diff < 0.5 else '⚠️'}")
    print(f"   Autocorr Diff:       {autocorr_diff:.4f} {'✅' if autocorr_diff < 0.1 else '⚠️'}")

    # Statistical tests (using first synthetic path)
    if synthetic_paths:
        synth_returns = synthetic_paths[0]["Close"].pct_change().dropna()
        tests = perform_statistical_tests(hist_returns, synth_returns)
        validation = perform_advanced_validation(hist_returns, synth_returns)

        print(f"\n🧪 STATISTICAL TESTS (p-values, Path 1):")
        ks_pvalue = tests["ks_test"]["pvalue"]
        mw_pvalue = tests["mann_whitney"]["pvalue"]
        lev_pvalue = tests["levene_test"]["pvalue"]

        print(f"   Kolmogorov-Smirnov:  {ks_pvalue:.4f} {'✅' if ks_pvalue > 0.05 else '⚠️'}")
        print(f"   Mann-Whitney U:      {mw_pvalue:.4f} {'✅' if mw_pvalue > 0.05 else '⚠️'}")
        print(f"   Levene (variance):   {lev_pvalue:.4f} {'✅' if lev_pvalue > 0.05 else '⚠️'}")

        # Advanced validation results
        print(f"\n🔬 ADVANCED VALIDATION (Path 1):")

        # Volatility clustering
        if "volatility_clustering" in validation:
            vol_clust = validation["volatility_clustering"]
            print(
                f"   Volatility Clustering: {vol_clust['difference']:.4f} {'✅' if vol_clust['passed'] else '⚠️'}"
            )
            print(
                f"     Historical: {vol_clust['historical']:.4f}, Synthetic: {vol_clust['synthetic']:.4f}"
            )

        # Rolling volatility
        if "rolling_volatility" in validation:
            roll_vol = validation["rolling_volatility"]
            print(
                f"   Rolling Volatility:   {roll_vol['relative_difference']:.4f} {'✅' if roll_vol['passed'] else '⚠️'}"
            )
            print(
                f"     Historical: {roll_vol['historical_mean']:.6f}, Synthetic: {roll_vol['synthetic_mean']:.6f}"
            )

        # Fat tails
        if "fat_tails" in validation:
            fat_tails = validation["fat_tails"]
            print(
                f"   Fat Tails (kurtosis): {fat_tails['difference']:.4f} {'✅' if fat_tails['passed'] else '⚠️'}"
            )
            print(
                f"     Historical: {fat_tails['historical_kurtosis']:.4f}, Synthetic: {fat_tails['synthetic_kurtosis']:.4f}"
            )

        # Extreme values
        if "extreme_values" in validation:
            extreme = validation["extreme_values"]
            print(
                f"   Extreme Values:       {extreme['max_difference']:.4f} {'✅' if extreme['passed'] else '⚠️'}"
            )
            print(f"     Max percentile difference between historical and synthetic")

    # Individual path summary
    print(f"\n📈 INDIVIDUAL PATH SUMMARY:")
    for i, stats in enumerate(synth_stats_list):
        print(
            f"   Path {i+1}: Mean={stats['mean']:.6f}, Vol={stats['std']:.6f}, Skew={stats['skewness']:.4f}"
        )

    # Overall validation summary
    if synthetic_paths:
        synth_returns = synthetic_paths[0]["Close"].pct_change().dropna()
        tests = perform_statistical_tests(hist_returns, synth_returns)
        validation = perform_advanced_validation(hist_returns, synth_returns)

        # Count passed tests
        passed_tests = 0
        total_tests = 0

        # Statistical tests
        for test_name, test_result in tests.items():
            total_tests += 1
            if test_result.get("pvalue", 0) > 0.05:
                passed_tests += 1

        # Advanced validation tests
        for val_name, val_result in validation.items():
            if isinstance(val_result, dict) and "passed" in val_result:
                total_tests += 1
                if val_result["passed"]:
                    passed_tests += 1

        quality_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\n📊 OVERALL VALIDATION SUMMARY:")
        print(f"   Tests Passed:        {passed_tests}/{total_tests} ({quality_score:.1f}%)")
        print(
            f"   Quality Rating:      {'Excellent' if quality_score >= 80 else 'Good' if quality_score >= 60 else 'Fair' if quality_score >= 40 else 'Poor'}"
        )

    print(f"\n{'='*60}")
    print("Legend: ✅ = Good match, ⚠️ = Potential issue")
    print(
        "Note: p-values > 0.05 indicate synthetic data is statistically similar to historical data"
    )
    print("Note: This tool validates that synthetic data maintains similar statistical properties")
    print("      to the original data, which is essential for Monte Carlo simulations.")
    print(
        "Advanced validation includes volatility clustering, rolling volatility, fat tails, and extreme values."
    )
    print(f"{'='*60}\n")


def main():
    """Main function to run synthetic data quality testing."""
    parser = argparse.ArgumentParser(
        description="Test synthetic data quality against real market data using project's data source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_synthetic_data_working.py AAPL
  python scripts/test_synthetic_data_working.py MSFT --paths 5
  python scripts/test_synthetic_data_working.py GOOGL --period 1y --paths 3
        """,
    )

    parser.add_argument("symbol", help="Stock symbol to analyze (e.g., AAPL, MSFT, GOOGL)")

    parser.add_argument(
        "--paths", type=int, default=3, help="Number of synthetic paths to generate (default: 3)"
    )

    parser.add_argument("--period", default="2y", help="Historical data period (default: 2y)")

    args = parser.parse_args()

    try:
        symbol = args.symbol.upper()
        logger.info(f"Starting synthetic data quality analysis for {symbol}")

        # Fetch historical data (will fallback to sample data if needed)
        historical_data = fetch_historical_data_with_hybrid_source(symbol, args.period)

        # Generate synthetic data paths
        synthetic_paths = generate_synthetic_data_paths(historical_data, symbol, args.paths)

        # Print comprehensive analysis
        print_analysis_summary(historical_data, synthetic_paths, symbol)

        logger.info("Analysis complete!")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
