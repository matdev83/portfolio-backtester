"""
Visual Inspection Tool for Synthetic Data Generator

This module provides comprehensive visual analysis tools to compare historical
and synthetic financial data. It includes:
- Statistical comparison plots
- Return distribution analysis
- Volatility clustering visualization
- Autocorrelation analysis
- GARCH parameter validation plots
- Multi-asset comparison dashboards

Key Features:
- Side-by-side comparison of historical vs synthetic data
- Statistical significance testing
- Interactive plotting capabilities
- Comprehensive reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from portfolio_backtester.monte_carlo.synthetic_data_generator import SyntheticDataGenerator

# Set up logging
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SyntheticDataVisualInspector:
    """
    Comprehensive visual inspection tool for synthetic financial data.

    Provides detailed comparison between historical and synthetic data
    with statistical analysis and visual validation.
    """

    def __init__(self, config: Dict):
        """
        Initialize the visual inspector.

        Args:
            config: Configuration dictionary for data generation
        """
        self.config = config
        self.generator = SyntheticDataGenerator(config)

        # Set up plotting parameters
        self.figsize = (15, 10)
        self.dpi = 100

    def generate_comparison_report(
        self,
        historical_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        num_synthetic_paths: int = 10,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Generate comprehensive comparison report for multiple tickers.

        Args:
            historical_data: Dictionary of ticker -> OHLC DataFrame
            tickers: List of tickers to analyze
            num_synthetic_paths: Number of synthetic paths to generate per ticker
            output_dir: Directory to save plots (optional)

        Returns:
            Dictionary containing analysis results for each ticker
        """
        results = {}

        for ticker in tickers:
            logger.info(f"Generating comparison report for {ticker}")

            if ticker not in historical_data:
                logger.warning(f"No historical data available for {ticker}")
                continue

            try:
                # Generate synthetic data
                synthetic_data = self._generate_multiple_synthetic_paths(
                    historical_data[ticker], num_synthetic_paths, ticker
                )

                # Perform analysis
                analysis_results = self._analyze_ticker_data(
                    historical_data[ticker], synthetic_data, ticker
                )

                # Generate plots
                self._create_comprehensive_plots(
                    historical_data[ticker], synthetic_data, ticker, output_dir
                )

                results[ticker] = analysis_results

            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                continue

        # Generate summary report
        if output_dir:
            self._create_summary_report(results, output_dir)

        return results

    def _generate_multiple_synthetic_paths(
        self, historical_data: pd.DataFrame, num_paths: int, asset_name: str
    ) -> List[pd.DataFrame]:
        """
        Generate multiple synthetic paths for comparison.

        Args:
            historical_data: Historical OHLC data
            num_paths: Number of synthetic paths to generate
            asset_name: Name of the asset

        Returns:
            List of synthetic data DataFrames
        """
        synthetic_paths = []

        # Calculate length based on historical data
        length = len(historical_data)

        for i in range(num_paths):
            try:
                # Set different random seed for each path
                original_seed = self.config.get("random_seed")
                if original_seed is not None:
                    np.random.seed(original_seed + i)

                synthetic_df = self.generator.generate_synthetic_prices(
                    historical_data, length, f"{asset_name}_path_{i+1}"
                )

                synthetic_paths.append(synthetic_df)

            except Exception as e:
                logger.warning(f"Failed to generate synthetic path {i+1} for {asset_name}: {e}")
                continue

        return synthetic_paths

    def _analyze_ticker_data(
        self, historical_data: pd.DataFrame, synthetic_data: List[pd.DataFrame], ticker: str
    ) -> Dict:
        """
        Perform comprehensive statistical analysis.

        Args:
            historical_data: Historical OHLC data
            synthetic_data: List of synthetic data DataFrames
            ticker: Ticker symbol

        Returns:
            Dictionary containing analysis results
        """
        # Calculate historical returns
        hist_returns = historical_data["Close"].pct_change(fill_method=None).dropna()

        # Calculate synthetic returns for each path
        synth_returns_list = []
        for synth_df in synthetic_data:
            synth_returns = synth_df["Close"].pct_change(fill_method=None).dropna()
            synth_returns_list.append(synth_returns)

        # Statistical analysis
        analysis: Dict[str, Any] = {
            "ticker": ticker,
            "historical_stats": self._calculate_statistics(hist_returns),
            "synthetic_stats": [],
            "statistical_tests": {},
            "garch_analysis": {},
        }

        # Individual synthetic path statistics
        for i, synth_returns in enumerate(synth_returns_list):
            stats_dict = self._calculate_statistics(synth_returns)
            stats_dict["path_id"] = i + 1
            analysis["synthetic_stats"].append(stats_dict)

        # Statistical tests
        analysis["statistical_tests"] = self._perform_statistical_tests(
            hist_returns, synth_returns_list
        )

        # GARCH analysis
        analysis["garch_analysis"] = self._analyze_garch_properties(
            hist_returns, synth_returns_list
        )

        return analysis

    def _calculate_statistics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive statistics for return series.

        Args:
            returns: Return series

        Returns:
            Dictionary of statistics
        """
        return {
            "mean": returns.mean(),
            "std": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "min": returns.min(),
            "max": returns.max(),
            "autocorr_lag1": returns.autocorr(lag=1) if len(returns) > 1 else 0,
            "autocorr_squared_lag1": (returns**2).autocorr(lag=1) if len(returns) > 1 else 0,
            "jarque_bera_stat": stats.jarque_bera(returns)[0],
            "jarque_bera_pvalue": stats.jarque_bera(returns)[1],
            "var_95": returns.quantile(0.05),
            "var_99": returns.quantile(0.01),
            "count": len(returns),
        }

    def _perform_statistical_tests(
        self, historical_returns: pd.Series, synthetic_returns_list: List[pd.Series]
    ) -> Dict:
        """
        Perform statistical tests comparing historical and synthetic data.

        Args:
            historical_returns: Historical return series
            synthetic_returns_list: List of synthetic return series

        Returns:
            Dictionary of test results
        """
        tests = {}

        for i, synth_returns in enumerate(synthetic_returns_list):
            path_tests = {}

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(historical_returns, synth_returns)
            path_tests["ks_test"] = {"statistic": ks_stat, "pvalue": ks_pvalue}

            # Anderson-Darling test
            try:
                ad_stat, ad_crit, ad_sig = stats.anderson_ksamp([historical_returns, synth_returns])
                path_tests["anderson_darling"] = {
                    "statistic": np.float64(ad_stat),
                    "critical_values": ad_crit.tolist() if hasattr(ad_crit, "tolist") else list(ad_crit),  # type: ignore[dict-item]
                    "significance_level": np.float64(ad_sig),
                }
            except Exception:
                path_tests["anderson_darling"] = {}

            # Mann-Whitney U test (for location)
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                historical_returns, synth_returns, alternative="two-sided"
            )
            path_tests["mann_whitney"] = {
                "statistic": np.float64(mw_stat),
                "pvalue": np.float64(mw_pvalue),
            }

            # Levene test (for variance)
            lev_stat, lev_pvalue = stats.levene(historical_returns, synth_returns)
            path_tests["levene_test"] = {
                "statistic": np.float64(lev_stat),
                "pvalue": np.float64(lev_pvalue),
            }

            tests[f"path_{i+1}"] = path_tests

        return tests

    def _analyze_garch_properties(
        self, historical_returns: pd.Series, synthetic_returns_list: List[pd.Series]
    ) -> Dict:
        """
        Analyze GARCH properties of historical vs synthetic data.

        Args:
            historical_returns: Historical return series
            synthetic_returns_list: List of synthetic return series

        Returns:
            Dictionary of GARCH analysis results
        """
        garch_analysis: Dict[str, Any] = {}

        # Historical GARCH properties
        hist_analysis = self.generator.analyze_asset_statistics(
            pd.DataFrame({"Close": (1 + historical_returns).cumprod()})
        )

        garch_analysis["historical"] = {
            "garch_params": (
                hist_analysis.garch_params.__dict__ if hist_analysis.garch_params else None
            ),
            "volatility_clustering": hist_analysis.autocorr_squared,
            "tail_index": hist_analysis.tail_index,
        }

        # Synthetic GARCH properties
        garch_analysis["synthetic"] = []

        for i, synth_returns in enumerate(synthetic_returns_list):
            try:
                synth_analysis = self.generator.analyze_asset_statistics(
                    pd.DataFrame({"Close": (1 + synth_returns).cumprod()})
                )

                synth_garch = {
                    "path_id": i + 1,
                    "garch_params": (
                        synth_analysis.garch_params.__dict__
                        if synth_analysis.garch_params
                        else None
                    ),
                    "volatility_clustering": synth_analysis.autocorr_squared,
                    "tail_index": synth_analysis.tail_index,
                }

                garch_analysis["synthetic"].append(synth_garch)

            except Exception as e:
                logger.warning(f"Failed GARCH analysis for synthetic path {i+1}: {e}")
                continue

        return garch_analysis

    def _create_comprehensive_plots(
        self,
        historical_data: pd.DataFrame,
        synthetic_data: List[pd.DataFrame],
        ticker: str,
        output_dir: Optional[str] = None,
    ):
        """
        Create comprehensive comparison plots.

        Args:
            historical_data: Historical OHLC data
            synthetic_data: List of synthetic data DataFrames
            ticker: Ticker symbol
            output_dir: Directory to save plots
        """
        # Calculate returns
        hist_returns = historical_data["Close"].pct_change(fill_method=None).dropna()
        synth_returns_list = [
            df["Close"].pct_change(fill_method=None).dropna() for df in synthetic_data
        ]

        # Create main comparison plot
        self._create_main_comparison_plot(hist_returns, synth_returns_list, ticker, output_dir)

        # Create statistical distribution plots
        self._create_distribution_plots(hist_returns, synth_returns_list, ticker, output_dir)

        # Create autocorrelation plots
        self._create_autocorrelation_plots(hist_returns, synth_returns_list, ticker, output_dir)

        # Create price path comparison
        self._create_price_path_plots(historical_data, synthetic_data, ticker, output_dir)

    def _create_main_comparison_plot(
        self,
        hist_returns: pd.Series,
        synth_returns_list: List[pd.Series],
        ticker: str,
        output_dir: Optional[str] = None,
    ):
        """
        Create main comparison plot with multiple subplots.

        Args:
            hist_returns: Historical returns
            synth_returns_list: List of synthetic returns
            ticker: Ticker symbol
            output_dir: Output directory for saving
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)

        # 1. Return time series comparison
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(hist_returns.index, hist_returns, label="Historical", alpha=0.8, linewidth=1)
        for i, synth_returns in enumerate(synth_returns_list[:3]):  # Show first 3 paths
            ax1.plot(
                synth_returns.index,
                synth_returns,
                label=f"Synthetic {i+1}",
                alpha=0.6,
                linewidth=0.8,
            )
        ax1.set_title(f"{ticker} - Return Time Series Comparison")
        ax1.set_ylabel("Returns")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Return distribution comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.hist(hist_returns, bins=50, alpha=0.7, label="Historical", density=True)
        for i, synth_returns in enumerate(synth_returns_list[:3]):
            ax2.hist(synth_returns, bins=50, alpha=0.5, label=f"Synthetic {i+1}", density=True)
        ax2.set_title(f"{ticker} - Return Distribution Comparison")
        ax2.set_xlabel("Returns")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Q-Q plot
        ax3 = fig.add_subplot(gs[1, 0])
        stats.probplot(np.asarray(hist_returns, dtype=np.float64), dist="norm", plot=ax3)  # type: ignore[call-overload]
        ax3.set_title(f"{ticker} - Historical Q-Q Plot")
        ax3.grid(True, alpha=0.3)

        # 4. Synthetic Q-Q plot (first path)
        ax4 = fig.add_subplot(gs[1, 1])
        if synth_returns_list:
            stats.probplot(np.asarray(synth_returns_list[0], dtype=np.float64), dist="norm", plot=ax4)  # type: ignore[call-overload]
        ax4.set_title(f"{ticker} - Synthetic Q-Q Plot")
        ax4.grid(True, alpha=0.3)

        # 5. Volatility comparison
        ax5 = fig.add_subplot(gs[1, 2:])
        hist_vol = hist_returns.rolling(window=20).std()
        ax5.plot(hist_vol.index, hist_vol, label="Historical", linewidth=2)
        for i, synth_returns in enumerate(synth_returns_list[:3]):
            synth_vol = synth_returns.rolling(window=20).std()
            ax5.plot(synth_vol.index, synth_vol, label=f"Synthetic {i+1}", alpha=0.7)
        ax5.set_title(f"{ticker} - 20-Day Rolling Volatility")
        ax5.set_ylabel("Volatility")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Statistics comparison table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")

        # Create statistics table
        stats_data = []
        hist_stats = self._calculate_statistics(hist_returns)
        stats_data.append(
            ["Historical"]
            + [
                f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                for v in [
                    hist_stats["mean"],
                    hist_stats["std"],
                    hist_stats["skewness"],
                    hist_stats["kurtosis"],
                    hist_stats["var_95"],
                    hist_stats["var_99"],
                ]
            ]
        )

        for i, synth_returns in enumerate(synth_returns_list[:5]):  # Show first 5 paths
            synth_stats = self._calculate_statistics(synth_returns)
            stats_data.append(
                [f"Synthetic {i+1}"]
                + [
                    f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                    for v in [
                        synth_stats["mean"],
                        synth_stats["std"],
                        synth_stats["skewness"],
                        synth_stats["kurtosis"],
                        synth_stats["var_95"],
                        synth_stats["var_99"],
                    ]
                ]
            )

        columns = ["Data", "Mean", "Std", "Skewness", "Kurtosis", "VaR 95%", "VaR 99%"]

        table = ax6.table(cellText=stats_data, colLabels=columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.suptitle(
            f"{ticker} - Comprehensive Synthetic Data Analysis", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_path / f"{ticker}_comprehensive_analysis.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

        plt.show()

    def _create_distribution_plots(
        self,
        hist_returns: pd.Series,
        synth_returns_list: List[pd.Series],
        ticker: str,
        output_dir: Optional[str] = None,
    ):
        """
        Create detailed distribution comparison plots.

        Args:
            hist_returns: Historical returns
            synth_returns_list: List of synthetic returns
            ticker: Ticker symbol
            output_dir: Output directory for saving
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Histogram comparison
        axes[0, 0].hist(hist_returns, bins=50, alpha=0.7, label="Historical", density=True)
        for i, synth_returns in enumerate(synth_returns_list[:3]):
            axes[0, 0].hist(
                synth_returns, bins=50, alpha=0.5, label=f"Synthetic {i+1}", density=True
            )
        axes[0, 0].set_title("Return Distribution Comparison")
        axes[0, 0].set_xlabel("Returns")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plots
        box_data = [hist_returns] + synth_returns_list[:5]
        box_labels = ["Historical"] + [
            f"Synthetic {i+1}" for i in range(min(5, len(synth_returns_list)))
        ]
        axes[0, 1].boxplot(box_data, labels=box_labels)
        axes[0, 1].set_title("Return Distribution Box Plots")
        axes[0, 1].set_ylabel("Returns")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Tail comparison (extreme values)
        axes[0, 2].hist(
            hist_returns[hist_returns < hist_returns.quantile(0.05)],
            bins=20,
            alpha=0.7,
            label="Historical Left Tail",
            density=True,
        )
        for i, synth_returns in enumerate(synth_returns_list[:2]):
            left_tail = synth_returns[synth_returns < synth_returns.quantile(0.05)]
            axes[0, 2].hist(
                left_tail, bins=20, alpha=0.5, label=f"Synthetic {i+1} Left Tail", density=True
            )
        axes[0, 2].set_title("Left Tail Comparison (Bottom 5%)")
        axes[0, 2].set_xlabel("Returns")
        axes[0, 2].set_ylabel("Density")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Cumulative distribution
        hist_sorted = np.sort(hist_returns)
        hist_cdf = np.arange(1, len(hist_sorted) + 1) / len(hist_sorted)
        axes[1, 0].plot(hist_sorted, hist_cdf, label="Historical", linewidth=2)

        for i, synth_returns in enumerate(synth_returns_list[:3]):
            synth_sorted = np.sort(synth_returns)
            synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)
            axes[1, 0].plot(synth_sorted, synth_cdf, label=f"Synthetic {i+1}", alpha=0.7)

        axes[1, 0].set_title("Cumulative Distribution Comparison")
        axes[1, 0].set_xlabel("Returns")
        axes[1, 0].set_ylabel("Cumulative Probability")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Variance comparison over time
        hist_var = hist_returns.rolling(window=20).var()
        axes[1, 1].plot(hist_var.index, hist_var, label="Historical", linewidth=2)
        for i, synth_returns in enumerate(synth_returns_list[:3]):
            synth_var = synth_returns.rolling(window=20).var()
            axes[1, 1].plot(synth_var.index, synth_var, label=f"Synthetic {i+1}", alpha=0.7)
        axes[1, 1].set_title("20-Day Rolling Variance")
        axes[1, 1].set_ylabel("Variance")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Skewness and Kurtosis comparison
        metrics_data = []
        hist_stats = self._calculate_statistics(hist_returns)
        metrics_data.append(["Historical", hist_stats["skewness"], hist_stats["kurtosis"]])

        for i, synth_returns in enumerate(synth_returns_list):
            synth_stats = self._calculate_statistics(synth_returns)
            metrics_data.append(
                [f"Synthetic {i+1}", synth_stats["skewness"], synth_stats["kurtosis"]]
            )

        metrics_df = pd.DataFrame(metrics_data, columns=["Data", "Skewness", "Kurtosis"])

        x_pos = np.arange(len(metrics_df))
        width = 0.35

        axes[1, 2].bar(
            x_pos - width / 2, metrics_df["Skewness"], width, label="Skewness", alpha=0.8
        )
        axes[1, 2].bar(
            x_pos + width / 2, metrics_df["Kurtosis"], width, label="Kurtosis", alpha=0.8
        )

        axes[1, 2].set_title("Skewness and Kurtosis Comparison")
        axes[1, 2].set_xlabel("Data Source")
        axes[1, 2].set_ylabel("Value")
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(metrics_df["Data"], rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f"{ticker} - Distribution Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_path / f"{ticker}_distribution_analysis.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

        plt.show()

    def _create_autocorrelation_plots(
        self,
        hist_returns: pd.Series,
        synth_returns_list: List[pd.Series],
        ticker: str,
        output_dir: Optional[str] = None,
    ):
        """
        Create autocorrelation analysis plots.

        Args:
            hist_returns: Historical returns
            synth_returns_list: List of synthetic returns
            ticker: Ticker symbol
            output_dir: Output directory for saving
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        max_lags = min(50, len(hist_returns) // 4)

        # 1. Returns autocorrelation
        hist_autocorr = [hist_returns.autocorr(lag=i) for i in range(1, max_lags + 1)]
        axes[0, 0].plot(
            range(1, max_lags + 1), hist_autocorr, "o-", label="Historical", linewidth=2
        )

        for i, synth_returns in enumerate(synth_returns_list[:3]):
            synth_autocorr = [synth_returns.autocorr(lag=j) for j in range(1, max_lags + 1)]
            axes[0, 0].plot(
                range(1, max_lags + 1), synth_autocorr, "o-", label=f"Synthetic {i+1}", alpha=0.7
            )

        axes[0, 0].set_title("Returns Autocorrelation")
        axes[0, 0].set_xlabel("Lag")
        axes[0, 0].set_ylabel("Autocorrelation")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # 2. Squared returns autocorrelation (volatility clustering)
        hist_sq_autocorr = [(hist_returns**2).autocorr(lag=i) for i in range(1, max_lags + 1)]
        axes[0, 1].plot(
            range(1, max_lags + 1), hist_sq_autocorr, "o-", label="Historical", linewidth=2
        )

        for i, synth_returns in enumerate(synth_returns_list[:3]):
            synth_sq_autocorr = [(synth_returns**2).autocorr(lag=j) for j in range(1, max_lags + 1)]
            axes[0, 1].plot(
                range(1, max_lags + 1), synth_sq_autocorr, "o-", label=f"Synthetic {i+1}", alpha=0.7
            )

        axes[0, 1].set_title("Squared Returns Autocorrelation (Volatility Clustering)")
        axes[0, 1].set_xlabel("Lag")
        axes[0, 1].set_ylabel("Autocorrelation")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # 3. Absolute returns autocorrelation
        hist_abs_autocorr = [hist_returns.abs().autocorr(lag=i) for i in range(1, max_lags + 1)]
        axes[1, 0].plot(
            range(1, max_lags + 1), hist_abs_autocorr, "o-", label="Historical", linewidth=2
        )

        for i, synth_returns in enumerate(synth_returns_list[:3]):
            synth_abs_autocorr = [
                synth_returns.abs().autocorr(lag=j) for j in range(1, max_lags + 1)
            ]
            axes[1, 0].plot(
                range(1, max_lags + 1),
                synth_abs_autocorr,
                "o-",
                label=f"Synthetic {i+1}",
                alpha=0.7,
            )

        axes[1, 0].set_title("Absolute Returns Autocorrelation")
        axes[1, 0].set_xlabel("Lag")
        axes[1, 0].set_ylabel("Autocorrelation")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # 4. Autocorrelation comparison at lag 1
        autocorr_data = []
        autocorr_data.append(
            [
                "Historical",
                hist_returns.autocorr(lag=1),
                (hist_returns**2).autocorr(lag=1),
                hist_returns.abs().autocorr(lag=1),
            ]
        )

        for i, synth_returns in enumerate(synth_returns_list):
            autocorr_data.append(
                [
                    f"Synthetic {i+1}",
                    synth_returns.autocorr(lag=1),
                    (synth_returns**2).autocorr(lag=1),
                    synth_returns.abs().autocorr(lag=1),
                ]
            )

        autocorr_df = pd.DataFrame(
            autocorr_data, columns=["Data", "Returns", "Squared Returns", "Absolute Returns"]
        )

        x_pos = np.arange(len(autocorr_df))
        width = 0.25

        axes[1, 1].bar(x_pos - width, autocorr_df["Returns"], width, label="Returns", alpha=0.8)
        axes[1, 1].bar(
            x_pos, autocorr_df["Squared Returns"], width, label="Squared Returns", alpha=0.8
        )
        axes[1, 1].bar(
            x_pos + width,
            autocorr_df["Absolute Returns"],
            width,
            label="Absolute Returns",
            alpha=0.8,
        )

        axes[1, 1].set_title("Autocorrelation Comparison at Lag 1")
        axes[1, 1].set_xlabel("Data Source")
        axes[1, 1].set_ylabel("Autocorrelation")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(autocorr_df["Data"], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        plt.suptitle(f"{ticker} - Autocorrelation Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_path / f"{ticker}_autocorrelation_analysis.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

        plt.show()

    def _create_price_path_plots(
        self,
        historical_data: pd.DataFrame,
        synthetic_data: List[pd.DataFrame],
        ticker: str,
        output_dir: Optional[str] = None,
    ):
        """
        Create price path comparison plots.

        Args:
            historical_data: Historical OHLC data
            synthetic_data: List of synthetic data DataFrames
            ticker: Ticker symbol
            output_dir: Output directory for saving
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Price paths comparison
        axes[0, 0].plot(
            historical_data.index, historical_data["Close"], label="Historical", linewidth=2
        )
        for i, synth_df in enumerate(synthetic_data[:5]):  # Show first 5 paths
            axes[0, 0].plot(synth_df.index, synth_df["Close"], label=f"Synthetic {i+1}", alpha=0.7)
        axes[0, 0].set_title("Price Paths Comparison")
        axes[0, 0].set_ylabel("Price")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Normalized price paths (starting from 1)
        hist_normalized = historical_data["Close"] / historical_data["Close"].iloc[0]
        axes[0, 1].plot(historical_data.index, hist_normalized, label="Historical", linewidth=2)
        for i, synth_df in enumerate(synthetic_data[:5]):
            synth_normalized = synth_df["Close"] / synth_df["Close"].iloc[0]
            axes[0, 1].plot(synth_df.index, synth_normalized, label=f"Synthetic {i+1}", alpha=0.7)
        axes[0, 1].set_title("Normalized Price Paths (Base = 1)")
        axes[0, 1].set_ylabel("Normalized Price")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Drawdown comparison
        hist_cummax = historical_data["Close"].cummax()
        hist_drawdown = (historical_data["Close"] - hist_cummax) / hist_cummax
        axes[1, 0].plot(historical_data.index, hist_drawdown, label="Historical", linewidth=2)

        for i, synth_df in enumerate(synthetic_data[:3]):
            synth_cummax = synth_df["Close"].cummax()
            synth_drawdown = (synth_df["Close"] - synth_cummax) / synth_cummax
            axes[1, 0].plot(synth_df.index, synth_drawdown, label=f"Synthetic {i+1}", alpha=0.7)

        axes[1, 0].set_title("Drawdown Comparison")
        axes[1, 0].set_ylabel("Drawdown")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        # 4. Final price distribution
        final_prices = [synth_df["Close"].iloc[-1] for synth_df in synthetic_data]
        hist_final = historical_data["Close"].iloc[-1]

        axes[1, 1].hist(
            final_prices, bins=20, alpha=0.7, density=True, label="Synthetic Final Prices"
        )
        axes[1, 1].axvline(
            hist_final,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Historical Final Price: {hist_final:.2f}",
        )
        axes[1, 1].set_title("Final Price Distribution")
        axes[1, 1].set_xlabel("Final Price")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"{ticker} - Price Path Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_path / f"{ticker}_price_path_analysis.png", dpi=self.dpi, bbox_inches="tight"
            )

        plt.show()

    def _create_summary_report(self, results: Dict, output_dir: str):
        """
        Create a summary report of all analyses.

        Args:
            results: Analysis results for all tickers
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create summary statistics table
        summary_data = []
        for ticker, analysis in results.items():
            hist_stats = analysis["historical_stats"]
            synth_stats_avg = self._average_synthetic_stats(analysis["synthetic_stats"])

            summary_data.append(
                {
                    "Ticker": ticker,
                    "Hist_Mean": hist_stats["mean"],
                    "Synth_Mean": synth_stats_avg["mean"],
                    "Hist_Std": hist_stats["std"],
                    "Synth_Std": synth_stats_avg["std"],
                    "Hist_Skew": hist_stats["skewness"],
                    "Synth_Skew": synth_stats_avg["skewness"],
                    "Hist_Kurt": hist_stats["kurtosis"],
                    "Synth_Kurt": synth_stats_avg["kurtosis"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "synthetic_data_summary.csv", index=False)

        logger.info(f"Summary report saved to {output_path / 'synthetic_data_summary.csv'}")

    def _average_synthetic_stats(self, synthetic_stats: List[Dict]) -> Dict:
        """
        Calculate average statistics across synthetic paths.

        Args:
            synthetic_stats: List of statistics dictionaries

        Returns:
            Dictionary of averaged statistics
        """
        if not synthetic_stats:
            return {}

        avg_stats = {}
        for key in synthetic_stats[0].keys():
            if key != "path_id" and isinstance(synthetic_stats[0][key], (int, float)):
                avg_stats[key] = np.mean([stats[key] for stats in synthetic_stats])

        return avg_stats
