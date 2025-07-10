"""
Validation Metrics for Synthetic Data Quality Assessment

This module provides comprehensive validation metrics to assess the quality
of synthetic financial data compared to real historical data. It includes
statistical tests, distribution comparisons, and stylized facts validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import kstest, jarque_bera, anderson
import warnings
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Container for validation test results."""
    test_name: str
    passed: bool
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    critical_value: Optional[float] = None
    details: Optional[Dict] = None


class SyntheticDataValidator:
    """
    Comprehensive validator for synthetic financial data quality.
    
    Performs various statistical tests and comparisons to ensure
    synthetic data preserves key properties of the original data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the validator.
        
        Args:
            config: Validation configuration dictionary
        """
        self.config = config
        self.validation_config = config.get('validation_config', {})
        self.results_history: List[ValidationResults] = []
    
    def validate_synthetic_data(
        self,
        original_data: pd.Series,
        synthetic_data: pd.Series,
        asset_name: str = "Unknown"
    ) -> Dict[str, ValidationResults]:
        """
        Perform comprehensive validation of synthetic data.
        
        Args:
            original_data: Original historical data
            synthetic_data: Synthetic data to validate
            asset_name: Name of the asset for logging
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Basic statistical properties validation
        results['basic_stats'] = self._validate_basic_statistics(
            original_data, synthetic_data, asset_name
        )
        
        # Distribution similarity tests
        results['distribution'] = self._validate_distribution_similarity(
            original_data, synthetic_data, asset_name
        )
        
        # Autocorrelation structure validation
        results['autocorrelation'] = self._validate_autocorrelation_structure(
            original_data, synthetic_data, asset_name
        )
        
        # Volatility clustering validation
        results['volatility_clustering'] = self._validate_volatility_clustering(
            original_data, synthetic_data, asset_name
        )
        
        # Rolling volatility validation
        results['rolling_volatility'] = self._validate_rolling_volatility(
            original_data, synthetic_data, asset_name
        )
        
        # Fat tail validation
        results['fat_tails'] = self._validate_fat_tails(
            original_data, synthetic_data, asset_name
        )
        
        # Extreme value validation
        results['extreme_values'] = self._validate_extreme_values(
            original_data, synthetic_data, asset_name
        )
        
        # Overall quality score
        results['overall_quality'] = self._calculate_overall_quality(results)
        
        # Store results in history
        self.results_history.extend(results.values())
        
        return results
    
    def _validate_basic_statistics(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        asset_name: str
    ) -> ValidationResults:
        """Validate basic statistical properties (mean, std, skewness, kurtosis)."""
        try:
            # Calculate statistics
            orig_stats = {
                'mean': original.mean(),
                'std': original.std(),
                'skewness': original.skew(),
                'kurtosis': original.kurtosis()
            }
            
            synth_stats = {
                'mean': synthetic.mean(),
                'std': synthetic.std(),
                'skewness': synthetic.skew(),
                'kurtosis': synthetic.kurtosis()
            }
            
            # Calculate relative differences
            tolerance = self.validation_config.get('basic_stats_tolerance', 0.2)
            differences = {}
            passed = True
            
            for stat_name in orig_stats:
                orig_val = orig_stats[stat_name]
                synth_val = synth_stats[stat_name]
                
                if abs(orig_val) > 1e-8:  # Avoid division by zero
                    rel_diff = abs(synth_val - orig_val) / abs(orig_val)
                else:
                    rel_diff = abs(synth_val - orig_val)
                
                differences[stat_name] = rel_diff
                
                if rel_diff > tolerance:
                    passed = False
            
            logger.debug(f"Basic Stats for {asset_name}:")
            logger.debug(f"  Original: {orig_stats}")
            logger.debug(f"  Synthetic: {synth_stats}")
            logger.debug(f"  Differences: {differences}")
            
            return ValidationResults(
                test_name="basic_statistics",
                passed=passed,
                details={
                    'original_stats': orig_stats,
                    'synthetic_stats': synth_stats,
                    'relative_differences': differences,
                    'tolerance': tolerance
                }
            )
            
        except Exception as e:
            logger.error(f"Basic statistics validation failed for {asset_name}: {e}")
            return ValidationResults(
                test_name="basic_statistics",
                passed=False,
                details={'error': str(e)}
            )
    
    def _validate_distribution_similarity(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        asset_name: str
    ) -> ValidationResults:
        """Validate distribution similarity using Kolmogorov-Smirnov test."""
        try:
            # Remove NaN values
            orig_clean = original.dropna()
            synth_clean = synthetic.dropna()
            
            if len(orig_clean) < 10 or len(synth_clean) < 10:
                logger.warning(f"Insufficient data for KS test for {asset_name}. Original: {len(orig_clean)}, Synthetic: {len(synth_clean)}")
                return ValidationResults(
                    test_name="distribution_similarity",
                    passed=False,
                    details={'error': 'Insufficient data for KS test'}
                )
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(orig_clean, synth_clean)
            
            # Check if distributions are similar
            threshold = self.validation_config.get('ks_test_pvalue_threshold', 0.05)
            passed = p_value > threshold
            
            logger.debug(f"KS Test for {asset_name}: Statistic={ks_statistic:.4f}, P-value={p_value:.4f}, Threshold={threshold:.4f}, Passed={passed}")
            
            return ValidationResults(
                test_name="distribution_similarity",
                passed=passed,
                p_value=float(p_value),
                statistic=float(ks_statistic),
                critical_value=threshold,
                details={
                    'test_type': 'Kolmogorov-Smirnov',
                    'interpretation': 'Higher p-value indicates more similar distributions'
                }
            )
            
        except Exception as e:
            logger.error(f"Distribution similarity validation failed for {asset_name}: {e}")
            return ValidationResults(
                test_name="distribution_similarity",
                passed=False,
                details={'error': str(e)}
            )
    
    def _validate_autocorrelation_structure(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        asset_name: str
    ) -> ValidationResults:
        """Validate autocorrelation structure."""
        try:
            # Calculate autocorrelations for multiple lags
            max_lags = min(20, len(original) // 4)
            lags = range(1, max_lags + 1)
            
            orig_autocorr = [original.autocorr(lag=lag) for lag in lags]
            synth_autocorr = [synthetic.autocorr(lag=lag) for lag in lags]
            
            # Filter out NaN values from autocorrelations
            valid_autocorr_pairs = [(o, s) for o, s in zip(orig_autocorr, synth_autocorr)
                                    if not (pd.isna(o) or pd.isna(s))]
            
            if not valid_autocorr_pairs:
                logger.warning(f"No valid autocorrelation values for {asset_name}. Skipping autocorrelation validation.")
                return ValidationResults(
                    test_name="autocorrelation_structure",
                    passed=False,
                    details={'error': 'No valid autocorrelation values'}
                )
            
            autocorr_diffs = [abs(o - s) for o, s in valid_autocorr_pairs]
            
            # Check if autocorrelation structure is preserved
            tolerance = self.validation_config.get('autocorr_max_deviation', 0.1)
            max_diff = max(autocorr_diffs)
            avg_diff = np.mean(autocorr_diffs)
            
            passed = max_diff < tolerance
            
            logger.debug(f"Autocorrelation for {asset_name}: Max Diff={max_diff:.4f}, Avg Diff={avg_diff:.4f}, Tolerance={tolerance:.4f}, Passed={passed}")
            
            return ValidationResults(
                test_name="autocorrelation_structure",
                passed=passed,
                details={
                    'max_difference': max_diff,
                    'avg_difference': avg_diff,
                    'tolerance': tolerance,
                    'original_autocorr': orig_autocorr[:5],  # First 5 lags
                    'synthetic_autocorr': synth_autocorr[:5]
                }
            )
            
        except Exception as e:
            logger.error(f"Autocorrelation validation failed for {asset_name}: {e}")
            return ValidationResults(
                test_name="autocorrelation_structure",
                passed=False,
                details={'error': str(e)}
            )
    
    def _validate_volatility_clustering(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        asset_name: str
    ) -> ValidationResults:
        """Validate volatility clustering (autocorrelation in squared returns)."""
        try:
            # Calculate squared returns
            orig_squared = original ** 2
            synth_squared = synthetic ** 2
            
            # Calculate autocorrelations for squared returns
            max_lags = min(10, len(original) // 4)
            lags = range(1, max_lags + 1)
            
            orig_vol_autocorr = [orig_squared.autocorr(lag=lag) for lag in lags]
            synth_vol_autocorr = [synth_squared.autocorr(lag=lag) for lag in lags]
            
            # Focus on first few lags (most important for volatility clustering)
            first_lag_orig = orig_vol_autocorr[0] if orig_vol_autocorr else 0
            first_lag_synth = synth_vol_autocorr[0] if synth_vol_autocorr else 0
            
            # Check if volatility clustering is preserved
            threshold = self.validation_config.get('volatility_clustering_threshold', 0.05)
            clustering_diff = abs(first_lag_orig - first_lag_synth)
            
            # Also check that synthetic data shows volatility clustering
            has_clustering = first_lag_synth > threshold
            clustering_preserved = clustering_diff < 0.1 # More lenient for clustering preservation
            
            passed = has_clustering and clustering_preserved
            
            logger.debug(f"Volatility Clustering for {asset_name}: Orig Lag1={first_lag_orig:.4f}, Synth Lag1={first_lag_synth:.4f}, Diff={clustering_diff:.4f}, Passed={passed}")
            
            return ValidationResults(
                test_name="volatility_clustering",
                passed=passed,
                details={
                    'original_lag1_autocorr': first_lag_orig,
                    'synthetic_lag1_autocorr': first_lag_synth,
                    'difference': clustering_diff,
                    'has_clustering': has_clustering,
                    'clustering_preserved': clustering_preserved,
                    'threshold': threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Volatility clustering validation failed for {asset_name}: {e}")
            return ValidationResults(
                test_name="volatility_clustering",
                passed=False,
                details={'error': str(e)}
            )
    
    def _validate_rolling_volatility(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        asset_name: str
    ) -> ValidationResults:
        """Validate rolling volatility similarity."""
        try:
            window = self.validation_config.get('rolling_volatility_window', 22) # Default to 22 trading days
            tolerance = self.validation_config.get('rolling_volatility_tolerance', 0.3) # 30% tolerance
            
            if len(original) < window or len(synthetic) < window:
                logger.warning(f"Insufficient data for rolling volatility validation for {asset_name}. Need at least {window} observations.")
                return ValidationResults(
                    test_name="rolling_volatility",
                    passed=False,
                    details={'error': 'Insufficient data for rolling volatility validation'}
                )
            
            orig_rolling_std = original.rolling(window=window).std().dropna()
            synth_rolling_std = synthetic.rolling(window=window).std().dropna()
            
            # Ensure both series have comparable length after rolling window
            min_len = min(len(orig_rolling_std), len(synth_rolling_std))
            if min_len == 0:
                logger.warning(f"No valid rolling standard deviations for {asset_name}. Skipping rolling volatility validation.")
                return ValidationResults(
                    test_name="rolling_volatility",
                    passed=False,
                    details={'error': 'No valid rolling standard deviations'}
                )
            
            orig_rolling_std = orig_rolling_std.iloc[-min_len:]
            synth_rolling_std = synth_rolling_std.iloc[-min_len:]
            
            # Calculate mean absolute percentage error (MAPE) or similar
            # Avoid division by zero for very low volatility periods
            diffs = np.abs(synth_rolling_std - orig_rolling_std)
            
            # Use a robust comparison, e.g., mean absolute difference relative to mean original volatility
            if orig_rolling_std.mean() > 1e-8:
                relative_diff = diffs.mean() / orig_rolling_std.mean()
            else:
                relative_diff = diffs.mean() # If original is near zero, just check absolute diff
            
            passed = relative_diff < tolerance
            
            logger.debug(f"Rolling Volatility for {asset_name}: Relative Diff={relative_diff:.4f}, Tolerance={tolerance:.4f}, Passed={passed}")
            
            return ValidationResults(
                test_name="rolling_volatility",
                passed=passed,
                details={
                    'window': window,
                    'original_mean_rolling_std': orig_rolling_std.mean(),
                    'synthetic_mean_rolling_std': synth_rolling_std.mean(),
                    'relative_difference': relative_diff,
                    'tolerance': tolerance
                }
            )
            
        except Exception as e:
            logger.error(f"Rolling volatility validation failed for {asset_name}: {e}")
            return ValidationResults(
                test_name="rolling_volatility",
                passed=False,
                details={'error': str(e)}
            )
    
    def _validate_fat_tails(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        asset_name: str
    ) -> ValidationResults:
        """Validate fat tail properties using Hill estimator and kurtosis."""
        try:
            # Estimate tail indices
            orig_tail_index = self._estimate_tail_index(original)
            synth_tail_index = self._estimate_tail_index(synthetic)
            
            # Compare tail indices
            tolerance = self.validation_config.get('tail_index_tolerance', 0.5)
            tail_diff = abs(orig_tail_index - synth_tail_index)
            
            passed_tail_index = tail_diff < tolerance
            
            # Additional fat tail tests using kurtosis
            orig_kurtosis = original.kurtosis()
            synth_kurtosis = synthetic.kurtosis()
            
            # Ensure kurtosis values are floats for comparison using pd.to_numeric
            orig_kurtosis_val = pd.to_numeric(orig_kurtosis, errors='coerce')
            synth_kurtosis_val = pd.to_numeric(synth_kurtosis, errors='coerce')

            # Check if both show excess kurtosis (fat tails)
            excess_kurtosis_orig = orig_kurtosis_val > 3
            excess_kurtosis_synth = synth_kurtosis_val > 3

            # A more robust check for kurtosis: ensure synthetic kurtosis is within a reasonable range of original
            kurtosis_tolerance = self.validation_config.get('kurtosis_tolerance', 1.0) # Absolute difference
            passed_kurtosis = abs(orig_kurtosis_val - synth_kurtosis_val) < kurtosis_tolerance
            
            # Overall pass if either tail index or kurtosis check passes (more lenient)
            passed = passed_tail_index or passed_kurtosis
            
            logger.debug(f"Fat Tails for {asset_name}: Orig Tail Index={orig_tail_index:.2f}, Synth Tail Index={synth_tail_index:.2f}, Diff={tail_diff:.2f}, Passed Tail Index={passed_tail_index}")
            logger.debug(f"  Orig Kurtosis={orig_kurtosis_val:.2f}, Synth Kurtosis={synth_kurtosis_val:.2f}, Passed Kurtosis={passed_kurtosis}, Overall Passed={passed}")
            
            return ValidationResults(
                test_name="fat_tails",
                passed=passed,
                details={
                    'original_tail_index': orig_tail_index,
                    'synthetic_tail_index': synth_tail_index,
                    'tail_index_difference': tail_diff,
                    'tail_index_tolerance': tolerance,
                    'original_kurtosis': orig_kurtosis,
                    'synthetic_kurtosis': synth_kurtosis,
                    'kurtosis_tolerance': kurtosis_tolerance,
                    'excess_kurtosis_orig': excess_kurtosis_orig,
                    'excess_kurtosis_synth': excess_kurtosis_synth
                }
            )
            
        except Exception as e:
            logger.error(f"Fat tail validation failed for {asset_name}: {e}")
            return ValidationResults(
                test_name="fat_tails",
                passed=False,
                details={'error': str(e)}
            )
    
    def _validate_extreme_values(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        asset_name: str
    ) -> ValidationResults:
        """Validate extreme value characteristics."""
        try:
            # Calculate extreme value statistics
            # Ensure enough data points for percentiles
            if len(original.dropna()) < 100 or len(synthetic.dropna()) < 100:
                logger.warning(f"Insufficient data for extreme value validation for {asset_name}. Need at least 100 observations.")
                return ValidationResults(
                    test_name="extreme_values",
                    passed=False,
                    details={'error': 'Insufficient data for extreme value validation'}
                )
            
            orig_percentiles = np.percentile(original.dropna(), [1, 5, 95, 99])
            synth_percentiles = np.percentile(synthetic.dropna(), [1, 5, 95, 99])
            
            # Compare extreme percentiles
            percentile_diffs = np.abs(orig_percentiles - synth_percentiles)
            
            # Check if extreme values are reasonably similar
            tolerance = self.validation_config.get('extreme_value_tolerance', 0.5) # Absolute difference tolerance
            max_percentile_diff = np.max(percentile_diffs)
            
            passed = max_percentile_diff < tolerance
            
            logger.debug(f"Extreme Values for {asset_name}: Max Percentile Diff={max_percentile_diff:.4f}, Tolerance={tolerance:.4f}, Passed={passed}")
            logger.debug(f"  Original Percentiles: {orig_percentiles}")
            logger.debug(f"  Synthetic Percentiles: {synth_percentiles}")
            
            return ValidationResults(
                test_name="extreme_values",
                passed=passed,
                details={
                    'original_percentiles': {
                        '1%': orig_percentiles[0],
                        '5%': orig_percentiles[1],
                        '95%': orig_percentiles[2],
                        '99%': orig_percentiles[3]
                    },
                    'synthetic_percentiles': {
                        '1%': synth_percentiles[0],
                        '5%': synth_percentiles[1],
                        '95%': synth_percentiles[2],
                        '99%': synth_percentiles[3]
                    },
                    'max_difference': max_percentile_diff,
                    'tolerance': tolerance
                }
            )
            
        except Exception as e:
            logger.error(f"Extreme value validation failed for {asset_name}: {e}")
            return ValidationResults(
                test_name="extreme_values",
                passed=False,
                details={'error': str(e)}
            )
    
    def _calculate_overall_quality(self, results: Dict[str, ValidationResults]) -> ValidationResults:
        """Calculate overall quality score based on individual test results."""
        try:
            # Count passed tests
            test_results = [r for r in results.values() if r.test_name != 'overall_quality']
            passed_tests = sum(1 for r in test_results if r.passed)
            total_tests = len(test_results)
            
            if total_tests == 0:
                logger.warning("No individual validation tests performed. Overall quality cannot be calculated.")
                return ValidationResults(
                    test_name="overall_quality",
                    passed=False,
                    details={'error': 'No tests performed'}
                )
            
            # Calculate quality score
            quality_score = passed_tests / total_tests
            
            # Determine overall pass/fail
            quality_threshold = self.validation_config.get('overall_quality_threshold', 0.7)
            passed = quality_score >= quality_threshold
            
            logger.debug(f"Overall Quality: Score={quality_score:.3f}, Passed Tests={passed_tests}/{total_tests}, Threshold={quality_threshold:.3f}, Overall Passed={passed}")
            
            return ValidationResults(
                test_name="overall_quality",
                passed=passed,
                details={
                    'quality_score': quality_score,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'threshold': quality_threshold,
                    'individual_results': {r.test_name: r.passed for r in test_results}
                }
            )
            
        except Exception as e:
            logger.error(f"Overall quality calculation failed: {e}")
            return ValidationResults(
                test_name="overall_quality",
                passed=False,
                details={'error': str(e)}
            )
    
    def _estimate_tail_index(self, data: pd.Series) -> float:
        """Estimate tail index using Hill estimator."""
        try:
            # Use absolute values for tail estimation
            abs_data = np.abs(data.dropna())
            
            if len(abs_data) < 20:
                logger.warning(f"Insufficient data ({len(abs_data)}) for Hill tail index estimation. Returning default 3.0.")
                return 3.0  # Default assumption
            
            # Sort in descending order
            sorted_data = np.sort(abs_data)[::-1]
            
            # Use top 10% for Hill estimator, but at least 10 observations
            k = max(10, int(0.1 * len(sorted_data)))
            
            # Ensure k does not exceed available data
            k = min(k, len(sorted_data) - 1)
            
            if k <= 0:
                logger.warning(f"Not enough data points for Hill estimator after filtering. Returning default 3.0.")
                return 3.0
            
            # Hill estimator
            # Ensure sorted_data[k] is not zero to avoid division by zero in log
            if sorted_data[k] == 0:
                logger.warning(f"Value at k-th position is zero in Hill estimator. Returning default 3.0.")
                return 3.0
            
            log_ratios = np.log(sorted_data[:k] / sorted_data[k])
            
            # Handle cases where log_ratios might contain inf or NaN
            if np.any(np.isinf(log_ratios)) or np.any(np.isnan(log_ratios)):
                logger.warning(f"Invalid log ratios in Hill estimator. Returning default 3.0.")
                return 3.0
            
            mean_log_ratios = np.mean(log_ratios)
            if mean_log_ratios == 0:
                logger.warning(f"Mean log ratios is zero in Hill estimator. Returning default 3.0.")
                return 3.0
            
            tail_index = 1.0 / mean_log_ratios
            
            # Bound the estimate to reasonable financial values
            return np.clip(tail_index, 1.5, 10.0)
            
        except Exception as e:
            logger.error(f"Hill tail index estimation failed: {e}. Returning default 3.0.")
            return 3.0  # Default fallback
    
    def generate_validation_report(
        self,
        results: Dict[str, ValidationResults],
        asset_name: str
    ) -> str:
        """Generate a comprehensive validation report."""
        report_lines = [
            f"=== Synthetic Data Validation Report for {asset_name} ===",
            ""
        ]
        
        # Overall quality
        overall = results.get('overall_quality')
        if overall and overall.details: # Add check for overall.details
            status = "PASSED" if overall.passed else "FAILED"
            score = overall.details.get('quality_score', 0) * 100
            report_lines.extend([
                f"Overall Quality: {status} ({score:.1f}%)",
                ""
            ])
        
        # Individual test results
        for test_name, result in results.items():
            if test_name == 'overall_quality':
                continue
            
            status = "PASS" if result.passed else "FAIL"
            report_lines.append(f"{test_name.replace('_', ' ').title()}: {status}")
            
            if result.p_value is not None:
                report_lines.append(f"  P-value: {result.p_value:.4f}")
            
            if result.details:
                for key, value in result.details.items():
                    # Only print simple values or nested dicts for stats
                    if key not in ['error'] and not isinstance(value, dict):
                        report_lines.append(f"  {key}: {value}")
                    elif isinstance(value, dict):
                        # For nested dictionaries like 'original_stats', print each item
                        report_lines.append(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            report_lines.append(f"    {sub_key}: {sub_value:.4f}" if isinstance(sub_value, (int, float)) else f"    {sub_key}: {sub_value}")
        
        report_lines.append("")
        return "\n".join(report_lines)
    
    def get_validation_summary(self) -> Dict:
        """Get summary statistics of all validation results."""
        if not self.results_history:
            return {"total_validations": 0}
        
        total_validations = len(self.results_history)
        passed_validations = sum(1 for r in self.results_history if r.passed)
        
        # Group by test type
        test_type_stats = {}
        for result in self.results_history:
            test_name = result.test_name
            if test_name not in test_type_stats:
                test_type_stats[test_name] = {'total': 0, 'passed': 0}
            
            test_type_stats[test_name]['total'] += 1
            if result.passed:
                test_type_stats[test_name]['passed'] += 1
        
        # Calculate pass rates
        for test_name in test_type_stats:
            stats = test_type_stats[test_name]
            stats['pass_rate'] = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'overall_pass_rate': passed_validations / total_validations if total_validations > 0 else 0,
            'test_type_statistics': test_type_stats
        }