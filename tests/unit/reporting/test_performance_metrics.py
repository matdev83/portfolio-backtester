import unittest
import pandas as pd
import numpy as np
from scipy.stats import linregress
from portfolio_backtester.reporting.performance_metrics import calculate_metrics
import warnings

class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        # Create a sample returns series for the portfolio and benchmark
        dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=24, freq='ME'))
        self.rets = pd.Series([0.02, -0.01, 0.03, -0.02] * 6, index=dates, name='Portfolio')
        self.bench_rets = pd.Series([0.01, -0.005, 0.015, -0.01] * 6, index=dates, name='Benchmark')
        self.bench_ticker_name = 'Benchmark'

    def test_calculate_metrics_smoke(self):
        # Smoke test to ensure the function runs without errors
        try:
            calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        except Exception as e:
            self.fail(f"calculate_metrics raised an exception: {e}")

    def test_total_return(self):
        metrics = calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        # Expected: (1.02 * 0.99 * 1.03 * 0.98)^6 - 1
        expected_return = (1.02 * 0.99 * 1.03 * 0.98)**6 - 1
        self.assertAlmostEqual(metrics['Total Return'], expected_return, places=4)

    def test_annualized_return(self):
        metrics = calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        # Expected: ((1 + total_return)^(12/24)) - 1
        total_return = (1.02 * 0.99 * 1.03 * 0.98)**6 - 1
        expected_ann_return = (1 + total_return)**(12/24) - 1
        self.assertAlmostEqual(metrics['Ann. Return'], expected_ann_return, places=4)

    def test_sharpe_ratio(self):
        # Test with a zero volatility series
        zero_vol_rets = pd.Series([0.01] * 24, index=self.rets.index)
        
        # Expect RuntimeWarning from scipy.stats.skew/kurtosis for constant data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning) # Ignore other potential warnings
            with self.assertWarns(RuntimeWarning):
                metrics = calculate_metrics(zero_vol_rets, self.bench_rets, self.bench_ticker_name)
        
        # self.assertTrue(np.isnan(metrics['Sharpe'])) # Old behavior
        # New behavior: Sharpe should be inf if mean return is positive and vol is zero
        # For this specific zero_vol_rets (all 0.01), annualized return is positive.
        self.assertEqual(metrics['Sharpe'], np.inf)


    def test_zero_denominator_metrics(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=252, freq='B')) # Approx 1 year daily
            bench_dummy = pd.Series(0.0, index=dates) # Dummy benchmark

            # --- Test Sharpe Ratio ---
            # Positive return, zero volatility
            rets_pos_zero_vol = pd.Series([0.0001] * 252, index=dates)
            metrics_pos_zero_vol = calculate_metrics(rets_pos_zero_vol, bench_dummy, "Bench")
            self.assertEqual(metrics_pos_zero_vol['Sharpe'], np.inf, "Sharpe: Positive return, zero vol should be inf")

            # Negative return, zero volatility
            rets_neg_zero_vol = pd.Series([-0.0001] * 252, index=dates)
            metrics_neg_zero_vol = calculate_metrics(rets_neg_zero_vol, bench_dummy, "Bench")
            self.assertEqual(metrics_neg_zero_vol['Sharpe'], -np.inf, "Sharpe: Negative return, zero vol should be -inf")

            # Zero return, zero volatility
            rets_zero_zero_vol = pd.Series([0.0] * 252, index=dates)
            metrics_zero_zero_vol = calculate_metrics(rets_zero_zero_vol, bench_dummy, "Bench")
            self.assertEqual(metrics_zero_zero_vol['Sharpe'], 0.0, "Sharpe: Zero return, zero vol should be 0.0")

            # --- Test Calmar Ratio ---
            # Positive return, zero max drawdown (e.g., all positive or zero returns)
            rets_pos_zero_mdd = pd.Series([0.0001] * 252, index=dates) # Also zero vol, but testing MDD effect
            metrics_pos_zero_mdd = calculate_metrics(rets_pos_zero_mdd, bench_dummy, "Bench")
            self.assertEqual(metrics_pos_zero_mdd['Calmar'], np.inf, "Calmar: Positive return, zero MDD should be inf")

            # Negative return, zero max drawdown (e.g., all returns are zero, ann_ret is zero, but if it were neg)
            # This case is tricky: if ann_ret is negative, but MDD is zero (e.g. flat returns then one drop, but we test with flat)
            # For purely flat returns (all 0.0), ann_ret is 0, mdd is 0, Calmar should be 0.0
            rets_zero_zero_mdd = pd.Series([0.0] * 252, index=dates)
            metrics_zero_zero_mdd = calculate_metrics(rets_zero_zero_mdd, bench_dummy, "Bench")
            self.assertEqual(metrics_zero_zero_mdd['Calmar'], 0.0, "Calmar: Zero return, zero MDD should be 0.0")

            # If we had a series that ends up with negative annualized return but somehow zero MDD (hard to construct naturally without manipulation)
            # For instance, if ann_ret was negative and mdd was epsilon. The code path for -np.inf in Calmar:
            # Create a series that has a negative mean but never drawdowns (e.g. starts high, small positive/zeros, ends lower)
            # This is slightly artificial for Calmar as MDD would not be zero if it ends lower than start after positive/zero returns.
            # The primary case for -np.inf is if ann_ret is negative and max_dd is epsilon.
            # Let's test the zero return, zero MDD case as it's most common for "flat" performance.

            # --- Test Sortino Ratio ---
            # Positive mean return, zero downside risk (all returns >= target_return, default 0)
            rets_pos_zero_dr = pd.Series([0.0001] * 252, index=dates)
            metrics_pos_zero_dr = calculate_metrics(rets_pos_zero_dr, bench_dummy, "Bench")
            self.assertEqual(metrics_pos_zero_dr['Sortino'], np.inf, "Sortino: Positive return, zero downside risk should be inf")

            # Negative mean return, zero downside risk (e.g. all returns are 0.0001, but target_return is 0.0002)
            # Or, more simply, all returns are 0.0, mean is 0, downside risk is 0.
            rets_zero_zero_dr = pd.Series([0.0] * 252, index=dates) # target_return = 0
            metrics_zero_zero_dr = calculate_metrics(rets_zero_zero_dr, bench_dummy, "Bench")
            self.assertEqual(metrics_zero_zero_dr['Sortino'], 0.0, "Sortino: Zero return, zero downside risk should be 0.0")

            # Negative mean return, zero downside risk (all returns positive, but mean < target)
            # e.g. r.mean() is -0.0001 (because target is higher than actual returns), downside_risk is 0
            # This would be -np.inf. Let's construct such a case for Sortino.
            # For Sortino, if r.mean() is negative, and downside_risk is zero -> -np.inf
            # This can happen if all returns are positive but less than a positive target_return,
            # or if all returns are negative and target_return is even more negative.
            # Using default target_return = 0: if all returns are positive, r.mean() is positive.
            # If all returns are 0, r.mean() is 0.
            # To get r.mean() < 0 with downside_risk == 0 (w.r.t target=0), all returns must be 0. This leads to Sortino = 0.
            # The -np.inf case for Sortino is when r.mean() < 0 AND downside_risk is zero.
            # This implies that all returns must be >= target_return (so no downside entries) AND the mean of returns must be < 0.
            # If target_return = 0, then all returns must be >=0. For mean to be < 0, this is not possible unless all returns are 0 (mean=0).
            # Let's test the case where mean return is negative, and downside risk is non-zero (standard case)
            # And then the specific case of negative mean return and zero downside risk (which should be -inf)
            # This requires a non-zero target for a more intuitive test of -inf.
            # The current Sortino implementation defaults target=0 for the Sortino function itself.
            # Let's use a series of small negative numbers, which will have non-zero downside risk.
            rets_neg_nonzero_dr = pd.Series([-0.0001] * 252, index=dates)
            metrics_neg_nonzero_dr = calculate_metrics(rets_neg_nonzero_dr, bench_dummy, "Bench")
            self.assertTrue(metrics_neg_nonzero_dr['Sortino'] < 0 and np.isfinite(metrics_neg_nonzero_dr['Sortino']), "Sortino: Negative return, non-zero downside risk should be finite negative")
            # The -np.inf for Sortino happens if mean return is negative AND downside deviation is zero.
            # For example, if target_return = 0.01, and all actual returns are 0.005.
            # Then target_returns are all -0.005. np.minimum(0, target_returns) are all -0.005. downside_risk > 0.
            # This means the -np.inf case for Sortino is hard to hit with target=0 if mean return is also negative.
            # If mean return is negative (e.g. all returns are -0.001) and target=0, then downside_risk > 0.
            # The current implementation of Sortino will return 0 if mean is 0 and downside is 0.
            # It will return inf if mean > 0 and downside is 0.
            # It will return -inf if mean < 0 and downside is 0. This is the path we need to test.
            # To make mean < 0 and downside_risk = 0 (w.r.t. target=0): all returns must be >= 0.
            # This is only possible if all returns are exactly 0. Then mean is 0, so Sortino is 0.
            # So the -np.inf path for Sortino with target=0 is effectively unreachable.
            # The implementation is correct for "if abs(annualized_mean_return) < EPSILON" then 0.0.
            # And "elif annualized_mean_return > 0" then np.inf.
            # And "else" (meaning annualized_mean_return < 0) then -np.inf.
            # This "else" path will be taken if, for example, all returns are 0.0001, but the r.mean() somehow becomes negative.
            # This won't happen.
            # The logic is sound: if downside is zero, result is based on sign of mean return.
            # A series of all zeros will have zero mean and zero downside -> Sortino = 0.0 (Correctly tested)
            # A series of all positive numbers will have positive mean and zero downside -> Sortino = np.inf (Correctly tested)
            # A series of all negative numbers will have negative mean and positive downside -> Sortino = finite negative.
            # The only way to get -np.inf for Sortino is if mean is negative AND downside is zero.
            # This implies returns are all >= target. If target is 0, returns are all >=0. For mean to be negative, they must be all 0.
            # So, the -np.inf path for Sortino with target=0 is practically non-hittable if data is consistent.
            # However, if `r.mean() * steps_per_year` was truly negative and `downside_risk` was zero, it *would* return -np.inf.
            # The tests for 0.0 and np.inf for Sortino are sufficient given its definition with target=0.


    def test_r_squared(self):
        metrics = calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        expected = np.corrcoef(self.rets, self.bench_rets)[0, 1] ** 2
        self.assertAlmostEqual(metrics['R^2'], expected, places=6)

    def test_k_ratio(self):
        metrics = calculate_metrics(self.rets, self.bench_rets, self.bench_ticker_name)
        log_eq = np.log((1 + self.rets).cumprod())
        idx = np.arange(len(log_eq))
        reg = linregress(idx, log_eq)
        expected = (reg.slope / reg.stderr) * np.sqrt(len(log_eq))
        self.assertAlmostEqual(metrics['K-Ratio'], expected, places=6)

if __name__ == '__main__':
    unittest.main()
