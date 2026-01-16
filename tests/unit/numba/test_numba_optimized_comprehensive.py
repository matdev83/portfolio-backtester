import numpy as np
import pytest
from portfolio_backtester.numba_optimized import (
    calculate_returns_fast,
    momentum_scores_fast_vectorized,
    returns_to_prices_fast,
    mdd_fast,
    rolling_mean_fast,
    rolling_cumprod_fast,
    rolling_downside_volatility_fast,
    ema_fast,
    atr_fast_fixed,
    compensated_skew_fast,
    compensated_kurtosis_fast,
    sharpe_fast,
    sortino_fast,
    rolling_sharpe_batch
)

class TestNumbaBasicOps:
    def test_calculate_returns_fast(self):
        prices = np.array([100.0, 101.0, 102.01, 100.0, np.nan, 105.0])
        # Returns:
        # 0: NaN
        # 1: 101/100 - 1 = 0.01
        # 2: 102.01/101 - 1 = 0.01
        # 3: 100/102.01 - 1 ~= -0.0197
        # 4: NaN (current nan)
        # 5: 105/nan - 1 = NaN (prev nan) -> logic: if not np.isnan(prices[i-1])...
        
        returns = calculate_returns_fast(prices)
        
        assert np.isnan(returns[0])
        assert returns[1] == pytest.approx(0.01)
        assert returns[2] == pytest.approx(0.01)
        assert returns[3] == pytest.approx(100.0/102.01 - 1.0)
        assert np.isnan(returns[4])
        assert np.isnan(returns[5])

    def test_momentum_scores_fast_vectorized(self):
        now = np.array([110.0, 90.0, 100.0, np.nan])
        then = np.array([100.0, 100.0, np.nan, 100.0])
        
        # 0: 110/100 - 1 = 0.1
        # 1: 90/100 - 1 = -0.1
        # 2: NaN (then is nan)
        # 3: NaN (now is nan)
        
        scores = momentum_scores_fast_vectorized(now, then)
        
        assert scores[0] == pytest.approx(0.1)
        assert scores[1] == pytest.approx(-0.1)
        assert np.isnan(scores[2])
        assert np.isnan(scores[3])

    def test_returns_to_prices_fast(self):
        returns = np.array([0.1, -0.1, 0.5])
        initial = 100.0
        
        # Prices:
        # 0: 100 * (1+0.1) = 110
        # 1: 110 * (1-0.1) = 99
        # 2: 99 * (1+0.5) = 148.5
        
        prices = returns_to_prices_fast(returns, initial)
        
        assert prices[0] == pytest.approx(110.0)
        assert prices[1] == pytest.approx(99.0)
        assert prices[2] == pytest.approx(148.5)
        
    def test_returns_to_prices_fast_clipping(self):
        # Test clipping safeguards
        returns = np.array([-0.9, 10.0]) # Extreme values
        initial = 100.0
        
        # -0.9 clipped to -0.5
        # 10.0 clipped to 5.0
        
        prices = returns_to_prices_fast(returns, initial)
        
        # 0: 100 * (1 - 0.5) = 50.0
        # 1: 50.0 * (1 + 5.0) = 300.0
        
        assert prices[0] == pytest.approx(50.0)
        assert prices[1] == pytest.approx(300.0)

    def test_mdd_fast(self):
        # 100 -> 110 -> 99 -> 120
        # Max: 100, 110, 110, 120
        # DD: 0, 0, (99/110)-1=-0.1, 0
        # Min DD: -0.1
        
        series = np.array([100.0, 110.0, 99.0, 120.0])
        mdd = mdd_fast(series)
        assert mdd == pytest.approx(-0.1)


class TestNumbaRolling:
    def test_rolling_mean_fast(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = 3
        # 0: NaN
        # 1: NaN
        # 2: mean(1,2,3) = 2.0
        # 3: mean(2,3,4) = 3.0
        # 4: mean(3,4,5) = 4.0
        
        result = rolling_mean_fast(data, window)
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0
        assert result[3] == 3.0
        assert result[4] == 4.0

    def test_rolling_cumprod_fast(self):
        # Returns: 0.1, 0.1
        data = np.array([0.1, 0.1, 0.1])
        window = 2
        # 0: NaN
        # 1: (1.1 * 1.1) - 1 = 0.21
        # 2: (1.1 * 1.1) - 1 = 0.21
        
        result = rolling_cumprod_fast(data, window)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(0.21)
        assert result[2] == pytest.approx(0.21)

    def test_rolling_downside_volatility_fast(self):
        # Only negative returns matter
        data = np.array([0.1, -0.1, -0.1, 0.1, -0.1])
        window = 3
        
        # 0: [0.1] (len 1 < 3//2=1? no, 1 >= 1. valid returns: [0.1]. neg: []. res=0)
        # Wait, window=3. i starts at 2.
        
        # 2: [0.1, -0.1, -0.1]. Valid: 3. >= 1. Neg: [-0.1, -0.1]. 
        # Mean neg: -0.1. Var: 0. Std: 0.
        
        # Let's verify with diverse negative returns
        data = np.array([0.1, -0.1, -0.2])
        # Neg: [-0.1, -0.2]. Mean: -0.15. 
        # Var: ((-0.1 - -0.15)^2 + (-0.2 - -0.15)^2) / (2-1)
        # Var: (0.0025 + 0.0025) / 1 = 0.005
        # Std: sqrt(0.005) ~= 0.07071
        
        result = rolling_downside_volatility_fast(data, 3)
        assert result[2] == pytest.approx(np.sqrt(0.005))


class TestNumbaIndicators:
    def test_ema_fast(self):
        data = np.array([10.0, 11.0, 12.0])
        window = 1 # Alpha = 2/(1+1) = 1.0. EMA = Price.
        result = ema_fast(data, window)
        np.testing.assert_allclose(result, data)
        
        window = 3 # Alpha = 2/4 = 0.5
        # 0: 10.0
        # 1: 0.5*11 + 0.5*10 = 5.5 + 5.0 = 10.5
        # 2: 0.5*12 + 0.5*10.5 = 6.0 + 5.25 = 11.25
        
        result = ema_fast(data, window)
        assert result[0] == 10.0
        assert result[1] == 10.5
        assert result[2] == 11.25

    def test_atr_fast_fixed(self):
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])
        window = 2
        
        # TR:
        # 0: NaN (no prev close). But implementation might handle index 0 specially?
        # Code: range(1, n). tr[0] is initialized to nan.
        # 1: Max(11-10=1, |11-9.5|=1.5, |10-9.5|=0.5) = 1.5
        # 2: Max(12-11=1, |12-10.5|=1.5, |11-10.5|=0.5) = 1.5
        
        # ATR (rolling mean of TR):
        # 0: NaN
        # 1: Window [0, 1]. TR[0] nan, TR[1] 1.5. Valid: 1. Sum: 1.5. Mean: 1.5.
        # 2: Window [1, 2]. TR[1] 1.5, TR[2] 1.5. Valid: 2. Sum: 3.0. Mean: 1.5.
        
        result = atr_fast_fixed(high, low, close, window)
        assert np.isnan(result[0])
        assert result[1] == 1.5
        assert result[2] == 1.5


class TestNumbaStats:
    def test_compensated_skew_fast(self):
        # Normal distribution should have skew near 0
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        skew = compensated_skew_fast(data)
        assert abs(skew) < 0.2 # Roughly symmetric

        # Right skewed
        data = np.concatenate([np.random.normal(0, 1, 100), np.array([10.0, 10.0, 10.0])])
        skew = compensated_skew_fast(data)
        assert skew > 0.0

    def test_compensated_kurtosis_fast(self):
        # Normal dist kurtosis should be near 0 (excess)
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        kurt = compensated_kurtosis_fast(data)
        assert abs(kurt) < 0.3 


class TestNumbaBatch:
    def test_sharpe_fast_batch(self):
        # 2 assets
        # Asset 0: Constant return 0.01 (Std 0 -> Sharpe 0 or handled?)
        # Asset 1: Return 0.01, 0.02, 0.01...
        
        # Code handles low variance: if variance > 1e-10... else 0.0
        
        T = 10
        N = 2
        returns = np.zeros((T, N))
        returns[:, 0] = 0.01 # Constant
        returns[:, 1] = np.random.normal(0.01, 0.01, T)
        
        window = 5
        
        result = sharpe_fast(returns, window)
        
        # Asset 0 should be 0.0 due to zero variance
        assert np.all(result[window-1:, 0] == 0.0)
        
        # Asset 1 should be non-zero
        assert np.all(np.isfinite(result[window-1:, 1]))

    def test_sortino_fast_batch(self):
        T = 10
        N = 1
        returns = np.zeros((T, N))
        returns[:, 0] = np.array([0.01, -0.01] * 5)
        
        window = 4
        result = sortino_fast(returns, window)
        
        assert np.all(np.isfinite(result[window-1:]))

    def test_rolling_sharpe_batch(self):
        # Similar to sharpe_fast but using the batch specific function
        T = 10
        N = 2
        returns = np.random.normal(0.01, 0.01, (T, N))
        window = 5
        
        result = rolling_sharpe_batch(returns, window)
        
        assert result.shape == (T, N)
        assert np.isnan(result[0, 0])
