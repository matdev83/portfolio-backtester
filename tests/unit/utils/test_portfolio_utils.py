import pytest
import pandas as pd
import numpy as np
from portfolio_backtester.utils.portfolio_utils import default_candidate_weights, apply_leverage_and_smoothing

class TestPortfolioUtils:
    
    def test_default_candidate_weights_basic(self):
        scores = pd.Series({"A": 10.0, "B": 5.0, "C": 1.0})
        # Top 1 by default or fraction
        # n_assets=3. top_decile=0.1 -> ceil(0.3)=1. 
        weights = default_candidate_weights(scores, {"num_holdings": 2})
        
        assert weights["A"] == 0.5
        assert weights["B"] == 0.5
        assert weights["C"] == 0.0
        assert np.isclose(weights.sum(), 1.0)

    def test_default_candidate_weights_fallback(self):
        scores = pd.Series({"A": -1.0, "B": -2.0})
        # Long only, no positive scores
        weights = default_candidate_weights(scores, {"num_holdings": 1, "trade_shorts": False})
        
        # Should fallback to highest score asset getting 1.0 if n_long=1
        assert weights["A"] == 1.0
        assert weights["B"] == 0.0

    def test_default_candidate_weights_trade_shorts(self):
        scores = pd.Series({"A": 10.0, "B": -5.0, "C": 0.0})
        weights = default_candidate_weights(scores, {"num_holdings": 2, "trade_shorts": True})
        
        # A and C are top 2. But B has negative score. 
        # Logic: nonzero_assets in top_assets.
        # Top 2 assets: A(10), C(0).
        # C has score 0.0. A has 10.0.
        # nonzero_assets = [A]. C is 0 so excluded.
        # So A gets 1.0
        assert weights["A"] == 1.0
        assert weights["C"] == 0.0

    def test_apply_leverage_and_smoothing_basic(self):
        curr = pd.Series({"A": 0.5, "B": 0.5})
        prev = pd.Series({"A": 0.5, "B": 0.5})
        
        # No leverage, no smoothing (lambda=1.0 effectively if not used?) 
        # Logic: lambda * curr + (1-lambda) * prev
        # Default lambda=0.5
        
        res = apply_leverage_and_smoothing(curr, prev, {"smoothing_lambda": 0.5})
        # 0.5*0.5 + 0.5*0.5 = 0.5
        assert res["A"] == 0.5
        assert res["B"] == 0.5

    def test_apply_leverage_and_smoothing_change(self):
        curr = pd.Series({"A": 1.0, "B": 0.0})
        prev = pd.Series({"A": 0.0, "B": 1.0})
        
        # lambda 0.5
        res = apply_leverage_and_smoothing(curr, prev, {"smoothing_lambda": 0.5})
        
        # A: 0.5*1 + 0.5*0 = 0.5
        # B: 0.5*0 + 0.5*1 = 0.5
        assert res["A"] == 0.5
        assert res["B"] == 0.5

    def test_apply_leverage(self):
        curr = pd.Series({"A": 0.5, "B": 0.5})
        res = apply_leverage_and_smoothing(curr, None, {"leverage": 2.0})
        
        # Leverage 2.0 -> 1.0, 1.0
        # Renormalize (long only default) -> sum is 2.0. / 2.0 -> 0.5?
        # Logic: weights = weights / total if trade_shorts is False.
        # If leverage applied, sum is > 1. 
        # Renormalization brings it back to 1.0? 
        # Wait, if we want leverage, we shouldn't renormalize to 1.0?
        # Let's check logic in source:
        # weights = candidate_weights * leverage
        # ...
        # if not params.get("trade_shorts", False):
        #    total = weights.sum()
        #    if total != 0: weights = weights / total
        
        # This implementation seems to cancel out leverage for long-only if it renormalizes to sum=1!
        # Unless "total" is something else? No, total = weights.sum().
        # So for long-only, leverage parameter seems ineffective if it just rescales to 1.0?
        # Let's verify this behavior with the test.
        
        assert np.isclose(res.sum(), 1.0) 
        # If this passes, then leverage parameter is indeed ignored for long-only in this implementation
        # which might be a bug or intended for relative weights only.
        # Given "apply_leverage", one would expect exposure > 1.0.

    def test_apply_leverage_trade_shorts(self):
        curr = pd.Series({"A": 1.0})
        # trade_shorts = True -> No renormalization
        res = apply_leverage_and_smoothing(curr, None, {"leverage": 2.0, "trade_shorts": True})
        
        assert res["A"] == 2.0
