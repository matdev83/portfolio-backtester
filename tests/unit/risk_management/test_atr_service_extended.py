import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.risk_management.atr_service import OptimizedATRService

class TestOptimizedATRService:
    @pytest.fixture
    def service(self):
        return OptimizedATRService(cache_size=10)

    @pytest.fixture
    def ohlc_data(self):
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        tickers = ["AAPL", "MSFT"]
        cols = pd.MultiIndex.from_product([tickers, ["High", "Low", "Close"]])
        
        # Simple data: High=105, Low=95, Close=100
        # True Range ~ 10 (approx)
        data = pd.DataFrame(index=dates, columns=cols)
        data.columns.names = ["Ticker", "Field"] # Critical for service logic
        for t in tickers:
            data[(t, "High")] = 105.0
            data[(t, "Low")] = 95.0
            data[(t, "Close")] = 100.0
            
        return data

    def test_calculate_atr_caching(self, service, ohlc_data):
        date = ohlc_data.index[-1]
        
        # First call
        res1 = service.calculate_atr(ohlc_data, date, atr_length=14)
        info1 = service.cache_info()
        assert info1["size"] == 1
        
        # Second call (same data)
        res2 = service.calculate_atr(ohlc_data, date, atr_length=14)
        info2 = service.cache_info()
        assert info2["size"] == 1 # Should hit cache
        
        pd.testing.assert_series_equal(res1, res2)

    def test_calculate_atr_cache_invalidation_data_change(self, service, ohlc_data):
        date = ohlc_data.index[-1]
        
        # First call
        service.calculate_atr(ohlc_data, date)
        
        # Modify data (change High for last day)
        data_mod = ohlc_data.copy()
        data_mod.loc[date, ("AAPL", "High")] = 200.0
        
        # Second call
        res2 = service.calculate_atr(data_mod, date)
        info2 = service.cache_info()
        
        # Should be a new entry because data hash changed
        assert info2["size"] == 2
        
        val = res2["AAPL"]
        if hasattr(val, "item"):
            val = val.item()
        assert val != 0.0 # ATR should change due to volatility spike

    def test_calculate_atr_insufficient_data(self, service, ohlc_data):
        # ATR length > data length
        res = service.calculate_atr(ohlc_data, ohlc_data.index[0], atr_length=50)
        
        assert "AAPL" in res.index
        assert np.isnan(res["AAPL"])
        assert np.isnan(res["MSFT"])

    def test_calculate_atr_missing_columns(self, service):
        dates = pd.date_range("2023-01-01", periods=10)
        # Missing 'Low'
        cols = pd.MultiIndex.from_product([["AAPL"], ["High", "Close"]])
        data = pd.DataFrame(100.0, index=dates, columns=cols)
        data.columns.names = ["Ticker", "Field"]
        
        res = service.calculate_atr(data, dates[-1])
        assert "AAPL" in res.index
        assert np.isnan(res["AAPL"])

    def test_calculate_atr_simple_format(self, service):
        # Single level columns (Close prices only fallback)
        dates = pd.date_range("2023-01-01", periods=20)
        data = pd.DataFrame(100.0, index=dates, columns=["AAPL", "MSFT"])
        
        # Add volatility in the LAST few days so rolling std is non-zero
        for i in range(15, 20):
            data.loc[dates[i], "AAPL"] = 100.0 + (10.0 * (i % 2))
        
        res = service.calculate_atr(data, dates[-1], atr_length=5)
        # Should use volatility based approximation
        assert not np.isnan(res["AAPL"])
        assert res["AAPL"] > 0.0

    def test_cache_clearing(self, service, ohlc_data):
        service.calculate_atr(ohlc_data, ohlc_data.index[-1])
        assert service.cache_info()["size"] > 0
        
        service.clear_cache()
        assert service.cache_info()["size"] == 0
