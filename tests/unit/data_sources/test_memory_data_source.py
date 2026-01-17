import pytest
import pandas as pd
import numpy as np
from portfolio_backtester.data_sources.memory_data_source import MemoryDataSource

class TestMemoryDataSource:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2020-01-01", "2020-01-05")
        
        # MultiIndex DataFrame for daily data
        tuples = [
            ("AAPL", "Open"), ("AAPL", "Close"),
            ("GOOG", "Open"), ("GOOG", "Close")
        ]
        index = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"])
        
        data = np.random.randn(len(dates), 4)
        daily_df = pd.DataFrame(data, index=dates, columns=index)
        
        # Simple DataFrame for benchmark
        benchmark_df = pd.DataFrame(
            {"SPY": np.random.randn(len(dates))},
            index=dates
        )
        
        return {
            "daily_data": daily_df,
            "benchmark_data": benchmark_df
        }

    def test_initialization(self, sample_data):
        config = {"data_frames": sample_data}
        source = MemoryDataSource(config)
        
        assert source.daily_data is not None
        assert source.benchmark_data is not None
        assert source.daily_data.equals(sample_data["daily_data"])

    def test_get_data_filtering(self, sample_data):
        config = {"data_frames": sample_data}
        source = MemoryDataSource(config)
        
        # Test date filtering
        start = "2020-01-02"
        end = "2020-01-04"
        
        # Test ticker filtering
        tickers = ["AAPL"]
        
        result = source.get_data(tickers, start, end)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3 # 2nd, 3rd, 4th
        assert "AAPL" in result.columns.get_level_values("Ticker")
        assert "GOOG" not in result.columns.get_level_values("Ticker")
        
        # Check values match original
        orig_slice = sample_data["daily_data"].loc[start:end, ("AAPL", slice(None))]
        pd.testing.assert_frame_equal(result, orig_slice)

    def test_get_benchmark_data(self, sample_data):
        config = {"data_frames": sample_data}
        source = MemoryDataSource(config)
        
        start = "2020-01-02"
        end = "2020-01-03"
        
        result = source.get_benchmark_data("SPY", start, end)
        
        assert len(result) == 2
        pd.testing.assert_frame_equal(
            result, 
            sample_data["benchmark_data"].loc[start:end]
        )

    def test_empty_config(self):
        source = MemoryDataSource({})
        assert source.get_data(["AAPL"], "2020-01-01", "2020-01-02").empty
        assert source.get_benchmark_data("SPY", "2020-01-01", "2020-01-02").empty
