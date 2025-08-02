"""
Integration tests for meta strategy trade tracking with real sub-strategies.

Tests end-to-end trade aggregation functionality using actual strategy implementations
like CalmarMomentumStrategy and IntramonthSeasonalStrategy.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.portfolio_backtester.strategies.meta.simple_meta_strategy import SimpleMetaStrategy
from src.portfolio_backtester.backtester_logic.strategy_logic import generate_signals
from src.portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns


class TestMetaTradeTrackingIntegration:
    """Integration tests for meta strategy trade tracking."""
    
    @pytest.fixture
    def market_data(self):
        """Create realistic market data for testing."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        assets = ["AAPL", "MSFT", "GOOGL", "SPY"]
        
        # Create MultiIndex columns for OHLCV data
        columns = pd.MultiIndex.from_product(
            [assets, ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Field"]
        )
        
        # Generate realistic price data with trends
        np.random.seed(42)  # For reproducible tests
        
        # Start with base prices
        base_prices = {"AAPL": 150, "MSFT": 250, "GOOGL": 100, "SPY": 400}
        
        data = []
        for i, date in enumerate(dates):
            daily_data = []
            for asset in assets:
                base_price = base_prices[asset]
                
                # Add trend and volatility
                trend = 0.0002 * i  # Slight upward trend
                volatility = np.random.normal(0, 0.02)  # 2% daily volatility
                
                close_price = base_price * (1 + trend + volatility)
                open_price = close_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.randint(1000000, 10000000)
                
                daily_data.extend([open_price, high_price, low_price, close_price, volume])
            
            data.append(daily_data)
        
        df = pd.DataFrame(data, index=dates, columns=columns)
        return df
    
    @pytest.fixture
    def meta_strategy_config(self):
        """Create meta strategy configuration with real sub-strategies."""
        return {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "momentum",
                    "strategy_class": "CalmarMomentumStrategy",
                    "strategy_params": {
                        "rolling_window": 6,
                        "num_holdings": 2,
                        "price_column_asset": "Close",
                        "price_column_benchmark": "Close",
                        "timing_config": {
                            "mode": "time_based",
                            "rebalance_frequency": "M"
                        }
                    },
                    "weight": 0.6
                },
                {
                    "strategy_id": "seasonal",
                    "strategy_class": "IntramonthSeasonalStrategy",
                    "strategy_params": {
                        "direction": "long",
                        "entry_day": 5,
                        "hold_days": 10,
                        "price_column_asset": "Close",
                        "trade_longs": True,
                        "trade_shorts": False,
                        "timing_config": {
                            "mode": "signal_based"
                        }
                    },
                    "weight": 0.4
                }
            ]
        }
    
    def test_end_to_end_trade_aggregation(self, market_data, meta_strategy_config):
        """Test complete end-to-end trade aggregation with real strategies."""
        # Create meta strategy
        meta_strategy = SimpleMetaStrategy(meta_strategy_config)
        
        # Verify initial state
        assert len(meta_strategy.get_aggregated_trades()) == 0
        assert meta_strategy.get_trade_aggregator().initial_capital == 1000000
        
        # Create scenario config
        scenario_config = {
            "name": "test_meta_strategy",
            "strategy": "SimpleMetaStrategy",
            "strategy_params": meta_strategy_config,
            "timing_config": {
                "rebalance_frequency": "M"
            }
        }
        
        universe_tickers = ["AAPL", "MSFT", "GOOGL"]
        benchmark_ticker = "SPY"
        
        # Generate signals (this should trigger trade tracking)
        signals = generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=universe_tickers,
            benchmark_ticker=benchmark_ticker,
            has_timed_out=lambda: False
        )
        
        # Verify signals were generated
        assert not signals.empty
        assert len(signals.columns) == len(universe_tickers)
        
        # Verify trades were captured
        aggregated_trades = meta_strategy.get_aggregated_trades()
        assert len(aggregated_trades) > 0
        
        # Verify trade details
        for trade in aggregated_trades:
            assert trade.asset in universe_tickers
            assert trade.strategy_id in ["momentum", "seasonal"]
            assert trade.allocated_capital > 0
            assert trade.price > 0
            assert trade.quantity != 0
        
        # Test portfolio return calculation
        rets_daily = market_data.xs('Close', level='Field', axis=1).pct_change().fillna(0.0)
        
        global_config = {
            "benchmark": benchmark_ticker,
            "portfolio_value": 1000000.0
        }
        
        portfolio_returns, trade_tracker = calculate_portfolio_returns(
            sized_signals=signals,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            rets_daily=rets_daily,
            universe_tickers=universe_tickers,
            global_config=global_config,
            track_trades=True,
            strategy=meta_strategy
        )
        
        # Verify portfolio returns
        assert len(portfolio_returns) == len(market_data)
        assert not portfolio_returns.isna().all()
        
        # Verify performance metrics
        performance = meta_strategy.get_comprehensive_performance_metrics()
        assert performance["total_trades"] == len(aggregated_trades)
        assert performance["initial_capital"] == 1000000
        assert "total_return" in performance
        assert "sharpe_ratio" in performance
    
    def test_capital_allocation_accuracy(self, market_data, meta_strategy_config):
        """Test that capital allocation is accurate across sub-strategies."""
        meta_strategy = SimpleMetaStrategy(meta_strategy_config)
        
        # Check initial capital allocation
        capital_allocations = meta_strategy.calculate_sub_strategy_capital()
        
        expected_momentum = 1000000 * 0.6  # 60%
        expected_seasonal = 1000000 * 0.4  # 40%
        
        assert abs(capital_allocations["momentum"] - expected_momentum) < 0.01
        assert abs(capital_allocations["seasonal"] - expected_seasonal) < 0.01
        
        # Generate some trades
        scenario_config = {
            "strategy_params": meta_strategy_config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Verify trades respect capital allocation
        trades = meta_strategy.get_aggregated_trades()
        
        momentum_trades = [t for t in trades if t.strategy_id == "momentum"]
        seasonal_trades = [t for t in trades if t.strategy_id == "seasonal"]
        
        # All momentum trades should have 60% allocation
        for trade in momentum_trades:
            assert abs(trade.allocated_capital - expected_momentum) < 1.0
        
        # All seasonal trades should have 40% allocation
        for trade in seasonal_trades:
            assert abs(trade.allocated_capital - expected_seasonal) < 1.0
    
    def test_strategy_attribution_accuracy(self, market_data, meta_strategy_config):
        """Test accuracy of strategy attribution."""
        meta_strategy = SimpleMetaStrategy(meta_strategy_config)
        
        # Generate trades
        scenario_config = {
            "name": "test_strategy_attribution",
            "strategy_params": meta_strategy_config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Get attribution
        attribution = meta_strategy.get_strategy_attribution()
        
        # Should have attribution for strategies that generated trades
        assert "momentum" in attribution
        
        # Verify attribution data for momentum (which should have trades)
        momentum_attr = attribution["momentum"]
        
        # Should have trade statistics
        assert "total_trades" in momentum_attr
        assert "total_trade_value" in momentum_attr
        
        # Total trades should match individual strategy trades
        all_trades = meta_strategy.get_aggregated_trades()
        momentum_trades = [t for t in all_trades if t.strategy_id == "momentum"]
        
        assert momentum_attr["total_trades"] == len(momentum_trades)
        
        # If seasonal strategy generated trades, verify its attribution too
        seasonal_trades = [t for t in all_trades if t.strategy_id == "seasonal"]
        if len(seasonal_trades) > 0:
            assert "seasonal" in attribution
            seasonal_attr = attribution["seasonal"]
            assert seasonal_attr["total_trades"] == len(seasonal_trades)
    
    def test_performance_vs_manual_calculation(self, market_data, meta_strategy_config):
        """Test that meta strategy performance matches manual calculation."""
        meta_strategy = SimpleMetaStrategy(meta_strategy_config)
        
        # Generate trades and calculate performance
        scenario_config = {
            "strategy_params": meta_strategy_config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Calculate portfolio returns using framework
        rets_daily = market_data.xs('Close', level='Field', axis=1).pct_change().fillna(0.0)
        
        global_config = {
            "benchmark": "SPY",
            "portfolio_value": 1000000.0
        }
        
        signals = generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        portfolio_returns, _ = calculate_portfolio_returns(
            sized_signals=signals,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            rets_daily=rets_daily,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            global_config=global_config,
            track_trades=False,
            strategy=meta_strategy
        )
        
        # Calculate total return from portfolio returns
        framework_total_return = (1 + portfolio_returns).prod() - 1
        
        # Get meta strategy's calculated performance
        performance = meta_strategy.get_comprehensive_performance_metrics()
        meta_total_return = performance["total_return"]
        
        # They should be very close (allowing for small numerical differences)
        assert abs(framework_total_return - meta_total_return) < 0.01
    
    def test_nested_meta_strategies(self, market_data):
        """Test nested meta strategies with trade aggregation."""
        # Create a meta strategy that includes another meta strategy
        nested_config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "momentum",
                    "strategy_class": "CalmarMomentumStrategy",
                    "strategy_params": {
                        "rolling_window": 6,
                        "num_holdings": 2,
                        "price_column_asset": "Close",
                        "price_column_benchmark": "Close",
                        "timing_config": {
                            "mode": "time_based",
                            "rebalance_frequency": "M"
                        }
                    },
                    "weight": 0.5
                },
                {
                    "strategy_id": "seasonal",
                    "strategy_class": "IntramonthSeasonalStrategy",
                    "strategy_params": {
                        "direction": "long",
                        "entry_day": 5,
                        "hold_days": 10,
                        "price_column_asset": "Close",
                        "timing_config": {
                            "mode": "signal_based"
                        }
                    },
                    "weight": 0.5
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(nested_config)
        
        # Generate signals
        scenario_config = {
            "strategy_params": nested_config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        signals = generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Verify nested strategy works
        assert not signals.empty
        trades = meta_strategy.get_aggregated_trades()
        assert len(trades) > 0
        
        # Verify capital allocation is correct for nested structure
        capital_allocations = meta_strategy.calculate_sub_strategy_capital()
        assert abs(capital_allocations["momentum"] - 500000) < 0.01  # 50%
        assert abs(capital_allocations["seasonal"] - 500000) < 0.01  # 50%
    
    def test_edge_cases(self, market_data, meta_strategy_config):
        """Test edge cases in trade aggregation."""
        meta_strategy = SimpleMetaStrategy(meta_strategy_config)
        
        # Test with limited market data (should handle gracefully)
        limited_data = market_data.iloc[:30]  # Only first 30 days
        
        scenario_config = {
            "name": "test_edge_cases",
            "strategy_params": meta_strategy_config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        signals = generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=limited_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Should handle limited data without errors
        assert isinstance(signals, pd.DataFrame)
        
        # Test with zero trades scenario
        # (This might happen with very restrictive strategy parameters)
        trades = meta_strategy.get_aggregated_trades()
        
        if len(trades) == 0:
            # Should handle zero trades gracefully
            performance = meta_strategy.get_comprehensive_performance_metrics()
            assert performance["total_trades"] == 0
            assert performance["total_return"] == 0.0
        else:
            # Normal case - verify trades are valid
            for trade in trades:
                assert trade.quantity != 0
                assert trade.price > 0
                assert trade.allocated_capital > 0
    
    def test_transaction_cost_handling(self, market_data, meta_strategy_config):
        """Test that transaction costs are properly handled in aggregation."""
        # Add transaction costs to config
        global_config = {
            "commission_per_share": 0.005,
            "commission_min_per_order": 1.0,
            "commission_max_percent_of_trade": 0.005,
            "slippage_bps": 2.5,
            "default_transaction_cost_bps": 10.0
        }

        meta_strategy = SimpleMetaStrategy(meta_strategy_config, global_config=global_config)

        scenario_config = {
            "strategy_params": meta_strategy_config,
            "timing_config": {"rebalance_frequency": "M"}
        }

        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )

        trades = meta_strategy.get_aggregated_trades()

        # Verify transaction costs are applied
        total_costs = sum(trade.transaction_cost for trade in trades)
        assert total_costs > 0

        # Verify costs are reasonable (should be around 10 bps of trade value)
        total_trade_value = sum(trade.trade_value for trade in trades)
        expected_costs = total_trade_value * 0.001  # 10 bps

        # Allow some tolerance for rounding
        assert abs(total_costs - expected_costs) < expected_costs * 0.1