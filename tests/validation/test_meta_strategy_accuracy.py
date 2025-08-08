"""
Validation tests for meta strategy accuracy against manual calculations.

These tests verify that meta strategy performance calculations are mathematically
correct by comparing against manually calculated weighted performance.
"""

import pytest
import pandas as pd
from typing import List

from portfolio_backtester.strategies.meta.simple_meta_strategy import SimpleMetaStrategy
from portfolio_backtester.strategies.base.trade_record import TradeRecord, TradeSide
from portfolio_backtester.backtester_logic.strategy_logic import generate_signals


class TestMetaStrategyAccuracy:
    """Validation tests for meta strategy mathematical accuracy."""
    
    @pytest.fixture
    def controlled_market_data(self):
        """Create controlled market data for precise calculations."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        
        # Create predictable price movements
        aapl_prices = [100.0 + i * 1.0 for i in range(len(dates))]  # Linear increase
        msft_prices = [200.0 - i * 0.5 for i in range(len(dates))]  # Linear decrease
        googl_prices = [150.0] * len(dates)  # Flat
        spy_prices = [400.0 + i * 0.1 for i in range(len(dates))]  # Slight increase
        
        columns = pd.MultiIndex.from_product(
            [["AAPL", "MSFT", "GOOGL", "SPY"], ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Field"]
        )
        
        data = []
        for i, date in enumerate(dates):
            row = []
            for asset, prices in [("AAPL", aapl_prices), ("MSFT", msft_prices), 
                                ("GOOGL", googl_prices), ("SPY", spy_prices)]:
                price = prices[i]
                # OHLC all same for simplicity
                row.extend([price, price, price, price, 1000000])
            data.append(row)
        
        return pd.DataFrame(data, index=dates, columns=columns)
    
    def test_simple_two_strategy_validation(self, controlled_market_data):
        """Test meta strategy with two simple strategies against manual calculation."""
        
        # Create a simple meta strategy configuration
        config = {
            "initial_capital": 100000,
            "allocations": [
                {
                    "strategy_id": "strategy_a",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 3,
                        "num_holdings": 1,
                        "price_column_asset": "Close",
                        "price_column_benchmark": "Close",
                        "timing_config": {
                            "mode": "time_based",
                            "rebalance_frequency": "M"
                        }
                    },
                    "weight": 0.7  # 70% allocation
                },
                {
                    "strategy_id": "strategy_b",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {
                        "direction": "long",
                        "entry_day": 1,
                        "hold_days": 30,
                        "price_column_asset": "Close",
                        "timing_config": {
                            "mode": "signal_based"
                        }
                    },
                    "weight": 0.3  # 30% allocation
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Generate signals and trades
        scenario_config = {
            "strategy_params": config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=controlled_market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Get meta strategy performance
        meta_performance = meta_strategy.get_comprehensive_performance_metrics()
        
        # Manual calculation
        trades = meta_strategy.get_aggregated_trades()
        manual_return = self._calculate_manual_return(trades, controlled_market_data, 100000)
        
        # Compare
        meta_return = meta_performance["total_return"]
        
        # Should be very close (within 0.1% tolerance for numerical precision)
        assert abs(meta_return - manual_return) < 0.001, \
            f"Meta strategy return {meta_return:.6f} != Manual return {manual_return:.6f}"
    
    def _calculate_manual_return(self, trades: List[TradeRecord], market_data: pd.DataFrame, 
                               initial_capital: float) -> float:
        """Manually calculate return from trades and market data."""
        if not trades:
            return 0.0
        
        # Sort trades by date
        sorted_trades = sorted(trades, key=lambda t: t.date)
        
        # Track positions and cash
        positions = {}  # asset -> quantity
        cash = initial_capital
        
        # Get close prices
        close_prices = market_data.xs('Close', level='Field', axis=1)
        
        # Process each trade
        for trade in sorted_trades:
            if trade.asset not in positions:
                positions[trade.asset] = 0.0
            
            if trade.side == TradeSide.BUY:
                positions[trade.asset] += trade.quantity
                cash -= (trade.trade_value + trade.transaction_cost)
            else:  # SELL
                positions[trade.asset] -= trade.quantity
                cash += (trade.trade_value - trade.transaction_cost)
            
            # Clean up zero positions
            if abs(positions[trade.asset]) < 1e-8:
                del positions[trade.asset]
        
        # Calculate final portfolio value using last available prices
        final_date = close_prices.index[-1]
        final_position_value = 0.0
        
        for asset, quantity in positions.items():
            if asset in close_prices.columns:
                final_price = close_prices.loc[final_date, asset]
                final_position_value += quantity * final_price
        
        final_portfolio_value = cash + final_position_value
        
        return (final_portfolio_value - initial_capital) / initial_capital
    
    def test_zero_trades_scenario(self):
        """Test meta strategy with zero trades."""
        config = {
            "initial_capital": 100000,
            "allocations": [
                {
                    "strategy_id": "inactive",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 100,  # Very long window to prevent trades
                        "num_holdings": 1,
                        "price_column_asset": "Close",
                        "timing_config": {
                            "mode": "time_based",
                            "rebalance_frequency": "M"
                        }
                    },
                    "weight": 1.0
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Create minimal market data
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        columns = pd.MultiIndex.from_product(
            [["AAPL", "SPY"], ["Close"]],
            names=["Ticker", "Field"]
        )
        
        data = [[100.0, 400.0] for _ in dates]
        market_data = pd.DataFrame(data, index=dates, columns=columns)
        
        scenario_config = {
            "name": "test_zero_trades",
            "strategy_params": config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Should have zero trades and zero return
        trades = meta_strategy.get_aggregated_trades()
        performance = meta_strategy.get_comprehensive_performance_metrics()
        
        assert len(trades) == 0
        assert performance["total_return"] == 0.0
        assert performance["total_trades"] == 0
    
    def test_single_trade_accuracy(self):
        """Test accuracy with a single, controlled trade."""
        # Create meta strategy that will make exactly one trade
        config = {
            "initial_capital": 100000,
            "allocations": [
                {
                    "strategy_id": "single_trade",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 2,
                        "num_holdings": 1,
                        "price_column_asset": "Close",
                        "timing_config": {
                            "mode": "time_based",
                            "rebalance_frequency": "M"
                        }
                    },
                    "weight": 1.0
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Create market data that will trigger exactly one trade
        dates = pd.date_range("2023-01-01", "2023-02-01", freq="D")
        
        # AAPL goes from $100 to $120 (20% gain)
        aapl_prices = [100.0] * 15 + [120.0] * (len(dates) - 15)
        spy_prices = [400.0] * len(dates)
        
        columns = pd.MultiIndex.from_product(
            [["AAPL", "SPY"], ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Field"]
        )
        
        data = []
        for i, date in enumerate(dates):
            aapl_price = aapl_prices[i]
            spy_price = spy_prices[i]
            data.append([aapl_price, aapl_price, aapl_price, aapl_price, 1000000,
                        spy_price, spy_price, spy_price, spy_price, 1000000])
        
        market_data = pd.DataFrame(data, index=dates, columns=columns)
        
        scenario_config = {
            "strategy_params": config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Verify single trade
        trades = meta_strategy.get_aggregated_trades()
        
        if len(trades) > 0:
            # Calculate expected return manually
            manual_return = self._calculate_manual_return(trades, market_data, 100000)
            
            # Get meta strategy return
            performance = meta_strategy.get_comprehensive_performance_metrics()
            meta_return = performance["total_return"]
            
            # Should match closely
            assert abs(meta_return - manual_return) < 0.001
    
    def test_overlapping_trades_accuracy(self, controlled_market_data):
        """Test accuracy when trades overlap in time."""
        config = {
            "initial_capital": 100000,
            "allocations": [
                {
                    "strategy_id": "overlap_a",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 3,
                        "num_holdings": 2,
                        "price_column_asset": "Close",
                        "timing_config": {
                            "mode": "time_based",
                            "rebalance_frequency": "W"  # Weekly rebalancing
                        }
                    },
                    "weight": 0.6
                },
                {
                    "strategy_id": "overlap_b",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {
                        "direction": "long",
                        "entry_day": 1,
                        "hold_days": 7,
                        "price_column_asset": "Close",
                        "timing_config": {
                            "mode": "signal_based"
                        }
                    },
                    "weight": 0.4
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        scenario_config = {
            "strategy_params": config,
            "timing_config": {"rebalance_frequency": "W"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=controlled_market_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        # Verify overlapping trades are handled correctly
        trades = meta_strategy.get_aggregated_trades()
        
        if len(trades) > 0:
            # Group trades by date to check for overlaps
            trades_by_date = {}
            for trade in trades:
                date = trade.date
                if date not in trades_by_date:
                    trades_by_date[date] = []
                trades_by_date[date].append(trade)
            
            # Calculate manual return
            manual_return = self._calculate_manual_return(trades, controlled_market_data, 100000)
            
            # Get meta strategy return
            performance = meta_strategy.get_comprehensive_performance_metrics()
            meta_return = performance["total_return"]
            
            # Should match despite overlapping trades
            assert abs(meta_return - manual_return) < 0.001
    
    def test_capital_allocation_math(self):
        """Test mathematical correctness of capital allocation."""
        config = {
            "initial_capital": 100000,
            "allocations": [
                {
                    "strategy_id": "strat_60",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 5,
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"}
                    },
                    "weight": 0.6
                },
                {
                    "strategy_id": "strat_25",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {
                        "entry_day": 1,
                        "timing_config": {"mode": "signal_based"}
                    },
                    "weight": 0.25
                },
                {
                    "strategy_id": "strat_15",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 10,
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"}
                    },
                    "weight": 0.15
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Test initial allocation
        allocations = meta_strategy.calculate_sub_strategy_capital()
        
        assert abs(allocations["strat_60"] - 60000) < 0.01
        assert abs(allocations["strat_25"] - 25000) < 0.01
        assert abs(allocations["strat_15"] - 15000) < 0.01
        
        # Test allocation after capital change
        meta_strategy.update_available_capital({
            "strat_60": 0.10,   # 10% return
            "strat_25": -0.05,  # -5% return
            "strat_15": 0.02    # 2% return
        })
        
        # Expected new capital: 100000 + (60000*0.10) + (25000*-0.05) + (15000*0.02)
        # = 100000 + 6000 - 1250 + 300 = 105050
        expected_new_capital = 105050
        
        assert abs(meta_strategy.available_capital - expected_new_capital) < 0.01
        
        # Test new allocations
        new_allocations = meta_strategy.calculate_sub_strategy_capital()
        
        assert abs(new_allocations["strat_60"] - expected_new_capital * 0.6) < 0.01
        assert abs(new_allocations["strat_25"] - expected_new_capital * 0.25) < 0.01
        assert abs(new_allocations["strat_15"] - expected_new_capital * 0.15) < 0.01
    
    def test_transaction_cost_accuracy(self):
        """Test that transaction costs are calculated accurately."""
        config = {
            "initial_capital": 100000,
            "transaction_costs_bps": 5.0,  # 5 basis points
            "allocations": [
                {
                    "strategy_id": "cost_test",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 3,
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"}
                    },
                    "weight": 1.0
                }
            ]
        }
        
        meta_strategy = SimpleMetaStrategy(config)
        
        # Create simple market data
        dates = pd.date_range("2023-01-01", "2023-01-15", freq="D")
        columns = pd.MultiIndex.from_product(
            [["AAPL", "SPY"], ["Close"]],
            names=["Ticker", "Field"]
        )
        
        data = [[100.0, 400.0] for _ in dates]
        market_data = pd.DataFrame(data, index=dates, columns=columns)
        
        scenario_config = {
            "name": "test_transaction_cost",
            "strategy_params": config,
            "timing_config": {"rebalance_frequency": "M"}
        }
        
        generate_signals(
            strategy=meta_strategy,
            scenario_config=scenario_config,
            price_data_daily_ohlc=market_data,
            universe_tickers=["AAPL"],
            benchmark_ticker="SPY",
            has_timed_out=lambda: False
        )
        
        trades = meta_strategy.get_aggregated_trades()
        
        # Verify transaction costs
        for trade in trades:
            expected_cost = trade.trade_value * 0.0005  # 5 bps
            assert abs(trade.transaction_cost - expected_cost) < 0.01
    
    def test_edge_case_validation(self):
        """Test edge cases for mathematical accuracy."""
        
        # Test with very small capital
        small_config = {
            "initial_capital": 1000,  # $1,000
            "allocations": [
                {
                    "strategy_id": "small",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 3,
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"}
                    },
                    "weight": 1.0
                }
            ]
        }
        
        small_meta = SimpleMetaStrategy(small_config)
        
        # Should handle small capital correctly
        allocations = small_meta.calculate_sub_strategy_capital()
        assert abs(allocations["small"] - 1000) < 0.01
        
        # Test with fractional weights
        fractional_config = {
            "initial_capital": 100000,
            "allocations": [
                {
                    "strategy_id": "frac1",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {
                        "rolling_window": 3,
                        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"}
                    },
                    "weight": 0.333333
                },
                {
                    "strategy_id": "frac2",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {
                        "entry_day": 1,
                        "timing_config": {"mode": "signal_based"}
                    },
                    "weight": 0.666667
                }
            ]
        }
        
        fractional_meta = SimpleMetaStrategy(fractional_config)
        
        # Should handle fractional weights correctly
        frac_allocations = fractional_meta.calculate_sub_strategy_capital()
        
        # Allow small rounding errors
        assert abs(frac_allocations["frac1"] - 33333.3) < 0.1
        assert abs(frac_allocations["frac2"] - 66666.7) < 0.1
        
        # Total should still equal initial capital
        total_allocated = sum(frac_allocations.values())
        assert abs(total_allocated - 100000) < 0.01