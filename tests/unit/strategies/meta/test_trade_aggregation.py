"""
Comprehensive unit tests for trade aggregation functionality.

Tests the mathematical correctness of trade aggregation, capital allocation scaling,
and performance calculation accuracy for meta strategies.
"""

import pytest
import pandas as pd
from datetime import timedelta

from portfolio_backtester.strategies.base.trade_aggregator import TradeAggregator
from portfolio_backtester.strategies.base.trade_record import TradeRecord, TradeSide


class TestTradeAggregationMathematics:
    """Test mathematical correctness of trade aggregation."""
    
    @pytest.fixture
    def aggregator(self):
        """Create a trade aggregator with $1M initial capital."""
        return TradeAggregator(1000000.0)
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        base_date = pd.Timestamp("2023-01-01")
        
        trades = [
            # Strategy 1: Buy AAPL
            TradeRecord(
                date=base_date,
                asset="AAPL",
                quantity=1000,
                price=150.0,
                side=TradeSide.BUY,
                strategy_id="momentum",
                allocated_capital=600000.0,  # 60% allocation
                transaction_cost=15.0,
                trade_value=150000.0
            ),
            # Strategy 2: Buy MSFT
            TradeRecord(
                date=base_date,
                asset="MSFT",
                quantity=500,
                price=200.0,
                side=TradeSide.BUY,
                strategy_id="seasonal",
                allocated_capital=400000.0,  # 40% allocation
                transaction_cost=10.0,
                trade_value=100000.0
            ),
            # Strategy 1: Sell half AAPL position
            TradeRecord(
                date=base_date + timedelta(days=30),
                asset="AAPL",
                quantity=500,
                price=160.0,
                side=TradeSide.SELL,
                strategy_id="momentum",
                allocated_capital=600000.0,
                transaction_cost=8.0,
                trade_value=80000.0
            )
        ]
        
        return trades
    
    def test_trade_recording_accuracy(self, aggregator, sample_trades):
        """Test that trades are recorded accurately."""
        for trade in sample_trades:
            aggregator.track_sub_strategy_trade(trade)
        
        recorded_trades = aggregator.get_aggregated_trades()
        
        # Verify all trades recorded
        assert len(recorded_trades) == 3
        
        # Verify trade details preserved
        for original, recorded in zip(sample_trades, recorded_trades):
            assert recorded.date == original.date
            assert recorded.asset == original.asset
            assert recorded.quantity == original.quantity
            assert recorded.price == original.price
            assert recorded.side == original.side
            assert recorded.strategy_id == original.strategy_id
            assert recorded.allocated_capital == original.allocated_capital
            assert recorded.transaction_cost == original.transaction_cost
    
    def test_capital_allocation_scaling(self, aggregator):
        """Test that trades are properly scaled by capital allocation."""
        # Strategy with 60% allocation ($600k) signals 25% position in AAPL
        # Should result in: 0.25 * $600k = $150k trade
        
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-01"),
            asset="AAPL",
            quantity=1000,  # $150k / $150 = 1000 shares
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=600000.0,
            transaction_cost=15.0,
            trade_value=150000.0
        )
        
        aggregator.track_sub_strategy_trade(trade)
        
        # Verify position reflects correct scaling
        positions = aggregator.get_current_positions()
        assert "AAPL" in positions
        
        aapl_position = positions["AAPL"]
        assert aapl_position.quantity == 1000
        assert abs(aapl_position.market_value - 150000.0) < 0.01
    
    def test_portfolio_value_calculation(self, aggregator):
        """Test portfolio value calculation with market data."""
        # Initial state
        initial_value = aggregator.calculate_portfolio_value(pd.Timestamp("2023-01-01"))
        assert initial_value == 1000000.0
        
        # Add a trade
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-01"),
            asset="AAPL",
            quantity=1000,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=600000.0,
            transaction_cost=15.0,
            trade_value=150000.0
        )
        
        aggregator.track_sub_strategy_trade(trade)
        
        # Create market data with price appreciation
        market_data = pd.DataFrame({
            "AAPL": [150.0, 160.0, 170.0]
        }, index=pd.date_range("2023-01-01", periods=3, freq="D"))
        
        # Update portfolio values with market data
        aggregator.update_portfolio_values_with_market_data(market_data)
        
        # Check portfolio values
        timeline = aggregator.get_portfolio_timeline()
        assert not timeline.empty
        
        # Day 1: $1M - $150k (trade) - $15 (cost) + $150k (position) = $999,985
        day1_value = timeline.loc[pd.Timestamp("2023-01-01"), "portfolio_value"]
        expected_day1 = 1000000.0 - 15.0  # Initial - transaction cost
        assert abs(day1_value - expected_day1) < 1.0
        
        # Day 2: Price up to $160, position worth $160k
        day2_value = timeline.loc[pd.Timestamp("2023-01-02"), "portfolio_value"]
        expected_day2 = (1000000.0 - 150000.0 - 15.0) + (1000 * 160.0)  # Cash + position value
        assert abs(day2_value - expected_day2) < 1.0
    
    def test_performance_calculation_accuracy(self, aggregator):
        """Test accuracy of performance calculations."""
        # Add trades that should result in known performance
        trades = [
            TradeRecord(
                date=pd.Timestamp("2023-01-01"),
                asset="AAPL",
                quantity=1000,
                price=100.0,
                side=TradeSide.BUY,
                strategy_id="test",
                allocated_capital=1000000.0,
                transaction_cost=10.0,
                trade_value=100000.0
            )
        ]
        
        for trade in trades:
            aggregator.track_sub_strategy_trade(trade)
        
        # Create market data showing 50% appreciation
        market_data = pd.DataFrame({
            "AAPL": [100.0, 150.0]
        }, index=pd.date_range("2023-01-01", periods=2, freq="D"))
        
        aggregator.update_portfolio_values_with_market_data(market_data)
        
        # Calculate performance
        performance = aggregator.calculate_weighted_performance()
        
        # Expected: $1M initial, bought $100k of AAPL at $100, now worth $150k
        # Portfolio value: $900k cash + $150k position - $10 cost = $1,049,990
        # Return: ($1,049,990 - $1,000,000) / $1,000,000 = 4.999%
        expected_return = 0.04999
        assert abs(performance["total_return"] - expected_return) < 0.001
    
    def test_multi_strategy_attribution(self, aggregator):
        """Test attribution across multiple strategies."""
        trades = [
            # Momentum strategy: profitable trade
            TradeRecord(
                date=pd.Timestamp("2023-01-01"),
                asset="AAPL",
                quantity=500,
                price=100.0,
                side=TradeSide.BUY,
                strategy_id="momentum",
                allocated_capital=600000.0,
                transaction_cost=5.0,
                trade_value=50000.0
            ),
            # Seasonal strategy: break-even trade
            TradeRecord(
                date=pd.Timestamp("2023-01-01"),
                asset="MSFT",
                quantity=200,
                price=200.0,
                side=TradeSide.BUY,
                strategy_id="seasonal",
                allocated_capital=400000.0,
                transaction_cost=4.0,
                trade_value=40000.0
            )
        ]
        
        for trade in trades:
            aggregator.track_sub_strategy_trade(trade)
        
        # Create market data: AAPL up 20%, MSFT flat
        market_data = pd.DataFrame({
            "AAPL": [100.0, 120.0],
            "MSFT": [200.0, 200.0]
        }, index=pd.date_range("2023-01-01", periods=2, freq="D"))
        
        aggregator.update_portfolio_values_with_market_data(market_data)
        
        # Test attribution
        attribution = aggregator.get_strategy_attribution()
        
        assert "momentum" in attribution
        assert "seasonal" in attribution
        
        # Momentum should have more trades value due to price appreciation
        momentum_stats = attribution["momentum"]
        seasonal_stats = attribution["seasonal"]
        
        assert momentum_stats["total_trades"] == 1
        assert seasonal_stats["total_trades"] == 1
        
        # Both strategies should have their respective trade values
        assert momentum_stats["total_trade_value"] == 50000.0
        assert seasonal_stats["total_trade_value"] == 40000.0
    
    def test_cash_balance_tracking(self, aggregator):
        """Test accurate cash balance tracking."""
        initial_cash = aggregator._cash_balance
        assert initial_cash == 1000000.0
        
        # Buy trade: reduces cash
        buy_trade = TradeRecord(
            date=pd.Timestamp("2023-01-01"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="test",
            allocated_capital=1000000.0,
            transaction_cost=10.0,
            trade_value=15000.0
        )
        
        aggregator.track_sub_strategy_trade(buy_trade)
        
        # Cash should be reduced by trade value + transaction cost
        expected_cash = 1000000.0 - 15000.0 - 10.0
        assert abs(aggregator._cash_balance - expected_cash) < 0.01
        
        # Sell trade: increases cash
        sell_trade = TradeRecord(
            date=pd.Timestamp("2023-01-02"),
            asset="AAPL",
            quantity=50,
            price=160.0,
            side=TradeSide.SELL,
            strategy_id="test",
            allocated_capital=1000000.0,
            transaction_cost=8.0,
            trade_value=8000.0
        )
        
        aggregator.track_sub_strategy_trade(sell_trade)
        
        # Cash should increase by trade value minus transaction cost
        expected_cash = (1000000.0 - 15000.0 - 10.0) + 8000.0 - 8.0
        assert abs(aggregator._cash_balance - expected_cash) < 0.01
    
    def test_position_consolidation(self, aggregator):
        """Test that positions are properly consolidated across trades."""
        # Multiple trades in same asset
        trades = [
            TradeRecord(
                date=pd.Timestamp("2023-01-01"),
                asset="AAPL",
                quantity=100,
                price=150.0,
                side=TradeSide.BUY,
                strategy_id="momentum",
                allocated_capital=600000.0,
                transaction_cost=5.0,
                trade_value=15000.0
            ),
            TradeRecord(
                date=pd.Timestamp("2023-01-02"),
                asset="AAPL",
                quantity=50,
                price=160.0,
                side=TradeSide.BUY,
                strategy_id="seasonal",
                allocated_capital=400000.0,
                transaction_cost=4.0,
                trade_value=8000.0
            ),
            TradeRecord(
                date=pd.Timestamp("2023-01-03"),
                asset="AAPL",
                quantity=25,
                price=155.0,
                side=TradeSide.SELL,
                strategy_id="momentum",
                allocated_capital=600000.0,
                transaction_cost=3.0,
                trade_value=3875.0
            )
        ]
        
        for trade in trades:
            aggregator.track_sub_strategy_trade(trade)
        
        positions = aggregator.get_current_positions()
        assert len(positions) == 1
        assert "AAPL" in positions
        
        aapl_position = positions["AAPL"]
        # Net position: 100 + 50 - 25 = 125 shares
        assert aapl_position.quantity == 125
        
        # Average price should be weighted: (100*150 + 50*160) / 150 = 153.33
        expected_avg_price = (100 * 150.0 + 50 * 160.0) / 150
        assert abs(aapl_position.average_price - expected_avg_price) < 0.01
    
    def test_zero_position_cleanup(self, aggregator):
        """Test that zero positions are properly cleaned up."""
        # Buy and then sell exact same amount
        buy_trade = TradeRecord(
            date=pd.Timestamp("2023-01-01"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="test",
            allocated_capital=1000000.0,
            transaction_cost=5.0,
            trade_value=15000.0
        )
        
        sell_trade = TradeRecord(
            date=pd.Timestamp("2023-01-02"),
            asset="AAPL",
            quantity=100,
            price=160.0,
            side=TradeSide.SELL,
            strategy_id="test",
            allocated_capital=1000000.0,
            transaction_cost=5.0,
            trade_value=16000.0
        )
        
        aggregator.track_sub_strategy_trade(buy_trade)
        
        # Should have position
        positions = aggregator.get_current_positions()
        assert len(positions) == 1
        assert "AAPL" in positions
        
        aggregator.track_sub_strategy_trade(sell_trade)
        
        # Position should be cleaned up
        positions = aggregator.get_current_positions()
        assert len(positions) == 0
    
    def test_returns_calculation(self, aggregator):
        """Test daily returns calculation."""
        # Add a trade
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-01"),
            asset="AAPL",
            quantity=1000,
            price=100.0,
            side=TradeSide.BUY,
            strategy_id="test",
            allocated_capital=1000000.0,
            transaction_cost=10.0,
            trade_value=100000.0
        )
        
        aggregator.track_sub_strategy_trade(trade)
        
        # Create market data with daily price changes
        market_data = pd.DataFrame({
            "AAPL": [100.0, 105.0, 110.0, 108.0]
        }, index=pd.date_range("2023-01-01", periods=4, freq="D"))
        
        aggregator.update_portfolio_values_with_market_data(market_data)
        
        timeline = aggregator.get_portfolio_timeline()
        returns = timeline["returns"]
        
        # Check that returns are calculated correctly
        assert not returns.isna().all()
        
        # Day 2 return should reflect 5% price increase on $100k position
        # Portfolio value change: $5k on $1M base ≈ 0.5% return
        day2_return = returns.iloc[1]  # Second day
        expected_return = 5000.0 / (1000000.0 - 10.0)  # $5k gain on initial portfolio
        assert abs(day2_return - expected_return) < 0.001