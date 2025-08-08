"""Tests for TradeAggregator class."""

import pandas as pd

from portfolio_backtester.strategies.base.trade_aggregator import TradeAggregator
from portfolio_backtester.strategies.base.trade_record import TradeRecord, TradeSide


class TestTradeAggregator:
    """Test cases for TradeAggregator class."""

    def test_trade_aggregator_initialization(self):
        """Test basic TradeAggregator initialization."""
        initial_capital = 100000.0
        aggregator = TradeAggregator(initial_capital)

        assert aggregator.initial_capital == initial_capital
        assert aggregator.current_capital == initial_capital
        assert len(aggregator.get_aggregated_trades()) == 0
        assert len(aggregator.get_current_positions()) == 0
        assert aggregator.calculate_portfolio_value(pd.Timestamp.now()) == initial_capital

    def test_track_single_buy_trade(self):
        """Test tracking a single buy trade."""
        aggregator = TradeAggregator(100000.0)

        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=15.0,
        )

        aggregator.track_sub_strategy_trade(trade)

        # Check trade history
        trades = aggregator.get_aggregated_trades()
        assert len(trades) == 1
        assert trades[0] == trade

        # Check positions
        positions = aggregator.get_current_positions()
        assert len(positions) == 1
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 100
        assert positions["AAPL"].average_price == 150.0

        # Check cash balance (should be reduced by trade value + costs)
        expected_cash = 100000.0 - (15000.0 + 15.0)  # 100000 - 15015
        assert abs(aggregator._cash_balance - expected_cash) < 0.01

    def test_track_single_sell_trade(self):
        """Test tracking a single sell trade."""
        aggregator = TradeAggregator(100000.0)

        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="MSFT",
            quantity=50,
            price=200.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=30000.0,
            transaction_cost=10.0,
        )

        aggregator.track_sub_strategy_trade(trade)

        # Check positions (sell creates negative position)
        positions = aggregator.get_current_positions()
        assert len(positions) == 1
        assert "MSFT" in positions
        assert positions["MSFT"].quantity == -50
        assert positions["MSFT"].is_short

        # Check cash balance (should be increased by trade value - costs)
        expected_cash = 100000.0 + (10000.0 - 10.0)  # 100000 + 9990
        assert abs(aggregator._cash_balance - expected_cash) < 0.01

    def test_track_multiple_trades_same_asset(self):
        """Test tracking multiple trades for the same asset."""
        aggregator = TradeAggregator(100000.0)

        # First buy
        trade1 = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="GOOGL",
            quantity=10,
            price=2500.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=25.0,
        )

        # Second buy at different price
        trade2 = TradeRecord(
            date=pd.Timestamp("2023-01-16"),
            asset="GOOGL",
            quantity=5,
            price=2600.0,
            side=TradeSide.BUY,
            strategy_id="seasonal",
            allocated_capital=30000.0,
            transaction_cost=13.0,
        )

        aggregator.track_sub_strategy_trade(trade1)
        aggregator.track_sub_strategy_trade(trade2)

        # Check trade history
        trades = aggregator.get_aggregated_trades()
        assert len(trades) == 2

        # Check position (should be averaged)
        positions = aggregator.get_current_positions()
        assert len(positions) == 1
        assert "GOOGL" in positions

        position = positions["GOOGL"]
        assert position.quantity == 15  # 10 + 5

        # Average price: (10 * 2500 + 5 * 2600) / 15 = 38000 / 15 = 2533.33
        expected_avg_price = (10 * 2500.0 + 5 * 2600.0) / 15
        assert abs(position.average_price - expected_avg_price) < 0.01

        # Check strategy contributions
        assert position.strategy_contributions["momentum"] == 10
        assert position.strategy_contributions["seasonal"] == 5

    def test_track_buy_then_sell_same_asset(self):
        """Test tracking buy then sell for same asset."""
        aggregator = TradeAggregator(100000.0)

        # Buy first
        buy_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AMZN",
            quantity=20,
            price=3000.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=70000.0,
            transaction_cost=60.0,
        )

        # Sell part of position
        sell_trade = TradeRecord(
            date=pd.Timestamp("2023-01-16"),
            asset="AMZN",
            quantity=8,
            price=3100.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=25000.0,
            transaction_cost=24.8,
        )

        aggregator.track_sub_strategy_trade(buy_trade)
        aggregator.track_sub_strategy_trade(sell_trade)

        # Check position (should be net long)
        positions = aggregator.get_current_positions()
        assert len(positions) == 1
        assert "AMZN" in positions

        position = positions["AMZN"]
        assert position.quantity == 12  # 20 - 8
        assert position.average_price == 3000.0  # Should keep original average
        assert position.strategy_contributions["momentum"] == 20
        assert position.strategy_contributions["seasonal"] == -8

    def test_track_trades_multiple_assets(self):
        """Test tracking trades across multiple assets."""
        aggregator = TradeAggregator(200000.0)

        trades = [
            TradeRecord(
                date=pd.Timestamp("2023-01-15"),
                asset="AAPL",
                quantity=100,
                price=150.0,
                side=TradeSide.BUY,
                strategy_id="momentum",
                allocated_capital=100000.0,
                transaction_cost=15.0,
            ),
            TradeRecord(
                date=pd.Timestamp("2023-01-15"),
                asset="MSFT",
                quantity=50,
                price=200.0,
                side=TradeSide.BUY,
                strategy_id="seasonal",
                allocated_capital=100000.0,
                transaction_cost=10.0,
            ),
            TradeRecord(
                date=pd.Timestamp("2023-01-16"),
                asset="GOOGL",
                quantity=5,
                price=2500.0,
                side=TradeSide.SELL,
                strategy_id="momentum",
                allocated_capital=100000.0,
                transaction_cost=12.5,
            ),
        ]

        for trade in trades:
            aggregator.track_sub_strategy_trade(trade)

        # Check positions
        positions = aggregator.get_current_positions()
        assert len(positions) == 3
        assert "AAPL" in positions
        assert "MSFT" in positions
        assert "GOOGL" in positions

        assert positions["AAPL"].quantity == 100
        assert positions["MSFT"].quantity == 50
        assert positions["GOOGL"].quantity == -5  # Short position

        # Check trade history
        all_trades = aggregator.get_aggregated_trades()
        assert len(all_trades) == 3

    def test_get_trades_by_strategy(self):
        """Test filtering trades by strategy."""
        aggregator = TradeAggregator(100000.0)

        momentum_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
        )

        seasonal_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="MSFT",
            quantity=50,
            price=200.0,
            side=TradeSide.BUY,
            strategy_id="seasonal",
            allocated_capital=50000.0,
        )

        aggregator.track_sub_strategy_trade(momentum_trade)
        aggregator.track_sub_strategy_trade(seasonal_trade)

        # Test filtering
        momentum_trades = aggregator.get_trades_by_strategy("momentum")
        seasonal_trades = aggregator.get_trades_by_strategy("seasonal")

        assert len(momentum_trades) == 1
        assert len(seasonal_trades) == 1
        assert momentum_trades[0] == momentum_trade
        assert seasonal_trades[0] == seasonal_trade

        # Test non-existent strategy
        empty_trades = aggregator.get_trades_by_strategy("nonexistent")
        assert len(empty_trades) == 0

    def test_get_trades_by_asset(self):
        """Test filtering trades by asset."""
        aggregator = TradeAggregator(100000.0)

        aapl_trade1 = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
        )

        aapl_trade2 = TradeRecord(
            date=pd.Timestamp("2023-01-16"),
            asset="AAPL",
            quantity=50,
            price=155.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=30000.0,
        )

        msft_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="MSFT",
            quantity=25,
            price=200.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
        )

        for trade in [aapl_trade1, aapl_trade2, msft_trade]:
            aggregator.track_sub_strategy_trade(trade)

        # Test filtering
        aapl_trades = aggregator.get_trades_by_asset("AAPL")
        msft_trades = aggregator.get_trades_by_asset("MSFT")

        assert len(aapl_trades) == 2
        assert len(msft_trades) == 1
        assert aapl_trade1 in aapl_trades
        assert aapl_trade2 in aapl_trades
        assert msft_trades[0] == msft_trade

    def test_calculate_weighted_performance(self):
        """Test performance calculation."""
        initial_capital = 100000.0
        aggregator = TradeAggregator(initial_capital)

        # Add some trades
        trade1 = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=15.0,
        )

        trade2 = TradeRecord(
            date=pd.Timestamp("2023-01-16"),
            asset="MSFT",
            quantity=50,
            price=200.0,
            side=TradeSide.BUY,
            strategy_id="seasonal",
            allocated_capital=50000.0,
            transaction_cost=10.0,
        )

        aggregator.track_sub_strategy_trade(trade1)
        aggregator.track_sub_strategy_trade(trade2)

        performance = aggregator.calculate_weighted_performance()

        assert "total_return" in performance
        assert "total_trades" in performance
        assert "buy_trades" in performance
        assert "sell_trades" in performance
        assert "total_pnl" in performance
        assert "current_value" in performance
        assert "cash_balance" in performance
        assert "initial_capital" in performance

        assert performance["total_trades"] == 2
        assert performance["buy_trades"] == 2
        assert performance["sell_trades"] == 0
        assert performance["initial_capital"] == initial_capital

    def test_get_strategy_attribution(self):
        """Test strategy attribution calculation."""
        aggregator = TradeAggregator(100000.0)

        # Add trades from different strategies
        momentum_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=15.0,
        )

        seasonal_trade = TradeRecord(
            date=pd.Timestamp("2023-01-16"),
            asset="MSFT",
            quantity=50,
            price=200.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=50000.0,
            transaction_cost=10.0,
        )

        aggregator.track_sub_strategy_trade(momentum_trade)
        aggregator.track_sub_strategy_trade(seasonal_trade)

        attribution = aggregator.get_strategy_attribution()

        assert "momentum" in attribution
        assert "seasonal" in attribution

        momentum_stats = attribution["momentum"]
        seasonal_stats = attribution["seasonal"]

        assert momentum_stats["total_trades"] == 1
        assert momentum_stats["buy_trades"] == 1
        assert momentum_stats["sell_trades"] == 0
        assert momentum_stats["total_trade_value"] == 15000.0
        assert momentum_stats["total_transaction_costs"] == 15.0

        assert seasonal_stats["total_trades"] == 1
        assert seasonal_stats["buy_trades"] == 0
        assert seasonal_stats["sell_trades"] == 1
        assert seasonal_stats["total_trade_value"] == 10000.0
        assert seasonal_stats["total_transaction_costs"] == 10.0

    def test_export_trades_to_dataframe(self):
        """Test exporting trades to DataFrame."""
        aggregator = TradeAggregator(100000.0)

        # Test empty case
        empty_df = aggregator.export_trades_to_dataframe()
        assert empty_df.empty

        # Add some trades
        trade1 = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
        )

        trade2 = TradeRecord(
            date=pd.Timestamp("2023-01-16"),
            asset="MSFT",
            quantity=50,
            price=200.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=50000.0,
        )

        aggregator.track_sub_strategy_trade(trade1)
        aggregator.track_sub_strategy_trade(trade2)

        df = aggregator.export_trades_to_dataframe()

        assert len(df) == 2
        assert "date" in df.columns
        assert "asset" in df.columns
        assert "quantity" in df.columns
        assert "price" in df.columns
        assert "side" in df.columns
        assert "strategy_id" in df.columns

        # Check data
        assert df.iloc[0]["asset"] == "AAPL"
        assert df.iloc[1]["asset"] == "MSFT"
        assert df.iloc[0]["side"] == "buy"
        assert df.iloc[1]["side"] == "sell"

    def test_invalid_trade_configurations(self):
        """Test that invalid trade configurations are properly rejected."""
        aggregator = TradeAggregator(100000.0)

        # Create a valid buy trade first
        valid_buy_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=15.0,
        )

        # Manually corrupt the quantity to create an invalid configuration
        # (bypassing the __post_init__ correction)
        valid_buy_trade.quantity = -100  # Make it negative after construction

        try:
            aggregator.track_sub_strategy_trade(valid_buy_trade)
            assert False, "Should have raised ValueError for buy trade with negative quantity"
        except ValueError as e:
            assert "buy trades must have positive quantities" in str(e).lower()

        # Create a valid sell trade and corrupt it
        valid_sell_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="MSFT",
            quantity=50,
            price=200.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=30000.0,
            transaction_cost=10.0,
        )

        # The TradeRecord __post_init__ will make this -50, so we need to make it positive again
        valid_sell_trade.quantity = 50  # Make it positive after construction

        try:
            aggregator.track_sub_strategy_trade(valid_sell_trade)
            assert False, "Should have raised ValueError for sell trade with positive quantity"
        except ValueError as e:
            assert "sell trades must have negative quantities" in str(e).lower()

        # Test trade with zero quantity
        zero_quantity_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="GOOGL",
            quantity=100,  # Start with valid quantity
            price=2500.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=25.0,
        )

        # Manually set to zero after construction
        zero_quantity_trade.quantity = 0

        try:
            aggregator.track_sub_strategy_trade(zero_quantity_trade)
            assert False, "Should have raised ValueError for trade with zero quantity"
        except ValueError as e:
            assert "trade quantity cannot be zero" in str(e).lower()

    def test_valid_buy_and_sell_scenarios(self):
        """Test that valid buy and sell scenarios work correctly."""
        aggregator = TradeAggregator(100000.0)

        # Valid buy trade (positive quantity)
        buy_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AAPL",
            quantity=100,  # Positive quantity for buy
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=15.0,
        )

        # Valid sell trade (negative quantity)
        sell_trade = TradeRecord(
            date=pd.Timestamp("2023-01-16"),
            asset="MSFT",
            quantity=-50,  # Negative quantity for sell
            price=200.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=30000.0,
            transaction_cost=10.0,
        )

        # Both should work without errors
        aggregator.track_sub_strategy_trade(buy_trade)
        aggregator.track_sub_strategy_trade(sell_trade)

        # Verify trades were recorded
        trades = aggregator.get_aggregated_trades()
        assert len(trades) == 2

        # Verify positions
        positions = aggregator.get_current_positions()
        assert len(positions) == 2
        assert "AAPL" in positions
        assert "MSFT" in positions
        assert positions["AAPL"].quantity == 100  # Long position
        assert positions["MSFT"].quantity == -50  # Short position

        # Verify cash balance changes
        # Buy: 100000 - (15000 + 15) = 84985
        # Sell: 84985 + (10000 - 10) = 94975
        expected_cash = 100000.0 - (15000.0 + 15.0) + (10000.0 - 10.0)
        assert abs(aggregator._cash_balance - expected_cash) < 0.01
