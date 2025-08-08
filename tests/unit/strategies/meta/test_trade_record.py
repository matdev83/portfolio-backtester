"""Tests for TradeRecord and PositionRecord classes."""

import pytest
import pandas as pd

from portfolio_backtester.strategies.base.trade_record import (
    TradeRecord, PositionRecord, TradeSide
)


class TestTradeRecord:
    """Test cases for TradeRecord class."""
    
    def test_trade_record_initialization(self):
        """Test basic TradeRecord initialization."""
        date = pd.Timestamp("2023-01-15")
        trade = TradeRecord(
            date=date,
            asset="AAPL",
            quantity=100,
            price=150.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0,
            transaction_cost=15.0
        )
        
        assert trade.date == date
        assert trade.asset == "AAPL"
        assert trade.quantity == 100
        assert trade.price == 150.0
        assert trade.side == TradeSide.BUY
        assert trade.strategy_id == "momentum"
        assert trade.allocated_capital == 50000.0
        assert trade.transaction_cost == 15.0
        assert trade.trade_value == 15000.0  # 100 * 150
        assert trade.is_buy
        assert not trade.is_sell
    
    def test_trade_record_sell_side(self):
        """Test TradeRecord with sell side."""
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="MSFT",
            quantity=50,
            price=200.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=30000.0
        )
        
        assert trade.side == TradeSide.SELL
        assert trade.quantity == -50  # Should be negative for sell
        assert trade.is_sell
        assert not trade.is_buy
        assert trade.trade_value == 10000.0  # 50 * 200
    
    def test_trade_record_net_value_buy(self):
        """Test net value calculation for buy trade."""
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="GOOGL",
            quantity=10,
            price=2500.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=100000.0,
            transaction_cost=25.0
        )
        
        # For buy: net_value = quantity * price + transaction_cost
        expected_net_value = 10 * 2500.0 + 25.0
        assert trade.net_value == expected_net_value
    
    def test_trade_record_net_value_sell(self):
        """Test net value calculation for sell trade."""
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AMZN",
            quantity=20,
            price=3000.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=80000.0,
            transaction_cost=60.0
        )
        
        # For sell: net_value = quantity * price - transaction_cost
        # Note: quantity will be negative for sell, so this becomes negative
        expected_net_value = -20 * 3000.0 - 60.0
        assert trade.net_value == expected_net_value
    
    def test_trade_record_quantity_sign_correction(self):
        """Test that quantity sign is corrected based on trade side."""
        # Buy with negative quantity should be corrected to positive
        buy_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="TSLA",
            quantity=-100,  # Negative quantity for buy
            price=800.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=100000.0
        )
        assert buy_trade.quantity == 100  # Should be corrected to positive
        
        # Sell with positive quantity should be corrected to negative
        sell_trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="NVDA",
            quantity=50,  # Positive quantity for sell
            price=400.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=50000.0
        )
        assert sell_trade.quantity == -50  # Should be corrected to negative
    
    def test_trade_record_to_dict(self):
        """Test conversion to dictionary."""
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="META",
            quantity=75,
            price=300.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=60000.0,
            transaction_cost=22.5
        )
        
        trade_dict = trade.to_dict()
        
        assert trade_dict['asset'] == "META"
        assert trade_dict['quantity'] == 75
        assert trade_dict['price'] == 300.0
        assert trade_dict['side'] == "buy"
        assert trade_dict['strategy_id'] == "momentum"
        assert trade_dict['allocated_capital'] == 60000.0
        assert trade_dict['transaction_cost'] == 22.5
        assert trade_dict['trade_value'] == 22500.0
    
    def test_trade_record_from_dict(self):
        """Test creation from dictionary."""
        trade_dict = {
            'date': '2023-01-15',
            'asset': 'CRM',
            'quantity': 40,
            'price': 150.0,
            'side': 'sell',
            'strategy_id': 'seasonal',
            'allocated_capital': 40000.0,
            'transaction_cost': 6.0,
            'trade_value': 6000.0
        }
        
        trade = TradeRecord.from_dict(trade_dict)
        
        assert trade.asset == "CRM"
        assert trade.quantity == -40  # Should be negative for sell
        assert trade.price == 150.0
        assert trade.side == TradeSide.SELL
        assert trade.strategy_id == "seasonal"
        assert trade.allocated_capital == 40000.0
        assert trade.transaction_cost == 6.0


class TestPositionRecord:
    """Test cases for PositionRecord class."""
    
    def test_position_record_initialization(self):
        """Test basic PositionRecord initialization."""
        date = pd.Timestamp("2023-01-15")
        position = PositionRecord(
            asset="AAPL",
            quantity=100,
            average_price=150.0,
            last_update=date,
            strategy_contributions={"momentum": 100}
        )
        
        assert position.asset == "AAPL"
        assert position.quantity == 100
        assert position.average_price == 150.0
        assert position.last_update == date
        assert position.strategy_contributions == {"momentum": 100}
        assert position.is_long
        assert not position.is_short
        assert not position.is_flat
        assert position.market_value == 15000.0  # 100 * 150
    
    def test_position_record_short_position(self):
        """Test PositionRecord with short position."""
        position = PositionRecord(
            asset="TSLA",
            quantity=-50,
            average_price=800.0,
            last_update=pd.Timestamp("2023-01-15"),
            strategy_contributions={"seasonal": -50}
        )
        
        assert position.is_short
        assert not position.is_long
        assert not position.is_flat
        assert position.market_value == -40000.0  # -50 * 800
    
    def test_position_record_flat_position(self):
        """Test PositionRecord with flat (zero) position."""
        position = PositionRecord(
            asset="MSFT",
            quantity=0.0,
            average_price=200.0,
            last_update=pd.Timestamp("2023-01-15"),
            strategy_contributions={}
        )
        
        assert position.is_flat
        assert not position.is_long
        assert not position.is_short
        assert position.market_value == 0.0
    
    def test_position_add_trade_first_trade(self):
        """Test adding first trade to empty position."""
        position = PositionRecord(
            asset="GOOGL",
            quantity=0.0,
            average_price=0.0,
            last_update=pd.Timestamp("2023-01-01"),
            strategy_contributions={}
        )
        
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="GOOGL",
            quantity=10,
            price=2500.0,
            side=TradeSide.BUY,
            strategy_id="momentum",
            allocated_capital=50000.0
        )
        
        position.add_trade(trade)
        
        assert position.quantity == 10
        assert position.average_price == 2500.0
        assert position.last_update == trade.date
        assert position.strategy_contributions["momentum"] == 10
    
    def test_position_add_trade_same_direction(self):
        """Test adding trade in same direction (averaging up/down)."""
        position = PositionRecord(
            asset="AMZN",
            quantity=20,
            average_price=3000.0,
            last_update=pd.Timestamp("2023-01-10"),
            strategy_contributions={"momentum": 20}
        )
        
        # Add another buy at different price
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="AMZN",
            quantity=10,
            price=3200.0,
            side=TradeSide.BUY,
            strategy_id="seasonal",
            allocated_capital=40000.0
        )
        
        position.add_trade(trade)
        
        # New average: (20 * 3000 + 10 * 3200) / 30 = 92000 / 30 = 3066.67
        assert position.quantity == 30
        assert abs(position.average_price - 3066.67) < 0.01
        assert position.strategy_contributions["momentum"] == 20
        assert position.strategy_contributions["seasonal"] == 10
    
    def test_position_add_trade_opposite_direction(self):
        """Test adding trade in opposite direction (partial close)."""
        position = PositionRecord(
            asset="NVDA",
            quantity=100,
            average_price=400.0,
            last_update=pd.Timestamp("2023-01-10"),
            strategy_contributions={"momentum": 100}
        )
        
        # Sell part of the position
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="NVDA",
            quantity=30,
            price=450.0,
            side=TradeSide.SELL,
            strategy_id="seasonal",
            allocated_capital=20000.0
        )
        
        position.add_trade(trade)
        
        # Quantity: 100 + (-30) = 70
        # Average price should remain the same for partial close
        assert position.quantity == 70
        assert position.average_price == 400.0  # Should remain original average
        assert position.strategy_contributions["momentum"] == 100
        assert position.strategy_contributions["seasonal"] == -30
    
    def test_position_add_trade_full_close(self):
        """Test adding trade that fully closes position."""
        position = PositionRecord(
            asset="CRM",
            quantity=50,
            average_price=150.0,
            last_update=pd.Timestamp("2023-01-10"),
            strategy_contributions={"momentum": 50}
        )
        
        # Sell entire position
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="CRM",
            quantity=50,
            price=160.0,
            side=TradeSide.SELL,
            strategy_id="momentum",
            allocated_capital=25000.0
        )
        
        position.add_trade(trade)
        
        assert position.quantity == 0
        assert position.average_price == 0.0  # Reset on full close
        assert position.is_flat
        assert position.strategy_contributions["momentum"] == 0  # 50 + (-50)
    
    def test_position_add_trade_wrong_asset(self):
        """Test that adding trade for wrong asset raises error."""
        position = PositionRecord(
            asset="AAPL",
            quantity=100,
            average_price=150.0,
            last_update=pd.Timestamp("2023-01-10"),
            strategy_contributions={"momentum": 100}
        )
        
        trade = TradeRecord(
            date=pd.Timestamp("2023-01-15"),
            asset="MSFT",  # Wrong asset
            quantity=50,
            price=200.0,
            side=TradeSide.BUY,
            strategy_id="seasonal",
            allocated_capital=25000.0
        )
        
        with pytest.raises(ValueError, match="Trade asset MSFT doesn't match position asset AAPL"):
            position.add_trade(trade)
    
    def test_position_to_dict(self):
        """Test conversion to dictionary."""
        position = PositionRecord(
            asset="META",
            quantity=75,
            average_price=300.0,
            last_update=pd.Timestamp("2023-01-15"),
            strategy_contributions={"momentum": 50, "seasonal": 25}
        )
        
        position_dict = position.to_dict()
        
        assert position_dict['asset'] == "META"
        assert position_dict['quantity'] == 75
        assert position_dict['average_price'] == 300.0
        assert position_dict['market_value'] == 22500.0
        assert position_dict['is_long']
        assert not position_dict['is_short']
        assert not position_dict['is_flat']
        assert position_dict['strategy_contributions'] == {"momentum": 50, "seasonal": 25}