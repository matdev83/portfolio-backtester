"""Tests for the unified commission calculator."""

import pandas as pd

from portfolio_backtester.trading.unified_commission_calculator import (
    UnifiedCommissionCalculator,
    TradeCommissionInfo,
    get_unified_commission_calculator,
)


class TestTradeCommissionInfo:
    """Test the TradeCommissionInfo dataclass."""

    def test_trade_commission_info_creation(self):
        """Test creating a TradeCommissionInfo instance."""
        info = TradeCommissionInfo(
            asset="AAPL",
            date=pd.Timestamp("2023-01-01"),
            quantity=100.0,
            price=150.0,
            trade_value=15000.0,
            commission_amount=7.5,
            slippage_amount=3.75,
            total_cost=11.25,
            commission_rate_bps=5.0,
            slippage_rate_bps=2.5,
        )

        assert info.asset == "AAPL"
        assert info.quantity == 100.0
        assert info.price == 150.0
        assert info.trade_value == 15000.0
        assert info.commission_amount == 7.5
        assert info.slippage_amount == 3.75
        assert info.total_cost == 11.25
        assert info.commission_rate_bps == 5.0
        assert info.slippage_rate_bps == 2.5

    def test_to_dict(self):
        """Test converting TradeCommissionInfo to dictionary."""
        info = TradeCommissionInfo(
            asset="MSFT",
            date=pd.Timestamp("2023-01-02"),
            quantity=50.0,
            price=250.0,
            trade_value=12500.0,
            commission_amount=6.25,
            slippage_amount=3.125,
            total_cost=9.375,
            commission_rate_bps=5.0,
            slippage_rate_bps=2.5,
        )

        result = info.to_dict()

        assert result["asset"] == "MSFT"
        assert result["quantity"] == 50.0
        assert result["price"] == 250.0
        assert result["trade_value"] == 12500.0
        assert result["commission_amount"] == 6.25
        assert result["slippage_amount"] == 3.125
        assert result["total_cost"] == 9.375


class TestUnifiedCommissionCalculator:
    """Test the UnifiedCommissionCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "commission_per_share": 0.005,
            "commission_min_per_order": 1.0,
            "commission_max_percent_of_trade": 0.005,
            "slippage_bps": 2.5,
            "default_transaction_cost_bps": 10.0,
        }
        self.calculator = UnifiedCommissionCalculator(self.config)

    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator.commission_per_share == 0.005
        assert self.calculator.commission_min_per_order == 1.0
        assert self.calculator.commission_max_percent == 0.005
        assert self.calculator.slippage_bps == 2.5
        assert self.calculator.default_transaction_cost_bps == 10.0

    def test_calculate_trade_commission_detailed(self):
        """Test detailed IBKR-style commission calculation."""
        result = self.calculator.calculate_trade_commission(
            asset="MSFT",
            date=pd.Timestamp("2023-01-02"),
            quantity=200.0,
            price=250.0,
            transaction_costs_bps=None,  # Use detailed calculation
        )

        assert result.asset == "MSFT"
        assert result.quantity == 200.0
        assert result.price == 250.0
        assert result.trade_value == 50000.0

        # Commission: 200 shares * $0.005 = $1.00, but min is $1.00
        # Max commission: 50000 * 0.005 = $250, so $1.00 applies
        expected_commission = 1.0

        # Slippage: 50000 * 0.025 / 100 = $12.50
        expected_slippage = 50000.0 * 0.025 / 100

        expected_total = expected_commission + expected_slippage

        assert abs(result.commission_amount - expected_commission) < 0.01
        assert abs(result.slippage_amount - expected_slippage) < 0.01
        assert abs(result.total_cost - expected_total) < 0.01

    def test_calculate_trade_commission_large_trade(self):
        """Test commission calculation for large trade hitting max commission."""
        result = self.calculator.calculate_trade_commission(
            asset="GOOGL",
            date=pd.Timestamp("2023-01-03"),
            quantity=10000.0,
            price=100.0,
            transaction_costs_bps=None,
        )

        # Trade value: 10000 * 100 = 1,000,000
        # Commission per share: 10000 * 0.005 = $50
        # Max commission: 1,000,000 * 0.005 = $5,000
        # So commission should be $50 (not capped)
        expected_commission = 50.0

        # Slippage: 1,000,000 * 0.025 / 100 = $250
        expected_slippage = 1000000.0 * 0.025 / 100

        assert abs(result.commission_amount - expected_commission) < 0.01
        assert abs(result.slippage_amount - expected_slippage) < 0.01

    def test_calculate_portfolio_commissions(self):
        """Test portfolio-level commission calculation."""
        # Create test data
        dates = pd.date_range("2023-01-01", periods=3, freq="D")

        # Turnover series
        turnover = pd.Series([0.1, 0.05, 0.08], index=dates)

        # Weights daily
        weights_daily = pd.DataFrame(
            {"AAPL": [0.5, 0.6, 0.4], "MSFT": [0.5, 0.4, 0.6]}, index=dates
        )

        # Price data
        price_data = pd.DataFrame(
            {"AAPL": [150.0, 152.0, 148.0], "MSFT": [250.0, 248.0, 252.0]}, index=dates
        )

        total_costs, breakdown, detailed_info = self.calculator.calculate_portfolio_commissions(
            turnover=turnover,
            weights_daily=weights_daily,
            price_data=price_data,
            portfolio_value=100000.0,
        )

        # Check that we get results
        assert len(total_costs) == 3
        assert "commission_costs" in breakdown
        assert "slippage_costs" in breakdown
        assert "total_costs" in breakdown
        assert len(detailed_info) == 3

        # Check that costs are reasonable
        assert all(cost >= 0 for cost in total_costs)
        assert all(cost >= 0 for cost in breakdown["commission_costs"])
        assert all(cost >= 0 for cost in breakdown["slippage_costs"])

    def test_get_commission_summary(self):
        """Test commission summary generation."""
        # Create mock detailed trade info
        detailed_info = {
            pd.Timestamp("2023-01-01"): {
                "AAPL": TradeCommissionInfo(
                    asset="AAPL",
                    date=pd.Timestamp("2023-01-01"),
                    quantity=100.0,
                    price=150.0,
                    trade_value=15000.0,
                    commission_amount=9.0,
                    slippage_amount=6.0,
                    total_cost=15.0,
                    commission_rate_bps=6.0,
                    slippage_rate_bps=4.0,
                )
            },
            pd.Timestamp("2023-01-02"): {
                "MSFT": TradeCommissionInfo(
                    asset="MSFT",
                    date=pd.Timestamp("2023-01-02"),
                    quantity=50.0,
                    price=250.0,
                    trade_value=12500.0,
                    commission_amount=7.5,
                    slippage_amount=5.0,
                    total_cost=12.5,
                    commission_rate_bps=6.0,
                    slippage_rate_bps=4.0,
                )
            },
        }

        summary = self.calculator.get_commission_summary(detailed_info)

        assert summary["total_trades"] == 2
        assert summary["total_commission"] == 16.5  # 9.0 + 7.5
        assert summary["total_slippage"] == 11.0  # 6.0 + 5.0
        assert summary["total_costs"] == 27.5  # 15.0 + 12.5
        assert summary["avg_commission_per_trade"] == 8.25  # 16.5 / 2
        assert summary["avg_slippage_per_trade"] == 5.5  # 11.0 / 2
        assert summary["avg_cost_per_trade"] == 13.75  # 27.5 / 2

    def test_get_commission_summary_empty(self):
        """Test commission summary with no trades."""
        summary = self.calculator.get_commission_summary({})

        assert summary["total_trades"] == 0
        assert summary["total_commission"] == 0.0
        assert summary["total_slippage"] == 0.0
        assert summary["total_costs"] == 0.0
        assert summary["avg_commission_per_trade"] == 0.0
        assert summary["avg_slippage_per_trade"] == 0.0
        assert summary["avg_cost_per_trade"] == 0.0


def test_get_unified_commission_calculator():
    """Test the factory function."""
    config = {"commission_per_share": 0.01, "slippage_bps": 5.0}

    calculator = get_unified_commission_calculator(config)

    assert isinstance(calculator, UnifiedCommissionCalculator)
    assert calculator.commission_per_share == 0.01
    assert calculator.slippage_bps == 5.0
