"""
Strategy data fixtures for generating standardized strategy configurations.

This module provides the StrategyDataFixture class with methods for generating
standard configuration patterns for different strategy types used in tests.
"""

from typing import Dict, Any
from functools import lru_cache


class StrategyDataFixture:
    """
    Fixture class for generating standardized strategy configurations.

    Provides methods to create configuration dictionaries for different
    strategy types with commonly used parameter combinations.
    """

    @staticmethod
    @lru_cache(maxsize=20)
    def momentum_config(
        lookback_months: int = 3,
        skip_months: int = 1,
        top_decile_fraction: float = 0.5,
        leverage: float = 1.0,
        trade_longs: bool = True,
        trade_shorts: bool = False,
    ) -> Dict[str, Any]:
        """
        Create standard momentum strategy configuration.

        Args:
            lookback_months: Number of months to look back for momentum calculation
            skip_months: Number of months to skip for momentum calculation
            top_decile_fraction: Fraction of top performers to select
            leverage: Strategy leverage
            trade_longs: Whether to trade long positions
            trade_shorts: Whether to trade short positions

        Returns:
            Dictionary with momentum strategy configuration
        """
        return {
            "strategy_params": {
                "lookback_months": lookback_months,
                "skip_months": skip_months,
                "top_decile_fraction": top_decile_fraction,
                "smoothing_lambda": 0.5,
                "leverage": leverage,
                "trade_longs": trade_longs,
                "trade_shorts": trade_shorts,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            },
            "num_holdings": None,
        }

    @staticmethod
    @lru_cache(maxsize=20)
    def calmar_momentum_config(
        lookback_months: int = 6,
        skip_months: int = 1,
        num_holdings: int = 10,
        leverage: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create Calmar momentum strategy configuration.

        Args:
            lookback_months: Number of months for Calmar ratio calculation
            skip_months: Number of months to skip
            num_holdings: Number of holdings to select
            leverage: Strategy leverage

        Returns:
            Dictionary with Calmar momentum strategy configuration
        """
        return {
            "strategy_params": {
                "lookback_months": lookback_months,
                "skip_months": skip_months,
                "num_holdings": num_holdings,
                "leverage": leverage,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            },
            "num_holdings": num_holdings,
        }

    @staticmethod
    @lru_cache(maxsize=20)
    def sortino_momentum_config(
        lookback_months: int = 6,
        skip_months: int = 1,
        num_holdings: int = 10,
        leverage: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create Sortino momentum strategy configuration.

        Args:
            lookback_months: Number of months for Sortino ratio calculation
            skip_months: Number of months to skip
            num_holdings: Number of holdings to select
            leverage: Strategy leverage

        Returns:
            Dictionary with Sortino momentum strategy configuration
        """
        return {
            "strategy_params": {
                "lookback_months": lookback_months,
                "skip_months": skip_months,
                "num_holdings": num_holdings,
                "leverage": leverage,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            },
            "num_holdings": num_holdings,
        }

    @staticmethod
    @lru_cache(maxsize=20)
    def ema_roro_config(
        fast_ema_days: int = 10,
        slow_ema_days: int = 20,
        leverage: float = 2.0,
        risk_off_leverage_multiplier: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Create EMA RoRo strategy configuration.

        Args:
            fast_ema_days: Fast EMA period in days
            slow_ema_days: Slow EMA period in days
            leverage: Base leverage
            risk_off_leverage_multiplier: Leverage multiplier during risk-off periods

        Returns:
            Dictionary with EMA RoRo strategy configuration
        """
        return {
            "fast_ema_days": fast_ema_days,
            "slow_ema_days": slow_ema_days,
            "leverage": leverage,
            "risk_off_leverage_multiplier": risk_off_leverage_multiplier,
        }

    @staticmethod
    @lru_cache(maxsize=20)
    def uvxy_rsi_config(
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        leverage: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create UVXY RSI strategy configuration.

        Args:
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            leverage: Strategy leverage

        Returns:
            Dictionary with UVXY RSI strategy configuration
        """
        return {
            "strategy_params": {
                "rsi_period": rsi_period,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
                "leverage": leverage,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            }
        }

    @staticmethod
    @lru_cache(maxsize=20)
    def momentum_dvol_sizer_config(
        lookback_months: int = 3,
        skip_months: int = 1,
        volatility_lookback: int = 20,
        target_volatility: float = 0.15,
        leverage: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create momentum strategy with volatility sizing configuration.

        Args:
            lookback_months: Momentum lookback period
            skip_months: Skip months for momentum
            volatility_lookback: Days for volatility calculation
            target_volatility: Target portfolio volatility
            leverage: Base leverage

        Returns:
            Dictionary with momentum volatility sizing strategy configuration
        """
        return {
            "strategy_params": {
                "lookback_months": lookback_months,
                "skip_months": skip_months,
                "volatility_lookback": volatility_lookback,
                "target_volatility": target_volatility,
                "leverage": leverage,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            }
        }

    @staticmethod
    @lru_cache(maxsize=20)
    def low_volatility_factor_config(
        lookback_months: int = 6,
        num_holdings: int = 20,
        leverage: float = 1.0,
        rebalance_frequency: str = "M",
    ) -> Dict[str, Any]:
        """
        Create low volatility factor strategy configuration.

        Args:
            lookback_months: Volatility calculation lookback
            num_holdings: Number of holdings to select
            leverage: Strategy leverage
            rebalance_frequency: Rebalancing frequency

        Returns:
            Dictionary with low volatility factor strategy configuration
        """
        return {
            "strategy_params": {
                "lookback_months": lookback_months,
                "num_holdings": num_holdings,
                "leverage": leverage,
                "trade_longs": True,
                "trade_shorts": False,
                "rebalance_frequency": rebalance_frequency,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            },
            "num_holdings": num_holdings,
        }

    @staticmethod
    def create_test_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        Create a collection of test scenarios for comprehensive testing.

        Returns:
            Dictionary mapping scenario names to strategy configurations
        """
        return {
            "momentum_simple": StrategyDataFixture.momentum_config(),
            "momentum_aggressive": StrategyDataFixture.momentum_config(
                lookback_months=1, leverage=2.0, trade_longs=True, trade_shorts=True
            ),
            "momentum_conservative": StrategyDataFixture.momentum_config(
                lookback_months=12, top_decile_fraction=0.2, leverage=0.5
            ),
            "calmar_standard": StrategyDataFixture.calmar_momentum_config(),
            "calmar_concentrated": StrategyDataFixture.calmar_momentum_config(
                num_holdings=5, leverage=1.5
            ),
            "sortino_standard": StrategyDataFixture.sortino_momentum_config(),
            "sortino_diversified": StrategyDataFixture.sortino_momentum_config(
                num_holdings=20, leverage=0.8
            ),
            "ema_roro_standard": StrategyDataFixture.ema_roro_config(),
            "ema_roro_conservative": StrategyDataFixture.ema_roro_config(
                leverage=1.0, risk_off_leverage_multiplier=0.2
            ),
            "uvxy_rsi_standard": StrategyDataFixture.uvxy_rsi_config(),
            "uvxy_rsi_sensitive": StrategyDataFixture.uvxy_rsi_config(
                rsi_oversold=20.0, rsi_overbought=80.0
            ),
            "dvol_sizer_standard": StrategyDataFixture.momentum_dvol_sizer_config(),
            "dvol_sizer_low_vol": StrategyDataFixture.momentum_dvol_sizer_config(
                target_volatility=0.10, volatility_lookback=30
            ),
            "low_vol_factor_standard": StrategyDataFixture.low_volatility_factor_config(),
            "low_vol_factor_concentrated": StrategyDataFixture.low_volatility_factor_config(
                num_holdings=10, leverage=1.2
            ),
        }

    @staticmethod
    def create_invalid_configs() -> Dict[str, Dict[str, Any]]:
        """
        Create invalid configurations for error handling tests.

        Returns:
            Dictionary mapping error scenario names to invalid configurations
        """
        return {
            "negative_lookback": StrategyDataFixture.momentum_config(lookback_months=-1),
            "zero_leverage": StrategyDataFixture.momentum_config(leverage=0.0),
            "invalid_fraction": StrategyDataFixture.momentum_config(top_decile_fraction=1.5),
            "missing_params": {
                "strategy_params": {
                    "leverage": 1.0,
                    # Missing required parameters
                }
            },
            "wrong_types": {
                "strategy_params": {
                    "lookback_months": "invalid",  # Should be int
                    "leverage": "invalid",  # Should be float
                    "trade_longs": "invalid",  # Should be bool
                }
            },
        }

    @staticmethod
    def get_expected_signals(strategy_type: str, scenario: str = "standard") -> Dict[str, Any]:
        """
        Get expected signal characteristics for different strategy types.

        Args:
            strategy_type: Type of strategy ('momentum', 'calmar', 'sortino', etc.)
            scenario: Scenario name ('standard', 'aggressive', 'conservative')

        Returns:
            Dictionary with expected signal characteristics
        """
        base_expectations = {
            "momentum": {
                "signal_range": (-2.0, 2.0),
                "sum_constraint": "leverage_bounded",
                "trade_longs": True,
                "trade_shorts": False,
                "rebalance_frequency": "monthly",
            },
            "calmar": {
                "signal_range": (0.0, 2.0),
                "sum_constraint": "leverage_bounded",
                "trade_longs": True,
                "trade_shorts": False,
                "rebalance_frequency": "monthly",
            },
            "sortino": {
                "signal_range": (0.0, 2.0),
                "sum_constraint": "leverage_bounded",
                "trade_longs": True,
                "trade_shorts": False,
                "rebalance_frequency": "monthly",
            },
            "ema_roro": {
                "signal_range": (-4.0, 4.0),
                "sum_constraint": "risk_adjusted",
                "trade_longs": True,
                "trade_shorts": True,
                "rebalance_frequency": "daily",
            },
            "uvxy_rsi": {
                "signal_range": (-1.0, 1.0),
                "sum_constraint": "leverage_bounded",
                "trade_longs": True,
                "trade_shorts": True,
                "rebalance_frequency": "daily",
            },
        }

        return base_expectations.get(strategy_type, {})

    @staticmethod
    def create_parameter_ranges() -> Dict[str, Dict[str, tuple]]:
        """
        Create parameter ranges for optimization and sensitivity testing.

        Returns:
            Dictionary mapping strategy types to parameter ranges
        """
        return {
            "momentum": {
                "lookback_months": (1, 12),
                "skip_months": (0, 3),
                "top_decile_fraction": (0.1, 1.0),
                "leverage": (0.1, 3.0),
                "smoothing_lambda": (0.0, 1.0),
            },
            "calmar": {
                "lookback_months": (3, 24),
                "skip_months": (0, 6),
                "num_holdings": (5, 50),
                "leverage": (0.5, 2.0),
            },
            "sortino": {
                "lookback_months": (3, 24),
                "skip_months": (0, 6),
                "num_holdings": (5, 50),
                "leverage": (0.5, 2.0),
            },
            "ema_roro": {
                "fast_ema_days": (5, 50),
                "slow_ema_days": (10, 100),
                "leverage": (1.0, 4.0),
                "risk_off_leverage_multiplier": (0.1, 1.0),
            },
            "uvxy_rsi": {
                "rsi_period": (7, 30),
                "rsi_oversold": (10.0, 40.0),
                "rsi_overbought": (60.0, 90.0),
                "leverage": (0.5, 2.0),
            },
        }
