"""Tests for SimpleMetaStrategy class."""

import pandas as pd
from unittest.mock import Mock, patch

from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.builtins.meta.simple_meta_strategy import SimpleMetaStrategy


class TestSimpleMetaStrategy:
    """Test cases for SimpleMetaStrategy."""

    def test_initialization(self):
        """Test SimpleMetaStrategy initialization."""
        config = {
            "initial_capital": 500000,
            "allocations": [
                {
                    "strategy_id": "momentum",
                    "strategy_class": "CalmarMomentumPortfolioStrategy",
                    "strategy_params": {"rolling_window": 6},
                    "weight": 0.7,
                },
                {
                    "strategy_id": "seasonal",
                    "strategy_class": "SeasonalSignalStrategy",
                    "strategy_params": {"entry_day": 5},
                    "weight": 0.3,
                },
            ],
        }

        strategy = SimpleMetaStrategy(config)

        assert strategy.initial_capital == 500000
        assert strategy.available_capital == 500000
        assert len(strategy.allocations) == 2
        assert strategy.strategy_params["min_allocation"] == 0.05  # Default
        assert strategy.strategy_params["rebalance_threshold"] == 0.05  # Default

    def test_allocate_capital(self):
        """Test fixed capital allocation."""
        config = {
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "MockStrategy",
                    "strategy_params": {},
                    "weight": 0.6,
                },
                {
                    "strategy_id": "strategy2",
                    "strategy_class": "MockStrategy",
                    "strategy_params": {},
                    "weight": 0.4,
                },
            ]
        }

        strategy = SimpleMetaStrategy(config)
        allocations = strategy.allocate_capital()

        assert allocations["strategy1"] == 0.6
        assert allocations["strategy2"] == 0.4

    @patch("portfolio_backtester.strategies._core.strategy_factory.StrategyFactory.create_strategy")
    def test_create_strategy_instance_uses_factory(self, mock_create):
        """Test that strategy creation uses the factory."""
        config = {
            "allocations": [
                {
                    "strategy_id": "test",
                    "strategy_class": "MockStrategy",
                    "strategy_params": {"param": "value"},
                    "weight": 1.0,
                }
            ]
        }

        mock_strategy = Mock()
        mock_create.return_value = mock_strategy

        strategy = SimpleMetaStrategy(config)
        result = strategy._create_strategy_instance("MockStrategy", {"param": "value"})

        mock_create.assert_called_once_with("MockStrategy", {"param": "value"})
        assert result == mock_strategy

    def test_tunable_parameters(self):
        """Test tunable parameters includes base parameters."""
        params = SimpleMetaStrategy.tunable_parameters()

        # Should include base meta strategy parameters
        expected_base_params = {"initial_capital", "min_allocation", "rebalance_threshold"}

        # tunable_parameters now returns dict, check keys
        assert isinstance(params, dict)
        assert expected_base_params.issubset(set(params.keys()))

    def test_generate_target_weights_returns_none_meta_opt_out(self) -> None:
        config = {
            "initial_capital": 1000000,
            "allocations": [
                {
                    "strategy_id": "strategy1",
                    "strategy_class": "FixedWeightPortfolioStrategy",
                    "strategy_params": {"weights": {"AAA": 1.0}},
                    "weight": 1.0,
                },
            ],
        }
        strategy = SimpleMetaStrategy(config)
        idx = pd.date_range("2024-01-02", periods=5, freq="B")
        universe = ["AAA"]
        asset = pd.DataFrame({universe[0]: 1.0}, index=idx)
        ctx = StrategyContext.from_standard_inputs(
            asset_data=asset,
            benchmark_data=asset,
            non_universe_data=pd.DataFrame(),
            rebalance_dates=pd.DatetimeIndex(idx),
            universe_tickers=universe,
            benchmark_ticker=universe[0],
            wfo_start_date=None,
            wfo_end_date=None,
            use_sparse_nan_for_inactive_rows=False,
        )
        assert strategy.generate_target_weights(ctx) is None
