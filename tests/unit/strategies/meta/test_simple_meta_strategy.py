"""Tests for SimpleMetaStrategy class."""

from unittest.mock import Mock, patch

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
