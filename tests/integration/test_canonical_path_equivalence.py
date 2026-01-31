import unittest
import unittest.mock
import pandas as pd
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.strategies._core.strategy_factory import StrategyFactory
from tests.base.integration_test_base import BaseIntegrationTest


class TestCanonicalPathEquivalence(BaseIntegrationTest):
    """
    Integration tests to ensure cross-path equivalence and consistent semantics
    using the canonical scenario configuration contract.
    """

    def setUp(self):
        super().setUp()
        self.normalizer = ScenarioNormalizer()
        self.strategy_manager = StrategyManager()

        # Define a representative scenario
        self.raw_scenario = {
            "name": "equivalence_test",
            "strategy": "SimpleMomentumPortfolioStrategy",
            "universe_config": {"type": "fixed", "tickers": ["AAPL", "MSFT", "GOOGL"]},
            "strategy_params": {"lookback_months": 12, "num_holdings": 2},
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }
        self.global_config = {
            "rebalance_frequency": "M",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        }
        self.canonical_config = self.normalizer.normalize(
            scenario=self.raw_scenario, global_config=self.global_config
        )

    def test_6_5_strategy_init_equivalence(self):
        """
        Task 6.5: Instantiate a representative strategy through both major instantiation paths
        and compare effective init configuration.
        """
        # Path A: StrategyManager (used in BacktestRunner)
        strategy_a = self.strategy_manager.get_strategy(
            self.canonical_config.strategy, self.canonical_config
        )

        # Path B: StrategyFactory directly (used in StrategyBacktester)
        strategy_b = StrategyFactory.create_strategy(
            self.canonical_config.strategy, self.canonical_config
        )

        # Verify both instances are created from the same canonical config
        self.assertEqual(strategy_a.canonical_config, self.canonical_config)
        self.assertEqual(strategy_b.canonical_config, self.canonical_config)

        # Verify effective strategy_params are identical
        self.assertEqual(strategy_a.strategy_params, strategy_b.strategy_params)

        # Verify providers and timing controllers are initialized consistently
        timing_a = strategy_a.get_timing_controller()
        timing_b = strategy_b.get_timing_controller()
        self.assertEqual(getattr(timing_a, "frequency", None), getattr(timing_b, "frequency", None))

        # Compare universe symbols from providers
        universe_a = strategy_a.get_universe_provider().get_universe_symbols({})
        universe_b = strategy_b.get_universe_provider().get_universe_symbols({})
        self.assertEqual(universe_a, universe_b)
        self.assertEqual(list(universe_a), ["AAPL", "MSFT", "GOOGL"])

    def test_6_4_dynamic_universe_equivalence(self):
        """
        Task 6.4: Add dynamic-universe regression test.
        Validate that a dynamic-universe scenario resolves the same universe across runtime modes.
        """
        # Define a scenario with a named universe
        dynamic_scenario = self.raw_scenario.copy()
        dynamic_scenario["universe_config"] = {"type": "named", "universe_name": "tech_giants"}

        # Patch load_named_universe in its source module
        # Since it is locally imported in universe_resolver, patching it in the source module
        # should work if the identity is preserved.
        with unittest.mock.patch(
            "portfolio_backtester.universe_loader.load_named_universe",
            return_value=["AAPL", "MSFT", "GOOGL"],
        ):

            canon_dynamic = self.normalizer.normalize(
                scenario=dynamic_scenario, global_config=self.global_config
            )

            # Instantiate strategies
            strat_a = self.strategy_manager.get_strategy(canon_dynamic.strategy, canon_dynamic)
            strat_b = StrategyFactory.create_strategy(canon_dynamic.strategy, canon_dynamic)

            # Verify universe resolution is consistent
            symbols_a = strat_a.get_universe_provider().get_universe_symbols({})
            symbols_b = strat_b.get_universe_provider().get_universe_symbols({})

            self.assertEqual(list(symbols_a), list(symbols_b))
            self.assertEqual(list(symbols_a), ["AAPL", "MSFT", "GOOGL"])

    def test_6_3_consistent_semantics_across_paths(self):
        """
        Task 6.3: integration test ensuring consistent semantics across backtest and optimization evaluation paths.
        """
        from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
        from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester

        # Use simple mock data for returns
        dates = pd.date_range(start="2020-01-01", periods=10, freq="BME")
        assets = ["AAPL", "MSFT", "GOOGL", "SPY"]  # Include benchmark
        mock_returns = pd.DataFrame(0.01, index=dates, columns=assets)
        mock_prices = (1 + mock_returns).cumprod()

        # Mock data manager to return our mock data
        with unittest.mock.patch(
            "portfolio_backtester.backtester_logic.backtest_runner.prepare_scenario_data"
        ) as mock_prepare:
            mock_prepare.return_value = (mock_prices, mock_returns)

            # Path A: BacktestRunner
            runner = BacktestRunner(
                self.global_config,
                data_cache=unittest.mock.MagicMock(),
                strategy_manager=self.strategy_manager,
            )

            res_a = runner.run_scenario(
                self.canonical_config,
                price_data_monthly_closes=mock_prices,
                price_data_daily_ohlc=mock_prices,
            )

            # Path B: StrategyBacktester
            tester = StrategyBacktester(self.global_config, data_source=unittest.mock.MagicMock())

            res_b_full = tester.backtest_strategy(
                self.canonical_config,
                monthly_data=mock_prices,
                daily_data=mock_prices,
                rets_full=mock_returns,
            )
            res_b = res_b_full.returns

            # Compare returns
            self.assertIsNotNone(res_a)
            self.assertIsNotNone(res_b)
            # Ignore frequency mismatch in comparison as different paths might handle it differently
            pd.testing.assert_series_equal(res_a, res_b, check_freq=False)  # type: ignore


if __name__ == "__main__":
    unittest.main()
