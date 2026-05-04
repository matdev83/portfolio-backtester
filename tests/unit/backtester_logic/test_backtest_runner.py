import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
from portfolio_backtester.data_sources.preloaded_frame_data_source import PreloadedFrameDataSource


@pytest.fixture
def mock_dependencies():
    global_config = {"benchmark": "SPY"}
    data_cache = MagicMock()
    strategy_manager = MagicMock()
    return global_config, data_cache, strategy_manager


@pytest.fixture
def runner(mock_dependencies):
    global_config, data_cache, strategy_manager = mock_dependencies
    return BacktestRunner(global_config, data_cache, strategy_manager)


def test_run_scenario_success(runner, mock_dependencies):
    global_config, data_cache, strategy_manager = mock_dependencies

    # Setup inputs
    scenario_config = {
        "name": "TestScenario",
        "strategy": "TestStrategy",
        "strategy_params": {},
        "universe": ["AAPL", "GOOG"],
    }
    # Initialize with columns matching universe so they aren't filtered out
    price_data_monthly = pd.DataFrame(columns=["AAPL", "GOOG"])
    price_data_daily = pd.DataFrame()
    rets_daily = pd.DataFrame()

    # Setup mocks
    mock_strategy = MagicMock()
    strategy_manager.get_strategy.return_value = mock_strategy
    # Strategy universe
    mock_strategy.get_universe.return_value = [("AAPL", {}), ("GOOG", {})]

    # Mock imported functions
    with (
        patch(
            "portfolio_backtester.backtester_logic.backtest_runner.prepare_scenario_data"
        ) as mock_prep,
        patch("portfolio_backtester.backtester_logic.backtest_runner.generate_signals") as mock_gen,
        patch("portfolio_backtester.backtester_logic.backtest_runner.size_positions") as mock_size,
        patch(
            "portfolio_backtester.backtester_logic.backtest_runner.calculate_portfolio_returns"
        ) as mock_calc,
    ):

        mock_prep.return_value = (price_data_monthly, rets_daily)
        mock_gen.return_value = pd.DataFrame()
        mock_size.return_value = pd.DataFrame()
        expected_rets = pd.Series([0.01, 0.02])
        mock_calc.return_value = (expected_rets, None)

        result = runner.run_scenario(
            scenario_config, price_data_monthly, price_data_daily, rets_daily
        )

        assert isinstance(result, pd.Series)
        assert result.equals(expected_rets)

        # Verify calls
        strategy_manager.get_strategy.assert_called_once()
        mock_prep.assert_called_once()
        mock_gen.assert_called_once()
        mock_size.assert_called_once()
        mock_calc.assert_called_once()


def test_run_scenario_missing_tickers(runner, mock_dependencies):
    global_config, data_cache, strategy_manager = mock_dependencies

    # Universe has ticker NOT in price data
    scenario_config = {
        "name": "TestScenario",
        "strategy": "TestStrategy",
        "strategy_params": {},
        "universe": ["AAPL", "UNKNOWN"],
    }
    # price_data_monthly columns only has AAPL
    price_data_monthly = pd.DataFrame(columns=["AAPL"])
    price_data_daily = pd.DataFrame()

    mock_strategy = MagicMock()
    strategy_manager.get_strategy.return_value = mock_strategy
    mock_strategy.get_universe.return_value = [("AAPL", {}), ("UNKNOWN", {})]

    with (
        patch(
            "portfolio_backtester.backtester_logic.backtest_runner.prepare_scenario_data"
        ) as mock_prep,
        patch("portfolio_backtester.backtester_logic.backtest_runner.generate_signals") as mock_gen,
        patch("portfolio_backtester.backtester_logic.backtest_runner.size_positions"),
        patch(
            "portfolio_backtester.backtester_logic.backtest_runner.calculate_portfolio_returns"
        ) as mock_calc,
    ):

        mock_prep.return_value = (price_data_monthly, None)
        mock_calc.return_value = (pd.Series(), None)

        runner.run_scenario(scenario_config, price_data_monthly, price_data_daily)

        # Check that UNKNOWN was filtered out
        # generate_signals call args: strategy, config, data, universe, ...
        args, _ = mock_gen.call_args
        universe_arg = args[3]
        assert "AAPL" in universe_arg
        assert "UNKNOWN" not in universe_arg


def test_run_scenario_no_tickers_left(runner, mock_dependencies):
    global_config, data_cache, strategy_manager = mock_dependencies

    scenario_config = {
        "name": "TestScenario",
        "strategy": "TestStrategy",
        "strategy_params": {},
        "universe": ["UNKNOWN"],
    }
    price_data_monthly = pd.DataFrame(columns=["AAPL"])

    mock_strategy = MagicMock()
    strategy_manager.get_strategy.return_value = mock_strategy
    mock_strategy.get_universe.return_value = [("UNKNOWN", {})]

    result = runner.run_scenario(scenario_config, price_data_monthly, pd.DataFrame())

    assert result is None


def test_run_backtest_mode(runner, mock_dependencies):
    global_config, data_cache, strategy_manager = mock_dependencies

    scenario_config = {"name": "TestScenario", "strategy": "TestStrategy", "strategy_params": {}}
    monthly_data = pd.DataFrame()
    daily_data = pd.DataFrame()
    rets_full = pd.DataFrame()

    # Mock StrategyBacktester
    with patch(
        "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
    ) as MockBacktester:
        mock_instance = MockBacktester.return_value
        mock_result = MagicMock()
        mock_result.returns = pd.Series()
        mock_result.trade_stats = {}
        mock_result.trade_history = []
        mock_result.performance_stats = {}
        mock_result.charts_data = {}
        mock_instance.backtest_strategy.return_value = mock_result

        result = runner.run_backtest_mode(scenario_config, monthly_data, daily_data, rets_full)

        assert isinstance(result, dict)
        assert "returns" in result
        assert "trade_stats" in result
        MockBacktester.assert_called_once()
        _args, _kwargs = MockBacktester.call_args
        assert isinstance(_args[1], PreloadedFrameDataSource)
        assert _args[1]._daily_frame is daily_data
        mock_instance.backtest_strategy.assert_called_once()


def test_run_backtest_mode_with_optuna(runner, mock_dependencies):
    global_config, data_cache, strategy_manager = mock_dependencies

    scenario_config = {
        "name": "TestScenario",
        "strategy": "TestStrategy",
        "strategy_params": {"a": 1},
    }

    with (
        patch(
            "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
        ) as MockBacktester,
        patch("optuna.load_study") as mock_load_study,
    ):

        mock_study = MagicMock()
        mock_study.best_params = {"a": 2, "b": 3}
        mock_load_study.return_value = mock_study

        mock_instance = MockBacktester.return_value
        mock_instance.backtest_strategy.return_value = MagicMock(
            returns=pd.Series(), trade_stats={}
        )

        runner.run_backtest_mode(
            scenario_config, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), study_name="TestStudy"
        )

        # Check if params were updated
        assert scenario_config["strategy_params"]["a"] == 2
        assert scenario_config["strategy_params"]["b"] == 3
