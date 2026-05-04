import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from portfolio_backtester.reporting.monte_carlo_analyzer import plot_stability_measures
from portfolio_backtester.reporting.table_generator import (
    generate_performance_table,
    generate_trade_statistics_table,
    _format_metric_value,
    _safe_metric,
    _collect_metrics,
)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_backtester():
    bt = MagicMock()
    bt.logger = MagicMock()
    bt.global_config = {"benchmark": "SPY"}
    # Setup minimal results structure for table generator
    bt.results = {"StratA": {"display_name": "Strategy A", "trade_stats": {}}}
    return bt


@pytest.fixture
def mock_opt_result():
    res = MagicMock()
    # Mocking attributes accessed by plot_stability_measures
    res.optimization_history = []
    return res


# -------------------------------------------------------------------------
# Monte Carlo Analyzer Robustness
# -------------------------------------------------------------------------


def test_plot_stability_measures_insufficient_history(mock_backtester, mock_opt_result):
    # Case 1: None history
    mock_opt_result.optimization_history = None
    plot_stability_measures(mock_backtester, "Test", mock_opt_result, pd.Series())
    mock_backtester.logger.info.assert_called_with(
        "Skipping trial P&L visualization: insufficient optimization history (need >=2)."
    )

    # Case 2: < 2 items
    mock_opt_result.optimization_history = [{"metrics": {}}]
    plot_stability_measures(mock_backtester, "Test", mock_opt_result, pd.Series())
    # Should log the same message


def test_plot_stability_measures_corrupted_trials(mock_backtester, mock_opt_result):
    # Trials with missing keys or wrong types
    mock_opt_result.optimization_history = [
        "Not a dict",  # Should be ignored
        {"metrics": None},  # Missing trial_returns
        {"metrics": {"trial_returns": None}},
        {
            "metrics": {
                "trial_returns": {"dates": ["2023-01-01", "2023-01-02"], "returns": [0.01, 0.02]}
            },
            "objective_value": None,  # Missing value -> skip
        },
        # One valid trial (but need 2 valid ones to plot)
        {
            "evaluation": 1,
            "metrics": {
                "trial_returns": {"dates": ["2023-01-01", "2023-01-02"], "returns": [0.01, 0.02]}
            },
            "objective_value": 1.5,
        },
    ]

    plot_stability_measures(mock_backtester, "Test", mock_opt_result, pd.Series())

    # Should log "<2 trials with stored returns data"
    mock_backtester.logger.info.assert_called()
    assert "Skipping trial P&L visualization: <2 trials" in str(
        mock_backtester.logger.info.call_args
    )


@patch("portfolio_backtester.reporting.monte_carlo_analyzer.plt")
def test_plot_stability_measures_plotting_exception(mock_plt, mock_backtester, mock_opt_result):
    # Provide valid data but simulate plotting failure
    valid_trial = {
        "evaluation": 1,
        "metrics": {
            "trial_returns": {
                "dates": pd.date_range("2023-01-01", periods=10),
                "returns": np.random.normal(0, 0.01, 10),
            }
        },
        "objective_value": 1.0,
    }
    mock_opt_result.optimization_history = [valid_trial, valid_trial]

    # Simulate exception during subplots creation
    mock_plt.subplots.side_effect = RuntimeError("Plotting backend failure")

    plot_stability_measures(mock_backtester, "Test", mock_opt_result, pd.Series())

    # Should catch exception and log error
    mock_backtester.logger.error.assert_called()
    assert "Error creating trial P&L visualization" in str(mock_backtester.logger.error.call_args)


# -------------------------------------------------------------------------
# Table Generator Robustness
# -------------------------------------------------------------------------


def test_safe_metric_robustness():
    s = pd.Series({"A": 1.0, "B": "invalid"})

    assert _safe_metric(s, "A") == 1.0
    assert pd.isna(_safe_metric(s, "Missing"))
    assert pd.isna(_safe_metric(s, "B"))  # Conversion to float fails


def test_format_metric_value_robustness():
    assert _format_metric_value("Total Return", 0.5) == "50.00%"
    assert _format_metric_value("Time in Market %", 0.6025) == "60.25%"
    assert _format_metric_value("Avg Gross Exposure", float("nan")) == "N/A"
    assert _format_metric_value("Avg Gross Exposure", 1.25) == "125.00%"
    assert _format_metric_value("Sharpe", 1.5) == "1.5000"
    assert _format_metric_value("Number of Trades (All)", 10.0) == "10"
    assert _format_metric_value("Unknown Metric", 1.23456) == "1.2346"


def test_generate_trade_statistics_table_empty(mock_backtester):
    console = MagicMock()
    generate_trade_statistics_table(console, "Strat", {})
    # Should not print table
    # Logic: if not trade_stats: return
    # Or if all_num_trades == 0
    console.print.assert_not_called()

    generate_trade_statistics_table(console, "Strat", {"all_num_trades": 0})
    console.print.assert_called()  # Prints "No trades found" message


def test_generate_performance_table_robustness(mock_backtester):
    console = MagicMock()
    period_returns = {"StratA": pd.Series(dtype=float)}  # Empty returns
    bench_returns = pd.Series([0.01], index=pd.date_range("2023-01-01", periods=1))

    # Should run without error even with empty strategy returns
    generate_performance_table(
        mock_backtester, console, period_returns, bench_returns, "Test Table", {}, "report_dir"
    )

    console.print.assert_called()


def test_collect_metrics_aligns_reference_to_strategy_window(mock_backtester):
    bench_returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03, 0.01],
        index=pd.date_range("2023-01-01", periods=5, freq="D"),
    )
    strategy_returns = pd.Series(
        [0.02, -0.01, 0.01],
        index=pd.date_range("2023-01-03", periods=3, freq="D"),
    )

    with patch("portfolio_backtester.reporting.performance_metrics.calculate_metrics") as mock_calc:
        mock_calc.return_value = pd.Series({"Sharpe": 1.0})

        _collect_metrics(
            mock_backtester,
            {"StratA": strategy_returns},
            bench_returns,
            {"StratA": 1},
        )

        # First call computes benchmark metrics and should use aligned window
        bench_call_args = mock_calc.call_args_list[0].args
        aligned_bench_rets = bench_call_args[0]
        assert list(aligned_bench_rets.index) == list(strategy_returns.index)

        # Second call computes strategy metrics and should pass same-window benchmark
        strategy_call_args = mock_calc.call_args_list[1].args
        strategy_bench_rets = strategy_call_args[1]
        assert list(strategy_bench_rets.index) == list(strategy_returns.index)


def test_generate_trade_statistics_infinite_values(mock_backtester):
    console = MagicMock()
    stats = {
        "all_num_trades": 10,
        "all_reward_risk_ratio": float("inf"),
        "long_reward_risk_ratio": float("-inf"),
    }

    generate_trade_statistics_table(console, "Strat", stats)

    # Should handle infs without crash
    console.print.assert_called()
