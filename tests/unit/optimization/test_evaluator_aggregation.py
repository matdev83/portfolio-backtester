import numpy as np
import pandas as pd

from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.backtesting.results import WindowResult


def _dummy_window_result(value: float) -> WindowResult:
    # Simple window result with one metric
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.Series(np.random.rand(5), index=dates)
    return WindowResult(
        window_returns=returns,
        metrics={"sharpe_ratio": value},
        train_start=dates[0],
        train_end=dates[2],
        test_start=dates[3],
        test_end=dates[4],
    )


def test_aggregate_objective_single_avg():
    evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False)
    obj = evaluator._aggregate_objective_values([1.0, 3.0, 5.0], window_lengths=[10, 10, 10])
    assert obj == 3.0


def test_aggregate_objective_weighted():
    evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False, aggregate_length_weighted=True)
    obj = evaluator._aggregate_objective_values([1.0, 3.0], window_lengths=[1, 3])
    # weighted average (1*1 +3*3)/4 = 2.5
    assert np.isclose(obj, 2.5)


def test_aggregate_metrics_simple():
    evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False)
    windows = [_dummy_window_result(1.0), _dummy_window_result(3.0)]
    agg = evaluator._aggregate_metrics(windows, window_lengths=[5, 5])
    assert agg["sharpe_ratio"] == 2.0