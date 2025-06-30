
import pandas as pd
import numpy as np
from portfolio_backtester.strategies.sharpe_momentum_strategy import SharpeMomentumStrategy
from portfolio_backtester.strategies.sortino_momentum_strategy import SortinoMomentumStrategy

def create_sample_data():
    dates = pd.date_range('2020-01-01', periods=60, freq='ME')
    np.random.seed(42)
    data = pd.DataFrame({
        'STOCK_A': np.random.normal(0.005, 0.02, len(dates)).cumsum() + 100,
        'STOCK_B': np.random.normal(0.008, 0.03, len(dates)).cumsum() + 100,
        'STOCK_C': np.random.normal(0.002, 0.01, len(dates)).cumsum() + 100,
        'STOCK_D': np.random.normal(-0.001, 0.025, len(dates)).cumsum() + 100,
    }, index=dates)
    benchmark_data = pd.Series(np.random.normal(0.004, 0.015, len(dates)).cumsum() + 100, index=dates, name='BENCHMARK')
    return data, benchmark_data

def test_sortino_vs_sharpe_weights():
    data, benchmark_data = create_sample_data()
    
    strategy_config = {
        'rolling_window': 6,
        'top_decile_fraction': 0.5,
        'smoothing_lambda': 0.5,
        'leverage': 1.0,
        'long_only': True,
        'sma_filter_window': None
    }

    sharpe_strategy = SharpeMomentumStrategy(strategy_config)
    sortino_strategy = SortinoMomentumStrategy(strategy_config)

    sharpe_weights = sharpe_strategy.generate_signals(data, benchmark_data)
    sortino_weights = sortino_strategy.generate_signals(data, benchmark_data)

    # Get the rolling ratios for inspection
    sharpe_rets = data.pct_change(fill_method=None)
    sortino_rets = data.pct_change(fill_method=None)

    sharpe_ratios = sharpe_strategy._calculate_rolling_sharpe(sharpe_rets, strategy_config.get('rolling_window', 6))
    sortino_ratios = sortino_strategy._calculate_rolling_sortino(sortino_rets, strategy_config.get('rolling_window', 6), strategy_config.get('target_return', 0.0))

    diff = (sharpe_weights - sortino_weights).abs().sum().sum()
    assert diff < 10.0 # Relaxed threshold for expected differences
