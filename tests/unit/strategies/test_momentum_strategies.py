
import unittest
import pandas as pd
import numpy as np
import pytest
from src.portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
from src.portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy
from src.portfolio_backtester.strategies.sortino_momentum_strategy import SortinoMomentumStrategy
from src.portfolio_backtester.strategies.momentum_dvol_sizer_strategy import MomentumDvolSizerStrategy
from src.portfolio_backtester.features.calmar_ratio import CalmarRatio
from src.portfolio_backtester.features.sortino_ratio import SortinoRatio

@pytest.fixture
def momentum_test_data():
    rebalance_dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))
    daily_start_date = rebalance_dates.min() - pd.DateOffset(months=12)
    daily_end_date = rebalance_dates.max()
    daily_dates = pd.date_range(start=daily_start_date, end=daily_end_date, freq='B')

    tickers = ['StockA', 'StockB', 'StockC', 'StockD']
    data_frames = []
    for ticker in tickers:
        if ticker == 'StockA':
            base_price = np.linspace(80, 210, len(daily_dates))
        elif ticker == 'StockB':
            base_price = np.linspace(120, 10, len(daily_dates))
        elif ticker == 'StockC':
            base_price = np.linspace(95, 115, len(daily_dates))
        else:
            base_price = np.full(len(daily_dates), 100)

        noise = np.random.normal(0, 0.5, size=len(daily_dates))
        close_prices = base_price + noise
        open_prices = close_prices - np.random.uniform(0, 0.5, size=len(daily_dates))
        high_prices = close_prices + np.random.uniform(0, 0.5, size=len(daily_dates))
        low_prices = close_prices - np.random.uniform(0, 0.5, size=len(daily_dates))
        volume = np.random.randint(1000, 5000, size=len(daily_dates))

        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=daily_dates)
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=['Ticker', 'Field'])
        data_frames.append(df)

    daily_ohlc_data = pd.concat(data_frames, axis=1)

    benchmark_base_price = np.linspace(90, 110, len(daily_dates))
    benchmark_noise = np.random.normal(0, 0.5, size=len(daily_dates))
    benchmark_close = benchmark_base_price + benchmark_noise
    benchmark_df = pd.DataFrame({
        'Open': benchmark_close - np.random.uniform(0,0.2, size=len(daily_dates)),
        'High': benchmark_close + np.random.uniform(0,0.2, size=len(daily_dates)),
        'Low': benchmark_close - np.random.uniform(0,0.2, size=len(daily_dates)),
        'Close': benchmark_close,
        'Volume': np.random.randint(10000, 50000, size=len(daily_dates))
    }, index=daily_dates)
    benchmark_df.columns = pd.MultiIndex.from_product([['SPY'], benchmark_df.columns], names=['Ticker', 'Field'])
    benchmark_ohlc_data = benchmark_df

    asset_monthly_closes = daily_ohlc_data.xs('Close', level='Field', axis=1).resample('ME').last()
    benchmark_monthly_closes = benchmark_ohlc_data.xs('Close', level='Field', axis=1).resample('ME').last()

    return {
        'rebalance_dates': rebalance_dates,
        'daily_dates': daily_dates,
        'daily_ohlc_data': daily_ohlc_data,
        'benchmark_ohlc_data': benchmark_ohlc_data,
        'asset_monthly_closes': asset_monthly_closes,
        'benchmark_monthly_closes': benchmark_monthly_closes
    }

@pytest.mark.parametrize("strategy_class, config", [
    (MomentumStrategy, {'strategy_params': {'lookback_months': 3, 'skip_months': 1, 'top_decile_fraction': 0.5, 'smoothing_lambda': 0.5, 'leverage': 1.0, 'long_only': True, 'price_column_asset': 'Close', 'price_column_benchmark': 'Close'}, 'num_holdings': None}),
    (CalmarMomentumStrategy, {'rolling_window': 6, 'top_decile_fraction': 0.1, 'smoothing_lambda': 0.5, 'leverage': 1.0, 'long_only': True, 'sma_filter_window': None}),
    (SortinoMomentumStrategy, {'rolling_window': 3, 'top_decile_fraction': 0.5, 'smoothing_lambda': 0.5, 'leverage': 1.0, 'long_only': True, 'target_return': 0.0}),
    (MomentumDvolSizerStrategy, {'lookback_months': 3})
])
def test_generate_signals_smoke(strategy_class, config, momentum_test_data):
    strategy = strategy_class(config)
    all_signal_weights = []
    for current_rebalance_date in momentum_test_data['rebalance_dates']:
        if current_rebalance_date < momentum_test_data['daily_dates'].min() + pd.DateOffset(months=config.get('lookback_months', config.get('rolling_window', 3))):
            continue

        historical_assets = momentum_test_data['daily_ohlc_data'][momentum_test_data['daily_ohlc_data'].index <= current_rebalance_date]
        historical_benchmark = momentum_test_data['benchmark_ohlc_data'][momentum_test_data['benchmark_ohlc_data'].index <= current_rebalance_date]

        try:
            weights_df = strategy.generate_signals(
                all_historical_data=historical_assets,
                benchmark_historical_data=historical_benchmark,
                current_date=current_rebalance_date
            )
            all_signal_weights.append(weights_df)
        except Exception as e:
            pytest.fail(f"generate_signals raised an exception on {current_rebalance_date}: {e}")

    assert len(all_signal_weights) > 0, "No signals were generated."
    final_weights_df = pd.concat(all_signal_weights)
    assert not final_weights_df.empty, "Concatenated weights DataFrame is empty."

class TestMomentumStrategySpecific:
    def test_top_performer_selection(self, momentum_test_data):
        strategy_config = {
            'strategy_params': {
                'lookback_months': 3,
                'skip_months': 0,
                'num_holdings': 1,
                'smoothing_lambda': 0.0,
                'leverage': 1.0,
                'long_only': True,
                'price_column_asset': 'Close',
                'price_column_benchmark': 'Close',
            },
            'num_holdings': 1,
        }
        strategy = MomentumStrategy(strategy_config)

        all_signal_weights = []
        valid_rebalance_dates = [
            d for d in momentum_test_data['rebalance_dates']
            if d >= momentum_test_data['daily_dates'].min() + pd.DateOffset(months=strategy_config['strategy_params']['lookback_months'])
        ]

        for current_rebalance_date in valid_rebalance_dates:
            historical_assets = momentum_test_data['daily_ohlc_data'][momentum_test_data['daily_ohlc_data'].index <= current_rebalance_date]
            historical_benchmark = momentum_test_data['benchmark_ohlc_data'][momentum_test_data['benchmark_ohlc_data'].index <= current_rebalance_date]

            weights_df = strategy.generate_signals(
                all_historical_data=historical_assets,
                benchmark_historical_data=historical_benchmark,
                current_date=current_rebalance_date
            )
            all_signal_weights.append(weights_df)

        assert len(all_signal_weights) > 0, "No signals generated for top performer test."
        final_weights_df = pd.concat(all_signal_weights)

        last_date_weights = final_weights_df.iloc[-1]
        assert abs(last_date_weights['StockA'] - 1.0) < 1e-2
        assert abs(last_date_weights['StockB']) < 1e-5
        assert abs(last_date_weights['StockC']) < 1e-5
        assert abs(last_date_weights['StockD']) < 1e-5

class TestCalmarMomentumStrategySpecific:
    def test_calculate_rolling_calmar(self, momentum_test_data):
        calmar_feature = CalmarRatio(rolling_window=6)
        rolling_calmar = calmar_feature.compute(momentum_test_data['asset_monthly_closes'])
        assert rolling_calmar.shape == momentum_test_data['asset_monthly_closes'].shape
        assert rolling_calmar.iloc[:5].isna().all().all()
        assert np.isfinite(rolling_calmar.iloc[6:]).all().all()

class TestSortinoMomentumStrategySpecific:
    def test_rolling_sortino_calculation(self, momentum_test_data):
        sortino_feature = SortinoRatio(rolling_window=3, target_return=0.0)
        rolling_sortino = sortino_feature.compute(momentum_test_data['asset_monthly_closes'])
        assert rolling_sortino.shape == momentum_test_data['asset_monthly_closes'].shape
        assert pd.isna(rolling_sortino.iloc[0]).all() or (rolling_sortino.iloc[0] == 0).all()
        final_sortino = rolling_sortino.iloc[-1]
        assert final_sortino['StockA'] > final_sortino['StockB']
