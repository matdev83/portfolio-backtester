import pytest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies import (
    MomentumStrategy,
    CalmarMomentumStrategy,
    SharpeMomentumStrategy,
    SortinoMomentumStrategy,
    VAMSMomentumStrategy,
    VAMSNoDownsideStrategy,
    MomentumDvolSizerStrategy
)
from src.portfolio_backtester.features.momentum import Momentum # For feature name generation

strategies_to_test_lag = [
    (MomentumStrategy, {"lookback_months": 6}),
    (CalmarMomentumStrategy, {"rolling_window": 6}),
    (SharpeMomentumStrategy, {"rolling_window": 6}),
    (SortinoMomentumStrategy, {"rolling_window": 6, "target_return": 0.0}),
    (VAMSMomentumStrategy, {"lookback_months": 6, "alpha": 0.5}), # Uses DPVAMSSignalGenerator
    (VAMSNoDownsideStrategy, {"lookback_months": 6}), # Uses VAMSSignalGenerator
    (MomentumDvolSizerStrategy, {"lookback_months": 6}),
]

@pytest.mark.parametrize("strategy_class, initial_params", strategies_to_test_lag)
def test_configurable_trading_lag(strategy_class, initial_params):
    dates = pd.date_range(start="2023-01-31", periods=3, freq="M")
    assets = ["AssetX", "AssetY"]

    mock_prices = pd.DataFrame(100.0, index=dates, columns=assets)
    mock_benchmark = pd.Series(100.0, index=dates, name="BENCH")

    mock_features = {}
    # Construct expected feature names based on strategy and its default generator params
    if strategy_class in [MomentumStrategy, MomentumDvolSizerStrategy]:
        lb = initial_params.get("lookback_months", 6)
        # Momentum feature with skip=0, no_suffix has name "momentum_{lb}m"
        mock_features[Momentum(lookback_months=lb).name] = pd.DataFrame(0.1, index=dates, columns=assets)
    elif strategy_class == CalmarMomentumStrategy:
        rw = initial_params.get("rolling_window", 6)
        mock_features[f"calmar_{rw}m"] = pd.DataFrame(0.1, index=dates, columns=assets)
    elif strategy_class == SharpeMomentumStrategy:
        rw = initial_params.get("rolling_window", 6)
        mock_features[f"sharpe_{rw}m"] = pd.DataFrame(0.1, index=dates, columns=assets)
    elif strategy_class == SortinoMomentumStrategy:
        rw = initial_params.get("rolling_window", 6)
        mock_features[f"sortino_{rw}m"] = pd.DataFrame(0.1, index=dates, columns=assets)
    elif strategy_class == VAMSMomentumStrategy: # DPVAMS
        lb = initial_params.get("lookback_months", 6)
        alpha = initial_params.get("alpha", 0.5)
        mock_features[f"dp_vams_{lb}m_{alpha:.2f}a"] = pd.DataFrame(0.1, index=dates, columns=assets)
    elif strategy_class == VAMSNoDownsideStrategy: # VAMS
        lb = initial_params.get("lookback_months", 6)
        mock_features[f"vams_{lb}m"] = pd.DataFrame(0.1, index=dates, columns=assets)

    # Config for lag=True
    # Ensure 'strategy_params' exists and is a dict for setdefault
    config_lag_true_dict = {"strategy_params": {**initial_params, "apply_trading_lag": True, "sma_filter_window": None}}
    strategy_lag_true = strategy_class(config_lag_true_dict)
    weights_lag_true = strategy_lag_true.generate_signals(mock_prices, mock_features, mock_benchmark)

    assert weights_lag_true.iloc[0].isnull().all(), \
        f"{strategy_class.__name__} with apply_trading_lag=True should have NaNs in the first row."
    if len(weights_lag_true) > 1 :
      assert not weights_lag_true.iloc[1:].isnull().values.all(), \
          f"{strategy_class.__name__} with apply_trading_lag=True should have some non-NaNs after first row for this mock data (unless all signals were zero)."

    # Config for lag=False
    config_lag_false_dict = {"strategy_params": {**initial_params, "apply_trading_lag": False, "sma_filter_window": None}}
    strategy_lag_false = strategy_class(config_lag_false_dict)
    weights_lag_false = strategy_lag_false.generate_signals(mock_prices, mock_features, mock_benchmark)

    # Check that the first row is NOT all NaNs (it should have some values)
    # This depends on BaseStrategy not producing all zeros/NaNs for the first period with these simple inputs.
    # Given a constant signal of 0.1, it should produce some weights.
    assert not weights_lag_false.iloc[0].isnull().values.all(), \
        f"{strategy_class.__name__} with apply_trading_lag=False should not have all NaNs in the first row."
    # A more robust check might be that it's not *all* NaN, rather than *no* NaNs.
    # If all signals were NaN, then even without lag, first row could be NaN.
    # But with constant 0.1 signals, it should produce weights.
    assert weights_lag_false.iloc[0].notnull().values.any(), \
        f"{strategy_class.__name__} with apply_trading_lag=False should have at least some non-NaNs in the first row."
