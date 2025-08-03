import pytest
import pandas as pd
import numpy as np

from portfolio_backtester.portfolio.position_sizer import (
    _normalize_weights,
    get_position_sizer,
    get_position_sizer_from_config,
    EqualWeightSizer,
)


def test_normalize_weights_zero_row():
    """All-zero rows should remain zeros without NaNs or division errors."""
    df = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    norm = _normalize_weights(df)
    # Expect exactly the same zeros frame
    pd.testing.assert_frame_equal(norm, df)


def test_normalize_weights_with_leverage():
    """Leverage should scale post-normalisation weights proportionally."""
    df = pd.DataFrame({"A": [1.0], "B": [1.0]})
    norm = _normalize_weights(df, leverage=2.0)
    expected = pd.DataFrame({"A": [1.0], "B": [1.0]})  # 0.5 each * 2
    pd.testing.assert_frame_equal(norm, expected)
    # Sum of absolute weights equals leverage factor
    assert np.isclose(norm.abs().sum(axis=1).iloc[0], 2.0)


def test_get_position_sizer_from_config_defaults_to_equal_weight():
    cfg = {}
    sizer = get_position_sizer_from_config(cfg)
    assert isinstance(sizer, EqualWeightSizer)


def test_get_position_sizer_from_config_specific():
    cfg = {"position_sizer": "equal_weight"}
    sizer = get_position_sizer_from_config(cfg)
    assert isinstance(sizer, EqualWeightSizer)


def test_get_position_sizer_unknown_raises():
    with pytest.raises(ValueError):
        get_position_sizer("non_existent")
