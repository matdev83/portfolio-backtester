import pandas as pd

from portfolio_backtester.utils.price_data_utils import (
    extract_current_prices,
    validate_price_data_sufficiency,
    normalize_price_series_to_dataframe,
)

def test_extract_current_prices_missing_date():
    dates = pd.date_range("2020-01-01", periods=3)
    df = pd.DataFrame({"A": [1,2,3]}, index=dates)
    result = extract_current_prices(df, pd.Timestamp("2021-01-01"), pd.Index(["A", "B"]))
    assert result.isna().all()
    assert list(result.index) == ["A", "B"]


def test_extract_current_prices_scalar():
    # DataFrame with single column returns scalar when using .loc on single date/col
    dates = pd.date_range("2020-01-01", periods=1)
    df = pd.DataFrame({"A": [10]}, index=dates)
    val = extract_current_prices(df, dates[0], pd.Index(["A"]))
    assert val.loc["A"] == 10


def test_validate_price_data_sufficiency():
    dates = pd.date_range("2020-01-01", periods=5)
    df = pd.DataFrame({"A": range(5)}, index=dates)
    ok, msg = validate_price_data_sufficiency(df, dates[-1], min_required_periods=3)
    assert ok is True
    ok, msg = validate_price_data_sufficiency(df, dates[-1], min_required_periods=10)
    assert ok is False


def test_normalize_price_series_to_dataframe():
    ser = pd.Series([1,2,3], index=["X","Y","Z"])
    df = normalize_price_series_to_dataframe(ser, target_columns=pd.Index(["X","Y","Z"]))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["X","Y","Z"]
    # When DataFrame input, ensure it's copied not same object
    df2 = normalize_price_series_to_dataframe(df)
    assert df2.equals(df)
    assert df2 is not df