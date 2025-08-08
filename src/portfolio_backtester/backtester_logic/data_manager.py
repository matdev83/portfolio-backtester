import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Legacy get_data_source function removed - replaced by DIP interfaces
# Use create_data_source from ..interfaces instead


def prepare_scenario_data(price_data_daily_ohlc, data_cache):
    daily_closes = None
    if price_data_daily_ohlc is not None:
        if isinstance(
            price_data_daily_ohlc.columns, pd.MultiIndex
        ) and "Close" in price_data_daily_ohlc.columns.get_level_values(1):
            daily_closes = price_data_daily_ohlc.xs("Close", level="Field", axis=1)
        elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
            daily_closes = price_data_daily_ohlc
        else:
            try:
                if "Close" in price_data_daily_ohlc.columns.get_level_values(-1):
                    daily_closes = price_data_daily_ohlc.xs("Close", level=-1, axis=1)
                else:
                    raise ValueError(
                        "Could not reliably extract 'Close' prices from price_data_daily_ohlc due to unrecognized column structure."
                    )
            except Exception as e:
                raise ValueError(
                    f"Error extracting 'Close' prices from price_data_daily_ohlc: {e}. Columns: {price_data_daily_ohlc.columns}"
                )

    if daily_closes is None or daily_closes.empty:
        raise ValueError("Daily close prices could not be extracted or are empty.")

    if isinstance(daily_closes, pd.Series):
        daily_closes = daily_closes.to_frame()

    monthly_closes = daily_closes.resample("BME").last()
    price_data_monthly_closes = (
        monthly_closes.to_frame() if isinstance(monthly_closes, pd.Series) else monthly_closes
    )

    rets_daily = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    rets_daily = rets_daily.to_frame() if isinstance(rets_daily, pd.Series) else rets_daily

    return price_data_monthly_closes, rets_daily
