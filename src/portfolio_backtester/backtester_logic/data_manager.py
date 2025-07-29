import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_data_source(global_config):
    from ..data_sources.stooq_data_source import StooqDataSource
    from ..data_sources.yfinance_data_source import YFinanceDataSource
    from ..data_sources.hybrid_data_source import HybridDataSource

    data_source_map = {
        "stooq": StooqDataSource,
        "yfinance": YFinanceDataSource,
        "hybrid": HybridDataSource,
    }

    ds_name = global_config.get("data_source", "hybrid").lower()
    data_source_class = data_source_map.get(ds_name)

    if data_source_class:
        logger.debug(f"Using {data_source_class.__name__}.")
        if ds_name == "hybrid":
            prefer_stooq = global_config.get("prefer_stooq", True)
            return HybridDataSource(
                cache_expiry_hours=24,
                prefer_stooq=prefer_stooq,
                negative_cache_timeout_hours=4
            )
        return data_source_class()
    else:
        logger.error(f"Unsupported data source: {ds_name}")
        raise ValueError(f"Unsupported data source: {ds_name}")

def prepare_scenario_data(price_data_daily_ohlc, data_cache):
    daily_closes = None
    if price_data_daily_ohlc is not None:
        if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and \
           'Close' in price_data_daily_ohlc.columns.get_level_values(1):
            daily_closes = price_data_daily_ohlc.xs('Close', level='Field', axis=1)
        elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
            daily_closes = price_data_daily_ohlc
        else:
            try:
                if 'Close' in price_data_daily_ohlc.columns.get_level_values(-1):
                    daily_closes = price_data_daily_ohlc.xs('Close', level=-1, axis=1)
                else:
                    raise ValueError("Could not reliably extract 'Close' prices from price_data_daily_ohlc due to unrecognized column structure.")
            except Exception as e:
                 raise ValueError(f"Error extracting 'Close' prices from price_data_daily_ohlc: {e}. Columns: {price_data_daily_ohlc.columns}")

    if daily_closes is None or daily_closes.empty:
        raise ValueError("Daily close prices could not be extracted or are empty.")

    if isinstance(daily_closes, pd.Series):
        daily_closes = daily_closes.to_frame()

    monthly_closes = daily_closes.resample("BME").last()
    price_data_monthly_closes = monthly_closes.to_frame() if isinstance(monthly_closes, pd.Series) else monthly_closes

    rets_daily = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    rets_daily = rets_daily.to_frame() if isinstance(rets_daily, pd.Series) else rets_daily

    return price_data_monthly_closes, rets_daily
