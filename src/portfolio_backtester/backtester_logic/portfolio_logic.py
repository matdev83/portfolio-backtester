import logging
import pandas as pd

from ..portfolio.rebalancing import rebalance

logger = logging.getLogger(__name__)

def calculate_portfolio_returns(sized_signals, scenario_config, price_data_daily_ohlc, rets_daily, universe_tickers, global_config):
    weights_monthly = rebalance(
        sized_signals, scenario_config["rebalance_frequency"]
    )

    weights_monthly = weights_monthly.reindex(columns=universe_tickers).fillna(0.0)

    weights_daily = weights_monthly.reindex(price_data_daily_ohlc.index, method="ffill")
    weights_daily = weights_daily.shift(1).fillna(0.0)

    if rets_daily is None:
        logger.error("rets_daily is None before reindexing in run_scenario.")
        return pd.Series(0.0, index=price_data_daily_ohlc.index)

    aligned_rets_daily = rets_daily.reindex(price_data_daily_ohlc.index).fillna(0.0)

    valid_universe_tickers_in_rets = [ticker for ticker in universe_tickers if ticker in aligned_rets_daily.columns]
    if len(valid_universe_tickers_in_rets) < len(universe_tickers):
        missing_tickers = set(universe_tickers) - set(valid_universe_tickers_in_rets)
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f"Tickers {missing_tickers} not found in aligned_rets_daily columns. Portfolio calculations might be affected.")

    if not valid_universe_tickers_in_rets:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning("No valid universe tickers found in daily returns. Gross portfolio returns will be zero.")
        daily_portfolio_returns_gross = pd.Series(0.0, index=weights_daily.index)
    else:
        daily_portfolio_returns_gross = (weights_daily[valid_universe_tickers_in_rets] * aligned_rets_daily[valid_universe_tickers_in_rets]).sum(axis=1)

    turnover = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1).fillna(0.0)

    from ..trading.transaction_costs import get_transaction_cost_model
    
    tx_cost_model = get_transaction_cost_model(global_config)
    transaction_costs, _ = tx_cost_model.calculate(
        turnover=turnover,
        weights_daily=weights_daily,
        price_data=price_data_daily_ohlc,
        portfolio_value=global_config.get("portfolio_value", 100000.0)
    )

    portfolio_rets_net = (daily_portfolio_returns_gross - transaction_costs).fillna(0.0)

    return portfolio_rets_net
