import logging
import pandas as pd

from ..portfolio.rebalancing import rebalance
from ..trading.trade_tracker import TradeTracker

logger = logging.getLogger(__name__)

def calculate_portfolio_returns(sized_signals, scenario_config, price_data_daily_ohlc, rets_daily, universe_tickers, global_config, track_trades=False):
    rebalance_frequency = scenario_config.get("timing_config", {}).get("rebalance_frequency", "M")
    weights_monthly = rebalance(
        sized_signals, rebalance_frequency
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

    from ..trading import get_transaction_cost_model
    
    tx_cost_model = get_transaction_cost_model(global_config)
    transaction_costs, _ = tx_cost_model.calculate(
        turnover=turnover,
        weights_daily=weights_daily,
        price_data=price_data_daily_ohlc,
        portfolio_value=global_config.get("portfolio_value", 100000.0)
    )

    portfolio_rets_net = (daily_portfolio_returns_gross - transaction_costs).fillna(0.0)
    
    # Initialize trade tracker if requested
    trade_tracker = None
    if track_trades:
        portfolio_value = global_config.get("portfolio_value", 100000.0)
        trade_tracker = TradeTracker(portfolio_value)
        
        # Track positions and calculate trade statistics
        _track_trades(trade_tracker, weights_daily, price_data_daily_ohlc, transaction_costs)
    
    return portfolio_rets_net, trade_tracker


def _track_trades(trade_tracker, weights_daily, price_data_daily_ohlc, transaction_costs):
    """Track trades using the trade tracker."""
    # Fallback to original implementation
    _track_trades_original(trade_tracker, weights_daily, price_data_daily_ohlc, transaction_costs)


def _track_trades_original(trade_tracker, weights_daily, price_data_daily_ohlc, transaction_costs):
    """Original trade tracking implementation (fallback)."""
    # Extract close prices
    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        close_prices = price_data_daily_ohlc.xs('Close', level='Field', axis=1)
    else:
        close_prices = price_data_daily_ohlc
    
    # Process each day
    for date in weights_daily.index:
        if date in close_prices.index:
            current_weights = weights_daily.loc[date]
            current_prices = close_prices.loc[date]
            
            # Calculate transaction cost per ticker (simplified)
            total_turnover = (weights_daily.loc[date] - weights_daily.shift(1).loc[date]).abs().sum()
            cost_per_ticker = transaction_costs.loc[date] / max(len(current_weights[current_weights != 0]), 1)
            
            # Update positions
            trade_tracker.update_positions(
                date, 
                current_weights, 
                current_prices, 
                cost_per_ticker
            )
            
            # Update MFE/MAE
            trade_tracker.update_mfe_mae(date, current_prices)
    
    # Close all positions at the end
    final_date = weights_daily.index[-1]
    final_prices = close_prices.loc[final_date] if final_date in close_prices.index else close_prices.iloc[-1]
    trade_tracker.close_all_positions(final_date, final_prices)